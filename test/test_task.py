"""评测**单通道任务模型**（--task root / stem / check_background）。

`test/test.py` 是给三通道联合模型用的；单任务模型输出只有 1 通道，通道含义由
checkpoint 里的 `task` 决定。本脚本只评测该通道：

- root：Dice / IoU / clDice，以及根数/总长（预测与 GT 都限定在 **GT 检查范围**；
  部署时这个范围应由单独的 check_background 模型提供，评测时用 GT 框即可）；
- stem：Dice / IoU / clDice；
- check_background：Dice / IoU / 像素准确率。

用法：

    python test/test_task.py \
        --model model_202610242351-stem \
        --data-dir datasets/root/test/GT

`--model` 可以是 `model/` 下的文件夹名（可省 `model_` 前缀），也可以是绝对路径。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, device as device_mod, image_io, metrics  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM,  # noqa: E402
                            build_target_masks, discover_pairs, load_annot)
from common.skeleton_stats import analyze_mask_ex  # noqa: E402

TASK_TO_CH = {"root": CH_ROOT, "stem": CH_STEM,
              "check_background": CH_CHECK}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="单任务模型文件夹/权重（如 model/model_2026...-stem）")
    p.add_argument("--data-dir", type=Path, default=config.TEST_DATA_DIR / "GT")
    p.add_argument("--size", type=int, default=None,
                   help="覆盖输入长边；切片模型这里是块边长（默认用权重里的设置）")
    p.add_argument("--mask-width", type=float, default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="CSV 输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    pths, names = ckpt.resolve_pths(args.model)
    if len(pths) != 1:
        sys.exit("[错误] 单任务评测只接受一个模型")
    device = device_mod.pick(cpu=args.cpu, gpu=args.gpu, need_gb=2.0)
    model, meta = ckpt.load_unet(pths[0], device=device)
    task = meta.get("task", "all")
    if task not in TASK_TO_CH:
        sys.exit(f"[错误] 这个权重不是单通道任务模型（task={task!r}）；"
                 f"三通道模型请用 test/test.py")
    ch = TASK_TO_CH[task]
    mask_width = config.MASK_LINE_WIDTH if args.mask_width is None else args.mask_width

    tile = ckpt.infer_tile([meta], args.size)
    if tile:
        size = tile
        print(f"[切片模型] 原始分辨率滑窗，块边长 {tile}")
    else:
        size = args.size or meta.get("size") or config.MAX_SIDE
        print(f"输入长边 {size}（模型 task={task}）")

    pairs = discover_pairs(args.data_dir)
    if not pairs:
        sys.exit(f"[错误] {args.data_dir} 下没有配对数据")

    rows = []
    dices, ious, cldices = [], [], []
    gt_counts, pred_counts, gt_totals, pred_totals = [], [], [], []
    t0 = time.time()
    for name, img_path, _ in pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        annot = load_annot(args.data_dir, name, (w0, h0), verbose=False)
        gt_masks, valid = build_target_masks(annot, (w0, h0), mask_width)
        if valid[ch] < 0.5:
            print(f"[跳过] {name} 没有 {task} 标注")
            continue
        gt = np.ascontiguousarray(gt_masks[:, :, ch])

        if tile:
            from common import predict
            prob = predict.tiled_probs(model, img, tile, device=device)
            pred = prob.numpy()[0, 0] > 0.5
        else:
            w1, h1 = image_io.target_size(w0, h0, size, config.STRIDE)
            x = image_io.to_model_input(image_io.resize_rgb(img, w1, h1)).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(x))
            pred = image_io.prob_to_orig_mask(
                prob, w0, h0, threshold=0.5, channel=0)

        bm = metrics.binary_metrics(pred, gt)
        cld = metrics.cldice(pred, gt) if task in ("root", "stem") else np.nan
        dices.append(bm["dice"]); ious.append(bm["iou"])
        if not np.isnan(cld):
            cldices.append(cld)

        row = [name, task, f"{bm['dice']:.4f}", f"{bm['iou']:.4f}",
               f"{bm['accuracy']:.4f}",
               f"{cld:.4f}" if not np.isnan(cld) else "-"]
        if task == "root":
            gt_root = gt
            pred_root = pred
            if valid[CH_CHECK] > 0:
                roi = np.ascontiguousarray(gt_masks[:, :, CH_CHECK])
                gt_root = gt_root & roi
                pred_root = pred_root & roi
            gs = analyze_mask_ex(gt_root, spur=config.PRED_SPUR_LENGTH,
                                 min_len=config.MIN_ROOT_LENGTH)
            ps = analyze_mask_ex(pred_root, spur=config.PRED_SPUR_LENGTH,
                                 min_len=config.MIN_ROOT_LENGTH)
            gt_counts.append(gs["count"]); pred_counts.append(ps["count"])
            gt_totals.append(gs["total"]); pred_totals.append(ps["total"])
            row += [gs["count"], ps["count"],
                    f"{gs['total']:.1f}", f"{ps['total']:.1f}"]
        rows.append(row)
        print(f"[{name}] {task} Dice={bm['dice']:.4f} clDice={cld:.3f}"
              if task in ("root", "stem")
              else f"[{name}] {task} Dice={bm['dice']:.4f}")

    if not rows:
        sys.exit("[错误] 没有可评测的样本")

    out = args.out
    if out is None:
        out = pths[0].parent / f"{pths[0].parent.name}_task_test.csv"
    header = ["图片名", "任务", "Dice", "IoU", "像素准确率", "clDice"]
    if task == "root":
        header += ["GT根数", "预测根数", "GT总长(px)", "预测总长(px)"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"{task}: Dice {np.mean(dices):.4f} | IoU {np.mean(ious):.4f}"
          + (f" | clDice {np.mean(cldices):.4f}" if cldices else ""))
    if task == "root":
        print(f"根数: GT {np.mean(gt_counts):.1f} vs 预测 {np.mean(pred_counts):.1f} "
              f"| MAE {np.mean(np.abs(np.array(gt_counts) - np.array(pred_counts))):.2f}")
        print(f"总长: GT {np.mean(gt_totals):.0f} vs 预测 {np.mean(pred_totals):.0f} "
              f"| MAE {np.mean(np.abs(np.array(gt_totals) - np.array(pred_totals))):.0f} px")
    print(f"耗时 {time.time() - t0:.1f}s | 结果已保存: {out}")


if __name__ == "__main__":
    main()
