"""评测 nnU-Net 的预测结果（三个二分类数据集分别输出 root/stem/check）。

nnU-Net 的 `nnUNetv2_predict` 输出 `preds/<case>.png`（0/1 标签图）。本脚本按
`case_map.json` 把 case 映回原图名，用项目自己的 `build_target_masks` 重新生成
三通道 GT（与训练/测试口径一致），然后算：

- 每通道 Dice / IoU / 像素准确率；
- root、stem 的 clDice；
- root 的根数/总长（与 test.py 一样：预测与 GT 都限定在 GT 检查范围内，
  走 `skeleton_stats.analyze_mask_ex` 的骨架→分链口径）。

用法：

    python tool/nnunet/eval_predictions.py \
        --pred-root  nnUNet_results/preds_root \
        --pred-stem  nnUNet_results/preds_stem \
        --pred-check nnUNet_results/preds_check \
        --case-map   nnUNet_raw/Dataset501_Root/case_map.json \
        --data-dir   datasets/root/test/GT \
        --out        nnUNet_results/nnunet_test_summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import image_io, metrics  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM,  # noqa: E402
                            build_target_masks, discover_pairs, load_annot)
from common.skeleton_stats import analyze_mask_ex  # noqa: E402

CHANNELS = (("root", CH_ROOT), ("stem", CH_STEM), ("check", CH_CHECK))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--pred-stem", type=Path, default=None)
    p.add_argument("--pred-check", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=config.TEST_DATA_DIR / "GT")
    p.add_argument("--case-map", required=True, type=Path,
                   help="export_dataset.py 生成的 case_map.json（任一数据集即可）")
    p.add_argument("--mask-width", type=float, default=config.MASK_LINE_WIDTH)
    p.add_argument("--out", type=Path, default=None, help="CSV 输出路径")
    return p.parse_args()


def _load_pred(path, shape):
    if not path.exists():
        return None
    # nnU-Net 输出/项目导出都用 label id 0/1（不是 0/255），所以按 >0 判定
    arr = np.asarray(Image.open(path).convert("L")) > 0
    h, w = shape
    if arr.shape != (h, w):
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8) * 255).resize(
            (w, h), Image.NEAREST)) > 127
    return arr


def main():
    args = parse_args()
    mapping = json.loads(args.case_map.read_text(encoding="utf-8"))["test"]
    pred_dirs = {"root": args.pred_root, "stem": args.pred_stem,
                 "check": args.pred_check}
    pairs = {name: path for name, path, _ in discover_pairs(args.data_dir)}

    rows = []
    for case, name in mapping.items():
        if name not in pairs:
            print(f"[警告] 测试集里找不到 {name}，跳过")
            continue
        img = image_io.load_rgb(pairs[name])
        h0, w0 = img.shape[:2]
        annot = load_annot(args.data_dir, name, (w0, h0), verbose=False)
        gt_masks, valid = build_target_masks(annot, (w0, h0), args.mask_width)

        row = {"name": name}
        for ch_name, c in CHANNELS:
            pred_dir = pred_dirs[ch_name]
            if pred_dir is None:
                continue
            pred = _load_pred(pred_dir / f"{case}.png", (h0, w0))
            if pred is None:
                print(f"[警告] {pred_dir} 里没有 {case}.png，跳过 {ch_name}")
                continue
            gt = np.ascontiguousarray(gt_masks[:, :, c])
            if valid[c] < 0.5:
                continue
            bm = metrics.binary_metrics(pred, gt)
            row[f"dice_{ch_name}"] = bm["dice"]
            row[f"iou_{ch_name}"] = bm["iou"]
            row[f"acc_{ch_name}"] = bm["accuracy"]
            if ch_name in ("root", "stem"):
                row[f"cldice_{ch_name}"] = metrics.cldice(pred, gt)

        # root 根数/总长：预测和 GT 都限定在 GT 检查范围内（与 test.py 同口径）
        if "root" in pred_dirs and pred_dirs["root"] is not None:
            pred = _load_pred(pred_dirs["root"] / f"{case}.png", (h0, w0))
            if pred is not None:
                gt_root = np.ascontiguousarray(gt_masks[:, :, CH_ROOT])
                if valid[CH_CHECK] > 0:
                    roi = np.ascontiguousarray(gt_masks[:, :, CH_CHECK])
                    pred = pred & roi
                    gt_root = gt_root & roi
                gt_st = analyze_mask_ex(gt_root, spur=config.PRED_SPUR_LENGTH,
                                        min_len=config.MIN_ROOT_LENGTH)
                pr_st = analyze_mask_ex(pred, spur=config.PRED_SPUR_LENGTH,
                                        min_len=config.MIN_ROOT_LENGTH)
                row.update({"gt_count": gt_st["count"],
                            "pred_count": pr_st["count"],
                            "gt_total": gt_st["total"],
                            "pred_total": pr_st["total"]})
        rows.append(row)

    if not rows:
        sys.exit("[错误] 没有任何可评测的样本")

    keys = []
    for ch_name, _ in CHANNELS:
        for k in (f"dice_{ch_name}", f"iou_{ch_name}", f"acc_{ch_name}",
                  f"cldice_{ch_name}"):
            if any(k in r for r in rows):
                keys.append(k)
    for k in ("gt_count", "pred_count", "gt_total", "pred_total"):
        if any(k in r for r in rows):
            keys.append(k)
    header = ["图片名"] + keys

    out = args.out
    if out is None:
        out = args.case_map.parent.parent / "nnunet_test_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for r in rows:
            wr.writerow([r["name"]] + [r.get(k, "") for k in keys])

    print(f"\n{'=' * 70}\n{len(rows)} 张测试图的平均指标")
    for ch_name, _ in CHANNELS:
        dk = f"dice_{ch_name}"
        if dk not in keys:
            continue
        dice = metrics.nanmean([r.get(dk, np.nan) for r in rows])
        iou = metrics.nanmean([r.get(f"iou_{ch_name}", np.nan) for r in rows])
        print(f"{ch_name:>6}: Dice {dice:.4f} | IoU {iou:.4f}", end="")
        ck = f"cldice_{ch_name}"
        if ck in keys:
            print(f" | clDice {metrics.nanmean([r.get(ck, np.nan) for r in rows]):.4f}",
                  end="")
        print()
    if "gt_count" in keys:
        gc = np.array([r.get("gt_count", np.nan) for r in rows], dtype=float)
        pc = np.array([r.get("pred_count", np.nan) for r in rows], dtype=float)
        gl = np.array([r.get("gt_total", np.nan) for r in rows], dtype=float)
        pl = np.array([r.get("pred_total", np.nan) for r in rows], dtype=float)
        print(f"根数: GT {np.nanmean(gc):.1f} vs 预测 {np.nanmean(pc):.1f} "
              f"| MAE {np.nanmean(np.abs(gc - pc)):.2f}")
        print(f"总长: GT {np.nanmean(gl):.0f} vs 预测 {np.nanmean(pl):.0f} "
              f"| MAE {np.nanmean(np.abs(gl - pl)):.0f} px")
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
