"""单通道任务模型的推理：root / stem / check_background 分开预测。

单任务模型只输出 1 通道，不能直接用 `inference.py`（那是三通道联合模型）。
本脚本把「任务模型 + 可选的 check_background 模型」拼成一个三通道包装器，
复用 `common/predict.predict` 的检查范围/ROI/根统计逻辑：

    root  + check 模型 -> root 概率在通道 0，check 在通道 2 -> ROI + 根统计
    stem  + check 模型 -> stem 概率在通道 1，check 在通道 2 -> 茎掩码
    check 单独         -> check 概率在通道 2 -> 检查范围框

用法：

    # 只跑 check
    python inference_task.py --model model_202610250930-check_background \
        --dir inference_pictures/某批图

    # 跑 root，并用 check 模型提供检查范围
    python inference_task.py --model model_202610250930 \
        --check-model model_202610250930-check_background \
        --dir inference_pictures/某批图

    # stem
    python inference_task.py --model model_202610250930-stem \
        --check-model model_202610250930-check_background \
        --dir inference_pictures/某批图
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, device as device_mod, image_io, naming, predict  # noqa: E402
from common.dataset import CH_CHECK, CH_ROOT, CH_STEM  # noqa: E402
from common.skeleton_stats import analyze_mask_ex  # noqa: E402

TASK_TO_CH = {"root": CH_ROOT, "stem": CH_STEM,
              "check_background": CH_CHECK}
COLORS = {"root": (255, 0, 0), "stem": (255, 165, 0),
          "check_background": (0, 255, 0)}


class TaskWithCheck(torch.nn.Module):
    """把 1 通道任务模型 + 可选 check 模型拼成 predict.predict 需要的 3 通道输出。"""

    def __init__(self, task_model, task_ch, check_model=None):
        super().__init__()
        self.task_model = task_model
        self.task_ch = task_ch
        self.check_model = check_model

    def forward(self, x):
        y = self.task_model(x)
        out = x.new_zeros(y.shape[0], 3, *y.shape[2:])
        out[:, self.task_ch] = y[:, 0]
        if self.check_model is not None:
            out[:, CH_CHECK] = self.check_model(x)[:, 0]
        return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="单任务模型文件夹/权重（model/ 下的名字或绝对路径）")
    p.add_argument("--check-model", default=None,
                   help="check_background 模型；root/stem 用它提供 ROI。"
                        "不给她则 root/stem 按全图统计/预测")
    p.add_argument("--dir", required=True, type=Path, help="输入图片文件夹")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="输出目录（默认 result/<图片文件夹名>_<task>）")
    p.add_argument("--size", type=int, default=None,
                   help="覆盖输入长边；切片模型这里是块边长（默认用权重里的设置）")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-mask", action="store_true",
                   help="额外保存 <名>_<task>_mask.png（0/255）")
    p.add_argument("--overlay-jpg", action="store_true",
                   help="overlay 存 JPEG（默认 PNG）")
    return p.parse_args()


def _overlay(img, mask, task, check_box, alpha=0.45):
    out = img.copy()
    cols = np.arange(out.shape[2])
    color = COLORS[task]
    if mask is not None and mask.any():
        lut = np.array(color, dtype=np.float32)
        blended = (out[mask].astype(np.float32) * (1 - alpha) + lut * alpha)
        out[mask] = blended.astype(np.uint8)
    if check_box is not None:
        h, w = out.shape[:2]
        im = Image.fromarray(out)
        ImageDraw.Draw(im).rectangle(
            [check_box[0], check_box[1], check_box[2] - 1, check_box[3] - 1],
            outline=(0, 255, 0), width=max(2, int(min(w, h) * 0.004)))
        out = np.asarray(im)
    return out


def main():
    args = parse_args()
    if not args.dir.is_dir():
        sys.exit(f"[错误] 图片文件夹不存在: {args.dir}")

    device = device_mod.pick(cpu=args.cpu, gpu=args.gpu, need_gb=6.0)
    pths, names = ckpt.resolve_pths(args.model)
    if len(pths) != 1:
        sys.exit("[错误] 单任务推理只接受一个 --model")
    task_model, meta = ckpt.load_unet(pths[0], device=device)
    task = meta.get("task", "all")
    if task not in TASK_TO_CH:
        sys.exit(f"[错误] {pths[0].parent.name} 是 {task} 模型；单任务推理要求 "
                 f"--task root/stem/check_background")
    task_ch = TASK_TO_CH[task]
    check_model = None
    check_meta = None
    if args.check_model:
        cpths, _ = ckpt.resolve_pths(args.check_model)
        if len(cpths) != 1:
            sys.exit("[错误] --check-model 只接受一个模型")
        check_model, check_meta = ckpt.load_unet(cpths[0], device=device)
        if check_meta.get("task") != "check_background":
            sys.exit(f"[错误] --check-model 必须是 check_background 模型，"
                     f"当前 task={check_meta.get('task')}")
    elif task in ("root", "stem"):
        print("[警告] 没有 --check-model：root/stem 不做检查范围限定，"
              "按整图预测/统计")

    # 尺寸/切片必须一致，否则 wrapper 里两个模型看到不同网格
    metas = [meta] + ([check_meta] if check_meta else [])
    sizes = {m.get("size") for m in metas if m.get("size")}
    crops = {m.get("crop") or 0 for m in metas}
    if len(sizes) > 1 or len(crops) > 1:
        sys.exit(f"[错误] 任务模型与 check 模型的训练尺度不一致："
                 f"size={sorted(sizes)}, crop={sorted(crops)}")
    tile = ckpt.infer_tile(metas, args.size)
    if tile:
        size = tile
        print(f"[切片模型] 原始分辨率滑窗，块边长 {tile}")
    else:
        size = args.size or meta.get("size") or config.MAX_SIDE
        print(f"输入长边 {size}")

    model = TaskWithCheck(task_model, task_ch, check_model)
    model.eval()

    imgs = sorted(p for p in args.dir.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS)
    if not imgs:
        sys.exit(f"[错误] {args.dir} 里没有图片")
    out_dir = args.out_dir or naming.create_unique_dir(
        config.RESULT_DIR, f"{args.dir.name}_{task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dir.name}_{task}.csv"
    overlay_fmt = "jpg" if args.overlay_jpg else "png"

    rows = []
    t0 = time.time()
    for p in imgs:
        img = image_io.load_rgb(p)
        h0, w0 = img.shape[:2]
        res = predict.predict(model, img, max_side=size, stride=config.STRIDE,
                              device=device, tile=tile)
        check_box = res["check_box"] if res["check_ok"] else None

        if task == "root":
            mask = res["mask_counted"]
            st = analyze_mask_ex(mask, spur=config.PRED_SPUR_LENGTH,
                                 min_len=config.MIN_ROOT_LENGTH)
            row = [p.name, task, int(mask.sum()), st["count"],
                   f"{st['total']:.1f}",
                   f"{st['total'] / st['count']:.1f}" if st["count"] else "0.0"]
            desc = (f"root 根数 {st['count']} 总长 {st['total']:.0f}px"
                    f" ROI={'是' if check_box else '否'}")
        elif task == "stem":
            mask = res["masks"][CH_STEM]
            row = [p.name, task, int(mask.sum())]
            if check_box is not None:
                roi = np.zeros((h0, w0), dtype=bool)
                roi[check_box[1]:check_box[3], check_box[0]:check_box[2]] = True
                mask = mask & roi
                row[2] = int(mask.sum())
            desc = f"stem 面积 {row[2]}px² ROI={'是' if check_box else '否'}"
        else:
            # check 任务的输出就是检查范围：优先用拟合的矩形
            if check_box is not None:
                mask = np.zeros((h0, w0), dtype=bool)
                mask[check_box[1]:check_box[3], check_box[0]:check_box[2]] = True
            else:
                prob = res["probs"][CH_CHECK] > 0.5
                mask = image_io.resize_bool_mask(prob, w0, h0)
            row = [p.name, task, int(mask.sum())]
            if check_box is not None:
                row += list(check_box)
            else:
                row += ["", "", "", ""]
            desc = f"check 面积 {row[2]}px² box={check_box}"

        ov = _overlay(img, mask, task, check_box)
        ov_path = out_dir / f"{p.stem}_{task}_overlay.{overlay_fmt}"
        Image.fromarray(ov).save(ov_path, quality=90) if overlay_fmt == "jpg" \
            else Image.fromarray(ov).save(ov_path)
        if args.save_mask:
            Image.fromarray((mask.astype(np.uint8) * 255)).save(
                out_dir / f"{p.stem}_{task}_mask.png", compress_level=1)

        rows.append(row)
        print(f"[{p.name}] {desc}")

    header = {
        "root": ["图片名", "任务", "根面积(px²)", "根数", "总长(px)", "平均根长(px)"],
        "stem": ["图片名", "任务", "茎面积(px²)"],
        "check_background": ["图片名", "任务", "检查范围面积(px²)",
                             "x0", "y0", "x1", "y1"],
    }[task]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)

    print(f"\n{task} 推理完成：{len(imgs)} 张，用时 {time.time() - t0:.1f}s")
    print(f"结果目录: {out_dir}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
