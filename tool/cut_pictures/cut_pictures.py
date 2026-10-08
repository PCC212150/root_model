"""按模型识别出的「检查范围(check_background)」裁剪图片：框内保留原样，框外全部涂黑。

**输出尺寸与输入完全一致**，框内像素停在原位 —— 坐标不变，所以 RSML 标注、掩码、
各类统计结果都还能直接对上，可以直接替换原图喂给训练/推理，不需要重新映射标注。
（要「只留框内那一块」的紧凑裁剪，本工具不做；那种切法坐标会偏移。）

检查范围的来源：**模型识别**，走的是和 inference.py / test.py 完全同一条流水线
（common/predict.py 的 _fit_rect：最大连通域 → 行/列覆盖量剖面拟合矩形 → 向外膨胀
CHECK_MARGIN_PX）。所以这里框出来的绿框 = overlay 图里看到的那个绿框，不会出现
「工具裁的框和推理时用的框不是同一个」这种口径分裂。

检查范围识别失败时（占图面 <5% 或 >99.9%）：**原图原样复制过去**，不涂黑也不跳过 ——
宁可交一张没裁的图，也不要交一张全黑的废图。这类图会在末尾单独列出来。

用法：
    python cut_pictures.py --dir "D:\\待裁剪图片"                      # 模型省略=最新
    python cut_pictures.py --dir "D:\\待裁剪图片" --model model_202609151347
    python cut_pictures.py --dir "D:\\待裁剪图片" --out "D:\\裁剪结果"
    python cut_pictures.py --dir "D:\\待裁剪图片" --dry-run            # 只识别框、不写图

输出：默认写到 result/{源文件夹名}_cut/（重名追加 -1、-2 …），内含
  - 裁剪后的图片（文件名与扩展名都不变）
  - cut_report.txt：逐张记录「原图 → 框坐标 → 框占图面比例 → 状态」，便于回溯

**显式给 --out 时是可续跑的**：目录已存在就直接往里写（不会再追加 -1），
**目标文件夹里已有同名的图会直接跳过、连推理都不跑**。所以第一次跑一半中断了、
或者后来往源文件夹里补了新图，再跑一次就只处理缺的那些，几百张图的批不会白跑第二遍。
（不加 --out 走默认路径时仍按项目规范追加 -1，不受影响。）

注意：输出图沿用**原扩展名**。源图是 jpg 的话，黑边与框边会被重新压缩一次
（想要无损就用 png 源图，或后续统一转 png）。
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import device as device_mod, ckpt, image_io, naming, predict  # noqa: E402
from PIL import Image  # noqa: E402

PROGRESS_EVERY = 10      # 每处理这么多张报一次进度


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="用模型识别出的检查范围裁剪图片：框内保留、框外涂黑，输出尺寸与输入一致",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python cut_pictures.py --dir "D:\\待裁剪图片"\n'
            '  python cut_pictures.py --dir "D:\\待裁剪图片" --model model_202609151347\n'
            '  python cut_pictures.py --dir "D:\\待裁剪图片" --out "D:\\裁剪结果" --dry-run\n'
        ),
    )
    p.add_argument("--dir", required=True,
                   help="待裁剪的图片文件夹（只处理这一层，不递归）")
    p.add_argument("--out", type=Path, default=None,
                   help="输出文件夹。默认 result/{源文件夹名}_cut")
    p.add_argument("--model", default=None,
                   help="模型文件夹名（可省 model_ 前缀）；省略=取 model/ 下最新的")
    p.add_argument("--size", type=int, default=None,
                   help="模型输入长边；默认用模型训练时的设置（从权重里读）")
    p.add_argument("--dry-run", action="store_true",
                   help="只识别检查范围并打印，不写任何图片")
    p.add_argument("--gpu", type=int, default=None,
                   help="用哪张 GPU（默认自动挑当前最空的一张）；等价于环境变量 ROOT_MODEL_GPU")

    p.add_argument("--cpu", action="store_true", help="强制用 CPU")
    return p.parse_args(argv)


def cut_one(img: np.ndarray, box) -> np.ndarray:
    """把 box 之外涂黑，返回与原图同尺寸的新数组。

    box = (x0, y0, x1, y1)，右/下为开区间（predict 返回的已经是原图坐标、且含 margin）。
    """
    out = np.zeros_like(img)                      # 一次分配全黑画布
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]         # 框内原样搬过来
    return out


def main():
    args = parse_args()
    img_dir = Path(args.dir)
    if not img_dir.is_dir():
        sys.exit(f"[错误] 图片文件夹不存在: {img_dir}")

    import torch
    device = device_mod.pick(cpu=args.cpu, gpu=getattr(args, "gpu", None))
    pths, names = ckpt.resolve_pths(args.model)
    model, metas = ckpt.load_models(pths, device)
    meta = metas[0]
    tag = f"集成 {len(names)} 个" if len(names) > 1 else "模型"
    print(f"{tag}: {' + '.join(names)} | 设备: {device}")
    size = args.size or meta.get("size") or config.MAX_SIDE
    if args.size is None and meta.get("size"):
        print(f"输入长边 {size}（用模型训练时的设置）")
    elif meta.get("size") and args.size != meta.get("size"):
        print(f"[警告] 输入长边 {args.size} 与模型训练时（{meta['size']}）不一致："
              f"尺度不匹配会让检查范围识别变差")

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS)
    if not imgs:
        sys.exit(f"[错误] 文件夹里没有图片(支持 {sorted(config.IMAGE_EXTS)})")
    stems = [p.stem for p in imgs]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        sys.exit(f"[错误] 文件夹里有同名不同扩展的图片 {dup}，输出会互相覆盖，"
                 f"请先改名或分开处理")
    print(f"图片 {len(imgs)} 张: {img_dir}")

    if args.dry_run:
        out_dir = None
        print("[dry-run] 不写任何文件\n")
    elif args.out is not None:
        # 显式给了 --out：目录存在就**直接续用**（不追加 -1），已存在的同名图跳过。
        # 这样中断了可以接着跑，补了新图再跑也只处理缺的那些。
        out_dir = Path(args.out)
        existed = out_dir.exists()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {out_dir}"
              + ("（已存在，续用；同名文件跳过）" if existed else "") + "\n")
    else:
        # 没给 --out：走默认路径，仍按项目规范追加 -1（默认路径是自动推出来的，
        # 复用容易把不同源文件夹的结果混在一起）
        out_dir = config.RESULT_DIR / f"{img_dir.name}_cut"
        out_dir = naming.create_unique_dir(out_dir.parent, out_dir.name)
        print(f"输出目录: {out_dir}\n")

    t0 = time.time()
    report, failed, skipped = [], [], []
    for k, p in enumerate(imgs, 1):
        # 目标文件已存在 → 跳过（**在推理之前跳过**，所以重跑几乎不花时间）
        if out_dir is not None and (out_dir / p.name).exists():
            skipped.append(p.name)
            report.append(f"{p.name}\t-\t-\t-\t已存在，跳过")
            continue
        img = image_io.load_rgb(p)
        h0, w0 = img.shape[:2]
        res = predict.predict(model, img, max_side=size, stride=config.STRIDE,
                              device=device, low_thresh=config.PRED_LOW_THRESHOLD)
        box, ok = res["check_box"], res["check_ok"]
        if ok:
            x0, y0, x1, y1 = box
            ratio = ((x1 - x0) * (y1 - y0)) / float(w0 * h0)
            status = "已裁剪"
            line = (f"{p.name}\t框=({x0},{y0})-({x1},{y1})\t"
                    f"{x1 - x0}x{y1 - y0}\t占图面 {ratio:.1%}\t{status}")
        else:
            ratio = float("nan")
            status = "检查范围未识别，原图复制"
            line = f"{p.name}\t框=无\t-\t-\t{status}"
            failed.append(p.name)
        report.append(line)

        if out_dir is not None:
            dst = out_dir / p.name
            if ok:
                Image.fromarray(cut_one(img, box)).save(dst)
            else:
                shutil.copy2(p, dst)              # 不涂黑也不跳过：原样交出去
        if k % PROGRESS_EVERY == 0 or k == len(imgs):
            print(f"  [{k}/{len(imgs)}] {p.name}: {status}"
                  + (f" 框占图面 {ratio:.1%}" if ok else ""))

    dt = time.time() - t0
    done = len(imgs) - len(skipped)
    print(f"\n共 {len(imgs)} 张：本次处理 {done} 张"
          + (f"，跳过已存在 {len(skipped)} 张" if skipped else "")
          + f"，耗时 {dt:.1f}s（{dt / max(done, 1):.2f}s/张）")
    if failed:
        print(f"[注意] {len(failed)} 张没识别出检查范围，已原样复制（没涂黑）：")
        for n in failed[:10]:
            print(f"    {n}")
        if len(failed) > 10:
            print(f"    …共 {len(failed)} 张，见 cut_report.txt")
    if out_dir is not None:
        rpt = out_dir / "cut_report.txt"
        head = (f"# 按检查范围裁剪报告  模型: {' + '.join(names)}  输入长边 {size}\n"
                f"# 框外涂黑，输出尺寸与原图一致（坐标不变）\n"
                f"# 本次 {len(imgs)} 张：处理 {done}，跳过已存在 {len(skipped)}，"
                f"未识别出检查范围 {len(failed)}（原图复制）\n")
        if rpt.exists():        # 续跑：追加，不覆盖上一次的记录
            with open(rpt, "a", encoding="utf-8") as f:
                f.write(f"\n# ---- {time.strftime('%Y-%m-%d %H:%M:%S')} 再次运行 ----\n")
                f.write(head + "\n".join(report) + "\n")
        else:
            rpt.write_text(head + "\n".join(report) + "\n", encoding="utf-8")
        print(f"结果目录: {out_dir}\n报告文件: {rpt}")
    else:
        print("\n".join(report))


if __name__ == "__main__":
    main()
