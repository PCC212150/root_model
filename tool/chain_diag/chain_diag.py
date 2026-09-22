"""误差分解诊断：把「总根长 / 根数」的误差拆成 **测量链路** 与 **模型** 两部分。

为什么要这个：`test.py` 只给「预测 vs 真值」一个数，回答不了「这误差是模型的锅，
还是测量链路的锅」。同一个数上，换算法、加分辨率、改损失，可能全是在给一把错了的
尺子调刻度。本工具在同一批图上同时算四条数，逐图对比：

    A  参考       RSML 折线的欧氏长度（根数 = 折线条数）—— 唯一的真值口径
    B0 纯链路     GT 折线按 MASK_LINE_WIDTH 画成掩码 -> 骨架 -> 分链，
                  **不剪枝、不过滤、不归一、不做 ROI/锚定**
                  A→B0 的差 = 「画粗再骨架化」本身损失掉的长度
    B1 完美模型   GT 掩码 ∩ GT 检查框 -> 骨架 -> 分链 -> 起点锚定到 GT 茎，
                  完全按部署口径。**这是整个系统在「掩码完美」时的上限**
    C  当前系统   模型预测 -> 与 B1 同一套部署口径

于是有：
    B0/A-1    链路固有偏差 —— 与模型无关，换模型也修不掉
    B1/A-1    系统上限 —— 模型完美时还剩多少误差
    C/A-1     当前实际
    C-B1      模型的真实贡献（这才是「换模型/加分辨率/改损失」能动的部分）

如果 B1 的误差远小于 C，说明**瓶颈在模型**，继续调链路是白费；
如果 B1 自己就偏很多，那**先修链路**，否则后面所有改进都在给错尺子调刻度。

用法（项目根目录下，pcc 环境）：
    python tool\\chain_diag\\chain_diag.py --dir datasets\\root\\test
    python tool\\chain_diag\\chain_diag.py --dir datasets\\root\\test --model model_202609192043
    python tool\\chain_diag\\chain_diag.py --dir datasets\\root\\test --no-model   # 只量链路
    python tool\\chain_diag\\chain_diag.py --dir datasets\\root\\train --limit 5
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming, predict  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM, build_target_masks,  # noqa: E402
                            discover_pairs, find_other)
from common.rsml_parse import parse_rsml, root_stats  # noqa: E402
from common.skeleton_stats import (analyze_mask_anchored,  # noqa: E402
                                   analyze_mask_ex)


def parse_args():
    p = argparse.ArgumentParser(description="把根长/根数误差拆成 测量链路 与 模型")
    p.add_argument("--dir", type=Path, required=True,
                   help="数据集目录（含 images/ 与 labels/roots/），路径含空格要加引号")
    p.add_argument("--model", default=None, help="模型文件夹名（可省 model_ 前缀）；缺省用最新")
    p.add_argument("--size", type=int, default=None,
                   help="模型输入长边；缺省用模型训练时的 --size（自动从权重读）")
    p.add_argument("--mask-width", type=float, default=config.MASK_LINE_WIDTH,
                   help=f"GT 折线画掩码的线宽(px，原图尺度)，默认 config 的 "
                        f"{config.MASK_LINE_WIDTH}")
    p.add_argument("--spur", type=float, default=config.PRED_SPUR_LENGTH,
                   help="B1/C 的骨架剪枝阈值(px)，默认 config.PRED_SPUR_LENGTH")
    p.add_argument("--min-len", type=float, default=config.MIN_ROOT_LENGTH,
                   help="B1/C 的最短根长(px)，默认 config.MIN_ROOT_LENGTH")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张（按文件名排序），0=全部")
    p.add_argument("--no-model", action="store_true",
                   help="不算 C（不加载模型），只量 A / B0 / B1 三条链路")
    p.add_argument("--out", type=Path, default=config.RESULT_DIR / "chain_diag",
                   help="结果目录（重名自动加 -1）")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _rel(v, base):
    """(v/base - 1)*100；base 为 0 时返回 nan。"""
    return (v / base - 1.0) * 100.0 if base else float("nan")


def _chain(mask, stem_mask, spur, min_len, anchor):
    """按部署口径跑一遍「掩码 -> 骨架 -> 分链 -> (锚定)」，返回 (根数, 总长)。

    anchor=False 时不锚定（B0 用）；anchor=True 时要求 stem_mask 给出来。
    两者都走 common.skeleton_stats 的同一实现，与 test.py / inference.py 无异。
    """
    if mask is None or not mask.any():
        return 0, 0.0
    if anchor and stem_mask is not None and stem_mask.any():
        st = analyze_mask_anchored(
            mask, stem_mask, spur=spur, min_len=min_len,
            factor=config.STEM_ANCHOR_FACTOR, min_px=config.STEM_ANCHOR_MIN_PX,
            max_px=config.STEM_ANCHOR_MAX_PX)
    else:
        st = analyze_mask_ex(mask, spur=spur, min_len=min_len, with_paths=False,
                             normalize_count=True)
    return st["count"], float(st["total"])


def main():
    args = parse_args()
    pairs = discover_pairs(args.dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit(f"[错误] {args.dir} 下没有「图片 + rsml」配对数据")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = folder = None
    size = args.size or config.MAX_SIDE
    if not args.no_model:
        pths, names = ckpt.resolve_pths(args.model)
        model, metas = ckpt.load_models(pths, device)
        for m in (model if isinstance(model, list) else [model]):
            m.eval()
        folder = Path(pths[0]).parent
        size = args.size or metas[0].get("size") or config.MAX_SIDE
        print(f"模型: {folder.name} | 设备: {device} | 输入长边 {size}")
    else:
        print("只量链路（--no-model）：不算 C，不加载模型")

    print(f"数据集: {args.dir} | 图片 {len(pairs)} 张 | 线宽 {args.mask_width:g}px | "
          f"B1/C 剪枝 {args.spur:g}px 最短根 {args.min_len:g}px\n")
    print(f"{'图片':<26} {'A参考':>9} {'B0纯链路':>9} {'B1完美':>9} {'C系统':>9}   "
          f"{'B0/A':>7} {'B1/A':>7} {'C/A':>7}   根数 A/B1/C")
    print("-" * 110)

    rows = []
    t0 = time.time()
    for i, (name, img_path, rsml_path) in enumerate(pairs, 1):
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        roots = parse_rsml(rsml_path)
        gt_n, _gt_lens, gt_total = root_stats(roots)

        # GT 三通道掩码（原图分辨率；与训练/评测同一实现）
        gt_masks, _valid = build_target_masks(
            rsml_path, find_other(args.dir, name), (w0, h0), (w0, h0),
            args.mask_width)
        gt_root = np.ascontiguousarray(gt_masks[:, :, CH_ROOT])
        gt_stem = np.ascontiguousarray(gt_masks[:, :, CH_STEM])
        gt_check = np.ascontiguousarray(gt_masks[:, :, CH_CHECK])

        # B0：纯链路 —— 不剪枝、不过滤、不归一、无 ROI、无锚定
        b0_n, b0_total = _chain(gt_root, None, 0.0, 0.0, anchor=False)
        # B1：完美模型 —— GT 根 ∩ GT 检查框，锚定到 GT 茎，部署口径
        b1_n, b1_total = _chain(gt_root & gt_check, gt_stem,
                                args.spur, args.min_len, anchor=True)

        if args.no_model:
            c_n, c_total = 0, float("nan")
        else:
            res = predict.predict(model, img, max_side=size, stride=config.STRIDE,
                                  device=device,
                                  low_thresh=config.PRED_LOW_THRESHOLD)
            c_n, c_total = _chain(res["mask_counted"], res["masks"][CH_STEM],
                                  args.spur, args.min_len, anchor=True)

        e0, e1, ec = _rel(b0_total, gt_total), _rel(b1_total, gt_total), \
            _rel(c_total, gt_total)
        fmt = "{:>7.1f}%" if not args.no_model else "{:>8.1f}"
        print(f"{name:<26} {gt_total:>9.0f} {b0_total:>9.0f} {b1_total:>9.0f} "
              f"{(c_total if not args.no_model else float('nan')):>9.0f}   "
              + fmt.format(e0) + " " + fmt.format(e1) + " "
              + (fmt.format(ec) if not args.no_model else "       -")
              + f"   {gt_n}/{b1_n}/{(c_n if not args.no_model else '-')}")

        rows.append({"name": name, "gt_n": gt_n, "gt_len": gt_total,
                     "b0_n": b0_n, "b0_len": b0_total,
                     "b1_n": b1_n, "b1_len": b1_total,
                     "c_n": c_n, "c_len": c_total})

    el = time.time() - t0
    g = np.array([r["gt_len"] for r in rows], dtype=float)
    b0 = np.array([r["b0_len"] for r in rows], dtype=float)
    b1 = np.array([r["b1_len"] for r in rows], dtype=float)
    gn = np.array([r["gt_n"] for r in rows], dtype=float)
    b1n = np.array([r["b1_n"] for r in rows], dtype=float)

    print("-" * 110)
    print(f"\n=== 汇总（{len(rows)} 张，均值）===")
    print(f"  总长 参考 {g.mean():.0f}px | 纯链路 {b0.mean():.0f}px "
          f"({_rel(b0.mean(), g.mean()):+.1f}%) | "
          f"完美模型 {b1.mean():.0f}px ({_rel(b1.mean(), g.mean()):+.1f}%)")
    print(f"  链路固有偏差（B0/A-1）: {_rel(b0.mean(), g.mean()):+.1f}%  "
          f"| 逐图绝对 {np.abs((b0 - g) / g).mean() * 100:.1f}%")
    print(f"  系统上限（B1/A-1）  : {_rel(b1.mean(), g.mean()):+.1f}%  "
          f"| 逐图绝对 {np.abs((b1 - g) / g).mean() * 100:.1f}%")
    print(f"  根数  参考 {gn.mean():.1f} 根 -> 完美模型 {b1n.mean():.1f} 根 "
          f"(MAE {np.abs(b1n - gn).mean():.2f} 根)")

    if not args.no_model:
        c = np.array([r["c_len"] for r in rows], dtype=float)
        cn = np.array([r["c_n"] for r in rows], dtype=float)
        print(f"  当前系统（C/A-1）    : {_rel(c.mean(), g.mean()):+.1f}%  "
              f"| 逐图绝对 {np.abs((c - g) / g).mean() * 100:.1f}%")
        print(f"  根数  参考 {gn.mean():.1f} 根 -> 系统 {cn.mean():.1f} 根 "
              f"(MAE {np.abs(cn - gn).mean():.2f} 根)")
        print(f"\n  --- 误差归属 ---")
        print(f"  链路固有 : 总长 {np.abs(b0.mean() - g.mean()):.0f}px  "
              f"({abs(_rel(b0.mean(), g.mean())):.1f}%)")
        print(f"  模型造成 : 总长 {np.abs(c.mean() - b1.mean()):.0f}px  "
              f"({abs(_rel(c.mean(), b1.mean())):.1f}%)")
        print(f"  当前总误差: 总长 {np.abs(c.mean() - g.mean()):.0f}px  "
              f"({abs(_rel(c.mean(), g.mean())):.1f}%)")

    out_dir = naming.unique_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    csv_path = out_dir / f"chain_diag_{naming.timestamp()}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# 误差分解（测量链路 vs 模型）  数据: {args.dir}  "
                 f"模型: {folder.name if folder else '未用（--no-model）'}\n")
        fh.write(f"# A 参考=RSML折线欧氏长度 | B0 纯链路=GT掩码不剪枝不锚定 | "
                 f"B1 完美模型=GT掩码走部署口径 | C 当前系统=模型预测走部署口径\n")
        fh.write(f"# 线宽 {args.mask_width:g}px | B1/C 剪枝 {args.spur:g}px "
                 f"最短根 {args.min_len:g}px | 输入长边 "
                 f"{size if folder else '—'}\n")
        wr = csv.writer(fh)
        cols = ["图片名", "A_参考总长(px)", "B0_纯链路总长(px)", "B1_完美模型总长(px)",
                "B0相对A(%)", "B1相对A(%)"]
        if not args.no_model:
            cols += ["C_系统总长(px)", "C相对A(%)", "C相对B1(%)"]
        cols += ["A_根数", "B1_根数"]
        if not args.no_model:
            cols += ["C_根数"]
        wr.writerow(cols)
        for r in rows:
            out = [r["name"], f"{r['gt_len']:.1f}", f"{r['b0_len']:.1f}",
                   f"{r['b1_len']:.1f}", f"{_rel(r['b0_len'], r['gt_len']):+.1f}",
                   f"{_rel(r['b1_len'], r['gt_len']):+.1f}"]
            if not args.no_model:
                out += [f"{r['c_len']:.1f}", f"{_rel(r['c_len'], r['gt_len']):+.1f}",
                        f"{_rel(r['c_len'], r['b1_len']):+.1f}"]
            out += [r["gt_n"], r["b1_n"]]
            if not args.no_model:
                out.append(r["c_n"])
            wr.writerow(out)
    print(f"\nCSV: {csv_path}")
    print(f"耗时 {el:.1f}s | 单图 {el / len(rows):.2f}s")


if __name__ == "__main__":
    main()
