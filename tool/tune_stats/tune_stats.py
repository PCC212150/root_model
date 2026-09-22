"""参数扫描：在带 RSML 真值的数据集上扫「掩码后处理」参数，按与真值的误差选最优。

统计口径（**与 test.py / inference.py 完全一致**）：
    根总数、总长：来自 common.skeleton_stats（lengths 之和）；
    掩码来源：predict.predict 返回的 mask_counted —— 即「根系概率 ∩ 模型识别出的检查范围」，
    这里沿用同一条流水线（概率层清 ROI 之外 -> 滞回阈值 -> 与 ROI 求交），并且把
    「起点锚定到茎」补回的那一段也计入长度（见 common.skeleton_stats.anchor_paths_to_stem），
    所以扫出来的最优参数就是部署时实际生效的参数。

真值侧：RSML 的根条数与折线总长（px），不再区分主根/侧根。

为什么快：掩码 -> 骨架 -> 邻接表这一步与阈值无关，同一张图只算一次（_skeleton_adj），
多组 spur/min_len 复用邻接表（_strands_from_adj 内部拷贝，不改入参）。

扫描维度：low × spur × min_len × erode × normalize × **anchor_min × anchor_max**。
**注意在训练集上扫、留出集复核，测试集只看不调**（否则等于拿测试集调参）——
用法示例里的 `--dir datasets\\root\\test` 是历史写法，实际应该指 train。

锚定那一维值得单独说：阈值 = `clamp(FACTOR×茎等效半径, anchor_min, anchor_max)`，
而本数据集茎的等效半径约 215~252px（原图尺度），×FACTOR(=6) 得 1291~1514 —— **恒被
anchor_max 截断**，所以调 FACTOR 是无效的，能动的只有 anchor_max。实测（2026-09-19）
两边都是 U 形、最低点落在 600（原值），所以锚定无需调整；要再试请先确认茎的等效半径。

用法（项目根目录下运行，pcc 环境）：
    python tool\\tune_stats\\tune_stats.py --dir datasets\\root\\train --dry-run
    python tool\\tune_stats\\tune_stats.py --dir datasets\\root\\train --limit 2
    python tool\\tune_stats\\tune_stats.py --dir datasets\\root\\train --sort total_len
    python tool\\tune_stats\\tune_stats.py --dir datasets\\root\\train \\
        --sort total_len --anchor-max 200,400,600,900,1400      # 扫锚定上限
    python tool\\tune_stats\\tune_stats.py --dir datasets\\root\\train --gt-mask  # 真值掩码上限实验
结果：result/tune_stats/tune_stats_{年月日时分}.txt（表格）与 .json（最优组合，可直接抄进 config.py）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming, predict  # noqa: E402
from common.dataset import CH_STEM, discover_pairs  # noqa: E402
from common.gt_mask import draw_mask_from_roots  # noqa: E402
from common.image_io import prob_to_orig_mask, prob_to_orig_mask_hysteresis  # noqa: E402
from common.rsml_parse import parse_rsml, root_stats  # noqa: E402
from common.skeleton_stats import (anchor_gain_for_trace,  # noqa: E402
                                   stem_anchor_tolerance, _skeleton_adj,
                                   _strands_from_adj)

# 粗估耗时用的常数（本机 RTX 5060、5472x3648 图、长边 1024 实测；只为 --dry-run 给个量级）
_SEC_INFER = 0.4        # 每张图前向 1 次
_SEC_SKEL = 0.35        # 每个 (low, erode) 的 掩码->骨架->邻接表
_SEC_COMBO = 0.06       # 每个 (spur, min_len, normalize) 的 剪枝+分链

KEYS = [("total", "根总数"), ("total_len", "总长")]
CURRENT = {"low": config.PRED_LOW_THRESHOLD, "spur": config.PRED_SPUR_LENGTH,
           "min_len": config.MIN_ROOT_LENGTH, "normalize": True,
           "anchor_min": config.STEM_ANCHOR_MIN_PX,
           "anchor_max": config.STEM_ANCHOR_MAX_PX}


def parse_args():
    p = argparse.ArgumentParser(description="扫描根系统计的后处理参数")
    p.add_argument("--dir", type=Path, required=True,
                   help="数据集目录（含 images/ 与 labels/roots/），路径含空格要加引号")
    p.add_argument("--model", default=None, help="模型文件夹名（可省 model_ 前缀）；缺省用最新")
    p.add_argument("--low", default="0.05,0.10,0.20",
                   help="滞回低阈值候选，逗号分隔（高阈值固定 0.5）")
    p.add_argument("--spur", default="20,30,45,60",
                   help="骨架剪枝长度候选(px)，逗号分隔")
    p.add_argument("--min-len", default="20", help="最短根长候选(px)，逗号分隔")
    p.add_argument("--erode", default="1", help="腐蚀次数候选，逗号分隔")
    p.add_argument("--normalize-count", default="both",
                   help="计数归一：on/off/both（both = 两种口径都扫，默认）")
    p.add_argument("--anchor-min", default=str(config.STEM_ANCHOR_MIN_PX),
                   help="锚定阈值下限候选(px，原图尺度)，逗号分隔。"
                        "锚定阈值 = clamp(FACTOR×茎等效半径, min, max)，"
                        f"FACTOR 固定取 config（{config.STEM_ANCHOR_FACTOR}）")
    p.add_argument("--anchor-max", default=str(config.STEM_ANCHOR_MAX_PX),
                   help="锚定阈值上限候选(px，原图尺度)，逗号分隔。**实测这个才是真正生效的那个**："
                        "本数据集茎的等效半径约 215~252px，×FACTOR(=6) 得 1291~1514，"
                        "全被上限截断 —— 所以 FACTOR 扫了也没用，能调的是这个上限")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 张（按文件名排序），0=全部")
    p.add_argument("--gt-mask", action="store_true",
                   help="用真值折线画的掩码代替模型输出（上限实验，不需要模型）")
    p.add_argument("--size", type=int, default=None,
                   help="模型输入长边像素；缺省用模型训练时的 --size（自动从权重读）")
    p.add_argument("--sort", default="score", choices=("score", "total", "total_len"),
                   help="排序依据：score=综合分，其余=单项误差")
    p.add_argument("--top", type=int, default=15, help="控制台只打印前 N 名（表格文件写全）")
    p.add_argument("--out", type=Path, default=config.RESULT_DIR / "tune_stats",
                   help="结果目录（重名自动加 -1）")
    p.add_argument("--dry-run", action="store_true", help="只打印组合数与预计耗时，不推理不写文件")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _floats(s):
    return [float(v) for v in str(s).split(",") if v.strip()]


def _ints(s):
    return [int(v) for v in str(s).split(",") if v.strip()]


def load_model(model_arg, device):
    """模型解析统一走 common/ckpt（原来这里、test.py、inference.py 各写了一份）。
    --model 同样支持逗号分隔的多模型集成。"""
    pths, names = ckpt.resolve_pths(model_arg)
    model, metas = ckpt.load_models(pths, device)   # 集成时 model 是列表
    for m in (model if isinstance(model, list) else [model]):
        m.eval()
    return model, Path(pths[0]).parent, metas[0]


def score_of(e):
    """综合分：根数误差 + 长度误差（长度是像素，除以 100 折到同一量级）。"""
    return e["total"] + e["total_len"] / 100.0


def clip_to_box(mask, box):
    """与 predict 里同样的「精确求交」：统计只在检查范围内。"""
    if box is None:
        return mask
    keep = np.zeros_like(mask)
    keep[box[1]:box[3], box[0]:box[2]] = True
    return mask & keep


def main():
    args = parse_args()
    lows = _floats(args.low)
    spurs = _floats(args.spur)
    min_lens = _floats(args.min_len)
    erodes = _ints(args.erode)
    norms = [True, False] if args.normalize_count == "both" else \
        [args.normalize_count != "off"]
    a_mins = _floats(args.anchor_min)
    a_maxs = _floats(args.anchor_max)
    combos = (len(lows) * len(spurs) * len(min_lens) * len(erodes) * len(norms)
              * len(a_mins) * len(a_maxs))

    pairs = discover_pairs(args.dir)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit(f"[错误] {args.dir} 下没有「图片 + rsml」配对数据")
    print(f"数据集: {args.dir} | 图片 {len(pairs)} 张" + (f"（limit={args.limit}）" if args.limit else ""))
    print(f"网格: low={lows} × spur={spurs} × min_len={min_lens} × erode={erodes} "
          f"× normalize={['on' if n else 'off' for n in norms]} "
          f"× anchor_min={a_mins} × anchor_max={a_maxs} = {combos} 组合")

    if args.dry_run:
        est = len(pairs) * (_SEC_COMBO * combos + _SEC_SKEL * len(lows) * len(erodes)
                            + (0 if args.gt_mask else _SEC_INFER))
        print(f"[dry-run] 预计耗时 ≈ {est / 60:.1f} 分钟"
              f"（模型前向 {len(pairs)} 次、骨架化 {len(pairs) * len(lows) * len(erodes)} 次）")
        print("[dry-run] 未推理、未写文件。去掉 --dry-run 即正式运行。")
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = folder = None
    size = args.size or config.MAX_SIDE
    if not args.gt_mask:
        model, folder, meta = load_model(args.model, device)
        # 切片训练的模型必须显式给 --size（权重里记的 size 是块边长，不是推理尺度）
        ckpt.require_explicit_size([meta], args.size, [folder.name])
        size = args.size or meta.get("size") or config.MAX_SIDE
        print(f"模型: {folder.name} | 设备: {device} | 输入长边 {size}"
              + ("（模型训练时的设置）" if args.size is None and meta.get("size") else ""))
        if meta.get("size") and args.size and args.size != meta["size"]:
            print(f"[警告] 输入长边 {args.size} 与模型训练时（{meta['size']}）不一致，"
                  f"扫出来的参数会与实际部署口径不符")
    else:
        print("上限实验：用真值折线画的掩码当输入（不加载模型，不做检查范围限定）")

    errors = {k: {} for k, _ in KEYS}   # (low,spur,min_len,erode,norm) -> [逐图绝对误差]
    t0 = time.time()
    checked = False

    for si, (name, img_path, rsml_path) in enumerate(pairs, 1):
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        roots = parse_rsml(rsml_path)
        n_gt, lens_gt, total_gt = root_stats(roots)
        gt_stats = {"total": n_gt, "total_len": total_gt}
        check_box = None
        stem_tree, stem = None, None
        if args.gt_mask:
            source_mask = draw_mask_from_roots(roots, (w0, h0), config.MASK_LINE_WIDTH)
            prob = None
        else:
            res = predict.predict(model, img, max_side=size, stride=config.STRIDE,
                                  device=device, low_thresh=0)
            # 概率图整张只前向一次，多组 low 阈值复用。prob_target 已在概率层清掉
            # 检查范围之外（与部署同口径），阈值化后再与 ROI 精确求交即可完全对齐。
            prob = torch.from_numpy(res["prob_target"]).unsqueeze(0).unsqueeze(0)
            check_box = res["check_box"] if res["check_ok"] else None
            # 起点锚定要用的茎掩码（与部署同口径，见 common.skeleton_stats）。
            # 容差按每个 (anchor_min, anchor_max) 组合现算 —— 见 stem_anchor_tolerance。
            stem = res["masks"][CH_STEM]
            if stem.any():
                from scipy.spatial import cKDTree
                ys_s, xs_s = np.nonzero(stem)
                stem_tree = cKDTree(np.column_stack([xs_s, ys_s]))
            else:
                stem = None

        for low in lows:
            if args.gt_mask:
                masks = [source_mask]
            else:
                masks = [clip_to_box(
                    prob_to_orig_mask_hysteresis(prob, w0, h0, high=0.5, low=low)
                    if low > 0 else prob_to_orig_mask(prob, w0, h0, 0.5),
                    check_box)]
            for erode in erodes:
                for m in masks:
                    adj = _skeleton_adj(m, erode)
                    for spur in spurs:
                        for min_len in min_lens:
                            for norm in norms:
                                if adj is None:
                                    entries, base_len = None, 0.0
                                else:
                                    entries = _strands_from_adj(adj, spur, min_len, norm)
                                    base_len = float(sum(e[0] for e in entries))
                                for a_min in a_mins:
                                    for a_max in a_maxs:
                                        key = (low, spur, min_len, erode, norm,
                                               a_min, a_max)
                                        if entries is None:
                                            pred = {"total": 0, "total_len": 0.0}
                                        else:
                                            # 起点锚定：补回被泡沫环挡住的那一段
                                            # （与部署同口径）
                                            extra = 0.0
                                            if stem_tree is not None:
                                                tol = stem_anchor_tolerance(
                                                    stem, config.STEM_ANCHOR_FACTOR,
                                                    a_min, a_max)
                                                extra = sum(
                                                    anchor_gain_for_trace(e[1], stem_tree, tol)
                                                    for e in entries)
                                            pred = {"total": len(entries),
                                                    "total_len": base_len + extra}
                                        for k, _ in KEYS:
                                            errors[k].setdefault(key, []).append(
                                                abs(pred[k] - gt_stats[k]))
                                        # 首次组合做一次「自建路径 == analyze_mask_ex」的
                                        # 等价性自检（比的是**未锚定**的长度：
                                        # analyze_mask_ex 不做锚定）
                                        if not checked and entries is not None:
                                            from common.skeleton_stats import analyze_mask_ex
                                            ref = analyze_mask_ex(m, spur=spur,
                                                                  min_len=min_len,
                                                                  erode_iters=erode,
                                                                  normalize_count=norm)
                                            if ref["count"] != pred["total"] or \
                                                    abs(ref["total"] - base_len) > 1e-6:
                                                sys.exit("[错误] 骨架复用路径与 analyze_mask_ex "
                                                         "结果不一致，请检查 "
                                                         "common/skeleton_stats.py 是否被改动")
                                            checked = True
        print(f"  [{si}/{len(pairs)}] {name}: GT 根 {gt_stats['total']} "
              f"总长 {gt_stats['total_len']:.0f}px | 已用 {time.time() - t0:.0f}s")

    # ---- 汇总：每个组合的逐图平均绝对误差 ----
    rows = []
    for key in sorted(errors["total"],
                      key=lambda k: (k[0], k[1], k[2], k[3], not k[4], k[5], k[6])):
        e = {k: float(np.mean(errors[k][key])) for k, _ in KEYS}
        e["score"] = score_of(e)
        rows.append((key, e))
    sort_key = {"score": "score", "total": "total", "total_len": "total_len"}[args.sort]
    rows.sort(key=lambda r: r[1][sort_key])

    out_dir = naming.unique_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    ts = naming.timestamp()
    txt = naming.unique_path(out_dir / f"tune_stats_{ts}.txt")
    best_key, best = rows[0]
    head = [f"# 数据集: {args.dir}   图片数 {len(pairs)}" + (f" (limit={args.limit})" if args.limit else ""),
            f"# 模型: {folder.name if folder else '（真值掩码实验）'} | 设备: {device} | "
            f"输入长边 {size if folder else '—（真值掩码实验在原图分辨率上做）'}",
            f"# 口径: 根系只在模型识别出的检查范围内统计、起点锚定到茎"
            f"（与 test.py / inference.py 一致；--gt-mask 模式不做这两项）",
            f"# 排序: {args.sort} | 综合分 = 根总数MAE + 总长MAE/100",
            f"# 网格: low={lows} spur={spurs} min_len={min_lens} erode={erodes} "
            f"normalize={['on' if n else 'off' for n in norms]} "
            f"anchor_min={a_mins} anchor_max={a_maxs} = {combos} 组合",
            f"# 锚定阈值 = clamp({config.STEM_ANCHOR_FACTOR}×茎等效半径, anchor_min, "
            f"anchor_max)；本数据集茎等效半径约 215~252px，×{config.STEM_ANCHOR_FACTOR} "
            f"远大于上限，所以实际生效的是 anchor_max",
            "#",
            "# 排名  low  spur  min_len  erode  normalize  a_min  a_max | "
            + "  ".join(f"{cn}MAE" for _, cn in KEYS) + " | score"]
    lines = list(head)
    for i, (key, e) in enumerate(rows, 1):
        low, spur, min_len, erode, norm, a_min, a_max = key
        lines.append(f"{i:5d} {low:5.2f} {spur:6.0f} {min_len:8.0f} {erode:6d}  "
                     f"{'on' if norm else 'off':9s} {a_min:6.0f} {a_max:6.0f} | "
                     + "  ".join(f"{e[k]:8.2f}" for k, _ in KEYS) + f" | {e['score']:7.2f}")
    cur_key = next((k for k in errors["total"]
                    if abs(k[0] - CURRENT["low"]) < 1e-9
                    and abs(k[1] - CURRENT["spur"]) < 1e-9
                    and abs(k[2] - CURRENT["min_len"]) < 1e-9
                    and k[4] == CURRENT["normalize"]
                    and abs(k[5] - CURRENT["anchor_min"]) < 1e-9
                    and abs(k[6] - CURRENT["anchor_max"]) < 1e-9), None)
    lines += ["", f"# 最优组合: --low {best_key[0]} --spur {best_key[1]:.0f} "
                  f"--min-len {best_key[2]:.0f} --erode {best_key[3]} "
                  f"--normalize-count {'on' if best_key[4] else 'off'} "
                  f"--anchor-min {best_key[5]:.0f} --anchor-max {best_key[6]:.0f}"]
    if cur_key is not None:
        ce = {k: float(np.mean(errors[k][cur_key])) for k, _ in KEYS}
        lines.append(f"# 与当前 config（low={CURRENT['low']}, spur={CURRENT['spur']:.0f}, "
                     f"min_len={CURRENT['min_len']:.0f}, normalize=on, "
                     f"anchor_min={CURRENT['anchor_min']:.0f}, "
                     f"anchor_max={CURRENT['anchor_max']:.0f}）对比: "
                     f"根总数MAE {ce['total']:.2f} -> {best['total']:.2f}, "
                     f"总长MAE {ce['total_len']:.2f} -> {best['total_len']:.2f}")
    el = time.time() - t0
    lines.append(f"# 耗时: 总 {el:.1f}s | 单图 {el / len(pairs):.2f}s | 组合平均 {el / combos:.3f}s")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    js = naming.unique_path(out_dir / f"tune_stats_{ts}.json")
    js.write_text(json.dumps({
        "low_thresh": best_key[0], "pred_spur_length": best_key[1],
        "min_root_length": best_key[2], "erode_iters": best_key[3],
        "normalize_count": best_key[4],
        "stem_anchor_min_px": best_key[5], "stem_anchor_max_px": best_key[6],
        "score": best["score"],
        "mae": {k: best[k] for k, _ in KEYS},
        "n_images": len(pairs), "data_dir": str(args.dir),
        "model": folder.name if folder else "gt-mask", "sort": args.sort,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== 前 {min(args.top, len(rows))} 名（按 {args.sort}）==")
    print(f"{'排名':>4s} {'low':>5s} {'spur':>5s} {'min_len':>7s} {'erode':>5s} {'norm':>4s} "
          f"{'a_min':>6s} {'a_max':>6s} | "
          + "  ".join(f"{cn+'MAE':>9s}" for _, cn in KEYS) + f" | {'score':>7s}")
    for i, (key, e) in enumerate(rows[:args.top], 1):
        print(f"{i:4d} {key[0]:5.2f} {key[1]:5.0f} {key[2]:7.0f} {key[3]:5d} "
              f"{'on' if key[4] else 'off':>4s} {key[5]:6.0f} {key[6]:6.0f} | "
              + "  ".join(f"{e[k]:9.2f}" for k, _ in KEYS) + f" | {e['score']:7.2f}")
    print(f"\n最优组合: low={best_key[0]} spur={best_key[1]:.0f} "
          f"min_len={best_key[2]:.0f} erode={best_key[3]} "
          f"normalize={'on' if best_key[4] else 'off'} "
          f"anchor_min={best_key[5]:.0f} anchor_max={best_key[6]:.0f}")
    print(f"表格: {txt}")
    print(f"最优参数(json): {js}")


if __name__ == "__main__":
    main()
