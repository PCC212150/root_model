"""在测试集（datasets/root/test，每组含 图片 + rsml + labelme json 真值）上评测已训练的模型。

用法（readme 两种写法均支持）：
    python test/test.py                                   # 自动使用最新模型
    python test/test.py --model model_202609091135
    python test/test.py --model_202609091135              # 兼容写法

输出：三类通道（根系 / 茎横截面 / 检查范围）各自的 IoU / Dice / 像素准确率 / clDice /
连通块，根系另加两个补充口径（**容差 Dice**：几像素的边界滑移不算错；**滞回口径 Dice**：
部署统计总长实际用的那张掩码），以及每张图「预测根数/总长/面积 vs RSML 真值」的对比、
平均绝对误差，和**预测 vs 真值的相关系数 R（长度、面积各一个）**—— MAE 量绝对误差，
R 量「模型有没有追踪植株间的差异」。口径细节见 common/metrics.py 与 CSV 末尾的 # 行。
结果保存至模型文件夹内 model_test_{年月日时分}.csv（UTF-8 BOM，Excel 可直接打开；
文件末尾是若干以 # 开头的汇总行）。
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
from common import ckpt, image_io, metrics, naming, predict  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM,  # noqa: E402
                            build_target_masks, discover_pairs, load_annot)
from common.rsml_parse import root_stats  # noqa: E402
from common.skeleton_stats import (analyze_mask_ex,  # noqa: E402
                                   continuation_flags)

# 要算 clDice / 连通块数的通道：**只给细结构**。检查范围是块状区域、骨架没有意义，
# 而且骨架化在 5472x3648 上要 0.3s/张，白花。
CLDICE_CH = (CH_ROOT, CH_STEM)


def preprocess_argv():
    """把 --model_xxx 兼容为 --model model_xxx。"""
    out, i = [], 0
    argv = sys.argv[1:]
    while i < len(argv):
        t = argv[i]
        if t.startswith("--model_") and "=" not in t:
            out += ["--model", t[2:]]
        else:
            out.append(t)
        i += 1
    return out


def parse_args():
    p = argparse.ArgumentParser(description="测试甘蔗根系 U-Net（三通道）")
    p.add_argument("--model", default=None,
                   help="模型文件夹名(可省略 model_ 前缀)；缺省自动取最新")
    p.add_argument("--size", type=int, default=None,
                   help="模型输入长边像素；缺省用模型训练时的 --size（自动从权重读），"
                        "再缺省 config.MAX_SIDE")
    p.add_argument("--data-dir", type=Path, default=config.TEST_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="长度换算：1 像素 = 多少毫米（默认用 config.MM_PER_PX）")
    p.add_argument("--mask-width", type=float, default=None,
                   help="评测时把真值折线画成多宽(px，原图尺度)；默认 config.MASK_LINE_WIDTH。"
                        "**换尺子实验用**：同一份预测换线宽 Dice 会差很多（5px→10px 实测 "
                        "0.50→0.62），要比两个模型就得用同一个值，别拿不同线宽的 Dice 互比")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(preprocess_argv())


def main():
    args = parse_args()
    # 评测尺子：真值折线画多宽。**换线宽 = 换尺子** —— 同一份预测 5px→10px 实测
    # Dice 0.50→0.62，所以不同线宽下的 Dice/IoU 一律不可互比（见 --mask-width 说明）。
    mask_width = args.mask_width or config.MASK_LINE_WIDTH
    if mask_width != config.MASK_LINE_WIDTH:
        print(f"[尺子] 真值线宽 {mask_width:g}px（config 是 {config.MASK_LINE_WIDTH:g}px）"
              f" —— 与其它线宽下的指标不可比")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    # --model 支持逗号分隔的多个模型（集成），见 ckpt.resolve_pths
    pths, names = ckpt.resolve_pths(args.model, args.out_dir)
    pth = pths[0]
    print(f"模型目录: {pth.parent}")
    if len(names) > 1:
        print(f"集成 {len(names)} 个: {' + '.join(names)}")
    model, metas = ckpt.load_models(pths, device)     # 集成时 model 是模型列表
    meta = metas[0]
    # 切片训练的模型走**原始分辨率滑窗**（见 ckpt.infer_tile 的实测对比）；0 = 老路径
    tile = ckpt.infer_tile(metas, args.size)
    print(f"权重: {pth.name} (保存于 epoch {meta.get('epoch', '?')}"
          + (f"，训练验证Dice {meta['val_dice']:.4f}" if meta.get("val_dice") else "")
          + f"，输出 {meta['out_ch']} 通道)")
    if tile:
        size = tile
        print(f"[切片模型] **原始分辨率滑窗**推理，块边长 {tile}（整图不缩放）"
              f"；像素指标也在原图分辨率上算")
    else:
        size = args.size or meta.get("size") or config.MAX_SIDE
        if args.size is None and meta.get("size"):
            print(f"输入长边 {size}（用模型训练时的设置）")
        elif meta.get("size") and args.size != meta.get("size"):
            print(f"[警告] 输入长边 {args.size} 与模型训练时（{meta['size']}）不一致："
                  f"尺度不匹配会明显掉精度，建议按训练尺度跑")

    pairs = discover_pairs(args.data_dir)
    if not pairs:
        print(f"[错误] {args.data_dir} 下没有「图片+标注」配对数据"
              f"（要 labels/other/<名>.json 或 labels/roots/<名>.rsml）")
        sys.exit(1)

    mm = config.MM_PER_PX if args.mm_per_px is None else args.mm_per_px
    names = list(config.CLASS_NAMES)
    rows = []
    agg = {k: [] for k in ("gt_roots", "pred_roots", "gt_total", "gt_total_raw",
                           "pred_total", "gt_cont",
                           "root_tol2", "root_tol4", "root_hyst",
                           "gt_area", "pred_area")}
    per_ch_metrics = {n: {"iou": [], "dice": [], "accuracy": [], "cldice": [],
                          "ncomp": []} for n in names}
    t_start = time.time()
    n_legacy = 0
    n_poly = 0
    for name, img_path, _annot_path in pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        annot = load_annot(args.data_dir, name, (w0, h0))
        n_legacy += annot.source == "rsml"
        # 多边形标注的图（2026-10-06 起支持）：root 通道按**真实轮廓填充**画，
        # 「GT总长-标注」会变成**轮廓周长**（不是根长）——下面集中告警一次。
        n_poly += bool(getattr(annot.lab, "root_polygons", None)
                       and any(annot.lab.root_polygons))
        roots = annot.roots
        gt_count, gt_lens, gt_total_raw = root_stats(roots)
        # 标注质量标记：交叉处断开重画的「续接片段」有多少条。
        # root_stats 数的就是折线条数（= RSML 的 ID 数），而续接片段会被当成额外的根
        # —— 所以这个数直接决定**该图的根数可不可信**。实测 11 张测试图里
        # plant_ S068-4_20251126ST 的 114 个 ID 中有 75~103 个是续接。
        # 详见 common/skeleton_stats.continuation_flags 与 tool/chain_diag/readme.md。
        n_cont = sum(continuation_flags(roots, max_gap=config.GT_CONT_MAX_GAP,
                                        max_angle=config.GT_CONT_MAX_ANGLE))
        # 像素指标**一律在原图分辨率上算**，与模型的输入尺寸无关。
        #
        # 原来（非滑窗时）是按**模型分辨率**算的，那让跨模型比较失效：GT 的线宽会跟着
        # 模型尺寸缩放（--size 1024 下 1.9px、2048 下 3.7px），**细的那份更难匹配**，
        # 于是「2048 vs 1024」比出来是 0.510 vs 0.507（看着几乎没差）；
        # 换成同一把尺子（都按 MASK_LINE_WIDTH=10 的原图尺度画 GT）是 0.571 vs 0.528。
        # 2026-09-25 改正 —— 代价是**与训练日志里的 val Dice 不再同一口径**
        # （那个按模型分辨率算，用于追一条曲线的趋势）；跨模型比较必须用这个。
        w1, h1 = w0, h0
        gt_masks, valid = build_target_masks(annot, (w1, h1), mask_width)
        # ---- 「理想掩码」对照：真值掩码走**与预测逐字相同**的那条流水线 ----
        # 为什么需要它：预测总长 = 「掩码 → 骨架 → 分链」的输出，而这条流水线本身
        # 不是恒等的（实测 骨架化 −1.9% / 剪枝 −3.4%，见 tool/chain_diag）。直接拿
        # 预测总长比 RSML 折线长，混着口径差；比这条流水线在**掩码完美**时的输出，
        # 剩下的才是模型真正的贡献。
        # 所以主指标是「预测 vs 理想」，副指标是「理想 vs 标注」（= 流水线固有偏差）。
        # 注：上面几个历史数字是 2026-09-22 在含起点锚定的口径下测的，锚定已于
        # 2026-10-05 整体删除（见 config.py），那之后的总长数字与以前不可比。
        gt_masks_orig, _ = build_target_masks(annot, (w0, h0), mask_width)
        gt_orig_root = np.ascontiguousarray(gt_masks_orig[:, :, CH_ROOT])
        if valid[CH_CHECK] > 0:      # 与预测同口径：只在检查范围内统计
            gt_orig_root = gt_orig_root & np.ascontiguousarray(
                gt_masks_orig[:, :, CH_CHECK])
        gt_ideal = analyze_mask_ex(
            gt_orig_root,
            spur=config.PRED_SPUR_LENGTH, min_len=config.MIN_ROOT_LENGTH)
        gt_total = gt_ideal["total"]
        gt = [np.ascontiguousarray(gt_masks[:, :, c]) for c in range(gt_masks.shape[2])]
        # 不变式：GT 根系也应限定在 GT 检查范围内（当前标注 100% 在框内，属校验性质）
        if valid[CH_CHECK] > 0:
            gt[CH_ROOT] = gt[CH_ROOT] & gt[CH_CHECK]

        res = predict.predict(model, img, max_side=size,
                              stride=config.STRIDE, device=device, tile=tile)
        # 预测也搬到原图分辨率才能和 GT 比（滑窗模式下它本来就在那一层，是恒等操作）
        def _to_orig(pb):
            return pb if pb.shape == (h0, w0) else \
                image_io.resize_bool_mask(pb, w0, h0)

        preds = [_to_orig(res["probs"][c] > 0.5) for c in range(len(res["probs"]))]
        # 与部署同口径：「掩码 → 骨架 → 分链」的直接输出，不做起点锚定
        st = analyze_mask_ex(
            res["mask_counted"],
            spur=config.PRED_SPUR_LENGTH, min_len=config.MIN_ROOT_LENGTH)
        pred_cnt, pred_lens, pred_total = st["count"], st["lengths"], st["total"]

        # clDice / 连通块数只算**细结构**通道：检查范围是块状区域，骨架没有意义，
        # 而且骨架化在 5472x3648 上要 0.3s/张，白花。
        ms = metrics.multi_channel_metrics(preds, gt, names=names, valid=valid,
                                           cldice_channels=CLDICE_CH)
        # root 的两个补充口径（只在 root 通道上算，见 metrics.tolerant_dice 与 CSV 页脚）：
        #   ① 容差 Dice：几像素的边界滑移不算错，剩下的才是断口/漏根这类结构性错误；
        #   ② 滞回口径 Dice：部署统计总长实际用的 mask_counted（低阈值 + 检查范围求交），
        #      它与上面那个 0.5 阈值的 Dice 不是同一张图。
        dice_tol2 = metrics.tolerant_dice(preds[CH_ROOT], gt[CH_ROOT], 2)
        dice_tol4 = metrics.tolerant_dice(preds[CH_ROOT], gt[CH_ROOT], 4)
        dice_hyst = metrics.binary_metrics(res["mask_counted"], gt[CH_ROOT])["dice"]
        agg["root_tol2"].append(dice_tol2)
        agg["root_tol4"].append(dice_tol4)
        agg["root_hyst"].append(dice_hyst)
        # 相关系数用的原始量：根系面积（GT 侧 = 真值根掩码 ∩ 检查框；预测侧 = mask_counted
        # —— 和「总长」用的是同一张统计掩码，两侧口径一致；也是推理 CSV 的「总根系面积」）
        agg["gt_area"].append(float(gt[CH_ROOT].sum()))
        agg["pred_area"].append(float(res["mask_counted"].sum()))
        for m in ms:
            for k in ("iou", "dice", "accuracy", "cldice", "ncomp"):
                per_ch_metrics[m["name"]][k].append(m[k])
        agg["gt_roots"].append(gt_count); agg["pred_roots"].append(pred_cnt)
        agg["gt_total"].append(gt_total)
        agg["gt_total_raw"].append(gt_total_raw)
        agg["gt_cont"].append(n_cont); agg["pred_total"].append(pred_total)
        len_str = ";".join(f"{v:.1f}" for v in pred_lens[:30]) or "-"

        def _f(v, fmt="{:.4f}"):
            return fmt.format(v) if not np.isnan(v) else "-"

        row = [name]
        for m in ms:
            row += [_f(m["iou"]), _f(m["dice"]), _f(m["accuracy"]),
                    _f(m["cldice"]), _f(m["ncomp"], "{:.0f}")]
        row += [gt_count, n_cont, pred_cnt,
                f"{gt_total_raw:.1f}", f"{gt_total:.1f}", f"{pred_total:.1f}", len_str]
        if mm and mm > 0:
            row += [f"{gt_total * mm:.1f}", f"{pred_total * mm:.1f}"]
        row += [f"{dice_tol2:.4f}", f"{dice_tol4:.4f}", f"{dice_hyst:.4f}",
                f"{agg['gt_area'][-1]:.0f}", f"{agg['pred_area'][-1]:.0f}"]
        rows.append(row)

        parts = []
        for m in ms:
            if np.isnan(m["iou"]):
                parts.append(f"{m['name']} 无真值")
                continue
            s = f"{m['name']} Dice={m['dice']:.4f}"
            if not np.isnan(m["cldice"]):
                # clDice 与连通块数必须和 Dice 并排看 —— 只看 Dice 会得出与肉眼
                # 相反的结论（实测 erode=3 让 MAE 变好但连通块 103 -> 156）。
                s += f" clDice={m['cldice']:.3f} 块={m['ncomp']:.0f}"
            parts.append(s)
        desc = " | ".join(parts)
        print(f"[{name}] {desc} | 根数 GT {gt_count}(含续接 {n_cont})/预测 {pred_cnt}"
              f" | 总长 GT {gt_total_raw:.0f}→理想 {gt_total:.0f} | 预测 {pred_total:.0f}")

    el = time.time() - t_start

    def avg(k):
        return float(np.mean(agg[k]))

    def mae(a, b):
        return float(np.abs(np.asarray(agg[a]) - np.asarray(agg[b])).mean())

    def corr(a, b):
        """预测值(b) 对 真值(a) 的 Pearson R 与回归斜率。

        为什么除 MAE 之外还要看 R：**MAE 量的是绝对误差，R 量的是「模型有没有追踪植株间的
        差异」** —— 做处理间对比时，后者才是关键。R 高但斜率明显 <1，说明排序对了、
        但系统性偏小（两列要一起读）。样本 <3 或某一列没有波动时返回 (nan, nan)。
        """
        x = np.asarray(agg[a], dtype=float)
        y = np.asarray(agg[b], dtype=float)
        if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            return float("nan"), float("nan")
        return (float(np.corrcoef(x, y)[0, 1]),
                float(np.polyfit(x, y, 1)[0]))

    r_len, k_len = corr("gt_total", "pred_total")
    r_ar, k_ar = corr("gt_area", "pred_area")

    header = ["图片名"]
    for n in names:
        header += [f"IoU({n})", f"Dice({n})", f"像素准确率({n})",
                   f"clDice({n})", f"连通块({n})"]
    header += ["GT根数(ID数)", "其中续接片段", "预测根数",
               "GT总长-标注(px)", "GT总长-理想(px)", "预测总长(px)",
               "预测各根长(px,降序,至多30条)"]
    if mm and mm > 0:
        header += ["GT总长(mm)", "预测总长(mm)"]
    header += ["Dice(root)@2px容差", "Dice(root)@4px容差", "Dice(root)滞回口径",
               "GT根系面积(px²)", "预测根系面积(px²)"]

    summary = [
        "",
        f"# 真值来源：json 里 root 折线 {len(pairs) - n_legacy} 图 | 旧格式 .rsml {n_legacy} 图"
        + ("（建议跑 tool\\merge_annot 统一）" if n_legacy else ""),
        f"# ===== 汇总（{len(pairs)} 图平均） =====",
        f"# 根数: GT平均 {avg('gt_roots'):.1f} vs 预测平均 {avg('pred_roots'):.1f} "
        f"(平均绝对误差 {mae('gt_roots', 'pred_roots'):.2f} 根) —— ⚠️ 见下条，"
        f"这个数只在无续接片段的图上有效",
        f"# 总长: 理想平均 {avg('gt_total'):.0f} px vs 预测平均 {avg('pred_total'):.0f} px "
        f"(平均绝对误差 {mae('gt_total', 'pred_total'):.0f} px)",
        f"#   对照①「标注原始总长」平均 {avg('gt_total_raw'):.0f} px —— "
        f"理想(掩码走同一条流水线) 比它高 "
        f"{(avg('gt_total') / max(avg('gt_total_raw'), 1) - 1) * 100:+.1f}%，"
        f"这部分是**流水线固有偏差**（骨架化/剪枝），换模型不会变；",
        f"#   对照②主指标用「预测 vs 理想」而不是「预测 vs 标注」，就是为了把这部分剔掉，"
        f"剩下的才是模型的贡献。",
    ]
    for n in names:
        dice = metrics.nanmean(per_ch_metrics[n]["dice"])
        iou = metrics.nanmean(per_ch_metrics[n]["iou"])
        acc = metrics.nanmean(per_ch_metrics[n]["accuracy"])
        if np.isnan(dice):
            summary.append(f"# {n}: 测试集无该通道真值")
            continue
        line = f"# {n}: Dice {dice:.4f} | IoU {iou:.4f} | 像素准确率 {acc:.4f}"
        cd = metrics.nanmean(per_ch_metrics[n]["cldice"])
        ncp = metrics.nanmean(per_ch_metrics[n]["ncomp"])
        if not np.isnan(cd):
            line += f" | **clDice {cd:.4f}** | **连通块 {ncp:.1f}**"
        summary.append(line)

    # ---- 根数只在「标注没有续接片段」的图上才可信 ----
    # root_stats 数的是 RSML 里 <root> 元素的个数 = 折线条数 = ID 数。而标注在交叉处
    # 断开重画（用户 2026-09-22 确认：交叉后看不出后续是哪个根），一条物理根会留下
    # 多个 ID。所以有续接片段的图，根数 MAE 量的是**标注习惯**，不是模型能力。
    clean = [i for i, c in enumerate(agg["gt_cont"]) if c == 0]
    if clean:
        rc = float(np.mean([abs(agg["gt_roots"][i] - agg["pred_roots"][i])
                            for i in clean]))
        summary += [
            f"# 根数 MAE（只算标注无续接片段的 {len(clean)}/{len(pairs)} 张）: {rc:.2f} 根"
            f" —— **这才是根数的有效读数**",
            f"#   其余 {len(pairs) - len(clean)} 张的 ID 数被交叉处的断开重画撑大了，"
            f"根数误差不可比（详见 tool/chain_diag/readme.md）",
        ]
    summary += [
        "# clDice / 连通块：clDice 量**骨架的连通性**（断一段就掉），连通块数是预测",
        "# 根掩码碎成了多少块。**这两个必须和 Dice 并排看** —— 像素 Dice/IoU 对「一条根",
        "# 断成几截」几乎不敏感（实测两个模型 Dice 只差 0.01，连通块数却差 3~5 倍）。",
        "# 只看 Dice 或只看总长 MAE 去调后处理，会得出与肉眼**相反**的结论：实测 erode=3",
        "# 让总长 MAE 从 2940 降到 2590，但连通块从 103 涨到 156（真值只有 4 块）。",
        "# 只对细结构通道（根系 / 茎）算 —— 检查范围是块状区域，骨架没有意义。",
    ]
    summary += [
        f"# 统计口径：输入长边 {size}；**真值线宽 {mask_width:g}px**（像素指标的尺子，"
        f"换线宽不可比）；根系只在模型识别出的检查范围内统计，且真值与预测**都走同一条"
        f"流水线**（掩码→骨架→分链）；低阈值 {config.PRED_LOW_THRESHOLD} / "
        f"剪枝 {config.PRED_SPUR_LENGTH}px / 最短根 {config.MIN_ROOT_LENGTH}px",
        f"# 2026-10-05 起「起点锚定」已整体删除（见 config.py 的历史注记）——"
        f"此前的总长/总长MAE 与今后不可比，锚定当年单独贡献约 +10.9 个百分点"
        f"（见 tool/chain_diag/）",
        f"# root 补充口径：容差 Dice @2px {avg('root_tol2'):.4f} / @4px {avg('root_tol4'):.4f}"
        f" —— 预测落在 GT 的 k px 邻域内就算对（几像素的边界滑移不算错），"
        f"剩下的才是断口/漏根这类结构性错误；",
        f"#   部署（滞回）口径 Dice {avg('root_hyst'):.4f} —— 统计总长用的那张 mask_counted"
        f"（低阈值 {config.PRED_LOW_THRESHOLD} + 检查范围求交），比上面 0.5 阈值那张胖，"
        f"两张不是同一张图；",
        f"# 相关性（{len(pairs)} 张）：长度 R={r_len:.3f}（斜率 {k_len:.2f}）| "
        f"面积 R={r_ar:.3f}（斜率 {k_ar:.2f}）—— 预测 vs 真值。R 量「模型有没有追踪"
        f"植株间的差异」（做处理间对比时比 MAE 更关键）；**R 高但斜率 <1 = 排序对、但系统性偏小**，"
        f"两个要一起读。面积用的是统计掩码（GT 根∩检查框 / 预测 mask_counted）；",
        f"# 单位换算：1 px = {mm} mm" if mm else "# 未做 mm 换算",
        f"# 测试总耗时 {el:.1f}s | 单图平均 {el / max(len(pairs), 1):.2f}s",
    ]
    if n_poly:
        summary.append(
            f"# ⚠️ 本批 {n_poly}/{len(pairs)} 张是**多边形标注**：root 通道按真实轮廓"
            f"**填充**画（不是 {mask_width:g}px 中心线）—— ① 像素指标与折线标注的批次"
            f"**不可比**（前景宽度不同）；② 「GT总长-标注」= 多边形**周长**、不是根长，"
            f"该列请忽略，总长以「GT总长-理想」（掩码→骨架口径）为准；③ 根数不受影响")

    # create_unique_file 而非 unique_path：两个测试进程同时启动时（一张卡一个），
    # 「先查存在、再 open(w)」会双双选中同一个文件名，**静默覆盖**对方的结果
    csv_path = naming.create_unique_file(pth.parent,
                                         f"model_test_{naming.timestamp()}.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
        for line in summary:
            wr.writerow([line])

    print()
    for n in names:
        dice = metrics.nanmean(per_ch_metrics[n]["dice"])
        if np.isnan(dice):
            print(f"{n}: 测试集无该通道真值")
            continue
        line = (f"{n}: Dice {dice:.4f} | IoU "
                f"{metrics.nanmean(per_ch_metrics[n]['iou']):.4f} | 像素准确率 "
                f"{metrics.nanmean(per_ch_metrics[n]['accuracy']):.4f}")
        cd = metrics.nanmean(per_ch_metrics[n]["cldice"])
        if not np.isnan(cd):
            line += (f" | clDice {cd:.4f}（连通性）| 连通块 "
                     f"{metrics.nanmean(per_ch_metrics[n]['ncomp']):.1f} 个")
        print(line)
    print(f"根数平均绝对误差 {mae('gt_roots', 'pred_roots'):.2f} 根"
          + (f"（只在无续接片段的 {len(clean)}/{len(pairs)} 张上算: {rc:.2f} 根）"
             if clean and len(clean) < len(pairs) else "")
          + f" | 总长平均绝对误差 {mae('gt_total', 'pred_total'):.0f} px "
            f"（vs 理想掩码；vs 原始标注 "
            f"{mae('gt_total_raw', 'pred_total'):.0f} px）")
    print(f"相关性 R（预测 vs 真值）: 长度 {r_len:.3f}（斜率 {k_len:.2f}）| "
          f"面积 {r_ar:.3f}（斜率 {k_ar:.2f}）")
    if n_poly:
        print(f"[注意] 本批 {n_poly}/{len(pairs)} 张是多边形标注：像素指标按"
              f"填充轮廓算（与折线标注不可比），「GT总长-标注」是周长、不是根长")
    print(f"测试总耗时 {el:.1f}s")
    print(f"结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
