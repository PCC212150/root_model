"""二值分割指标：IoU / Dice / 像素准确率 / clDice / 连通块数。

**为什么除了 Dice 还要 clDice 和连通块数**（2026-09-24 加的）：
像素级 Dice/IoU 对「一条根断成几截」几乎不敏感 —— 断一处只损失几个像素，面积和
Dice 都几乎不动。但对本项目的下游（骨架 -> 根数 / 总长）那是致命的。

实测代价：两个模型像素 Dice 只差 0.01，**骨架连通块数却差 3~5 倍**
（新切片模型 103~156 块 vs 旧整图模型 18~66 块，真值只有 4 块）。
更糟的是调参时会得出与肉眼**相反**的结论：tune_stats 推荐 erode=3（总长 MAE 更好
2590 vs 2940），但连通块从 103 涨到 156 —— 肉眼就是「根断断续续」。

所以**看后处理参数时必须同时看这三列**，只看 Dice 或只看 MAE 都会被带偏。
"""
import numpy as np
from skimage.measure import label as _label
from skimage.morphology import skeletonize as _skeletonize


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centerline Dice（Shit et al., CVPR 2021）：预测与真值的**骨架**之间的 Dice。

        Tprec = |skel(pred) ∩ gt| / |skel(pred)|    预测骨架有多少落在真值里（罚多预测）
        Tsens = |skel(gt) ∩ pred| / |skel(gt)|      真值骨架有多少被预测覆盖（罚漏预测）
        clDice = 2·Tprec·Tsens / (Tprec + Tsens)

    它量的是「连通性」：一条根断一截，Tsens 直接掉；凭空多一条，Tprec 掉。
    **注意 clDice 对「并根」不敏感**（两根并成一根时 Tprec/Tsens 都接近 1），
    所以它和「断」是单向的 —— 这一点在把它当损失用时踩过（见 config.LOSS_CLDICE_W）。
    这里只当**评测指标**用，专门补 Dice 测不到的那一半。
    """
    sp = _skeletonize(pred) if pred.any() else np.zeros_like(pred)
    sl = _skeletonize(gt) if gt.any() else np.zeros_like(gt)
    n_p, n_l = int(sp.sum()), int(sl.sum())
    if n_p == 0 and n_l == 0:
        return 1.0
    if n_p == 0 or n_l == 0:
        return 0.0
    tprec = float((sp & gt).sum()) / n_p
    tsens = float((sl & pred).sum()) / n_l
    return 2.0 * tprec * tsens / (tprec + tsens) if tprec + tsens > 0 else 0.0


def n_components(mask: np.ndarray) -> int:
    """8 邻接连通块数 —— 最直白的「断成几截」读数。

    真值通常只有个位数（折线画出来是连续的），而模型输出动辄上百。
    这个数比 clDice 更直观：直接就是「碎成多少块」。
    """
    if not mask.any():
        return 0
    return int(_label(mask, connectivity=2).max())


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt: 同形状 bool 掩码。返回 {'iou','dice','accuracy'}。

    clDice / 连通块数不在这里算 —— 它们要在**二维**掩码上做骨架化，
    而这里为了算 tp/fp/fn 已经把输入拉平了。见 multi_channel_metrics。
    """
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(pred.size - tp - fp - fn)
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 0.0
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / float(pred.size) if pred.size else 0.0
    return {"iou": iou, "dice": dice, "accuracy": acc}


def multi_channel_metrics(preds, gts, names=None, valid=None,
                          cldice_channels=None) -> list:
    """逐通道算指标：preds/gts 为同长度的掩码列表（形状一致）。

    valid: 可选的逐通道 0/1 列表；为 0 的通道返回 nan（该通道没有真值）。
    cldice_channels: 要算 clDice / 连通块数的通道下标；None = 全部。
        默认调用方一般只给「细结构」通道（根系/茎），因为块状通道（检查范围）
        的骨架没有意义、而且骨架化很贵。
    返回 [{'name','iou','dice','accuracy','cldice','ncomp'}, ...]，顺序与输入一致。
    """
    out = []
    nan = float("nan")
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        name = names[i] if names else str(i)
        if valid is not None and not valid[i]:
            out.append({"name": name, "iou": nan, "dice": nan, "accuracy": nan,
                        "cldice": nan, "ncomp": nan})
            continue
        do_cd = cldice_channels is None or i in cldice_channels
        m = binary_metrics(pred, gt)
        m["name"] = name
        # 两个键**始终存在**，不算的通道填 nan —— 否则调用方按固定键取值会 KeyError，
        # 而且「有没有算」这件事应该由数值表达，不该由键在不在表达。
        m["cldice"] = cldice(pred, gt) if do_cd else nan
        m["ncomp"] = float(n_components(pred)) if do_cd else nan
        out.append(m)
    return out


def nanmean(values) -> float:
    """忽略 nan 求均值；全为 nan 时返回 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))
