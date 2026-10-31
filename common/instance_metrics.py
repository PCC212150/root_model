"""实例分割指标：ROI 内 IoU / Dice、COCO 风格 AP、根数与根长。

本模块不依赖 torch，只处理 numpy 掩码；输入是 `common.instance.GTInstances`
和 `PredInstance` 列表。语义分割的 Dice/clDice 不在这里，仍由
`common.metrics` / `test/test.py` 负责。

坐标约定：GT 的 `label_map` 是全图 uint16；预测是局部裁剪 mask + 全局 bbox。
所有匹配先在 ROI（默认 GT check 区域）内求交，和 `test/test_task.py` 的 root
评测口径一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import config
from common.instance import GTInstances, PredInstance, pred_from_crop_mask
from common.skeleton_stats import analyze_mask_ex


def _mask_length(mask: np.ndarray) -> float:
    if mask is None or not np.asarray(mask).any():
        return 0.0
    st = analyze_mask_ex(np.asarray(mask, dtype=bool),
                         spur=config.PRED_SPUR_LENGTH,
                         min_len=config.MIN_ROOT_LENGTH,
                         normalize_count=False)
    return float(st["total"])


def roi_areas(gt: GTInstances, roi: np.ndarray) -> np.ndarray:
    """每个 GT 实例在 ROI 内的面积。"""
    n = len(gt.instances)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    ids = np.asarray([inst.instance_id for inst in gt.instances], dtype=np.int64)
    # label_map 可能因为空实例跳号；用 bincount 后按 instance_id 取
    counts = np.bincount(gt.label_map[roi].ravel(), minlength=int(ids.max()) + 1)
    for k, iid in enumerate(ids):
        if iid < len(counts):
            out[k] = counts[iid]
    return out


def _clip_pred_to_roi(pred: PredInstance, roi: np.ndarray):
    x0, y0, x1, y1 = pred.bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(roi.shape[1], int(x1)), min(roi.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    m = pred.mask & roi[y0:y1, x0:x1]
    if not m.any():
        return None
    return pred_from_crop_mask(m, (x0, y0), pred.score, pred.source, pred.meta)


def _iou_dice(preds, gt: GTInstances, roi: np.ndarray,
              gt_area: np.ndarray):
    """返回 (P,G) IoU、(P,G) Dice、预测 ROI 面积。"""
    n_pred = len(preds)
    n_gt = len(gt.instances)
    iou = np.zeros((n_pred, n_gt), dtype=np.float64)
    dice = np.zeros((n_pred, n_gt), dtype=np.float64)
    pred_area = np.zeros(n_pred, dtype=np.float64)
    if n_pred == 0 or n_gt == 0:
        return iou, dice, pred_area
    for p, pred in enumerate(preds):
        x0, y0, x1, y1 = pred.bbox
        m = pred.mask
        pa = float(m.sum())
        pred_area[p] = pa
        if pa <= 0:
            continue
        gt_crop = gt.label_map[y0:y1, x0:x1]
        vals, counts = np.unique(gt_crop[m].ravel(), return_counts=True)
        inter = np.zeros(n_gt, dtype=np.float64)
        # instance_id -> 行号
        for iid, c in zip(vals.tolist(), counts.tolist()):
            if iid == 0:
                continue
            for k, inst in enumerate(gt.instances):
                if inst.instance_id == iid:
                    inter[k] = c
                    break
        denom = gt_area + pa - inter
        valid = denom > 0
        iou[p, valid] = inter[valid] / denom[valid]
        dice[p, valid] = 2.0 * inter[valid] / (
            gt_area[valid] + pa + 1e-12)
    return iou, dice, pred_area


def greedy_match(iou: np.ndarray, scores: np.ndarray, threshold: float):
    """按 score 降序、每个预测挑当前 IoU 最大的未匹配 GT。返回 [(p,g)]。"""
    if iou.size == 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    used = set()
    pairs = []
    n_gt = iou.shape[1]
    for p in order.tolist():
        if n_gt == 0:
            break
        row = iou[p]
        best_g, best_iou = -1, -1.0
        for g in range(n_gt):
            if g in used:
                continue
            if row[g] > best_iou:
                best_g, best_iou = g, float(row[g])
        if best_g >= 0 and best_iou >= threshold:
            used.add(best_g)
            pairs.append((p, best_g))
    return pairs


def _pr_ap(tp: np.ndarray, fp: np.ndarray, n_gt: int):
    if n_gt <= 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(mrec, grid, side="left")
    return float(mpre[idx].mean())


def ap_from_images(evals: Iterable["ImageEval"], thresholds=None) -> dict:
    """把所有图的预测按 score 合并，算 COCO 风格 AP50/75/50-95。"""
    evals = list(evals)
    thresholds = (np.arange(0.5, 1.0, 0.05)
                  if thresholds is None else np.asarray(thresholds, dtype=float))
    n_gt = int(sum(e.n_gt_valid for e in evals))
    result = {"ap50": float("nan"), "ap75": float("nan"),
              "ap50_95": float("nan"), "n_gt": n_gt}
    if n_gt == 0:
        return result
    # 所有图的预测全局排序；每个图独立维护已匹配 GT
    entries = []
    for image_idx, e in enumerate(evals):
        for p, s in enumerate(e.scores):
            entries.append((float(s), image_idx, p))
    entries.sort(key=lambda x: (-x[0], x[1], x[2]))
    aps = []
    for thr in thresholds:
        used = [set() for _ in evals]
        tp, fp = [], []
        for score, image_idx, p in entries:
            e = evals[image_idx]
            if e.iou.shape[1] == 0:
                tp.append(0)
                fp.append(1)
                continue
            row = e.iou[p]
            best_g, best_iou = -1, -1.0
            for g in range(e.iou.shape[1]):
                if g in used[image_idx]:
                    continue
                if row[g] > best_iou:
                    best_g, best_iou = g, float(row[g])
            if best_g >= 0 and best_iou >= thr:
                used[image_idx].add(best_g)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        ap = _pr_ap(np.asarray(tp), np.asarray(fp), n_gt)
        aps.append(ap)
        if abs(thr - 0.5) < 1e-9:
            result["ap50"] = ap
        if abs(thr - 0.75) < 1e-9:
            result["ap75"] = ap
    result["ap50_95"] = float(np.nanmean(aps)) if aps else float("nan")
    return result


@dataclass
class ImageEval:
    image: str
    n_gt_raw: int
    n_gt_valid: int
    n_pred_raw: int
    n_pred: int
    n_pred_outside_roi: int
    scores: np.ndarray
    iou: np.ndarray
    dice: np.ndarray
    pred_lengths: np.ndarray
    gt_lengths: np.ndarray
    gt_line_lengths: np.ndarray
    gt_valid: np.ndarray
    preds: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    row: dict = field(default_factory=dict)


def evaluate_image(gt: GTInstances, preds, *, image_name=None,
                   roi_mode: str = "gt_check", score_thresh: float = 0.0,
                   match_iou: float = 0.5, nms_suppressed: int = 0):
    """单图实例评测；返回 ImageEval（含 per-image 汇总行）。"""
    image_name = image_name or gt.stem
    h, w = gt.height, gt.width
    if roi_mode == "gt_check" and gt.roi_mask is not None:
        roi = gt.roi_mask
    else:
        roi = np.ones((h, w), dtype=bool)

    gt_area = roi_areas(gt, roi)
    gt_valid = gt_area > 0
    n_gt_valid = int(gt_valid.sum())

    n_pred_raw = len(preds)
    clipped = []
    n_outside = 0
    for p in preds:
        cp = _clip_pred_to_roi(p, roi)
        if cp is None:
            n_outside += 1
        else:
            clipped.append(cp)

    # AP 用所有有效预测的 score（包括低于工作点的），工作点只影响匹配/根数/长度
    iou, dice, pred_area = _iou_dice(clipped, gt, roi, gt_area)
    if iou.size and not np.all(gt_valid):
        iou[:, ~gt_valid] = 0.0
        dice[:, ~gt_valid] = 0.0
    scores = np.asarray([p.score for p in clipped], dtype=np.float64)
    pred_lengths = np.asarray([_mask_length(p.mask) for p in clipped],
                              dtype=np.float64)
    for p, ln in zip(clipped, pred_lengths):
        p.meta["length_px"] = float(ln)
    if roi_mode != "none" and gt.roi_mask is not None:
        gt_lengths = []
        for inst in gt.instances:
            x0, y0, x1, y1 = inst.bbox
            m = inst.mask & roi[y0:y1, x0:x1]
            gt_lengths.append(_mask_length(m))
        gt_lengths = np.asarray(gt_lengths, dtype=np.float64)
    else:
        gt_lengths = np.asarray([inst.skel_length for inst in gt.instances],
                                dtype=np.float64)
    gt_line_lengths = np.asarray([inst.line_length for inst in gt.instances],
                                 dtype=np.float64)

    ap = ap_from_images([ImageEval(
        image=image_name, n_gt_raw=len(gt.instances), n_gt_valid=n_gt_valid,
        n_pred_raw=n_pred_raw, n_pred=len(clipped),
        n_pred_outside_roi=n_outside, scores=scores, iou=iou, dice=dice,
        pred_lengths=pred_lengths, gt_lengths=gt_lengths,
        gt_line_lengths=gt_line_lengths, gt_valid=gt_valid,
        preds=clipped)])

    keep = np.where(scores >= float(score_thresh))[0]
    matches = greedy_match(iou[keep], scores[keep], match_iou) if len(keep) else []
    # 映射回 clipped 索引
    matched = [(int(keep[p]), int(g)) for p, g in matches]
    matched_iou = [float(iou[p, g]) for p, g in matched]
    matched_dice = [float(dice[p, g]) for p, g in matched]
    matched_len = [abs(float(pred_lengths[p]) - float(gt_lengths[g]))
                   for p, g in matched]

    n_pred = int(len(keep))
    pred_total = float(pred_lengths[keep].sum()) if len(keep) else 0.0
    gt_total = float(gt_lengths[gt_valid].sum()) if n_gt_valid else 0.0
    count_abs = abs(n_pred - n_gt_valid)
    sem_area = int(gt.semantic_mask.sum())
    if iou.size and n_gt_valid:
        best_per_gt = iou[:, gt_valid].max(axis=0)
        mean_best_iou = float(np.mean(best_per_gt))
    else:
        mean_best_iou = float("nan")
    row = {
        "image": image_name,
        "n_gt_raw": len(gt.instances),
        "n_gt_valid": n_gt_valid,
        "semantic_area_px": sem_area,
        "n_pred_raw": n_pred_raw,
        "n_nms_suppressed": int(nms_suppressed),
        "n_pred": n_pred,
        "n_pred_outside_roi": n_outside,
        "n_matched": len(matched),
        "n_unmatched_gt": max(0, n_gt_valid - len(matched)),
        "n_unmatched_pred": max(0, n_pred - len(matched)),
        "mean_best_iou": mean_best_iou,
        "mean_matched_iou": float(np.mean(matched_iou)) if matched_iou else float("nan"),
        "mean_matched_dice": float(np.mean(matched_dice)) if matched_dice else float("nan"),
        "count_abs_err": int(count_abs),
        "count_signed_err": int(n_pred - n_gt_valid),
        "gt_skel_total_px": gt_total,
        "gt_line_total_px": float(gt_line_lengths.sum()),
        "pred_total_px": pred_total,
        "total_len_abs_err": abs(pred_total - gt_total),
        "matched_len_abs_err": float(np.mean(matched_len)) if matched_len else float("nan"),
        "ap50": ap["ap50"],
        "ap75": ap["ap75"],
        "ap50_95": ap["ap50_95"],
        "coverage_gt_mask": float(gt.info.get("coverage_gt_mask", float("nan"))),
        "unassigned_mask_px": int(gt.unassigned_mask_px),
        "overlap_px": int(gt.overlap_px),
        "unassigned_ratio": float(gt.unassigned_mask_px / max(1, sem_area)),
        "overlap_ratio": float(gt.overlap_px / max(1, sem_area)),
        "roi": roi_mode,
    }
    e = ImageEval(
        image=image_name, n_gt_raw=len(gt.instances), n_gt_valid=n_gt_valid,
        n_pred_raw=n_pred_raw, n_pred=n_pred, n_pred_outside_roi=n_outside,
        scores=scores, iou=iou, dice=dice, pred_lengths=pred_lengths,
        gt_lengths=gt_lengths, gt_line_lengths=gt_line_lengths,
        gt_valid=gt_valid, preds=clipped, matches=matched, row=row)
    return e


def aggregate(evals, *, prompt_mode="", backend="", prompt_variant="",
              runtime_s=float("nan")) -> dict:
    """把所有 ImageEval 汇总成 summary 行。"""
    evals = list(evals)
    if not evals:
        return {}
    rows = [e.row for e in evals]
    n_gt = sum(r["n_gt_valid"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    ap = ap_from_images(evals)
    # 匹配 IoU/Dice/长度误差按匹配对数加权，避免 1 根图与 40 根图等权
    def weighted(key):
        num = sum(r[key] * r["n_matched"] for r in rows
                  if not np.isnan(r[key]) and r["n_matched"] > 0)
        den = sum(r["n_matched"] for r in rows
                  if not np.isnan(r[key]) and r["n_matched"] > 0)
        return num / den if den else float("nan")

    mi = weighted("mean_matched_iou")
    md = weighted("mean_matched_dice")
    ml = weighted("matched_len_abs_err")
    return {
        "prompt_mode": prompt_mode,
        "backend": backend,
        "prompt_variant": prompt_variant,
        "n_images": len(rows),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "ap50": ap["ap50"],
        "ap75": ap["ap75"],
        "ap50_95": ap["ap50_95"],
        "mean_best_iou": float(
            sum(r["mean_best_iou"] * r["n_gt_valid"] for r in rows
                if not np.isnan(r["mean_best_iou"]))
            / max(1, sum(r["n_gt_valid"] for r in rows
                         if not np.isnan(r["mean_best_iou"])))),
        "mean_matched_iou": float(mi),
        "mean_matched_dice": float(md),
        "count_mae": float(np.mean([r["count_abs_err"] for r in rows])),
        "count_bias": float(np.mean([r["count_signed_err"] for r in rows])),
        "total_len_mae": float(np.mean([r["total_len_abs_err"] for r in rows])),
        "matched_len_mae": float(ml),
        "gt_skel_total_px": float(sum(r["gt_skel_total_px"] for r in rows)),
        "gt_line_total_px": float(sum(r["gt_line_total_px"] for r in rows)),
        "pred_total_px": float(sum(r["pred_total_px"] for r in rows)),
        "coverage_gt_mask": float(np.mean([r["coverage_gt_mask"] for r in rows])),
        "unassigned_mask_px": int(sum(r["unassigned_mask_px"] for r in rows)),
        "overlap_px": int(sum(r["overlap_px"] for r in rows)),
        "unassigned_ratio": float(sum(r["unassigned_mask_px"] for r in rows)
                                  / max(1, sum(r["semantic_area_px"] for r in rows))),
        "overlap_ratio": float(sum(r["overlap_px"] for r in rows)
                               / max(1, sum(r["semantic_area_px"] for r in rows))),
        "runtime_s": float(runtime_s),
    }
