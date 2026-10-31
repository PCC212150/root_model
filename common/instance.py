"""逐根实例 GT / 提示构造（不依赖 torch）。

现有训练/语义评测把一根根标注合并成一个 root 通道；本模块保留“一条 root
linestrip = 一个实例”的口径，供 SAM 系列的实例分割评测使用：

    JSON(root linestrip + polygon)
        -> polygon 按几何代价归到最近的中心线
        -> 每个实例的 polygon 并集 ∩ masks/<stem>.png
        -> label_map + 逐实例裁剪 mask
        -> GT / 骨架 prompt

为什么不直接用 `masks/*.png` 的连通块：根系在根颈处大量粘连，连通块会把
多根并成一块；这里用标注中心线做实例身份，polygon 只负责补出真实宽度。

注意：实例是**标注中心线实例**，不一定是生物根。交叉处用户会断开重画，
`common/skeleton_stats.py` 的 continuation 说明也适用于这里。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import config
from common import gt_mask
from common.labelme import parse_other, parse_root_shapes
from common.skeleton_stats import analyze_mask_ex

EPS = 1e-8
NEG_INF = 1e9


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------
def polyline_length(points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _resample_polyline(points, step: float = 10.0, max_points: int = 64):
    """按弧长等距重采样折线；用于代价计算，不用于画图。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= EPS:
        return pts[:1]
    n = min(max_points, max(2, int(math.ceil(total / max(step, EPS))) + 1))
    dists = np.linspace(0.0, total, n)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for d in dists:
        j = int(np.searchsorted(cum, d, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t = (d - cum[j]) / max(seg[j], EPS)
        out.append(pts[j] + t * (pts[j + 1] - pts[j]))
    return np.asarray(out, dtype=np.float64)


def _subsample_vertices(poly, max_points: int = 64):
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
    return pts[np.unique(idx)]


def _points_in_polygon(points, poly) -> np.ndarray:
    """射线法判断点是否在多边形内；points (N,2)。"""
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    if len(pts) == 0 or len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + EPS) + xi)
        inside ^= cond
        xj, yj = xi, yi
    return inside


def _point_to_polyline_dist(points, line) -> np.ndarray:
    """每个点到折线的最短距离；points (N,2)，line (M,2)。"""
    pts = np.asarray(points, dtype=np.float64)
    line = np.asarray(line, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(line) < 2:
        return np.full(len(pts), np.inf)
    best = np.full(len(pts), np.inf)
    for a, b in zip(line, line[1:]):
        ab = b - a
        denom = float(ab @ ab)
        if denom <= EPS:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


def _bbox_of_points(points, w: int, h: int, margin: float = 0.0):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return None
    x0 = max(0, int(math.floor(pts[:, 0].min() - margin)))
    y0 = max(0, int(math.floor(pts[:, 1].min() - margin)))
    x1 = min(w, int(math.ceil(pts[:, 0].max() + margin)))
    y1 = min(h, int(math.ceil(pts[:, 1].max() + margin)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _union_bbox(boxes, w: int, h: int, margin: float = 0.0):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x0 = max(0, min(b[0] for b in boxes) - int(margin))
    y0 = max(0, min(b[1] for b in boxes) - int(margin))
    x1 = min(w, max(b[2] for b in boxes) + int(margin))
    y1 = min(h, max(b[3] for b in boxes) + int(margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def bbox_gap(a, b) -> float:
    """两个 bbox 的间隔；相交时为 0。"""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(math.hypot(dx, dy))


# ---------------------------------------------------------------------------
# polygon -> linestrip 分组
# ---------------------------------------------------------------------------
@dataclass
class Grouping:
    groups: dict                 # line index -> polygon indices
    cost: np.ndarray             # (L,P)
    unassigned_lines: list
    unassigned_polygons: list
    line_only: list
    flags: dict                  # line index -> list[str]
    info: dict


def group_polygons(lines, polygons, *, buffer_px: float = 12.0,
                   max_bbox_gap: float = 300.0,
                   accept_cost: float = 0.8,
                   hard_cost: float = 1.5) -> Grouping:
    """把 polygon 碎片归到最近的中心线。

    代价 = 0.55*(1-cover) + 0.35*(1-near) + 0.10*dcent。
    - cover: 中心线采样点落在 polygon 内的比例；
    - near: polygon 顶点距中心线 <= buffer_px 的比例；
    - dcent: 质心距离/200，裁到 [0,1]。
    先用 Hungarian 保证每条线有一个 polygon，再把多余 polygon 并入最近的线。
    """
    L, P = len(lines), len(polygons)
    groups = {i: [] for i in range(L)}
    flags = {i: [] for i in range(L)}
    if L == 0:
        return Grouping({}, np.zeros((0, P)), [], list(range(P)), [],
                        flags, {"n_lines": 0, "n_polygons": P})
    if P == 0:
        return Grouping(groups, np.zeros((L, 0)), list(range(L)), [], list(range(L)),
                        {i: ["line_only"] for i in range(L)},
                        {"n_lines": L, "n_polygons": 0})

    line_samples = [_resample_polyline(l, step=10.0, max_points=64) for l in lines]
    line_cent = [np.asarray(l, dtype=np.float64).mean(axis=0) for l in lines]
    poly_samples = [_subsample_vertices(p, 64) for p in polygons]
    poly_cent = [np.asarray(p, dtype=np.float64).mean(axis=0) for p in polygons]
    line_boxes = [_bbox_of_points(l, 10 ** 9, 10 ** 9) for l in lines]
    poly_boxes = [_bbox_of_points(p, 10 ** 9, 10 ** 9) for p in polygons]

    cost = np.full((L, P), 10.0, dtype=np.float64)
    for i in range(L):
        ls = line_samples[i]
        lb = line_boxes[i]
        if len(ls) == 0 or lb is None:
            continue
        for j in range(P):
            pb = poly_boxes[j]
            if pb is None:
                continue
            gap = bbox_gap(lb, pb)
            if gap > max_bbox_gap:
                continue
            cover = float(_points_in_polygon(ls, polygons[j]).mean())
            near = float((_point_to_polyline_dist(poly_samples[j], lines[i])
                          <= buffer_px).mean())
            dcent = min(1.0, float(np.linalg.norm(line_cent[i] - poly_cent[j])) / 200.0)
            cost[i, j] = 0.55 * (1.0 - cover) + 0.35 * (1.0 - near) + 0.10 * dcent

    assigned = set()
    if L <= P:
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows.tolist(), cols.tolist()):
            groups[r].append(c)
            assigned.add(c)
            if cost[r, c] > accept_cost:
                flags[r].append("low_confidence")
    else:
        # 线比 polygon 多：每个 polygon 先归到最近的线，剩下的线走 line_only
        for c in range(P):
            r = int(np.argmin(cost[:, c]))
            groups[r].append(c)
            assigned.add(c)

    unassigned_polygons = []
    for c in range(P):
        if c in assigned:
            continue
        r = int(np.argmin(cost[:, c]))
        best = float(cost[r, c])
        if best <= hard_cost:
            groups[r].append(c)
            if best > accept_cost:
                flags[r].append("low_confidence")
            assigned.add(c)
        else:
            unassigned_polygons.append(c)

    unassigned_lines = []
    line_only = []
    for i in range(L):
        if not groups[i]:
            unassigned_lines.append(i)
            line_only.append(i)
            flags[i].append("line_only")

    return Grouping(groups=groups, cost=cost,
                    unassigned_lines=unassigned_lines,
                    unassigned_polygons=unassigned_polygons,
                    line_only=line_only, flags=flags,
                    info={"n_lines": L, "n_polygons": P,
                          "n_assigned_polygons": P - len(unassigned_polygons),
                          "n_unassigned_polygons": len(unassigned_polygons),
                          "n_line_only": len(line_only)})


# ---------------------------------------------------------------------------
# 实例 GT
# ---------------------------------------------------------------------------
@dataclass
class GTInstance:
    instance_id: int
    line_index: int
    polygon_indices: list
    line_points: np.ndarray
    line_length: float
    bbox: tuple
    mask: np.ndarray
    area: int
    skel_length: float
    n_strands: int
    group_cost: float
    flags: list = field(default_factory=list)


@dataclass
class GTInstances:
    stem: str
    width: int
    height: int
    instances: list
    label_map: np.ndarray
    semantic_mask: np.ndarray
    roi_mask: np.ndarray | None
    unassigned_mask_px: int
    overlap_px: int
    polygon_outside_mask_px: int
    info: dict


def _read_semantic_mask(mask_path: Path, width: int, height: int):
    if mask_path is None or not Path(mask_path).is_file():
        return None
    with Image.open(mask_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    if arr.shape != (height, width):
        raise ValueError(f"{mask_path}: 掩码尺寸 {arr.shape[::-1]} 与图片 "
                         f"{(width, height)} 不一致")
    return arr > 127


def _rasterize_polygons(polygons, bbox, width: int, height: int):
    x0, y0, x1, y1 = bbox
    img = Image.new("L", (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        pts = [(float(x) - x0, float(y) - y0) for x, y in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def _rasterize_line(line, bbox, width: int, height: int, line_width: int = 1):
    x0, y0, x1, y1 = bbox
    img = Image.new("L", (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(img)
    pts = [(float(x) - x0, float(y) - y0) for x, y in line]
    if len(pts) >= 2:
        draw.line(pts, fill=255, width=max(1, int(line_width)), joint="curve")
    return np.asarray(img, dtype=np.uint8) > 0


def _roi_from_other(json_path, width: int, height: int):
    other = parse_other(json_path, image_size=(width, height), verbose=False)
    if other.check_rect is None:
        return None
    x0, y0, x1, y1 = other.check_rect
    return gt_mask.draw_rects_at([(x0, y0, x1, y1)], (width, height))


def build_gt_instances(json_path, *, data_dir=None, image_size=None,
                       mask_line_width: int = config.MASK_LINE_WIDTH,
                       verbose: bool = False) -> GTInstances:
    """从一张图的 JSON + masks PNG 生成逐根实例 GT。

    data_dir 默认取 json 所在目录；掩码路径为 <data_dir>/masks/<stem>.png。
    没有 masks PNG 时退回 polygon 并集（会写 info["mask_fallback"]=True）。
    """
    json_path = Path(json_path)
    data_dir = Path(data_dir) if data_dir is not None else json_path.parent
    if image_size is None:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = data_dir / f"{json_path.stem}{ext}"
            if cand.is_file():
                img_path = cand
                break
        if img_path is None:
            raise FileNotFoundError(f"找不到 {json_path.stem} 的图片")
        with Image.open(img_path) as im:
            width, height = im.size
    else:
        width, height = int(image_size[0]), int(image_size[1])
    mask_path = data_dir / "masks" / f"{json_path.stem}.png"

    shapes = parse_root_shapes(json_path, image_size=(width, height), verbose=verbose)
    lines = shapes.lines
    polygons = shapes.polygons
    semantic = _read_semantic_mask(mask_path, width, height)
    mask_fallback = semantic is None
    if semantic is None:
        # 兜底：没有像素掩码时用 polygon 并集（只应在旧数据上发生）
        semantic = np.zeros((height, width), dtype=bool)
        for poly in polygons:
            b = _bbox_of_points(poly, width, height)
            if b is None:
                continue
            semantic[b[1]:b[3], b[0]:b[2]] |= _rasterize_polygons([poly], b, width, height)
        if verbose:
            print(f"[实例] {json_path.name}: 没有 masks PNG，root 用 polygon 并集")

    grouping = group_polygons(lines, polygons)

    label_map = np.zeros((height, width), dtype=np.uint16)
    best_dist = np.full((height, width), np.inf, dtype=np.float32)
    votes = np.zeros((height, width), dtype=np.uint8)
    outside_px = 0
    skipped_lines = []
    infos = []

    # 每条线一个实例；线没有 polygon 时用宽度线带兜底
    inst_id = 0
    for line_idx in range(len(lines)):
        poly_ids = grouping.groups.get(line_idx, [])
        line = lines[line_idx]
        bbox = _bbox_of_points(line, width, height, margin=max(4, mask_line_width))
        if bbox is None:
            skipped_lines.append(line_idx)
            continue
        poly_boxes = []
        for j in poly_ids:
            b = _bbox_of_points(polygons[j], width, height)
            if b is not None:
                poly_boxes.append(b)
        if poly_boxes:
            bbox = _union_bbox([bbox] + poly_boxes, width, height, margin=2)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if not poly_ids:
            # line_only：用 config.MASK_LINE_WIDTH 的带子补出实例（再裁到语义掩码）
            poly_mask = _rasterize_line(line, bbox, width, height,
                                        line_width=mask_line_width)
            flags = list(grouping.flags.get(line_idx, []))
            group_cost = 0.0
        else:
            poly_mask = _rasterize_polygons(
                [polygons[j] for j in poly_ids], bbox, width, height)
            flags = list(grouping.flags.get(line_idx, []))
            group_cost = float(max(grouping.cost[line_idx, j] for j in poly_ids))
        line_mask = _rasterize_line(line, bbox, width, height, line_width=1)
        sem_crop = semantic[y0:y1, x0:x1]
        valid = poly_mask & sem_crop
        if not valid.any() and poly_ids:
            # polygon 全在掩码外：仍保留线带，避免整条线消失
            valid = _rasterize_line(line, bbox, width, height,
                                    line_width=mask_line_width) & sem_crop
            flags.append("polygon_outside_mask")
        outside_px += int((poly_mask & ~sem_crop).sum())
        if not valid.any():
            skipped_lines.append(line_idx)
            continue
        dist = ndimage.distance_transform_edt(~line_mask).astype(np.float32)
        lab = label_map[y0:y1, x0:x1]
        bst = best_dist[y0:y1, x0:x1]
        take = valid & ((lab == 0) | (dist < bst))
        lab[take] = inst_id + 1
        bst[take] = dist[take]
        votes[y0:y1, x0:x1] += valid.astype(np.uint8)

        infos.append({
            "instance_id": inst_id + 1,
            "line_index": line_idx,
            "polygon_indices": list(poly_ids),
            "line_points": np.asarray(line, dtype=np.float64),
            "line_length": polyline_length(line),
            "bbox": bbox,
            "group_cost": group_cost,
            "flags": flags,
        })
        inst_id += 1

    # 语义掩码之外的 label 清零；实例 mask 从最终 label_map 裁剪
    label_map[semantic == 0] = 0
    instances = []
    for info in infos:
        x0, y0, x1, y1 = info["bbox"]
        mask = label_map[y0:y1, x0:x1] == info["instance_id"]
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(mask)
        bx0, by0 = x0 + int(xs.min()), y0 + int(ys.min())
        bx1, by1 = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
        mask = mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
        st = analyze_mask_ex(mask, spur=config.PRED_SPUR_LENGTH,
                             min_len=config.MIN_ROOT_LENGTH, normalize_count=False)
        instances.append(GTInstance(
            instance_id=info["instance_id"],
            line_index=info["line_index"],
            polygon_indices=info["polygon_indices"],
            line_points=info["line_points"],
            line_length=float(info["line_length"]),
            bbox=(bx0, by0, bx1, by1),
            mask=mask,
            area=int(mask.sum()),
            skel_length=float(st["total"]),
            n_strands=int(st["count"]),
            group_cost=float(info["group_cost"]),
            flags=list(info["flags"]),
        ))

    unassigned = int((semantic & (label_map == 0)).sum())
    overlap = int((votes > 1).sum())
    roi = _roi_from_other(json_path, width, height)
    info = {
        "stem": json_path.stem,
        "n_lines": len(lines),
        "n_polygons": len(polygons),
        "n_instances": len(instances),
        "n_assigned_polygons": grouping.info.get("n_assigned_polygons", 0),
        "n_unassigned_polygons": grouping.info.get("n_unassigned_polygons", 0),
        "n_line_only": grouping.info.get("n_line_only", 0),
        "n_skipped_lines": len(skipped_lines),
        "skipped_lines": skipped_lines,
        "unassigned_polygons": grouping.unassigned_polygons,
        "mask_fallback": mask_fallback,
        "coverage_gt_mask": (float((semantic & (label_map > 0)).sum())
                              / max(1, int(semantic.sum()))),
    }
    return GTInstances(stem=json_path.stem, width=width, height=height,
                       instances=instances, label_map=label_map,
                       semantic_mask=semantic, roi_mask=roi,
                       unassigned_mask_px=unassigned, overlap_px=overlap,
                       polygon_outside_mask_px=outside_px, info=info)


# ---------------------------------------------------------------------------
# 提示构造
# ---------------------------------------------------------------------------
@dataclass
class Prompt:
    crop_box: tuple              # 原图坐标 (x0,y0,x1,y1)
    box: np.ndarray              # crop 内 xyxy
    points: np.ndarray | None    # crop 内 Nx2
    labels: np.ndarray | None    # N
    source: str
    meta: dict = field(default_factory=dict)


def crop_for_box(image: np.ndarray, bbox, *, margin_px: float = 16.0,
                 margin_ratio: float = 0.1, min_size: int = 32):
    """按 bbox 裁图，返回 (crop, 局部 bbox, 原点)。"""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return None, None, None
    bw, bh = x1 - x0, y1 - y0
    margin = max(float(margin_px), float(margin_ratio) * max(bw, bh))
    cx0 = max(0, int(math.floor(x0 - margin)))
    cy0 = max(0, int(math.floor(y0 - margin)))
    cx1 = min(w, int(math.ceil(x1 + margin)))
    cy1 = min(h, int(math.ceil(y1 + margin)))
    if cx1 - cx0 < min_size:
        pad = (min_size - (cx1 - cx0) + 1) // 2
        cx0, cx1 = max(0, cx0 - pad), min(w, cx1 + pad)
    if cy1 - cy0 < min_size:
        pad = (min_size - (cy1 - cy0) + 1) // 2
        cy0, cy1 = max(0, cy0 - pad), min(h, cy1 + pad)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, None, None
    crop = image[cy0:cy1, cx0:cx1]
    local = np.array([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0],
                     dtype=np.float32)
    return crop, local, (cx0, cy0, cx1, cy1)


def _sample_line_points(line, n: int = 3) -> np.ndarray:
    pts = np.asarray(line, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return pts[np.unique(idx)]


def _clip_points(points, origin, crop_box):
    if points is None or len(points) == 0:
        return None
    ox, oy = origin[0], origin[1]
    out = np.asarray(points, dtype=np.float64) - np.array([ox, oy])
    return out.astype(np.float32)


def gt_prompts(gt: GTInstances, image: np.ndarray, *, margin_px: float = 16.0,
               margin_ratio: float = 0.1, with_points: bool = True,
               with_negatives: bool = True, max_negatives: int = 4):
    """GT 实例的框/点提示（oracle；只用于上限评测）。"""
    prompts = []
    for inst in gt.instances:
        crop, box, origin = crop_for_box(
            image, inst.bbox, margin_px=margin_px, margin_ratio=margin_ratio)
        if crop is None:
            continue
        pos = _sample_line_points(inst.line_points, 3) if with_points else None
        pts = []
        labels = []
        if pos is not None and len(pos):
            p = _clip_points(pos, origin, crop.shape)
            if p is not None:
                pts.append(p)
                labels.append(np.ones(len(p), dtype=np.int64))
        if with_negatives and gt.label_map is not None:
            x0, y0, x1, y1 = origin
            lab = gt.label_map[y0:y1, x0:x1]
            other = (lab > 0) & (lab != inst.instance_id)
            ys, xs = np.nonzero(other)
            if len(xs):
                take = np.linspace(0, len(xs) - 1,
                                   min(max_negatives, len(xs))).round().astype(int)
                neg = np.stack([xs[take], ys[take]], axis=1).astype(np.float32)
                pts.append(neg)
                labels.append(np.zeros(len(neg), dtype=np.int64))
        p_all = np.concatenate(pts, axis=0) if pts else None
        l_all = np.concatenate(labels, axis=0) if labels else None
        prompts.append(Prompt(crop_box=origin, box=box, points=p_all,
                              labels=l_all, source="gt_box",
                              meta={"instance_id": inst.instance_id,
                                    "line_length": inst.line_length,
                                    "skel_length": inst.skel_length}))
    return prompts


def skeleton_prompts(pred_mask: np.ndarray, image: np.ndarray, *, spur=None,
                     min_len=None, spacing: float = 50.0,
                     margin_px: float = 12.0, margin_ratio: float = 0.05,
                     max_points: int = 8, max_negatives: int = 4):
    """从预测 root 掩码的逐根骨架生成提示；不读取任何 GT。"""
    spur = config.PRED_SPUR_LENGTH if spur is None else spur
    min_len = config.MIN_ROOT_LENGTH if min_len is None else min_len
    st = analyze_mask_ex(pred_mask, spur=spur, min_len=min_len,
                         with_paths=True, spacing=spacing,
                         normalize_count=True)
    prompts = []
    for idx, path in enumerate(st.get("paths", [])):
        pts = np.asarray(path, dtype=np.float64)
        if len(pts) < 2:
            continue
        bbox = _bbox_of_points(pts, pred_mask.shape[1], pred_mask.shape[0])
        if bbox is None:
            continue
        crop, box, origin = crop_for_box(
            image, bbox, margin_px=margin_px, margin_ratio=margin_ratio)
        if crop is None:
            continue
        if len(pts) > max_points:
            take = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
            pos = pts[np.unique(take)]
        else:
            pos = pts
        p = _clip_points(pos, origin, crop.shape)
        point_list = [p] if p is not None and len(p) else []
        label_list = [np.ones(len(p), dtype=np.int64)] if p is not None and len(p) else []
        if max_negatives > 0:
            x0, y0, x1, y1 = origin
            bg = ~pred_mask[y0:y1, x0:x1]
            ys, xs = np.nonzero(bg)
            if len(xs):
                take = np.linspace(0, len(xs) - 1,
                                   min(max_negatives, len(xs))).round().astype(int)
                neg = np.stack([xs[take], ys[take]], axis=1).astype(np.float32)
                point_list.append(neg)
                label_list.append(np.zeros(len(neg), dtype=np.int64))
        prompts.append(Prompt(
            crop_box=origin, box=box,
            points=np.concatenate(point_list, axis=0) if point_list else None,
            labels=np.concatenate(label_list, axis=0) if label_list else None,
            source="skeleton",
            meta={"path_index": idx, "path_length": polyline_length(pts),
                  "n_strands": len(st.get("paths", []))}))
    return prompts


# ---------------------------------------------------------------------------
# 预测实例
# ---------------------------------------------------------------------------
@dataclass
class PredInstance:
    mask: np.ndarray             # crop bool
    bbox: tuple                  # 原图 xyxy
    score: float
    source: str
    meta: dict = field(default_factory=dict)

    @cached_property
    def area(self) -> int:
        return int(self.mask.sum())


def pred_from_crop_mask(mask, origin, score: float, source: str, meta=None):
    """把 crop 内的 bool 掩码变成 PredInstance；空掩码返回 None。"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim > 2:
        mask = mask.reshape(-1, *mask.shape[-2:]).any(axis=0)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = mask[y0:y1, x0:x1]
    ox, oy = int(origin[0]), int(origin[1])
    return PredInstance(mask=crop, bbox=(ox + x0, oy + y0, ox + x1, oy + y1),
                        score=float(score), source=source,
                        meta=dict(meta or {}))


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------
def _color(i: int):
    # 固定、可复现的高对比色
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
               (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
               (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40)]
    return np.array(palette[i % len(palette)], dtype=np.uint8)


def save_overlay(path, image: np.ndarray, gt: GTInstances | None = None,
                 preds=None, *, scale: float = 0.5, alpha: float = 0.45):
    """把 GT 实例 / 预测实例叠到缩放后的原图上；不写全分辨率大图。"""
    h, w = image.shape[:2]
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    base = np.asarray(Image.fromarray(image).resize((sw, sh), Image.BILINEAR))
    out = base.copy().astype(np.float32)

    def blend(mask, color):
        if mask is None or not mask.any():
            return
        m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (sw, sh), Image.NEAREST)) > 127
        out[m] = out[m] * (1 - alpha) + color.astype(np.float32) * alpha

    if gt is not None:
        if gt.label_map is not None:
            small = np.asarray(Image.fromarray(gt.label_map.astype(np.uint16)).resize(
                (sw, sh), Image.NEAREST))
            for inst in gt.instances:
                blend(small == inst.instance_id, _color(inst.instance_id))
    if preds:
        for i, p in enumerate(preds):
            x0, y0, x1, y1 = [int(round(v * scale)) for v in p.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(sw, x1), min(sh, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.asarray(Image.fromarray(p.mask.astype(np.uint8) * 255).resize(
                (x1 - x0, y1 - y0), Image.NEAREST)) > 127
            if m.any():
                sub = out[y0:y1, x0:x1]
                sub[m] = sub[m] * (1 - alpha) + _color(i + 3).astype(np.float32) * alpha
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path, quality=92)
