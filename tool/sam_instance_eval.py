"""SAM 系列零样本实例分割评测。

三种提示方式：
- gt_box   : GT 每根 bbox（oracle 上限；不作为部署结果）
- text     : SAM3 文本提示，整图按 tile 切块，完全零样本
- skeleton : 现有语义模型 -> 骨架 -> 逐根 box/点 -> SAM 精修（可部署）

用法见 `python tool/sam_instance_eval.py -h`。示例：
    python tool/sam_instance_eval.py --check-gt \
        --data-dir datasets/root/test/GT --out-dir result/instance_eval/gt_check

    python tool/sam_instance_eval.py --prompt gt_box --backend sam \
        --limit-images 1 --limit-instances 5 \
        --out-dir result/instance_eval/smoke_gtbox_sam

    python tool/sam_instance_eval.py --prompt skeleton --backend sam2 \
        --semantic-model model_202610291403 \
        --out-dir result/instance_eval/full_skeleton_sam2

不修改训练/语义评测；SAM1/2/MedSAM/SAM3 都按原生 promptable 方式运行。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common.instance import (  # noqa: E402
    GTInstances, PredInstance, build_gt_instances, gt_prompts,
    pred_from_crop_mask, save_overlay, skeleton_prompts,
)
from common.instance_metrics import aggregate, evaluate_image  # noqa: E402


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def discover_pairs(data_dir):
    data_dir = Path(data_dir)
    out = []
    for jp in sorted(data_dir.glob("*.json")):
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = data_dir / f"{jp.stem}{ext}"
            if cand.is_file():
                img = cand
                break
        if img is not None:
            out.append({"name": jp.stem, "json": jp, "image": img})
    return out


def load_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def subset_gt(gt: GTInstances, n: int) -> GTInstances:
    if n <= 0 or n >= len(gt.instances):
        return gt
    ids = [inst.instance_id for inst in gt.instances[:n]]
    label = gt.label_map.copy()
    label[~np.isin(label, ids)] = 0
    label[gt.semantic_mask == 0] = 0
    sem = gt.semantic_mask
    info = dict(gt.info)
    info["n_instances"] = len(ids)
    info["coverage_gt_mask"] = float((sem & (label > 0)).sum() / max(1, sem.sum()))
    return GTInstances(
        stem=gt.stem, width=gt.width, height=gt.height,
        instances=gt.instances[:n], label_map=label,
        semantic_mask=sem, roi_mask=gt.roi_mask,
        unassigned_mask_px=int((sem & (label == 0)).sum()),
        overlap_px=gt.overlap_px,
        polygon_outside_mask_px=gt.polygon_outside_mask_px,
        info=info)


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(ua + ub - inter, 1e-12)


def select_mask(out, prompt_box):
    masks = out.get("masks")
    if masks is None or len(masks) == 0:
        return None
    if len(masks) == 1:
        return 0
    scores = np.asarray(out.get("scores", np.zeros(len(masks)))).reshape(-1)
    boxes = np.asarray(out.get("boxes", np.zeros((len(masks), 4)))).reshape(-1, 4)
    rank = []
    for i in range(len(masks)):
        rank.append(float(scores[i] if i < len(scores) else 0.0)
                    + 0.5 * box_iou(boxes[i], prompt_box))
    return int(np.argmax(rank))


def _pred_iou(a: PredInstance, b: PredInstance) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    # 只裁到 bbox 交集；IoU 的并集面积直接由两个 mask 面积相加减
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ca = a.mask[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    cb = b.mask[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    inter = int((ca & cb).sum())
    union = int(a.area + b.area - inter)
    return inter / max(union, 1)


def nms_preds(preds, thr: float):
    if not preds or thr >= 1.0:
        return list(preds), 0
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    kept = []
    suppressed = 0
    for i in order:
        p = preds[i]
        if any(_pred_iou(p, q) >= thr for q in kept):
            suppressed += 1
            continue
        kept.append(p)
    return kept, suppressed


def text_tiles(w: int, h: int, tile: int, overlap: int):
    if tile <= 0 or (w <= tile and h <= tile):
        return [(0, 0, w, h)]
    tw, th = min(tile, w), min(tile, h)
    ox, oy = max(1, tw - overlap), max(1, th - overlap)

    def starts(total, size, step):
        if total <= size:
            return [0]
        xs = list(range(0, total - size + 1, step))
        if xs[-1] != total - size:
            xs.append(total - size)
        return xs

    return [(x, y, x + tw, y + th)
            for y in starts(h, th, oy) for x in starts(w, tw, ox)]


# ---------------------------------------------------------------------------
# 语义模型 / 骨架提示
# ---------------------------------------------------------------------------
def load_semantic(args, device):
    from common import ckpt, predict
    pths, names = ckpt.resolve_pths(args.semantic_model)
    models, metas = ckpt.load_models(pths, device=device)
    tile = ckpt.infer_tile(metas, args.size)
    size = args.size or metas[0].get("size") or config.MAX_SIDE
    print(f"[skeleton] 语义模型: {', '.join(names)} | size={size} | tile={tile or 0}")
    return {"models": models, "metas": metas, "tile": tile, "size": size,
            "predict": predict}


def semantic_mask(image, sem):
    res = sem["predict"].predict(
        sem["models"], image, max_side=sem["size"], stride=config.STRIDE,
        device=sem["device"], tile=sem["tile"])
    return res["mask_counted"]


def run_gt_box(backend, image, gt, args):
    prompts = gt_prompts(
        gt, image, margin_px=args.crop_margin_px,
        margin_ratio=args.crop_margin_ratio,
        with_points=not args.gt_box_only,
        with_negatives=not args.gt_box_only)
    preds = []
    for pr in prompts:
        x0, y0, x1, y1 = pr.crop_box
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        backend.set_image(crop)
        out = backend.predict(box=pr.box, points=pr.points, labels=pr.labels,
                              multimask_output=False)
        idx = select_mask(out, pr.box)
        if idx is None:
            continue
        score = float(out["scores"][idx]) if len(out["scores"]) else 0.0
        pred = pred_from_crop_mask(out["masks"][idx], (x0, y0), score, "gt_box",
                                   meta={"prompt_instance": pr.meta.get("instance_id"),
                                         "prompt_variant": "gt_box"})
        if pred is not None:
            preds.append(pred)
    kept, suppressed = nms_preds(preds, args.nms_iou)
    return kept, suppressed, len(prompts)


def run_skeleton(backend, image, gt, sem, args):
    pred_mask = semantic_mask(image, sem)
    prompts = skeleton_prompts(
        pred_mask, image, spur=config.PRED_SPUR_LENGTH,
        min_len=config.MIN_ROOT_LENGTH, spacing=50.0,
        margin_px=args.skeleton_margin_px, margin_ratio=0.05,
        max_points=8, max_negatives=0 if args.gt_box_only else 4)
    preds = []
    for pr in prompts:
        x0, y0, x1, y1 = pr.crop_box
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        backend.set_image(crop)
        points = None if args.gt_box_only else pr.points
        labels = None if args.gt_box_only else pr.labels
        out = backend.predict(box=pr.box, points=points, labels=labels,
                              multimask_output=False)
        idx = select_mask(out, pr.box)
        if idx is None:
            continue
        m = out["masks"][idx]
        if args.clip_to_pred_mask:
            m = m & pred_mask[y0:y1, x0:x1]
        score = float(out["scores"][idx]) if len(out["scores"]) else 0.0
        pred = pred_from_crop_mask(m, (x0, y0), score, "skeleton",
                                   meta={"path_index": pr.meta.get("path_index"),
                                         "prompt_variant": "skeleton"})
        if pred is not None:
            preds.append(pred)
    kept, suppressed = nms_preds(preds, args.nms_iou)
    return kept, suppressed, len(prompts)


def run_text(backend, image, variants, args):
    h, w = image.shape[:2]
    tiles = text_tiles(w, h, args.text_tile, args.text_overlap)
    out_by_variant = {v: [] for v in variants}
    for (x0, y0, x1, y1) in tiles:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        backend.set_image(crop)
        for variant in variants:
            out = backend.predict(text=variant)
            masks = out.get("masks", [])
            scores = out.get("scores", [])
            for i, m in enumerate(masks):
                score = float(scores[i]) if i < len(scores) else 0.0
                pred = pred_from_crop_mask(
                    m, (x0, y0), score, "text",
                    meta={"prompt_variant": variant, "tile": (x0, y0, x1, y1)})
                if pred is not None:
                    out_by_variant[variant].append(pred)
    result = {}
    for variant, preds in out_by_variant.items():
        kept, suppressed = nms_preds(preds, args.nms_iou)
        result[variant] = (kept, suppressed, len(tiles) * len(variants))
    return result


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_rows(path: Path, rows):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8-sig")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def gt_row(gt: GTInstances):
    rows = []
    sem = max(1, int(gt.semantic_mask.sum()))
    for inst in gt.instances:
        rows.append({
            "image": gt.stem,
            "instance_id": inst.instance_id,
            "line_index": inst.line_index,
            "polygon_indices": ";".join(str(i) for i in inst.polygon_indices),
            "n_polygons": len(inst.polygon_indices),
            "area_px": inst.area,
            "bbox_x0": inst.bbox[0], "bbox_y0": inst.bbox[1],
            "bbox_x1": inst.bbox[2], "bbox_y1": inst.bbox[3],
            "line_length_px": round(inst.line_length, 3),
            "skel_length_px": round(inst.skel_length, 3),
            "n_strands": inst.n_strands,
            "group_cost": round(inst.group_cost, 4),
            "flags": ";".join(inst.flags),
            "semantic_area_px": sem,
            "coverage_gt_mask": round(gt.info.get("coverage_gt_mask", 0.0), 6),
            "unassigned_mask_px": gt.unassigned_mask_px,
            "overlap_px": gt.overlap_px,
            "n_skipped_lines": gt.info.get("n_skipped_lines", 0),
            "skipped_lines": ";".join(str(i) for i in gt.info.get("skipped_lines", [])),
        })
    return rows


def write_gt_csv(path: Path, gts):
    rows = []
    for gt in gts:
        rows.extend(gt_row(gt))
    write_rows(path, rows)


def write_pred_csv(path: Path, evals, mode, backend, variant):
    rows = []
    for e in evals:
        for i, p in enumerate(e.preds):
            rows.append({
                "image": e.image,
                "pred_id": i,
                "source": p.source,
                "score": round(float(p.score), 6),
                "area_px": p.area,
                "bbox_x0": p.bbox[0], "bbox_y0": p.bbox[1],
                "bbox_x1": p.bbox[2], "bbox_y1": p.bbox[3],
                "length_px": round(float(p.meta.get("length_px", 0.0)), 3),
                "prompt_mode": mode,
                "prompt_variant": p.meta.get("prompt_variant", variant),
                "backend": backend,
            })
    write_rows(path, rows)


def write_match_csv(path: Path, evals, mode, backend, variant):
    rows = []
    for e in evals:
        gt_ids = getattr(e, "_gt_ids", None)
        for p_idx, g_idx in e.matches:
            pred = e.preds[p_idx]
            rows.append({
                "image": e.image,
                "pred_id": p_idx,
                "gt_id": int(gt_ids[g_idx]) if gt_ids else g_idx,
                "iou": round(float(e.iou[p_idx, g_idx]), 6),
                "dice": round(float(e.dice[p_idx, g_idx]), 6),
                "score": round(float(pred.score), 6),
                "pred_length_px": round(float(pred.meta.get("length_px", 0.0)), 3),
                "gt_skel_length_px": round(float(e.gt_lengths[g_idx]), 3),
                "length_abs_err": round(abs(float(pred.meta.get("length_px", 0.0))
                                            - float(e.gt_lengths[g_idx])), 3),
                "prompt_mode": mode,
                "prompt_variant": pred.meta.get("prompt_variant", variant),
                "backend": backend,
            })
    write_rows(path, rows)


def prepare_out_dir(arg: str | None, default: Path):
    p = Path(arg) if arg else default
    if p.exists() and any(p.iterdir()):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = p.parent / f"{p.name}_{stamp}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_config(path: Path, args, extra=None):
    cfg = {"argv": sys.argv, "args": vars(args)}
    try:
        cfg["git_commit"] = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        cfg["git_commit"] = None
    try:
        import torch
        cfg["torch"] = torch.__version__
        cfg["cuda"] = torch.version.cuda
        cfg["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            cfg["device_name"] = torch.cuda.get_device_name(0)
            cfg["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # noqa: BLE001
        cfg["torch_error"] = f"{type(exc).__name__}: {exc}"
    if extra:
        cfg.update(extra)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM 系列零样本实例分割评测")
    p.add_argument("--prompt", choices=("gt_box", "text", "skeleton"),
                   default="gt_box")
    p.add_argument("--backend", choices=("sam", "medsam", "sam2", "sam3"),
                   default="sam")
    p.add_argument("--data-dir", default=str(config.TEST_DATA_DIR / "GT"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--semantic-model", default="model_202610291403")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--sam2-config",
                   default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam3-conf", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--limit-images", type=int, default=0)
    p.add_argument("--limit-instances", type=int, default=0)
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--overlay-max-images", type=int, default=3)
    p.add_argument("--overlay-scale", type=float, default=0.5)
    p.add_argument("--roi", choices=("gt_check", "none"), default="gt_check")
    p.add_argument("--score-thresh", type=float, default=None)
    p.add_argument("--nms-iou", type=float, default=None)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--crop-margin-px", type=float, default=16.0)
    p.add_argument("--crop-margin-ratio", type=float, default=0.1)
    p.add_argument("--skeleton-margin-px", type=float, default=12.0)
    p.add_argument("--clip-to-pred-mask", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--gt-box-only", action="store_true",
                   help="gt_box/skeleton 只用框，不加点提示")
    p.add_argument("--text-tile", type=int, default=1008)
    p.add_argument("--text-overlap", type=int, default=256)
    p.add_argument("--text-prompts", default="sugarcane root",
                   help="逗号分隔；text 模式会逐个 prompt 评测")
    p.add_argument("--check-gt", action="store_true")
    p.add_argument("--check-backends", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.prompt == "text" and args.backend != "sam3":
        p.error("--prompt text 只支持 --backend sam3")
    if args.score_thresh is None:
        args.score_thresh = 0.5 if args.prompt == "text" else 0.0
    if args.nms_iou is None:
        args.nms_iou = 0.5 if args.prompt in ("text", "skeleton") else 1.0
    return args


def pick_device(args):
    if args.cpu:
        return "cpu"
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[错误] CUDA 不可用；调试可加 --cpu")
    return f"cuda:{args.gpu}" if args.gpu is not None else "cuda:0"


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    pairs = discover_pairs(args.data_dir)
    if args.limit_images:
        pairs = pairs[:args.limit_images]
    if not pairs:
        raise SystemExit(f"[错误] {args.data_dir} 下没有图片+JSON 配对")
    default_name = f"{args.prompt}_{args.backend}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = prepare_out_dir(args.out_dir, config.RESULT_DIR / "instance_eval" / default_name)
    print(f"输出目录: {out_dir}")

    if args.check_backends:
        from common.sam_prompt import backend_status
        status = backend_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        run_config(out_dir / "run_config.json", args, {"backend_status": status})
        return

    gts = []
    for pair in pairs:
        gt = build_gt_instances(pair["json"], data_dir=args.data_dir,
                                verbose=args.verbose)
        if args.limit_instances:
            gt = subset_gt(gt, args.limit_instances)
        gts.append(gt)
        print(f"[GT] {gt.stem}: 线 {gt.info['n_lines']} / polygon "
              f"{gt.info['n_polygons']} / 实例 {len(gt.instances)} | "
              f"coverage {gt.info['coverage_gt_mask']:.3f} | "
              f"unassigned {gt.unassigned_mask_px}")
        if gt.info.get("n_skipped_lines"):
            print(f"  [警告] {gt.info['n_skipped_lines']} 条线在掩码里没有任何像素，"
                  f"已跳过: {gt.info.get('skipped_lines')}")

    write_gt_csv(out_dir / "gt_instances.csv", gts)
    if args.check_gt:
        print(f"GT 检查完成，用时 {time.time() - t0:.1f}s")
        run_config(out_dir / "run_config.json", args,
                   {"gts": [g.info for g in gts]})
        return

    device = pick_device(args)
    from common.sam_prompt import build_backend
    backend = build_backend(args.backend, device=device,
                            checkpoint=args.checkpoint,
                            sam2_config=args.sam2_config,
                            sam3_confidence=args.sam3_conf)
    print(f"[backend] {args.backend} 已加载")

    sem = None
    if args.prompt == "skeleton":
        sem = load_semantic(args, device)
        sem["device"] = device

    variants = ([v.strip() for v in args.text_prompts.split(",") if v.strip()]
                if args.prompt == "text" else [args.prompt])
    evals_by_variant = {v: [] for v in variants}
    n_overlays = 0
    for idx, pair in enumerate(pairs):
        image = load_image(pair["image"])
        gt = gts[idx]
        if args.prompt == "gt_box":
            preds, suppressed, n_prompts = run_gt_box(backend, image, gt, args)
            per_variant = {args.prompt: (preds, suppressed, n_prompts)}
        elif args.prompt == "skeleton":
            preds, suppressed, n_prompts = run_skeleton(backend, image, gt, sem, args)
            per_variant = {args.prompt: (preds, suppressed, n_prompts)}
        else:
            per_variant = run_text(backend, image, variants, args)

        for variant, (preds, suppressed, n_prompts) in per_variant.items():
            e = evaluate_image(
                gt, preds, image_name=pair["name"], roi_mode=args.roi,
                score_thresh=args.score_thresh, match_iou=args.match_iou,
                nms_suppressed=suppressed)
            e.row["prompt_mode"] = args.prompt
            e.row["prompt_variant"] = variant
            e.row["backend"] = args.backend
            e.row["n_prompts"] = n_prompts
            e._gt_ids = [inst.instance_id for inst in gt.instances]
            evals_by_variant[variant].append(e)
            print(f"[{pair['name']}] {args.prompt}/{args.backend}/{variant}: "
                  f"GT {e.row['n_gt_valid']} / pred {e.row['n_pred']} / "
                  f"matched {e.row['n_matched']} / AP50 {e.row['ap50']:.3f}")
            if args.save_overlays and n_overlays < args.overlay_max_images:
                od = out_dir / "overlays"
                od.mkdir(parents=True, exist_ok=True)
                tag = variant.replace(" ", "_").replace("/", "_")
                save_overlay(od / f"{args.prompt}_{args.backend}_{tag}_{pair['name']}_gt.jpg",
                             image, gt=gt, scale=args.overlay_scale)
                save_overlay(od / f"{args.prompt}_{args.backend}_{tag}_{pair['name']}_pred.jpg",
                             image, preds=preds, scale=args.overlay_scale)
                n_overlays += 1

    all_evals = []
    all_rows = []
    summaries = []
    for variant, evals in evals_by_variant.items():
        all_evals.extend(evals)
        all_rows.extend(e.row for e in evals)
        summaries.append(aggregate(
            evals, prompt_mode=args.prompt, backend=args.backend,
            prompt_variant=variant, runtime_s=time.time() - t0))
    write_rows(out_dir / "per_image.csv", all_rows)
    write_rows(out_dir / "summary.csv", summaries)
    write_pred_csv(out_dir / "pred_instances.csv", all_evals, args.prompt,
                   args.backend, args.prompt if args.prompt != "text" else "all")
    write_match_csv(out_dir / "matches.csv", all_evals, args.prompt,
                    args.backend, args.prompt if args.prompt != "text" else "all")
    run_config(out_dir / "run_config.json", args, {
        "gts": [g.info for g in gts],
        "summary": summaries,
    })

    print("\n" + "=" * 72)
    for s in summaries:
        print(f"{s.get('prompt_mode')}/{s.get('backend')}/{s.get('prompt_variant')}: "
              f"AP50 {s.get('ap50', float('nan')):.4f} | "
              f"AP50-95 {s.get('ap50_95', float('nan')):.4f} | "
              f"IoU {s.get('mean_matched_iou', float('nan')):.4f} | "
              f"根数MAE {s.get('count_mae', float('nan')):.2f} | "
              f"总长MAE {s.get('total_len_mae', float('nan')):.1f}px")
    print(f"用时 {time.time() - t0:.1f}s | 结果: {out_dir}")


if __name__ == "__main__":
    main()
