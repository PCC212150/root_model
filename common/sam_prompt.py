"""原生 promptable SAM 后端（SAM1 / MedSAM / SAM2 / SAM3）。

与 `common/sam_seg.py` 等**语义**包装器不同：这里保留 prompt encoder /
mask decoder，按原生方式接收 box / point / text prompt，输出一组实例掩码。
仅用于实例评测（`tool/sam_instance_eval.py`），不参与训练。

依赖：
- SAM1/MedSAM: `Net/segment-anything-main`（已入库，无需 pip install）
- SAM2: `Net/sam2-main`，需要 hydra-core / omegaconf / iopath
- SAM3: `Net/sam3`，需要 timm / ftfy / regex / iopath / hydra-core

上游包用 `sys.path` 显式前置，避免被环境里同名的 pip 包顶掉。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config

PROJECT_ROOT = config.PROJECT_ROOT


def _ensure_package(pkg_name: str, rel_path: str):
    """把仓库内上游包放到 sys.path 最前，并确认 import 到的是它。"""
    path = (PROJECT_ROOT / rel_path).resolve()
    if not path.is_dir():
        raise RuntimeError(f"找不到上游包目录: {path}")
    s = str(path)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    mod = __import__(pkg_name)
    mod_path = Path(mod.__file__).resolve()
    if path not in mod_path.parents and mod_path != path:
        raise RuntimeError(
            f"{pkg_name} 实际来自 {mod_path}，不是仓库内的 {path}；"
            f"请检查环境里是否有同名 pip 包。")
    return mod


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _scores_to_prob(scores) -> np.ndarray:
    s = _to_numpy(scores).astype(np.float64).reshape(-1)
    if s.size and (np.nanmin(s) < 0.0 or np.nanmax(s) > 1.0):
        s = 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
    return s


def _mask_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pack_masks(masks, scores, boxes=None, prompt_box=None) -> dict:
    """把后端原始输出统一成 (N,H,W) bool + (N,) score + (N,4) xyxy。"""
    masks = _to_numpy(masks)
    if masks.ndim == 4:
        # (B,C,H,W) -> (B,H,W)；SAM1/SAM2 的 C 通常为 1/3
        masks = masks.reshape(-1, *masks.shape[-2:])
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.size == 0:
        return {"masks": np.zeros((0, 0, 0), dtype=bool),
                "scores": np.zeros(0, dtype=np.float64),
                "boxes": np.zeros((0, 4), dtype=np.float32)}
    masks = masks.astype(bool)
    probs = _scores_to_prob(scores)
    if probs.size < len(masks):
        probs = np.pad(probs, (0, len(masks) - probs.size), constant_values=0.0)
    probs = probs[:len(masks)]

    if boxes is None:
        out_boxes = []
        for m in masks:
            b = _mask_box(m)
            if b is None and prompt_box is not None:
                b = tuple(float(v) for v in prompt_box)
            out_boxes.append(b if b is not None else (0.0, 0.0, 0.0, 0.0))
        boxes = np.asarray(out_boxes, dtype=np.float32)
    else:
        boxes = _to_numpy(boxes).astype(np.float32).reshape(-1, 4)
        if len(boxes) != len(masks):
            boxes = np.resize(boxes, (len(masks), 4))
    return {"masks": masks, "scores": probs, "boxes": boxes}


class PromptableBackend:
    name = "base"
    supports_text = False

    def set_image(self, image: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, *, box=None, points=None, labels=None, text=None,
                multimask_output=True) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SAM1
# ---------------------------------------------------------------------------
class SAM1Backend(PromptableBackend):
    name = "sam"

    def __init__(self, device="cuda", checkpoint=None, variant="vit_b"):
        _ensure_package("segment_anything", "Net/segment-anything-main")
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = Path(checkpoint or config.SAM_PRETRAINED_CKPT)
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM 权重不存在: {ckpt}")
        try:
            model = sam_model_registry[variant](checkpoint=str(ckpt))
        except Exception as exc:  # noqa: BLE001 - 只对 weights_only 错误回退
            # torch>=2.6 的 weights_only 默认值可能拦住官方 checkpoint
            msg = str(exc)
            if "weights_only" not in msg and "Unsupported global" not in msg:
                raise
            model = sam_model_registry[variant](checkpoint=None)
            sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
                sd = sd["model"]
            model.load_state_dict(sd)
        self.device = device
        self.model = model.to(device).eval()
        self.predictor = SamPredictor(self.model)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)

    @torch.inference_mode()
    def predict(self, *, box=None, points=None, labels=None, text=None,
                multimask_output=True) -> dict:
        masks, scores, _ = self.predictor.predict(
            point_coords=None if points is None else np.asarray(points),
            point_labels=None if labels is None else np.asarray(labels),
            box=None if box is None else np.asarray(box),
            multimask_output=bool(multimask_output))
        return pack_masks(masks, scores, prompt_box=box)


# ---------------------------------------------------------------------------
# MedSAM（ViT-B 架构 + 官方 min-max 预处理）
# ---------------------------------------------------------------------------
class MedSAMBackend(PromptableBackend):
    name = "medsam"

    def __init__(self, device="cuda", checkpoint=None):
        _ensure_package("segment_anything", "Net/segment-anything-main")
        from segment_anything import sam_model_registry

        ckpt = Path(checkpoint or config.MEDSAM_PRETRAINED_CKPT)
        if not ckpt.is_file():
            raise FileNotFoundError(f"MedSAM 权重不存在: {ckpt}")
        try:
            model = sam_model_registry["vit_b"](checkpoint=str(ckpt))
        except Exception as exc:  # noqa: BLE001 - 只对 weights_only 错误回退
            msg = str(exc)
            if "weights_only" not in msg and "Unsupported global" not in msg:
                raise
            model = sam_model_registry["vit_b"](checkpoint=None)
            sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
                sd = sd["model"]
            model.load_state_dict(sd)
        self.device = device
        self.model = model.to(device).eval()
        self._embedding = None
        self._hw = None

    def set_image(self, image: np.ndarray) -> None:
        from PIL import Image

        h, w = image.shape[:2]
        self._hw = (h, w)
        arr = np.asarray(Image.fromarray(image).resize(
            (1024, 1024), Image.BILINEAR), dtype=np.float32)
        mn = float(arr.min())
        mx = float(arr.max())
        arr = (arr - mn) / max(mx - mn, 1e-8)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self._embedding = self.model.image_encoder(t)

    @torch.inference_mode()
    def predict(self, *, box=None, points=None, labels=None, text=None,
                multimask_output=True) -> dict:
        if self._embedding is None or self._hw is None:
            raise RuntimeError("必须先 set_image")
        h, w = self._hw
        scale = torch.tensor([1024.0 / w, 1024.0 / h, 1024.0 / w, 1024.0 / h],
                             device=self.device)
        box_t = None
        if box is not None:
            box_t = torch.as_tensor(box, dtype=torch.float32,
                                    device=self.device).reshape(1, 4) * scale
        points_t = None
        if points is not None:
            p = torch.as_tensor(points, dtype=torch.float32,
                                device=self.device).reshape(-1, 2)
            p = p * torch.tensor([1024.0 / w, 1024.0 / h], device=self.device)
            points_t = (p.unsqueeze(0), torch.as_tensor(
                labels, dtype=torch.int64, device=self.device).reshape(1, -1))
        sparse, dense = self.model.prompt_encoder(
            points=points_t, boxes=box_t, masks=None)
        low_res, iou_pred = self.model.mask_decoder(
            image_embeddings=self._embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        prob = torch.sigmoid(low_res)
        prob = F.interpolate(prob, size=(h, w), mode="bilinear",
                             align_corners=False)
        mask = (prob[0, 0] > 0.5).cpu().numpy()
        score = torch.sigmoid(iou_pred).reshape(-1)[0].item() if iou_pred is not None else 1.0
        return pack_masks(mask[None], [score], prompt_box=box)


# ---------------------------------------------------------------------------
# SAM2
# ---------------------------------------------------------------------------
class SAM2Backend(PromptableBackend):
    name = "sam2"

    def __init__(self, device="cuda", checkpoint=None,
                 config_file="configs/sam2.1/sam2.1_hiera_t.yaml"):
        _ensure_package("sam2", "Net/sam2-main")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = Path(checkpoint or config.SAM2_PRETRAINED_CKPT)
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM2 权重不存在: {ckpt}")
        model = build_sam2(config_file, str(ckpt), device=device)
        self.device = device
        self.model = model.to(device).eval()
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)

    @torch.inference_mode()
    def predict(self, *, box=None, points=None, labels=None, text=None,
                multimask_output=True) -> dict:
        masks, scores, _ = self.predictor.predict(
            point_coords=None if points is None else np.asarray(points),
            point_labels=None if labels is None else np.asarray(labels),
            box=None if box is None else np.asarray(box),
            multimask_output=bool(multimask_output))
        return pack_masks(masks, scores, prompt_box=box)


# ---------------------------------------------------------------------------
# SAM3（text / geometric box）
# ---------------------------------------------------------------------------
class SAM3Backend(PromptableBackend):
    name = "sam3"
    supports_text = True

    def __init__(self, device="cuda", checkpoint=None, confidence=0.01):
        _ensure_package("sam3", "Net/sam3")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        ckpt = Path(checkpoint or config.SAM3_PRETRAINED_CKPT)
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM3 权重不存在: {ckpt}")
        model = build_sam3_image_model(
            checkpoint_path=str(ckpt), load_from_HF=False,
            enable_segmentation=True, enable_inst_interactivity=False,
            device=device, eval_mode=True, compile=False)
        self.device = device
        self.model = model
        self.processor = Sam3Processor(model, resolution=1008, device=device,
                                       confidence_threshold=float(confidence))
        self._state = None

    def set_image(self, image: np.ndarray) -> None:
        # Sam3Processor 对 numpy 用 shape[-2:] 取 H/W，HWC 输入会拿到 (W,3)；
        # 传 PIL 才能正确记录 original_height/original_width。
        from PIL import Image
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        self._state = self.processor.set_image(pil)

    @torch.inference_mode()
    def predict(self, *, box=None, points=None, labels=None, text=None,
                multimask_output=True) -> dict:
        if self._state is None:
            raise RuntimeError("必须先 set_image")
        self.processor.reset_all_prompts(self._state)
        if text is not None:
            state = self.processor.set_text_prompt(str(text), self._state)
        elif box is not None:
            h, w = self._state["original_height"], self._state["original_width"]
            x0, y0, x1, y1 = [float(v) for v in np.asarray(box).reshape(-1)[:4]]
            cx, cy = (x0 + x1) * 0.5 / w, (y0 + y1) * 0.5 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            state = self.processor.add_geometric_prompt(
                [cx, cy, bw, bh], True, self._state)
        else:
            raise ValueError("SAM3 需要 text 或 box prompt（本适配器不支持单独点提示）")
        self._state = state
        masks = state.get("masks")
        if masks is None:
            return {"masks": np.zeros((0, 0, 0), dtype=bool),
                    "scores": np.zeros(0, dtype=np.float64),
                    "boxes": np.zeros((0, 4), dtype=np.float32)}
        return pack_masks(masks, state.get("scores", []),
                          boxes=state.get("boxes"), prompt_box=box)


def build_backend(name: str, device="cuda", *, checkpoint=None,
                  sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
                  sam3_confidence=0.01) -> PromptableBackend:
    name = name.lower()
    if name == "sam":
        return SAM1Backend(device=device, checkpoint=checkpoint)
    if name == "medsam":
        return MedSAMBackend(device=device, checkpoint=checkpoint)
    if name == "sam2":
        return SAM2Backend(device=device, checkpoint=checkpoint,
                           config_file=sam2_config)
    if name == "sam3":
        return SAM3Backend(device=device, checkpoint=checkpoint,
                           confidence=sam3_confidence)
    raise ValueError(f"未知 backend: {name!r}（可选 sam/medsam/sam2/sam3）")


def backend_status() -> dict:
    """不构建模型，只报告路径/依赖是否可见。"""
    out = {}
    for name, path in (("segment_anything", "Net/segment-anything-main"),
                       ("sam2", "Net/sam2-main"),
                       ("sam3", "Net/sam3")):
        p = PROJECT_ROOT / path
        item = {"path": str(p), "exists": p.is_dir()}
        try:
            mod = _ensure_package(name, path)
            item["import"] = str(Path(mod.__file__).resolve())
        except Exception as exc:  # noqa: BLE001 - 状态报告需要把失败原因带出来
            item["import_error"] = f"{type(exc).__name__}: {exc}"
        out[name] = item
    return out
