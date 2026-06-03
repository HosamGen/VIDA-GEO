# sam3_predictor.py
"""
Reusable SAM3 text-prompt segmentation predictor.

Goal:
- Load SAM3 once (optionally at server startup)
- Provide a clean Python API for:
    - segmenting an image given a TEXT prompt (e.g., "park")
    - returning a binary mask as uint8 (0/255) in the ORIGINAL image resolution

Notes:
- SAM3's image API (Sam3Processor) internally resizes inputs (commonly 1008x1008),
  so we always resize the predicted mask back to the original image size with NEAREST.
- The code is defensive about output structure and supports unioning multiple instance masks.

Environment vars (optional):
- SAM3_DEVICE: "cuda" / "cpu" / "cuda:0"
- SAM3_SCORE_THRESH: float (default 0.25)
- SAM3_MASK_THRESH: float (default 0.5)
- SAM3_COMPILE: "1" to enable torch.compile if supported by your SAM3 build

Run inside the sam3 conda env where `sam3` package is importable.
"""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image

# SAM3 imports (repo/package)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


@dataclass(frozen=True)
class Sam3SegResult:
    """Result of a SAM3 segmentation request."""
    mask_uint8: np.ndarray              # HxW uint8 values 0/255 at ORIGINAL image size
    num_instances: int
    kept_indices: List[int]             # indices kept after score thresholding
    scores: List[float]                 # kept scores (sorted desc)
    prompt: str


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "NO")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() != "" else default


class Sam3Predictor:
    """
    A persistent predictor that runs SAM3 text-prompt segmentation.
    Thread-safe via a lock (SAM3 uses internal state objects).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        score_thresh: float = 0.25,
        mask_thresh: float = 0.5,
        compile_model: Optional[bool] = None,
    ) -> None:
        self.device_str = device or _env_str("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.score_thresh = float(score_thresh if score_thresh is not None else _env_float("SAM3_SCORE_THRESH", 0.25))
        self.mask_thresh = float(mask_thresh if mask_thresh is not None else _env_float("SAM3_MASK_THRESH", 0.5))

        if compile_model is None:
            compile_model = _env_bool("SAM3_COMPILE", False)
        self.compile_model = bool(compile_model)

        self.device = torch.device(self.device_str if (self.device_str != "cuda" or torch.cuda.is_available()) else "cpu")

        self._lock = threading.RLock()
        self._loaded = False

        self.model = None
        self.processor = None

    def load(self) -> None:
        """Idempotent model load."""
        with self._lock:
            if self._loaded:
                return

            # build_sam3_image_model signature can vary by repo version; handle optional compile kwarg
            kwargs: Dict[str, Any] = dict(device=str(self.device), eval_mode=True)
            if self.compile_model:
                kwargs["compile"] = True

            try:
                self.model = build_sam3_image_model(**kwargs)
            except TypeError:
                # Older versions may not support compile kwarg
                kwargs.pop("compile", None)
                self.model = build_sam3_image_model(**kwargs)

            self.processor = Sam3Processor(self.model)
            self._loaded = True

    def debug_state(self) -> Dict[str, Any]:
        return {
            "loaded": bool(self._loaded),
            "device": str(self.device),
            "score_thresh": float(self.score_thresh),
            "mask_thresh": float(self.mask_thresh),
            "compile_model": bool(self.compile_model),
            "torch_version": getattr(torch, "__version__", "unknown"),
        }

    @staticmethod
    def _empty_mask(w: int, h: int) -> np.ndarray:
        return np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def mask_to_png_bytes(mask_uint8: np.ndarray) -> bytes:
        if mask_uint8.dtype != np.uint8:
            mask_uint8 = mask_uint8.astype(np.uint8)
        if mask_uint8.ndim != 2:
            raise ValueError("mask_uint8 must be HxW (single-channel)")
        buf = io.BytesIO()
        Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def save_mask_png(mask_uint8: np.ndarray, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        Image.fromarray(mask_uint8.astype(np.uint8), mode="L").save(tmp_path, format="PNG")
        os.replace(tmp_path, out_path)

    def segment(
        self,
        image: Image.Image,
        prompt: str,
        score_thresh: Optional[float] = None,
        mask_thresh: Optional[float] = None,
        union: bool = True,
    ) -> Sam3SegResult:
        """
        Segment using a TEXT prompt.
        Returns mask at the original image resolution.

        - score_thresh: filter instances by SAM3 score
        - mask_thresh: threshold mask probabilities to binary
        - union: if True, union all kept masks; else take the top-1 mask
        """
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt must be a non-empty string")

        with self._lock:
            self.load()
            assert self.processor is not None

            st = float(self.score_thresh if score_thresh is None else score_thresh)
            mt = float(self.mask_thresh if mask_thresh is None else mask_thresh)

            img = image.convert("RGB")
            orig_w, orig_h = img.size

            with torch.inference_mode():
                state = self.processor.set_image(img)
                out = self.processor.set_text_prompt(state=state, prompt=str(prompt))

                masks = out.get("masks", None)    # expected (N,1,H,W)
                scores = out.get("scores", None)  # expected (N,)

            if masks is None or scores is None or (not torch.is_tensor(masks)) or (not torch.is_tensor(scores)):
                # Return empty but be explicit
                return Sam3SegResult(
                    mask_uint8=self._empty_mask(orig_w, orig_h),
                    num_instances=0,
                    kept_indices=[],
                    scores=[],
                    prompt=str(prompt),
                )

            if masks.ndim != 4 or masks.shape[1] != 1:
                # Unexpected shape; try to coerce but fail loudly if not possible
                raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)} (expected N,1,H,W)")

            n = int(masks.shape[0])
            if n == 0:
                return Sam3SegResult(
                    mask_uint8=self._empty_mask(orig_w, orig_h),
                    num_instances=0,
                    kept_indices=[],
                    scores=[],
                    prompt=str(prompt),
                )

            masks_f = masks.detach().cpu().float()[:, 0, :, :]  # (N,H,W)
            scores_f = scores.detach().cpu().float()            # (N,)

            keep = scores_f >= st
            kept_idx = keep.nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(kept_idx, int):
                kept_idx = [kept_idx]

            if len(kept_idx) == 0:
                return Sam3SegResult(
                    mask_uint8=self._empty_mask(orig_w, orig_h),
                    num_instances=0,
                    kept_indices=[],
                    scores=[],
                    prompt=str(prompt),
                )

            # Sort kept by score desc
            kept_scores = scores_f[kept_idx]
            order = torch.argsort(kept_scores, descending=True)
            kept_idx_sorted = [kept_idx[int(i)] for i in order]
            kept_scores_sorted = [float(kept_scores[int(i)].item()) for i in order]

            if union:
                bin_masks = (masks_f[kept_idx_sorted] >= mt)  # (K,H,W) bool
                union_mask = bin_masks.any(dim=0).numpy().astype(np.uint8) * 255  # (H,W)
            else:
                top_i = kept_idx_sorted[0]
                union_mask = (masks_f[top_i] >= mt).numpy().astype(np.uint8) * 255

            # Resize to original image size (SAM3 processor uses internal resized image)
            mask_img = Image.fromarray(union_mask, mode="L").resize((orig_w, orig_h), resample=Image.NEAREST)
            mask_u8 = np.array(mask_img, dtype=np.uint8)

            return Sam3SegResult(
                mask_uint8=mask_u8,
                num_instances=len(kept_idx_sorted),
                kept_indices=kept_idx_sorted,
                scores=kept_scores_sorted,
                prompt=str(prompt),
            )

    def segment_path(
        self,
        image_path: str,
        prompt: str,
        score_thresh: Optional[float] = None,
        mask_thresh: Optional[float] = None,
        union: bool = True,
    ) -> Sam3SegResult:
        """Convenience wrapper: load image from disk and segment."""
        with Image.open(image_path) as im:
            return self.segment(im, prompt=prompt, score_thresh=score_thresh, mask_thresh=mask_thresh, union=union)
