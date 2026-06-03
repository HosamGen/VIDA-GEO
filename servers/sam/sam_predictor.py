# predictor.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


def _validate_points(points: np.ndarray, w: int, h: int, name: str = "points") -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be shape (N,2); got {points.shape}")
    xs = points[:, 0]
    ys = points[:, 1]
    if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
        bad = points[(xs < 0) | (xs >= w) | (ys < 0) | (ys >= h)]
        raise ValueError(f"{name} out of bounds for image (w={w}, h={h}). Bad: {bad.tolist()}")


def mask_to_png_bytes(mask_bool: np.ndarray) -> bytes:
    """
    Convert boolean mask -> PNG bytes (grayscale 0/255) without writing to disk.
    """
    import cv2

    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("Failed to encode mask PNG")
    return buf.tobytes()


@dataclass
class PointSegmentationResult:
    masks: np.ndarray      # (K,H,W) bool
    scores: np.ndarray     # (K,) float
    best_index: int


class SamService:
    """
    Loads SAM once and serves segmentations.
    Thread-safe via a lock (simplest reliable approach).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: Optional[str] = None,
        auto_generator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()

        self.sam = sam
        self.predictor = SamPredictor(self.sam)

        # Auto generator can be customized; leaving defaults keeps it simple.
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **(auto_generator_kwargs or {}))

        self._lock = threading.Lock()

    def segment_points(
        self,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = False,
    ) -> PointSegmentationResult:
        """
        points_xy: (N,2) float or int, in pixel coords (x,y)
        labels: (N,) int {0,1} background/foreground
        """
        h, w = image_rgb.shape[:2]
        _validate_points(points_xy, w, h, "points_xy")

        if labels.ndim != 1 or labels.shape[0] != points_xy.shape[0]:
            raise ValueError(f"labels must be shape (N,) matching points; got {labels.shape}")

        with self._lock, torch.no_grad():
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=points_xy.astype(np.float32),
                point_labels=labels.astype(np.int32),
                multimask_output=multimask_output,
            )

        # predictor returns masks shape (K,H,W)
        best_index = int(np.argmax(scores))
        return PointSegmentationResult(masks=masks.astype(bool), scores=scores, best_index=best_index)

    def segment_auto(
        self,
        image_rgb: np.ndarray,
        topk: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns top-K masks by area (descending).
        Each item includes: segmentation(bool HxW), area, bbox (xywh), and other SAM fields.
        """
        if topk < 1:
            topk = 1

        with self._lock, torch.no_grad():
            masks = self.mask_generator.generate(image_rgb)

        masks_sorted = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
        return masks_sorted[:topk]