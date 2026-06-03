#!/usr/bin/env python3
"""
oem_inprocess_segmenter.py — in-process (no subprocess) OEM-lightweight inference.

Why this exists
---------------
The upstream oem-lightweight demo script (eval_oem_lightweight.py) is a *demo*:
it loads the model, profiles FLOPs, does TTA, computes IoU, plots figures, and
exits. Calling it via subprocess for every request is therefore very slow.

This module loads the OEM model *once* and exposes a small predict/score API
that you can call repeatedly from FastAPI (or any Python code) without
re-loading weights.

It intentionally avoids importing oem-lightweight's `config.py` to dodge its
"path contains repo_name" assumption. We instead hardcode the small constants
we need (num_classes + mean/std + class colors) taken from the repo.

Upstream references:
- eval_oem_lightweight.py uses `oem_lightweight.model.sparsemask/fasterseg`
  and `SegEvaluator.evaluate()` for prediction. citeturn2view0turn4view1
- Default config enables TTA (`use_tta=True`). citeturn2view1turn4view1
- The demo data loader resizes inputs to 1024×1024. citeturn4view0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import numpy as np

# These come from oem-lightweight/config.py (stats + class_colors). citeturn2view1
OEM_MEAN = (0.4325, 0.4483, 0.3879)
OEM_STD  = (0.0195, 0.0169, 0.0179)

# Class order in the model output (0..7) matches config.class_names/class_colors. citeturn2view1turn4view1
# 0 bareland, 1 rangeland, 2 developed space, 3 road, 4 tree, 5 water, 6 agriculture land, 7 buildings
OEM_CLASS_NAMES = (
    "bareland",
    "rangeland",
    "developed_space",
    "road",
    "tree",
    "water",
    "agriculture",
    "building",
)

# Colours saved by the demo are based on config.class_colors. citeturn2view1turn4view1
OEM_CLASS_COLORS_RGB = np.array(
    [
        (128, 0, 0),      # bareland
        (0, 255, 0),      # rangeland
        (192, 192, 192),  # developed space
        (255, 255, 255),  # road
        (49, 139, 87),    # tree
        (0, 0, 255),      # water
        (127, 255, 0),    # agriculture land
        (255, 0, 0),      # buildings
    ],
    dtype=np.uint8,
)

NAME_TO_IDX = {n: i for i, n in enumerate(OEM_CLASS_NAMES)}


@dataclass(frozen=True)
class ScoreResult:
    greenery_score: float
    tree_fraction: float
    rangeland_fraction: float
    agriculture_fraction: float
    class_fractions: Dict[str, float]
    total_pixels: int
    greenery_classes_used: Sequence[str]
    image_size: Tuple[int, int]  # (W, H)


def _resolve_relative(p: str, base_dir: Path) -> str:
    pp = Path(p)
    return str((base_dir / pp).resolve()) if not pp.is_absolute() else str(pp)


def _safe_import_torch():
    # Torch import is intentionally delayed so this module can be imported even
    # in environments where torch isn't installed (e.g., for static analysis).
    import torch  # noqa: F401
    return torch


def _load_image_bgr(path: str, target_size: int) -> np.ndarray:
    """
    Load image and resize to (target_size, target_size).

    Uses OpenCV if available (fast), otherwise falls back to PIL.
    Matches OEM demo behaviour, which resizes both image and label to 1024×1024. citeturn4view0
    """
    try:
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise ValueError("cv2.imread returned None")

        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return img

    except Exception:
        # Fallback: PIL -> RGB -> BGR
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        if pil.size != (target_size, target_size):
            pil = pil.resize((target_size, target_size), resample=Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.uint8)  # RGB
        return arr[:, :, ::-1]  # BGR


class OemLightweightSegmenter:
    """
    Loads OEM-lightweight model once and provides fast per-image inference.

    Supported models:
      - sparsemask
      - fasterseg
    """

    def __init__(
        self,
        oem_repo: str,
        model: str,
        arch: str,
        weights: str,
        device: str = "cuda",
        use_tta: bool = False,
        input_size: int = 1024,
        use_fp16: bool = False,
    ) -> None:
        self.oem_repo = Path(oem_repo).expanduser().resolve()
        self.model_kind = model.lower().strip()
        self.arch = _resolve_relative(arch, self.oem_repo)
        self.weights = _resolve_relative(weights, self.oem_repo)
        self.input_size = int(input_size)
        self.use_tta = bool(use_tta)
        self.use_fp16 = bool(use_fp16)

        torch = _safe_import_torch()

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For fixed-size inference, this often improves performance on CUDA.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Ensure the OEM repo modules can be imported
        import sys
        sys.path.insert(0, str(self.oem_repo))
        sys.path.insert(0, str(self.oem_repo / "fasterseg_api"))  # fasterseg uses non-package imports citeturn5view1
        sys.path.insert(0, str(self.oem_repo / "sparsemask_api"))
        sys.path.insert(0, str(self.oem_repo / "oem_lightweight"))

        self._torch = torch
        self._model = self._load_model()
        self._model.to(self.device)
        self._model.eval()

        mean = torch.tensor(OEM_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor(OEM_STD,  dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

    def _load_model(self):
        torch = self._torch
        if self.model_kind == "sparsemask":
            # Equivalent to oem_lightweight.model.sparsemask(...) citeturn4view2turn5view0
            import numpy as _np
            from sparsemask_api.sparse_mask_eval_mode import SparseMask

            mask = _np.load(self.arch)
            model = SparseMask(mask, backbone_name="mobilenet_v2", depth=64, in_channels=3, num_classes=8)

            weights_obj = torch.load(self.weights, map_location="cpu")
            state_dict = weights_obj.get("state_dict", weights_obj)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            self._model_name = "SparseMask"
            return model

        if self.model_kind == "fasterseg":
            # Equivalent to oem_lightweight.model.fasterseg(...) citeturn4view2turn5view1
            from fasterseg_api.model_seg import Network_Multi_Path_Infer as Network

            state = torch.load(self.arch, map_location="cpu")
            model = Network(
                [state["alpha_1_0"].detach(), state["alpha_1_1"].detach(), state["alpha_1_2"].detach()],
                [None, state["beta_1_1"].detach(), state["beta_1_2"].detach()],
                [state["ratio_1_0"].detach(), state["ratio_1_1"].detach(), state["ratio_1_2"].detach()],
                num_classes=8,
                layers=16,
                Fch=12,
                width_mult_list=[4.0 / 12, 6.0 / 12, 8.0 / 12, 10.0 / 12, 1.0],
                stem_head_width=(8.0 / 12, 8.0 / 12),
                ignore_skip=False,
            )
            model.build_structure([2, 1])

            weights_dict = torch.load(self.weights, map_location="cpu")
            sd = model.state_dict()
            # Only load matching keys (same as upstream). citeturn4view2
            weights_dict = {k: v for k, v in weights_dict.items() if k in sd}
            sd.update(weights_dict)
            model.load_state_dict(sd)

            self._model_name = "FasterSeg"
            return model

        raise ValueError(f"Unknown model '{self.model_kind}'. Expected 'sparsemask' or 'fasterseg'.")

    def _preprocess(self, image_bgr: np.ndarray):
        """
        Convert a (H,W,3) uint8 BGR image into a normalized float tensor (1,3,H,W).
        Matches SegEvaluator.evaluate() logic regarding channel order. citeturn4view1
        """
        torch = self._torch

        # For FasterSeg only, upstream flips channels (BGR->RGB). citeturn4view1
        img = image_bgr
        if self._model_name == "FasterSeg":
            img = img[:, :, ::-1]

        # Torch expects contiguous array for from_numpy.
        img = np.ascontiguousarray(img)

        x = torch.from_numpy(img).to(device=self.device, dtype=torch.float32)  # (H,W,3)
        x = x.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,H,W) in [0,1]
        x = (x - self._mean) / self._std
        return x

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict_label_map_from_path(self, image_path: str) -> np.ndarray:
        img_bgr = _load_image_bgr(image_path, target_size=self.input_size)
        return self.predict_label_map(img_bgr)

    def predict_label_map(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Returns (H,W) uint8 label map with values 0..7 following OEM_CLASS_NAMES.
        """
        torch = self._torch

        x = self._preprocess(image_bgr)

        with torch.inference_mode():
            if self.use_fp16 and self.device.type == "cuda":
                autocast = torch.cuda.amp.autocast
            else:
                # Dummy context manager
                from contextlib import nullcontext
                autocast = lambda: nullcontext()

            with autocast():
                if self.use_tta:
                    # Upstream TTA runs model twice (original + horizontal flip). citeturn4view1
                    # NOTE: evaluator applies exp() after summing, but exp is monotonic so argmax is unchanged.
                    logits = self._model(x)
                    logits_flip = self._model(x.flip(-1)).flip(-1)
                    logits = logits + logits_flip
                else:
                    logits = self._model(x)

                # For SparseMask only, upstream interpolates logits to image size. citeturn4view1
                if self._model_name == "SparseMask":
                    h, w, _ = image_bgr.shape
                    logits = torch.nn.functional.interpolate(
                        logits, size=(h, w), mode="bilinear", align_corners=True
                    )

            pred = logits.argmax(dim=1).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
            return pred

    def score_label_map(
        self,
        label_map: np.ndarray,
        greenery_classes: Optional[Iterable[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> ScoreResult:
        if label_map.ndim != 2:
            raise ValueError(f"label_map must be (H,W), got shape {label_map.shape}")

        total_pixels = int(label_map.size)
        counts = np.bincount(label_map.reshape(-1), minlength=len(OEM_CLASS_NAMES)).astype(np.int64)
        fracs = counts / max(1, total_pixels)

        class_fractions = {OEM_CLASS_NAMES[i]: float(fracs[i]) for i in range(len(OEM_CLASS_NAMES))}

        if greenery_classes is None:
            greenery_classes = ("tree", "rangeland", "agriculture")
        greenery_classes_norm = [c.strip().lower() for c in greenery_classes if str(c).strip()]
        green_ids = [NAME_TO_IDX[c] for c in greenery_classes_norm if c in NAME_TO_IDX]
        greenery_score = float(fracs[green_ids].sum()) if green_ids else 0.0

        if image_size is None:
            h, w = label_map.shape
            image_size = (w, h)

        return ScoreResult(
            greenery_score=greenery_score,
            tree_fraction=float(fracs[NAME_TO_IDX["tree"]]),
            rangeland_fraction=float(fracs[NAME_TO_IDX["rangeland"]]),
            agriculture_fraction=float(fracs[NAME_TO_IDX["agriculture"]]),
            class_fractions=class_fractions,
            total_pixels=total_pixels,
            greenery_classes_used=tuple(greenery_classes_norm),
            image_size=image_size,
        )

    def score_image_path(
        self,
        image_path: str,
        greenery_classes: Optional[Iterable[str]] = None,
        save_pred_path: Optional[str] = None,
    ) -> Dict:
        """
        High-level helper matching the JSON schema used by the existing API.
        """
        label_map = self.predict_label_map_from_path(image_path)
        res = self.score_label_map(label_map, greenery_classes=greenery_classes)

        if save_pred_path:
            save_pred_path = str(Path(save_pred_path).expanduser().resolve())
            Path(save_pred_path).parent.mkdir(parents=True, exist_ok=True)
            rgb = OEM_CLASS_COLORS_RGB[label_map]  # (H,W,3)
            from PIL import Image
            Image.fromarray(rgb, mode="RGB").save(save_pred_path)

        payload = {
            "greenery_score": res.greenery_score,
            "tree_fraction": res.tree_fraction,
            "rangeland_fraction": res.rangeland_fraction,
            "agriculture_fraction": res.agriculture_fraction,
            # Keep backward-compat with the older CLI wrapper that included a
            # "background" key (even though the model outputs 8 classes).
            "class_fractions": {"background": 0.0, **res.class_fractions},
            "total_pixels": res.total_pixels,
            "greenery_classes_used": list(res.greenery_classes_used),
            "image_size": [res.image_size[0], res.image_size[1]],
        }
        if save_pred_path:
            payload["pred_image_path"] = save_pred_path
        return payload
