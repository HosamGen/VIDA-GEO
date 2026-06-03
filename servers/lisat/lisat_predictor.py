# lisat_predictor.py
"""Reusable LISAt predictor.

This module is meant to behave like your working CLI script (infer_lisat.py), but in a
long-lived process (FastAPI/Uvicorn).

Key fixes vs the earlier predictor version:
1) Robust image preprocessing fallback:
   - ImageProcessor.load_and_preprocess_image() can *throw* (e.g., when cv2.imread returns None
     and repo code immediately accesses .shape). In that case, we catch the exception,
     re-encode the source image to a clean 8-bit RGB PNG via PIL, and retry.
2) When we retry using a temp re-encoded file, we also pass that path forward as
   inputs["image_path"] so any downstream code that uses image_path stays consistent.
3) Added additional debug_state fields so /health can confirm which module file is loaded,
   what Python executable is used, and CUDA visibility.

Put this file in the ROOT of LISAt_code (same level as `model/`, `dataloaders/`, `utils.py`).
"""

from __future__ import annotations

import io
import os
import re
import sys
import pathlib
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image


# --- make the repo importable regardless of where you run from ---
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# LISAt imports (repo-local)
from model.LISAT_eval import load_pretrained_model_LISAT  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from model.llava.constants import DEFAULT_IMAGE_TOKEN  # noqa: E402
from dataloaders.base_dataset import ImageProcessor  # noqa: E402
from dataloaders.utils import replace_image_tokens, tokenize_and_pad  # noqa: E402
from utils import prepare_input  # noqa: E402


@dataclass(frozen=True)
class LisatOutput:
    """Result of a segmentation request."""

    mask_uint8: np.ndarray  # HxW, values 0/255
    object_present: bool
    generated_text: str = ""


# ------------------------
# Helpers: dtype/device
# ------------------------
def _parse_dtype(dtype_str: str, device: str) -> tuple[torch.dtype, str]:
    s = (dtype_str or "auto").strip().lower()

    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16, "bf16"
    if s in {"fp16", "float16", "half"}:
        return torch.float16, "fp16"
    if s in {"fp32", "float32", "full"}:
        return torch.float32, "fp32"

    # auto
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def _parse_device(device_str: Optional[str]) -> str:
    if device_str:
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_device_map(device_map_str: Optional[str]) -> Optional[str]:
    s = (device_map_str or "auto").strip().lower()
    if s in {"none", "null", "no", "false", "0"}:
        return None
    return device_map_str or "auto"


# ------------------------
# Helpers: memory strings
# ------------------------
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGTP]i?B|B)\s*$", re.IGNORECASE)

_UNIT_MULT = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "PB": 10**15,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
    "PIB": 2**50,
}


def _parse_mem_to_bytes(s: Optional[str]) -> Optional[int]:
    """Parse strings like "20GiB", "18000MiB", "22GB" into bytes."""

    if not s:
        return None
    s = s.strip()
    if s.startswith("="):
        s = s[1:].strip()

    m = _MEM_RE.match(s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    mult = _UNIT_MULT.get(unit)
    if mult is None:
        return None
    return int(val * mult)


def _safe_gpu_budget_bytes(requested: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Returns (gpu_budget_bytes, cpu_budget_bytes) with headroom applied."""

    if not requested or not torch.cuda.is_available():
        return None, None

    req_bytes = _parse_mem_to_bytes(requested)
    if req_bytes is None:
        return None, None

    headroom_str = os.getenv("LISAT_GPU_HEADROOM", "2GiB")
    headroom = _parse_mem_to_bytes(headroom_str) or 0
    total = torch.cuda.get_device_properties(0).total_memory
    cap = max(0, total - headroom)
    safe = min(req_bytes, cap)

    cpu_str = os.getenv("LISAT_MAX_CPU_MEMORY", "128GiB")
    cpu_bytes = _parse_mem_to_bytes(cpu_str)
    return safe, cpu_bytes


def _reencode_to_safe_png(src_path: str) -> str:
    """Re-encode any readable image into a clean 8-bit RGB PNG (temp file)."""

    with Image.open(src_path) as im:
        im = im.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        im.save(tmp_path, format="PNG", optimize=True)
        return tmp_path


class LisatPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        image_size: int = 1024,
        default_max_new_tokens: int = 512,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        dtype: str = "auto",
        enable_tf32: bool = True,
        use_autocast: bool = True,
    ) -> None:
        self.model_path = model_path or os.getenv("LISAT_MODEL_PATH", "checkpoints/LISAt-7b")
        self.image_size = int(image_size)
        self.default_max_new_tokens = int(default_max_new_tokens)

        self.device = _parse_device(device or os.getenv("LISAT_DEVICE"))
        self.device_map = _parse_device_map(device_map or os.getenv("LISAT_DEVICE_MAP"))
        self.torch_dtype, self.dtype_tag = _parse_dtype(dtype or os.getenv("LISAT_DTYPE", "auto"), self.device)

        self.use_autocast = bool(use_autocast) and (os.getenv("LISAT_AUTOCAST", "1") != "0")

        if enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        self._load()

    def _load(self) -> None:
        kwargs: Dict[str, Any] = dict(
            model_path=self.model_path,
            device_map=self.device_map if self.device_map is not None else None,
            device=self.device,
        )

        # Only apply max_memory/offload when using device_map dispatch
        if self.device_map is not None:
            max_gpu_str = os.getenv("LISAT_MAX_GPU_MEMORY")
            gpu_budget, cpu_budget = _safe_gpu_budget_bytes(max_gpu_str)
            if gpu_budget is not None:
                if cpu_budget is not None:
                    kwargs["max_memory"] = {0: int(gpu_budget), "cpu": int(cpu_budget)}
                else:
                    kwargs["max_memory"] = {0: int(gpu_budget), "cpu": os.getenv("LISAT_MAX_CPU_MEMORY", "128GiB")}

            offload_folder = os.getenv("LISAT_OFFLOAD_FOLDER")
            if offload_folder:
                os.makedirs(offload_folder, exist_ok=True)
                kwargs["offload_folder"] = offload_folder
                kwargs["offload_state_dict"] = True

        # Always TRY torch_dtype first (works if loader uses **kwargs)
        kwargs["torch_dtype"] = self.torch_dtype

        def _try_load_with_fallback(k: Dict[str, Any]):
            while True:
                try:
                    return load_pretrained_model_LISAT(**k)
                except TypeError as e:
                    msg = str(e)

                    if "torch_dtype" in msg:
                        td = k.pop("torch_dtype", None)
                        if td is not None:
                            k["dtype"] = td
                        continue

                    if "dtype" in msg:
                        k.pop("dtype", None)
                        continue

                    removed = False
                    for bad in ("max_memory", "offload_folder", "offload_state_dict"):
                        if bad in msg:
                            k.pop(bad, None)
                            removed = True
                            break
                    if removed:
                        continue

                    raise

        tokenizer, model, vision_tower, _ = _try_load_with_fallback(kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.vision_tower = vision_tower

        # Mirror infer_lisat.py behavior: try casting vision tower dtype even if dispatched.
        try:
            self.vision_tower = self.vision_tower.to(dtype=self.torch_dtype)
        except Exception:
            pass

        # Reduce peak memory if supported
        use_cache_env = os.getenv("LISAT_USE_CACHE", "1").strip().lower()
        if use_cache_env in {"0", "false", "no"}:
            try:
                self.model.config.use_cache = False
            except Exception:
                pass

        # If NOT dispatched (device_map=None), safe to move/cast
        if self.device_map is None:
            try:
                self.model = self.model.to(self.device)
            except Exception:
                pass
            try:
                self.vision_tower = self.vision_tower.to(self.device)
            except Exception:
                pass
            try:
                self.model = self.model.to(self.torch_dtype)
            except Exception:
                pass
            try:
                self.vision_tower = self.vision_tower.to(self.torch_dtype)
            except Exception:
                pass

        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.img_processor = ImageProcessor(self.vision_tower.image_processor, self.image_size)

        try:
            p = next(self.model.parameters())
            self._loaded_param_dtype = str(p.dtype)
        except Exception:
            self._loaded_param_dtype = "unknown"

    def _autocast_ctx(self):
        if not self.use_autocast:
            return nullcontext()
        if not (self.device.startswith("cuda") and torch.cuda.is_available()):
            return nullcontext()
        if self.torch_dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.torch_dtype)

    @torch.inference_mode()
    def segment_path(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> LisatOutput:
        max_new_tokens = int(max_new_tokens or self.default_max_new_tokens)

        tmp_path: Optional[str] = None
        used_path = image_path

        def _try_preprocess(p: str):
            try:
                img, img_clip, sam_shape = self.img_processor.load_and_preprocess_image(p)
                return img, img_clip, sam_shape, None
            except Exception as e:
                return None, None, None, e

        try:
            image, image_clip, sam_mask_shape, err = _try_preprocess(image_path)

            # If preprocessing crashed OR returned None, re-encode then retry.
            if err is not None or image is None or image_clip is None or sam_mask_shape is None:
                tmp_path = _reencode_to_safe_png(image_path)
                used_path = tmp_path
                image, image_clip, sam_mask_shape, err2 = _try_preprocess(tmp_path)
                if err2 is not None:
                    orig_err = f"{type(err).__name__}: {err}" if err is not None else "None"
                    re_err = f"{type(err2).__name__}: {err2}"
                    raise RuntimeError(
                        "Image preprocessing failed even after re-encoding. "
                        f"image_path={image_path}; original_error={orig_err}; after_reencode_error={re_err}"
                    ) from err2

            if image is None or image_clip is None or sam_mask_shape is None:
                raise RuntimeError(
                    "Image preprocessing failed: load_and_preprocess_image returned None (even after re-encoding). "
                    f"image={type(image)}, image_clip={type(image_clip)}, sam_mask_shape={sam_mask_shape}"
                )

            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            conversation_list = [conv.get_prompt()]

            if getattr(self.model.config, "mm_use_im_start_end", False):
                conversation_list = replace_image_tokens(conversation_list)

            input_ids, _ = tokenize_and_pad(conversation_list, self.tokenizer, padding="left")
            if input_ids is None:
                raise RuntimeError("tokenize_and_pad returned None for input_ids")

            inputs = {
                "image_path": used_path,
                "images_clip": torch.stack([image_clip], dim=0),
                "images": torch.stack([image], dim=0),
                "input_ids": input_ids,
                "sam_mask_shape_list": [sam_mask_shape],
            }
            inputs = prepare_input(
                inputs,
                self.dtype_tag,
                is_cuda=(self.device.startswith("cuda") and torch.cuda.is_available()),
            )

            # Defensive checks (so we fail with a clear error, not NoneType.shape)
            for k in ("images_clip", "images", "input_ids", "sam_mask_shape_list"):
                if k not in inputs or inputs[k] is None:
                    raise RuntimeError(f"prepare_input produced missing/None key: {k}")

            with self._autocast_ctx():
                output_ids, pred_masks, object_presence = self.model.evaluate(
                    inputs["images_clip"],
                    inputs["images"],
                    inputs["input_ids"],
                    inputs["sam_mask_shape_list"],
                    max_new_tokens=max_new_tokens,
                )

            if pred_masks is None or len(pred_masks) == 0 or pred_masks[0] is None:
                raise RuntimeError("model.evaluate returned empty/None pred_masks")

            pm = pred_masks[0]
            if pm.dim() == 3:
                pm = pm[0]

            mask = (pm.detach().float().cpu().numpy() > 0).astype(np.uint8) * 255

            # Optional: decode generated text
            if output_ids is None or inputs["input_ids"] is None:
                text = ""
            else:
                real_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                text = (
                    self.tokenizer.batch_decode(real_ids, skip_special_tokens=True)[0]
                    if real_ids.numel()
                    else ""
                )

            return LisatOutput(mask_uint8=mask, object_present=bool(object_presence[0]), generated_text=text)

        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def mask_to_png_bytes(mask_uint8: np.ndarray) -> bytes:
        if mask_uint8.dtype != np.uint8:
            mask_uint8 = mask_uint8.astype(np.uint8)
        if mask_uint8.ndim != 2:
            raise ValueError("mask_uint8 must be HxW")
        buf = io.BytesIO()
        Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def save_mask_png(mask_uint8: np.ndarray, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        Image.fromarray(mask_uint8.astype(np.uint8), mode="L").save(tmp_path, format="PNG")
        os.replace(tmp_path, out_path)

    def debug_state(self) -> dict:
        # Include extra info to help confirm which module is loaded and what env the worker sees.
        gpu_name = None
        total_mem = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = int(torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                pass

        return {
            "predictor_file": str(pathlib.Path(__file__).resolve()),
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": total_mem,
            "model_path": self.model_path,
            "device": self.device,
            "device_map": self.device_map,
            "requested_dtype": str(self.torch_dtype),
            "prepare_input_tag": self.dtype_tag,
            "first_param_dtype": self._loaded_param_dtype,
            "image_size": self.image_size,
            "default_max_new_tokens": self.default_max_new_tokens,
            "autocast": bool(self.use_autocast),
            "max_gpu_memory_env": os.getenv("LISAT_MAX_GPU_MEMORY"),
            "gpu_headroom_env": os.getenv("LISAT_GPU_HEADROOM", "2GiB"),
            "offload_folder_env": os.getenv("LISAT_OFFLOAD_FOLDER"),
            "use_cache_env": os.getenv("LISAT_USE_CACHE", "1"),
        }
