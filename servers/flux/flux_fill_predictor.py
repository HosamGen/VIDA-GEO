# flux_fill_predictor.py
#
# Persistent FLUX Fill predictor + FastAPI-friendly helpers.
#
# This file supports TWO backends:
#   1) backend="flux"      -> uses black-forest-labs/flux (your original setup)
#   2) backend="diffusers" -> uses 🤗 Diffusers FluxFillPipeline with optional quantization
#
# Goal of the diffusers backend:
#   - allow quantizing the *big* components (transformer + T5) so the whole stack can fit on GPU,
#     enabling FLUX_OFFLOAD=0 for lower latency.
#
from __future__ import annotations

import io
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# -------------------------
# Optional imports (so the file can be used even if you only installed one backend)
# -------------------------

_HAS_FLUX = False
try:
    # black-forest-labs/flux (official inference repo) internals
    from flux.util import load_t5, load_clip, load_flow_model, load_ae
    from flux.sampling import get_noise, prepare_fill, get_schedule, denoise, unpack

    _HAS_FLUX = True
except Exception:
    # Don't crash on import; only error if backend="flux" is actually used.
    _HAS_FLUX = False


# -------------------------
# Utilities
# -------------------------


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert tensor [1, 3, H, W] in ~[-1, 1] to PIL."""
    x = x.detach().float().clamp(-1, 1)
    x = (127.5 * (x + 1.0)).clamp(0, 255)
    x = x[0].permute(1, 2, 0).cpu().byte().numpy()
    return Image.fromarray(x)


def _pad_to_multiple_pil(
    im: Image.Image,
    multiple: int,
    mode: str,
    constant: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad a PIL image to have (W,H) divisible by `multiple`.

    mode:
      - "edge": replicate border pixels (good for RGB)
      - "constant": pad with a constant value (good for masks)

    Returns:
      (padded_image, (pad_left, pad_top, pad_right, pad_bottom))
    """
    if multiple <= 1:
        return im, (0, 0, 0, 0)

    w, h = im.size
    new_w = int(np.ceil(w / multiple) * multiple)
    new_h = int(np.ceil(h / multiple) * multiple)

    pad_w = new_w - w
    pad_h = new_h - h
    if pad_w == 0 and pad_h == 0:
        return im, (0, 0, 0, 0)

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    arr = np.array(im)
    if arr.ndim == 2:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    if mode == "edge":
        arr2 = np.pad(arr, pad_spec, mode="edge")
    elif mode == "constant":
        arr2 = np.pad(arr, pad_spec, mode="constant", constant_values=constant)
    else:
        raise ValueError(f"Unknown pad mode: {mode}")

    return Image.fromarray(arr2), (pad_left, pad_top, pad_right, pad_bottom)


def _crop_padding(im: Image.Image, pads: tuple[int, int, int, int], orig_size: tuple[int, int]) -> Image.Image:
    """Crop away symmetric padding to original (W,H)."""
    pad_left, pad_top, pad_right, pad_bottom = pads
    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return im
    ow, oh = orig_size
    return im.crop((pad_left, pad_top, pad_left + ow, pad_top + oh))


# -------------------------
# Config + Predictor
# -------------------------


@dataclass
class FluxFillConfig:
    # Common
    device: str = "cuda"

    # If True, keep the full stack on CPU while idle and move pieces to GPU per stage.
    # This minimizes persistent VRAM usage but adds CPU<->GPU transfer time.
    offload: bool = True

    # Compute dtype on CUDA.
    # For Ada GPUs, bf16 is generally recommended for stability.
    cuda_dtype: str = "bf16"  # one of {"bf16", "fp16"}

    # Prompt length
    t5_max_length: int = 128

    # Inference defaults
    default_num_steps: int = 50
    default_guidance: float = 30.0

    # Shape safety
    size_multiple: int = 16
    auto_pad_to_multiple: bool = True

    # Memory knobs
    empty_cache_between_stages: bool = True

    # Backend selection
    backend: str = "flux"  # "flux" or "diffusers"

    # ---- Diffusers backend knobs ----
    # Base model to load for FluxFillPipeline
    diffusers_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"

    # Quantization mode:
    #   - "none": full-precision (bf16/fp16) modules
    #   - "nf4":  load pre-quantized NF4 transformer + T5 from `diffusers_nf4_repo_id`
    #   - "bnb8": quantize transformer + T5 to 8-bit with bitsandbytes
    #   - "bnb4": quantize transformer + T5 to 4-bit NF4 with bitsandbytes
    diffusers_quant: str = "none"

    # Repo containing NF4 checkpoints for subfolders: transformer, text_encoder_2
    diffusers_nf4_repo_id: str = "diffusers/FLUX.1-Fill-dev-nf4"

    # If None, we use t5_max_length (to match your current behavior).
    diffusers_max_sequence_length: Optional[int] = None


class FluxFillPredictor:
    """Persistent FLUX Fill predictor."""

    def __init__(self, cfg: Optional[FluxFillConfig] = None):
        self.cfg = cfg or FluxFillConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        # Serialize load/warmup/fill
        self._lock = threading.RLock()
        self._loaded = False

        # flux backend modules
        self._t5 = None
        self._clip = None
        self._model = None
        self._ae = None

        # diffusers backend pipeline
        self._pipe = None

        # Reproducible-ish random seed generation when caller doesn't provide one
        self._rng = torch.Generator(device="cpu")

    # -------------------------
    # Helpers
    # -------------------------

    def _cuda_compute_dtype(self) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.bfloat16
        dt = (self.cfg.cuda_dtype or "bf16").lower().strip()
        if dt in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.bfloat16

    @staticmethod
    def _validate_inputs(img_path: str, mask_path: str) -> Tuple[int, int]:
        p_img = Path(img_path)
        p_mask = Path(mask_path)

        if not p_img.exists():
            raise FileNotFoundError(f"Conditioning image not found: {p_img}")
        if not p_mask.exists():
            raise FileNotFoundError(f"Mask image not found: {p_mask}")

        with Image.open(p_img) as im:
            w, h = im.size
        with Image.open(p_mask) as m:
            mw, mh = m.size

        if (w, h) != (mw, mh):
            raise ValueError(f"Mask size must match image size. image={w}x{h}, mask={mw}x{mh}.")

        return w, h

    def _maybe_pad_to_multiple_paths(
        self,
        img_path: str,
        mask_path: str,
    ) -> tuple[str, str, tuple[int, int], tuple[int, int, int, int], Optional[tempfile.TemporaryDirectory]]:
        """Path-based auto-padding (used for the flux backend because prepare_fill expects paths)."""
        orig_w, orig_h = self._validate_inputs(img_path, mask_path)
        orig_size = (orig_w, orig_h)

        multiple = int(self.cfg.size_multiple or 1)
        auto_pad = bool(self.cfg.auto_pad_to_multiple)

        if not auto_pad:
            if (orig_w % multiple) != 0 or (orig_h % multiple) != 0:
                raise ValueError(
                    f"Image/mask size must be divisible by {multiple}. Got {orig_w}x{orig_h}. "
                    "Either pad/crop your ROI, or enable auto padding (FLUX_AUTO_PAD=1)."
                )
            return img_path, mask_path, orig_size, (0, 0, 0, 0), None

        if (orig_w % multiple) == 0 and (orig_h % multiple) == 0:
            return img_path, mask_path, orig_size, (0, 0, 0, 0), None

        td = tempfile.TemporaryDirectory()

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im_pad, pads = _pad_to_multiple_pil(im, multiple, mode="edge")

        with Image.open(mask_path) as m:
            m = m.convert("L")
            m_pad, pads2 = _pad_to_multiple_pil(m, multiple, mode="constant", constant=0)

        if pads2 != pads:
            pads = pads2

        used_img = os.path.join(td.name, "img_padded.png")
        used_mask = os.path.join(td.name, "mask_padded.png")
        im_pad.save(used_img, format="PNG")
        m_pad.save(used_mask, format="PNG")

        return used_img, used_mask, orig_size, pads, td

    def _empty_cache(self) -> None:
        if self.cfg.empty_cache_between_stages and self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------
    # Loading
    # -------------------------

    def load(self) -> None:
        """Load all model components/pipeline (idempotent)."""
        with self._lock:
            if self._loaded:
                return

            backend = (self.cfg.backend or "flux").lower().strip()
            if backend == "diffusers":
                self._load_diffusers()
            else:
                self._load_flux()

            self._loaded = True

    def _load_flux(self) -> None:
        if not _HAS_FLUX:
            raise ImportError(
                "backend=flux requested but the 'flux' package couldn't be imported. "
                "Install black-forest-labs/flux (or make sure it's on PYTHONPATH)."
            )

        name = "flux-dev-fill"

        # When offload=True, load weights onto CPU to avoid persistent VRAM usage.
        base_device = "cpu" if self.cfg.offload else str(self.device)

        self._t5 = load_t5(base_device, max_length=self.cfg.t5_max_length).eval()
        self._clip = load_clip(base_device).eval()
        self._model = load_flow_model(name, device=base_device).eval()
        self._ae = load_ae(name, device=base_device).eval()

    def _load_diffusers(self) -> None:
        # Lazy imports so users who only want the flux backend aren't forced to install these.
        try:
            from diffusers import FluxFillPipeline, FluxTransformer2DModel
            from transformers import T5EncoderModel
        except Exception as e:
            raise ImportError(
                "backend=diffusers requested but diffusers/transformers are not installed. "
                "Install: pip install -U diffusers transformers accelerate bitsandbytes"
            ) from e

        dtype = self._cuda_compute_dtype() if self.device.type == "cuda" else torch.bfloat16
        quant = (self.cfg.diffusers_quant or "none").lower().strip()

        if quant == "nf4":
            # Load base pipeline in bf16/fp16
            base = FluxFillPipeline.from_pretrained(self.cfg.diffusers_model_id, torch_dtype=dtype)

            # Load pre-quantized NF4 checkpoints (transformer + T5)
            transformer = FluxTransformer2DModel.from_pretrained(
                self.cfg.diffusers_nf4_repo_id,
                subfolder="transformer",
                torch_dtype=dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.cfg.diffusers_nf4_repo_id,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
            )

            pipe = FluxFillPipeline.from_pipe(
                base,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=dtype,
            )

        elif quant in {"bnb8", "bnb4"}:
            # On-the-fly quantization with bitsandbytes
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

            if quant == "bnb8":
                tconf = TransformersBitsAndBytesConfig(load_in_8bit=True)
                dconf = DiffusersBitsAndBytesConfig(load_in_8bit=True)
            else:
                # 4-bit NF4; compute in bf16/fp16
                tconf = TransformersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
                dconf = DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )

            transformer = FluxTransformer2DModel.from_pretrained(
                self.cfg.diffusers_model_id,
                subfolder="transformer",
                torch_dtype=dtype,
                quantization_config=dconf,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.cfg.diffusers_model_id,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                quantization_config=tconf,
            )

            pipe = FluxFillPipeline.from_pretrained(
                self.cfg.diffusers_model_id,
                torch_dtype=dtype,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
            )

        else:
            # Full precision pipeline
            pipe = FluxFillPipeline.from_pretrained(self.cfg.diffusers_model_id, torch_dtype=dtype)

        # Device placement:
        # - If offload=True: use diffusers CPU offload helpers
        # - If offload=False: keep the entire pipeline on GPU (this is what you want for low latency)
        if self.device.type == "cuda":
            if self.cfg.offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(self.device)

        # Less overhead in API mode
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        self._pipe = pipe

    # -------------------------
    # Inference
    # -------------------------

    def fill(
        self,
        img_cond_path: str,
        img_mask_path: str,
        prompt: str,
        guidance: Optional[float] = None,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, int]:
        """Returns: (PIL_image, used_seed)"""
        with self._lock:
            self.load()

            used_guidance = float(guidance if guidance is not None else self.cfg.default_guidance)
            used_steps = int(num_steps if num_steps is not None else self.cfg.default_num_steps)

            if seed is None:
                seed = int(self._rng.seed())
            used_seed = int(seed)

            backend = (self.cfg.backend or "flux").lower().strip()
            if backend == "diffusers":
                return self._fill_diffusers(
                    img_cond_path=img_cond_path,
                    img_mask_path=img_mask_path,
                    prompt=prompt,
                    guidance=used_guidance,
                    num_steps=used_steps,
                    seed=used_seed,
                )

            return self._fill_flux(
                img_cond_path=img_cond_path,
                img_mask_path=img_mask_path,
                prompt=prompt,
                guidance=used_guidance,
                num_steps=used_steps,
                seed=used_seed,
            )

    def _fill_flux(
        self,
        img_cond_path: str,
        img_mask_path: str,
        prompt: str,
        guidance: float,
        num_steps: int,
        seed: int,
    ) -> Tuple[Image.Image, int]:
        # NOTE: This is your original (flux repo) fill path.
        # It is kept intact to avoid changing behavior.

        # Auto-pad to prevent internal patch reshape issues for arbitrary ROIs
        td = None
        used_img_path, used_mask_path, orig_size, pads, td = self._maybe_pad_to_multiple_paths(img_cond_path, img_mask_path)
        width, height = Image.open(used_img_path).size

        torch_device = self.device
        dtype = self._cuda_compute_dtype() if torch_device.type == "cuda" else torch.bfloat16

        try:
            with torch.inference_mode():
                # 1) noise on serving device
                x = get_noise(
                    num_samples=1,
                    height=height,
                    width=width,
                    device=torch_device,
                    dtype=dtype,
                    seed=seed,
                )

                # 2) prepare_fill (encode image/mask + text)
                if self.cfg.offload and torch_device.type == "cuda":
                    self._t5 = self._t5.to(torch_device)
                    self._clip = self._clip.to(torch_device)
                    self._ae = self._ae.to(torch_device)

                inp = prepare_fill(
                    self._t5,
                    self._clip,
                    x,
                    prompt=prompt,
                    ae=self._ae,
                    img_cond_path=used_img_path,
                    mask_path=used_mask_path,
                )

                timesteps = get_schedule(
                    num_steps,
                    inp["img"].shape[1],
                    shift=True,  # matches CLI behavior for dev model
                )

                # 3) denoise with main model on GPU
                if self.cfg.offload and torch_device.type == "cuda":
                    self._t5 = self._t5.to("cpu")
                    self._clip = self._clip.to("cpu")
                    self._ae = self._ae.to("cpu")
                    self._empty_cache()
                    self._model = self._model.to(torch_device)

                x = denoise(self._model, **inp, timesteps=timesteps, guidance=guidance)

                # 4) decode with AE decoder on GPU
                if self.cfg.offload and torch_device.type == "cuda":
                    self._model = self._model.to("cpu")
                    self._empty_cache()
                    # Keeping compatibility with the original approach (decoder on GPU only)
                    self._ae.decoder.to(torch_device)

                # Some flux versions expect float32 here; keep as-is for compatibility.
                x = unpack(x.float(), height, width)

                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=dtype,
                    enabled=(torch_device.type == "cuda"),
                ):
                    x = self._ae.decode(x)

                out_img = _tensor_to_pil(x)
                out_img = _crop_padding(out_img, pads, orig_size)

                # 5) return everything to CPU if offload
                if self.cfg.offload and torch_device.type == "cuda":
                    self._ae = self._ae.to("cpu")
                    self._empty_cache()

            return out_img, int(seed)

        except torch.cuda.OutOfMemoryError as e:
            if self.device.type == "cuda":
                try:
                    self._empty_cache()
                except Exception:
                    pass
            raise RuntimeError(
                "CUDA OOM during FLUX fill. Try fewer steps, smaller ROI, or enable offload. "
                "If you want low latency without offload, use backend=diffusers + quantization."
            ) from e

        finally:
            if td is not None:
                try:
                    td.cleanup()
                except Exception:
                    pass

    def _fill_diffusers(
        self,
        img_cond_path: str,
        img_mask_path: str,
        prompt: str,
        guidance: float,
        num_steps: int,
        seed: int,
    ) -> Tuple[Image.Image, int]:
        if self._pipe is None:
            raise RuntimeError("Diffusers pipeline is not loaded")

        # Load inputs
        w, h = self._validate_inputs(img_cond_path, img_mask_path)
        orig_size = (w, h)

        with Image.open(img_cond_path) as im:
            im = im.convert("RGB")
            img = im.copy()
        with Image.open(img_mask_path) as m:
            m = m.convert("L")
            mask = m.copy()

        pads = (0, 0, 0, 0)
        if self.cfg.auto_pad_to_multiple:
            multiple = int(self.cfg.size_multiple or 1)
            if multiple > 1 and ((w % multiple) != 0 or (h % multiple) != 0):
                img, pads = _pad_to_multiple_pil(img, multiple, mode="edge")
                mask, pads2 = _pad_to_multiple_pil(mask, multiple, mode="constant", constant=0)
                if pads2 != pads:
                    pads = pads2

        width, height = img.size

        # Diffusers uses a CPU generator in their examples for reproducibility
        gen = torch.Generator("cpu").manual_seed(int(seed))

        max_seq = int(self.cfg.diffusers_max_sequence_length or self.cfg.t5_max_length)

        try:
            with torch.inference_mode():
                out = self._pipe(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    height=int(height),
                    width=int(width),
                    guidance_scale=float(guidance),
                    num_inference_steps=int(num_steps),
                    max_sequence_length=max_seq,
                    generator=gen,
                ).images[0]

            out = _crop_padding(out, pads, orig_size)
            return out, int(seed)

        except torch.cuda.OutOfMemoryError as e:
            if self.device.type == "cuda":
                try:
                    self._empty_cache()
                except Exception:
                    pass
            raise RuntimeError(
                "CUDA OOM during diffusers FLUX fill. Try fewer steps, smaller ROI, or enable offload. "
                "If you are already quantized, you may be hitting activation memory at very large resolutions."
            ) from e

    def fill_png_bytes(self, *args, **kwargs) -> Tuple[bytes, int]:
        img, used_seed = self.fill(*args, **kwargs)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), used_seed

    def warmup(self, size: int = 256, steps: int = 1) -> None:
        """Small warmup to reduce first-request latency."""
        self.load()

        size = int(size)
        steps = int(steps)

        img = Image.fromarray(np.full((size, size, 3), 127, dtype=np.uint8), mode="RGB")
        mask = Image.fromarray(np.zeros((size, size), dtype=np.uint8), mode="L")
        m = np.array(mask)
        s0 = size // 4
        s1 = size - s0
        m[s0:s1, s0:s1] = 255
        mask = Image.fromarray(m, mode="L")

        # Keep warmup using the public fill() interface (paths)
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "warm_img.png")
            mask_path = os.path.join(td, "warm_mask.png")
            img.save(img_path, format="PNG")
            mask.save(mask_path, format="PNG")

            _ = self.fill(
                img_cond_path=img_path,
                img_mask_path=mask_path,
                prompt="warmup",
                guidance=1.0,
                num_steps=steps,
                seed=0,
            )
