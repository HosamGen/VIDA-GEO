# serve_flux_fill_api.py
#
# Minimal FastAPI server for FLUX Fill.
#
# Supports two backends via env:
#   FLUX_BACKEND=flux       (default) -> black-forest-labs/flux internals
#   FLUX_BACKEND=diffusers            -> 🤗 Diffusers FluxFillPipeline (+ optional quantization)
#
from __future__ import annotations

import logging
import os
import socket
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field

from flux_fill_predictor import FluxFillPredictor, FluxFillConfig

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="FLUX Fill API", version="2.1")

_predictor: Optional[FluxFillPredictor] = None

# Background init/warmup state (optional)
_init_thread: Optional[threading.Thread] = None
_init_error: Optional[str] = None
_warmup_done: bool = False

# Serialize requests explicitly (even though predictor also has its own lock)
_request_lock = threading.Lock()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) not in ("0", "false", "False")


def get_predictor() -> FluxFillPredictor:
    global _predictor
    if _predictor is None:
        # Common
        device = os.getenv("FLUX_DEVICE", "cuda")
        offload = _env_bool("FLUX_OFFLOAD", "1")
        cuda_dtype = os.getenv("FLUX_CUDA_DTYPE", "bf16")

        t5_max_length = int(os.getenv("FLUX_T5_MAX_LENGTH", "128"))
        default_num_steps = int(os.getenv("FLUX_DEFAULT_STEPS", "50"))
        default_guidance = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "30"))

        # Shape safety
        size_multiple = int(os.getenv("FLUX_SIZE_MULTIPLE", "16"))
        auto_pad = _env_bool("FLUX_AUTO_PAD", "1")

        # Memory knobs
        empty_cache = _env_bool("FLUX_EMPTY_CACHE", "1")

        # Backend + quantization
        backend = os.getenv("FLUX_BACKEND", "flux")
        model_id = os.getenv("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-Fill-dev")
        quant = os.getenv("FLUX_QUANT", "none")  # none|nf4|bnb8|bnb4
        nf4_repo = os.getenv("FLUX_NF4_REPO", "diffusers/FLUX.1-Fill-dev-nf4")

        # Optional override; if unset/0, predictor uses t5_max_length
        max_seq_raw = os.getenv("FLUX_MAX_SEQUENCE_LENGTH", "")
        max_seq = None
        if max_seq_raw.strip():
            try:
                v = int(max_seq_raw)
                if v > 0:
                    max_seq = v
            except Exception:
                max_seq = None

        cfg = FluxFillConfig(
            device=device,
            offload=offload,
            cuda_dtype=cuda_dtype,
            t5_max_length=t5_max_length,
            default_num_steps=default_num_steps,
            default_guidance=default_guidance,
            size_multiple=size_multiple,
            auto_pad_to_multiple=auto_pad,
            empty_cache_between_stages=empty_cache,
            backend=backend,
            diffusers_model_id=model_id,
            diffusers_quant=quant,
            diffusers_nf4_repo_id=nf4_repo,
            diffusers_max_sequence_length=max_seq,
        )
        _predictor = FluxFillPredictor(cfg)

    return _predictor


def _background_load_and_optionally_warmup(do_warmup: bool, size: int, steps: int) -> None:
    global _init_error, _warmup_done
    try:
        p = get_predictor()
        log.info(
            "FLUX: background init starting (backend=%s, quant=%s, offload=%s)",
            p.cfg.backend,
            getattr(p.cfg, "diffusers_quant", "none"),
            p.cfg.offload,
        )
        p.load()
        log.info("FLUX: model loaded (background).")

        if do_warmup:
            log.info("FLUX: warmup starting (size=%s, steps=%s)...", size, steps)
            p.warmup(size=size, steps=steps)
            _warmup_done = True
            log.info("FLUX: warmup complete.")
        else:
            _warmup_done = False
            log.info("FLUX: warmup skipped.")

    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}"
        log.exception("FLUX: background init failed: %s", _init_error)


@app.on_event("startup")
def _startup():
    """Optional eager-load + warmup on server start.

    export FLUX_EAGER_LOAD=1
    export FLUX_WARMUP=1
    """
    global _init_thread

    eager = _env_bool("FLUX_EAGER_LOAD", "0")
    if not eager:
        log.info("FLUX: eager load disabled (FLUX_EAGER_LOAD=0). Cold start happens on first request.")
        return

    do_warm = _env_bool("FLUX_WARMUP", "0")
    size = int(os.getenv("FLUX_WARMUP_SIZE", "256"))
    steps = int(os.getenv("FLUX_WARMUP_STEPS", "1"))

    if _init_thread is None or not _init_thread.is_alive():
        _init_thread = threading.Thread(
            target=_background_load_and_optionally_warmup,
            args=(do_warm, size, steps),
            daemon=True,
        )
        _init_thread.start()


class FillPathRequest(BaseModel):
    img_cond_path: str = Field(..., description="Path to conditioning image")
    img_mask_path: str = Field(..., description="Path to mask image (same WxH)")
    prompt: str
    guidance: float = 30.0
    num_steps: int = 50
    seed: Optional[int] = None


class WarmupRequest(BaseModel):
    size: int = 256
    steps: int = 1


@app.get("/health")
def health():
    p = get_predictor()
    return {
        "ok": True,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "loaded": getattr(p, "_loaded", False),
        "device": str(p.device),
        "backend": p.cfg.backend,
        "diffusers_quant": getattr(p.cfg, "diffusers_quant", None),
        "offload": bool(p.cfg.offload),
        "cuda_dtype": getattr(p.cfg, "cuda_dtype", None),
        "auto_pad_to_multiple": bool(p.cfg.auto_pad_to_multiple),
        "size_multiple": int(p.cfg.size_multiple),
        "empty_cache_between_stages": bool(p.cfg.empty_cache_between_stages),
        "warmup_done": bool(_warmup_done),
        "init_error": _init_error,
        "init_thread_alive": bool(_init_thread.is_alive()) if _init_thread is not None else False,
    }


@app.post("/init")
def init_model():
    """Trigger background model load (no warmup). Useful to pay cold-start once."""
    global _init_thread
    if _init_thread is not None and _init_thread.is_alive():
        return {"ok": True, "started": False, "message": "init already running"}
    _init_thread = threading.Thread(
        target=_background_load_and_optionally_warmup,
        args=(False, 0, 0),
        daemon=True,
    )
    _init_thread.start()
    return {"ok": True, "started": True}


@app.post("/warmup")
def warmup(req: WarmupRequest):
    """Trigger background warmup (loads model if needed)."""
    global _init_thread
    if _init_thread is not None and _init_thread.is_alive():
        return {"ok": True, "started": False, "message": "init/warmup already running"}
    _init_thread = threading.Thread(
        target=_background_load_and_optionally_warmup,
        args=(True, int(req.size), int(req.steps)),
        daemon=True,
    )
    _init_thread.start()
    return {"ok": True, "started": True, "size": int(req.size), "steps": int(req.steps)}


@app.post("/fill/png")
def fill_png(req: FillPathRequest):
    try:
        with _request_lock:
            predictor = get_predictor()
            png_bytes, used_seed = predictor.fill_png_bytes(
                img_cond_path=req.img_cond_path,
                img_mask_path=req.img_mask_path,
                prompt=req.prompt,
                guidance=req.guidance,
                num_steps=req.num_steps,
                seed=req.seed,
            )
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"X-FLUX-Seed": str(used_seed)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FLUX fill failed: {type(e).__name__}: {e}")


@app.post("/fill/upload")
def fill_upload(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    guidance: float = Form(30.0),
    num_steps: int = Form(50),
    seed: Optional[int] = Form(None),
):
    """Multipart endpoint so you can do curl -F image=@... -F mask=@..."""
    try:
        with _request_lock:
            predictor = get_predictor()

            with tempfile.TemporaryDirectory() as td:
                img_path = os.path.join(td, image.filename or "image.png")
                mask_path = os.path.join(td, mask.filename or "mask.png")

                with open(img_path, "wb") as f:
                    f.write(image.file.read())
                with open(mask_path, "wb") as f:
                    f.write(mask.file.read())

                png_bytes, used_seed = predictor.fill_png_bytes(
                    img_cond_path=img_path,
                    img_mask_path=mask_path,
                    prompt=prompt,
                    guidance=guidance,
                    num_steps=num_steps,
                    seed=seed,
                )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"X-FLUX-Seed": str(used_seed)},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FLUX fill failed: {type(e).__name__}: {e}")
