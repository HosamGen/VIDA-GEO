"""LISAt Segmentation Service API (FastAPI).

This is a debug-friendly version of serve_lisat_api.py.

Differences vs the original:
- Imports LisatPredictor from lisat_predictor.py (same as before), but logs the
  resolved module file path at startup so you can confirm Uvicorn isn't importing
  an unexpected module.
- On 500s, logs full stack traces via logger.exception (Uvicorn sometimes hides
  these depending on log settings).

Run (inside the lisat env):
  cd /path/to/LISAt_code
  export LISAT_MODEL_PATH="checkpoints/LISAt-7b"
  uvicorn serve_lisat_api_v2:app --host 127.0.0.1 --port 8001 --workers 1 --log-level info
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from lisat_predictor import LisatPredictor


logger = logging.getLogger("lisat_api")


class SegmentPathRequest(BaseModel):
    image_path: str = Field(..., description="Path to an image accessible to the LISAt service")
    prompt: str = Field(..., description="Natural language segmentation prompt")
    out_mask_path: Optional[str] = Field(
        None,
        description=(
            "If set, write mask PNG to this path (directories are created). "
            "If not set, defaults to outputs/<image_stem>/mask.png"
        ),
    )
    max_new_tokens: Optional[int] = Field(None, description="Override max_new_tokens (optional)")
    return_mask_base64: bool = Field(False, description="If true, include base64 PNG in JSON")


class SegmentResult(BaseModel):
    mask_path: Optional[str] = None
    object_present: bool
    text: str
    mask_png_base64: Optional[str] = None


_predictor: Optional[LisatPredictor] = None
_gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor

    # Log which lisat_predictor module is being imported
    import lisat_predictor as lp  # noqa

    logger.info("Using lisat_predictor from: %s", getattr(lp, "__file__", "<unknown>"))
    logger.info("Server CWD: %s", os.getcwd())

    _predictor = LisatPredictor(
        model_path=os.getenv("LISAT_MODEL_PATH", "checkpoints/LISAt-7b"),
        device=os.getenv("LISAT_DEVICE"),
        device_map=os.getenv("LISAT_DEVICE_MAP", "auto"),
        dtype=os.getenv("LISAT_DTYPE", "auto"),
        image_size=int(os.getenv("LISAT_IMAGE_SIZE", "1024")),
        default_max_new_tokens=int(os.getenv("LISAT_MAX_NEW_TOKENS", "512")),
        use_autocast=(os.getenv("LISAT_AUTOCAST", "1") != "0"),
    )

    logger.info("Predictor loaded. /health should be OK.")

    yield


app = FastAPI(title="LISAt Segmentation Service", lifespan=lifespan)


def _resolve_existing_file(p: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _resolve_output_path(image_path: Path, out_mask_path: Optional[str]) -> Path:
    if out_mask_path:
        out_path = Path(out_mask_path).expanduser().resolve()
    else:
        out_path = (Path("outputs") / image_path.stem / "mask.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


@app.get("/health")
def health():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", **_predictor.debug_state()}


@app.post("/segment/path", response_model=SegmentResult)
async def segment_path(req: SegmentPathRequest) -> SegmentResult:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        img_path = _resolve_existing_file(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")

    out_path = _resolve_output_path(img_path, req.out_mask_path)

    try:
        async with _gpu_lock:
            out = _predictor.segment_path(
                image_path=str(img_path),
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )

        _predictor.save_mask_png(out.mask_uint8, str(out_path))

        mask_b64 = None
        if req.return_mask_base64:
            png_bytes = _predictor.mask_to_png_bytes(out.mask_uint8)
            mask_b64 = base64.b64encode(png_bytes).decode("ascii")

        return SegmentResult(
            mask_path=str(out_path),
            object_present=out.object_present,
            text=out.generated_text,
            mask_png_base64=mask_b64,
        )
    except Exception as e:
        logger.exception("LISAt inference failed")
        raise HTTPException(status_code=500, detail=f"LISAt inference failed: {type(e).__name__}: {e}")


@app.post("/segment/png")
async def segment_png(req: SegmentPathRequest) -> Response:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        img_path = _resolve_existing_file(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")

    try:
        async with _gpu_lock:
            out = _predictor.segment_path(
                image_path=str(img_path),
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
        png_bytes = _predictor.mask_to_png_bytes(out.mask_uint8)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logger.exception("LISAt inference failed")
        raise HTTPException(status_code=500, detail=f"LISAt inference failed: {type(e).__name__}: {e}")


@app.post("/segment/upload")
async def segment_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: Optional[int] = Form(None),
) -> Response:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    img_bytes = await image.read()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        f.write(img_bytes)
        f.flush()

        try:
            async with _gpu_lock:
                out = _predictor.segment_path(
                    image_path=f.name,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
            png_bytes = _predictor.mask_to_png_bytes(out.mask_uint8)
            return Response(content=png_bytes, media_type="image/png")
        except Exception as e:
            logger.exception("LISAt inference failed")
            raise HTTPException(status_code=500, detail=f"LISAt inference failed: {type(e).__name__}: {e}")
