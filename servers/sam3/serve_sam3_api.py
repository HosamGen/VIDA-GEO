# serve_sam3_api.py
"""
FastAPI server for SAM3 text-prompt segmentation.

Run (inside sam3 env):
  cd /path/to/sam3_repo_or_where_files_live
  uvicorn serve_sam3_api:app --host 127.0.0.1 --port 8005 --workers 1

Endpoints:
  GET  /health
  POST /segment/png     (JSON; returns image/png bytes)
  POST /segment/path    (JSON; optionally writes mask to disk and/or returns base64)
  POST /segment/upload  (multipart; returns image/png bytes)

Environment vars (optional):
  SAM3_DEVICE="cuda" / "cpu"
  SAM3_SCORE_THRESH="0.25"
  SAM3_MASK_THRESH="0.5"
  SAM3_COMPILE="0" or "1"

Why --workers 1:
  Multiple workers would each load a copy of the model (wastes VRAM/RAM).
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from sam3_predictor import Sam3Predictor


# --------------------------
# Request/Response Models
# --------------------------
class SegmentRequest(BaseModel):
    image_path: str = Field(..., description="Path to an image accessible to the SAM3 service")
    prompt: str = Field(..., description="Text concept prompt, e.g. 'park'")
    out_mask_path: Optional[str] = Field(
        None,
        description="If set, write mask PNG to this path (directories are created). "
                    "If not set, defaults to outputs/<image_stem>/sam3_mask.png",
    )
    score_thresh: Optional[float] = Field(None, description="Override score threshold for this request")
    mask_thresh: Optional[float] = Field(None, description="Override mask threshold for this request")
    union: bool = Field(True, description="Union all kept instance masks; if false, take top-1")
    return_mask_base64: bool = Field(
        False,
        description="If true, include base64-encoded PNG bytes in the response JSON.",
    )


class SegmentResult(BaseModel):
    mask_path: Optional[str] = None
    object_present: bool
    num_instances: int
    scores: List[float] = []
    kept_indices: List[int] = []
    prompt: str
    mask_png_base64: Optional[str] = None


# --------------------------
# App + Global State
# --------------------------
_predictor: Optional[Sam3Predictor] = None
_gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    We create the predictor on startup.
    By default, it loads lazily on first request; you can force eager load by setting SAM3_EAGER_LOAD=1.
    """
    global _predictor
    _predictor = Sam3Predictor(
        device=os.getenv("SAM3_DEVICE"),
        score_thresh=float(os.getenv("SAM3_SCORE_THRESH", "0.25")),
        mask_thresh=float(os.getenv("SAM3_MASK_THRESH", "0.5")),
        compile_model=(os.getenv("SAM3_COMPILE", "0") not in ("0", "false", "False")),
    )

    eager = os.getenv("SAM3_EAGER_LOAD", "0") not in ("0", "false", "False")
    if eager:
        # Eagerly load weights once at startup
        _predictor.load()

    yield


app = FastAPI(title="SAM3 Segmentation Service", version="1.0", lifespan=lifespan)


# --------------------------
# Helpers
# --------------------------
def _resolve_existing_file(p: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _resolve_output_path(image_path: Path, out_mask_path: Optional[str]) -> Path:
    if out_mask_path:
        out_path = Path(out_mask_path).expanduser().resolve()
    else:
        out_path = (Path("outputs") / image_path.stem / "sam3_mask.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# --------------------------
# Endpoints
# --------------------------
@app.get("/health")
def health():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")
    return {"status": "ok", **_predictor.debug_state()}


@app.post("/segment/path", response_model=SegmentResult)
async def segment_path(req: SegmentRequest) -> SegmentResult:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")

    try:
        img_path = _resolve_existing_file(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")

    out_path = _resolve_output_path(img_path, req.out_mask_path)

    try:
        async with _gpu_lock:
            result = _predictor.segment_path(
                image_path=str(img_path),
                prompt=req.prompt,
                score_thresh=req.score_thresh,
                mask_thresh=req.mask_thresh,
                union=req.union,
            )

        # Save mask to disk on server
        _predictor.save_mask_png(result.mask_uint8, str(out_path))

        mask_b64 = None
        if req.return_mask_base64:
            png_bytes = _predictor.mask_to_png_bytes(result.mask_uint8)
            mask_b64 = base64.b64encode(png_bytes).decode("ascii")

        return SegmentResult(
            mask_path=str(out_path),
            object_present=bool(result.num_instances > 0),
            num_instances=int(result.num_instances),
            scores=list(result.scores),
            kept_indices=list(result.kept_indices),
            prompt=result.prompt,
            mask_png_base64=mask_b64,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {type(e).__name__}: {e}")


@app.post("/segment/png")
async def segment_png(req: SegmentRequest) -> Response:
    """
    Same as /segment/path, but returns the mask PNG bytes directly.
    Useful for curl: use --output mask.png
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")

    try:
        img_path = _resolve_existing_file(req.image_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {e}")

    try:
        async with _gpu_lock:
            result = _predictor.segment_path(
                image_path=str(img_path),
                prompt=req.prompt,
                score_thresh=req.score_thresh,
                mask_thresh=req.mask_thresh,
                union=req.union,
            )
        png_bytes = _predictor.mask_to_png_bytes(result.mask_uint8)
        return Response(content=png_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {type(e).__name__}: {e}")


@app.post("/segment/upload")
async def segment_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    score_thresh: Optional[float] = Form(None),
    mask_thresh: Optional[float] = Form(None),
    union: bool = Form(True),
) -> Response:
    """
    Upload image bytes (no need for server to access your file path).
    Returns the mask as image/png bytes.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")

    img_bytes = await image.read()

    # Write upload to temp file because PIL can open bytes directly, but we want consistent path handling
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        f.write(img_bytes)
        f.flush()

        try:
            async with _gpu_lock:
                result = _predictor.segment_path(
                    image_path=f.name,
                    prompt=prompt,
                    score_thresh=score_thresh,
                    mask_thresh=mask_thresh,
                    union=union,
                )
            png_bytes = _predictor.mask_to_png_bytes(result.mask_uint8)
            return Response(content=png_bytes, media_type="image/png")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {type(e).__name__}: {e}")
