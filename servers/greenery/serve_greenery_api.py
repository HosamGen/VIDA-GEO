#!/usr/bin/env python3
"""
serve_greenery_api_fast.py — FastAPI server using *in-process* OEM-lightweight inference.

This version avoids calling eval_oem_lightweight.py via subprocess per request.
Instead it loads the model once at startup and reuses it.

Environment variables (same as your current server):
    OEM_REPO
    OEM_MODEL             sparsemask (default) or fasterseg
    OEM_ARCH
    OEM_WEIGHTS
    OEM_DEVICE            cuda (default) or cpu
    OEM_GREENERY_CLASSES  default: tree,rangeland,agriculture

Extra env vars:
    OEM_USE_TTA           0/1 (default 0). Turning on TTA roughly doubles compute. citeturn2view1turn4view1
    OEM_USE_FP16          0/1 (default 0). CUDA only.
    OEM_INPUT_SIZE        default 1024 (matches upstream resize). citeturn4view0

Run:
    export OEM_REPO=/path/to/oem-lightweight
    export OEM_MODEL=sparsemask
    export OEM_ARCH=models/SparseMask/mask_thres_0.001.npy
    export OEM_WEIGHTS=models/SparseMask/checkpoint_63750.pth.tar
    export OEM_DEVICE=cuda
    python -m uvicorn serve_greenery_api_fast:app --host 127.0.0.1 --port 8006 --workers 1
"""

import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from greenery_score import OemLightweightSegmenter

app = FastAPI(title="Greenery Scoring API (fast)", version="2.0")

# Config from env (import-time)
_OEM_REPO: str    = os.environ.get("OEM_REPO", str(Path(__file__).parent))
_OEM_MODEL: str   = os.environ.get("OEM_MODEL", "sparsemask")
_OEM_ARCH: str    = os.environ.get("OEM_ARCH",  "models/SparseMask/mask_thres_0.001.npy")
_OEM_WEIGHTS: str = os.environ.get("OEM_WEIGHTS", "models/SparseMask/checkpoint_63750.pth.tar")
_DEVICE: str      = os.environ.get("OEM_DEVICE", "cuda")
_GREENERY_CLASSES_DEFAULT: str = os.environ.get("OEM_GREENERY_CLASSES", "tree,rangeland,agriculture")
_USE_TTA: bool    = os.environ.get("OEM_USE_TTA", "0").strip() in ("1", "true", "True", "yes", "YES")
_USE_FP16: bool   = os.environ.get("OEM_USE_FP16", "0").strip() in ("1", "true", "True", "yes", "YES")
_INPUT_SIZE: int  = int(os.environ.get("OEM_INPUT_SIZE", "1024"))

_segmenter: Optional[OemLightweightSegmenter] = None


@app.on_event("startup")
def _startup():
    global _segmenter
    if _segmenter is None:
        _segmenter = OemLightweightSegmenter(
            oem_repo=_OEM_REPO,
            model=_OEM_MODEL,
            arch=_OEM_ARCH,
            weights=_OEM_WEIGHTS,
            device=_DEVICE,
            use_tta=_USE_TTA,
            input_size=_INPUT_SIZE,
            use_fp16=_USE_FP16,
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "inprocess",
        "oem_repo": _OEM_REPO,
        "model": _OEM_MODEL,
        "device": _DEVICE,
        "use_tta": _USE_TTA,
        "use_fp16": _USE_FP16,
        "input_size": _INPUT_SIZE,
    }


class ScorePathRequest(BaseModel):
    image_path: str
    greenery_classes: str = _GREENERY_CLASSES_DEFAULT
    # Kept for parity with the upload endpoint. If provided, we'll write the
    # color-coded prediction PNG to this path.
    save_pred_path: Optional[str] = None


@app.post("/score/path")
def score_by_path(req: ScorePathRequest):
    p = Path(req.image_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {p}")

    if _segmenter is None:
        raise HTTPException(status_code=500, detail="Segmenter not initialized (startup not run?)")

    classes = [c.strip() for c in req.greenery_classes.split(",") if c.strip()]
    try:
        result = _segmenter.score_image_path(
            image_path=str(p),
            greenery_classes=classes,
            save_pred_path=req.save_pred_path,
        )
        return JSONResponse(content=result)
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/score/upload")
async def score_by_upload(
    image: UploadFile = File(...),
    save_pred_path: str = None,
):
    data = await image.read()
    suffix = Path(image.filename or "upload.png").suffix or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if _segmenter is None:
            raise HTTPException(status_code=500, detail="Segmenter not initialized (startup not run?)")

        result = _segmenter.score_image_path(
            image_path=tmp_path,
            greenery_classes=[c.strip() for c in _GREENERY_CLASSES_DEFAULT.split(",") if c.strip()],
            save_pred_path=save_pred_path,
        )
        return JSONResponse(content=result)
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    """CLI entrypoint (mirrors the old server's override pattern).

    Priority:
        CLI args > env vars > defaults
    """
    import argparse

    global _OEM_REPO, _OEM_MODEL, _OEM_ARCH, _OEM_WEIGHTS, _DEVICE
    global _GREENERY_CLASSES_DEFAULT, _USE_TTA, _USE_FP16, _INPUT_SIZE

    ap = argparse.ArgumentParser(description="Serve OEM greenery scoring API (fast, in-process)")
    ap.add_argument("--oem_repo", default=None)
    ap.add_argument("--model", default=None, choices=["sparsemask", "fasterseg"])
    ap.add_argument("--arch", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--greenery_classes", default=None)
    ap.add_argument("--use_tta", action="store_true", help="Enable test-time augmentation (slower)")
    ap.add_argument("--use_fp16", action="store_true", help="Enable FP16 autocast on CUDA")
    ap.add_argument("--input_size", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8006)
    args = ap.parse_args()

    if args.oem_repo:
        _OEM_REPO = str(Path(args.oem_repo).expanduser().resolve())
    if args.model:
        _OEM_MODEL = args.model
    if args.arch:
        _OEM_ARCH = args.arch
    if args.weights:
        _OEM_WEIGHTS = args.weights
    if args.device:
        _DEVICE = args.device
    if args.greenery_classes:
        _GREENERY_CLASSES_DEFAULT = args.greenery_classes
    if args.input_size:
        _INPUT_SIZE = int(args.input_size)

    # These flags override env vars if explicitly provided
    if args.use_tta:
        _USE_TTA = True
    if args.use_fp16:
        _USE_FP16 = True

    print(f"Starting greenery API (fast) on {args.host}:{args.port}")
    print(f"  OEM repo : {_OEM_REPO}")
    print(f"  Model    : {_OEM_MODEL}")
    print(f"  Device   : {_DEVICE}")
    print(f"  TTA      : {_USE_TTA}")
    print(f"  FP16     : {_USE_FP16}")
    print(f"  Size     : {_INPUT_SIZE}")
    print(f"  Classes  : {_GREENERY_CLASSES_DEFAULT}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
