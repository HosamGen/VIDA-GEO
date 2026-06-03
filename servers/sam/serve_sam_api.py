# server.py
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from contextlib import asynccontextmanager
import logging

from sam_predictor import SamService, mask_to_png_bytes

logger = logging.getLogger("sam_api")

sam_service: Optional[SamService] = None

# Public SAM checkpoint URLs (ungated — no HF token needed), keyed by model type.
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_CHECKPOINT_URLS["default"] = SAM_CHECKPOINT_URLS["vit_h"]


def ensure_checkpoint(checkpoint_path: str, model_type: str) -> str:
    """Download the public SAM checkpoint to `checkpoint_path` if it's missing.

    SAM weights are ungated and served from a stable public URL, so this is a
    convenience so users can just run the server without a manual download.
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    url = SAM_CHECKPOINT_URLS.get(model_type)
    if url is None:
        raise RuntimeError(
            f"No known download URL for SAM_MODEL_TYPE='{model_type}'. "
            f"Set SAM_CHECKPOINT to an existing file, or use one of {list(SAM_CHECKPOINT_URLS)}."
        )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    logger.info("SAM checkpoint not found at %s — downloading %s ...", checkpoint_path, url)
    import urllib.request
    tmp = checkpoint_path + ".part"
    urllib.request.urlretrieve(url, tmp)   # public URL, no auth
    os.replace(tmp, checkpoint_path)
    logger.info("SAM checkpoint downloaded to %s", checkpoint_path)
    return checkpoint_path


def read_image_rgb_from_upload(upload: UploadFile) -> np.ndarray:
    """
    Reads UploadFile -> RGB np.uint8 HxWx3
    """
    try:
        raw = upload.file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")


def parse_json_field(field: Optional[str], name: str) -> Optional[Any]:
    if field is None or field == "":
        return None
    try:
        return json.loads(field)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in '{name}': {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan which initializes the SamService on startup.

    Mirrors the style used in serve_lisat_api.py: logs the resolved module
    path, reads env vars, constructs the service and attaches lightweight
    debug/meta info for the /health endpoint.
    """
    global sam_service

    # Log which sam_predictor module file is being used
    import sam_predictor as sp  # noqa

    logger.info("Using sam_predictor from: %s", getattr(sp, "__file__", "<unknown>"))
    logger.info("Server CWD: %s", os.getcwd())

    checkpoint = os.environ.get("SAM_CHECKPOINT") or "checkpoints/sam_vit_h_4b8939.pth"
    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    device = os.environ.get("SAM_DEVICE")  # optional: "cpu", "cuda", "cuda:0", etc.

    # Download the (ungated) checkpoint automatically if it isn't on disk yet.
    checkpoint = ensure_checkpoint(checkpoint, model_type)

    sam_service = SamService(
        checkpoint_path=checkpoint,
        model_type=model_type,
        device=device,
    )

    # Attach some metadata to the service for health/debug responses.
    try:
        sam_service._meta = {"checkpoint": checkpoint, "model_type": model_type, "device": device}
    except Exception:
        # Non-fatal if attribute can't be set
        logger.exception("Failed to attach meta to sam_service")

    logger.info("SAM service loaded.")

    yield

    # No explicit shutdown required here; allow GC to cleanup if needed.


app = FastAPI(title="SAM Segmentation API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if sam_service is None:
        raise HTTPException(status_code=503, detail="SAM service not initialized")
    payload: Dict[str, Any] = {"status": "ok"}
    # If the service exposes a debug_state method, include it (similar to LISAt)
    if hasattr(sam_service, "debug_state"):
        try:
            payload.update(sam_service.debug_state())
        except Exception:
            logger.exception("sam_service.debug_state failed")
    elif hasattr(sam_service, "_meta"):
        payload.update(getattr(sam_service, "_meta", {}))
    return payload


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),

    # Mode
    auto: bool = Form(False),
    auto_topk: int = Form(5),

    # Point prompting
    points: Optional[str] = Form(None),      # JSON: [[x,y], ...]
    labels: Optional[str] = Form(None),      # JSON: [1,0,1,...]  (optional if only foreground points)
    neg_points: Optional[str] = Form(None),  # JSON: [[x,y], ...] background points

    multimask: bool = Form(False),

    # Output control
    response_type: str = Form("json"),       # "json" or "png"
    return_all_candidates: bool = Form(False),
):
    """
    POST /segment (multipart/form-data)

    - image: file upload
    - auto=true: use automatic mask generator (top-K by area)
    - else: use points and (optional) neg_points + labels
    """
    global sam_service
    if sam_service is None:
        raise HTTPException(status_code=500, detail="SAM service not initialized")

    image_rgb = read_image_rgb_from_upload(image)
    h, w = image_rgb.shape[:2]

    # Auto mode
    if auto:
        masks = sam_service.segment_auto(image_rgb=image_rgb, topk=auto_topk)

        # If they want raw PNG bytes, return the largest mask only (top-1)
        if response_type.lower() == "png":
            if len(masks) == 0:
                raise HTTPException(status_code=500, detail="No masks returned by auto generator")
            png_bytes = mask_to_png_bytes(masks[0]["segmentation"].astype(bool))
            return Response(content=png_bytes, media_type="image/png")

        # JSON response (base64 PNG per mask)
        items: List[Dict[str, Any]] = []
        for m in masks:
            seg = m["segmentation"].astype(bool)
            png_b64 = base64.b64encode(mask_to_png_bytes(seg)).decode("utf-8")
            items.append(
                {
                    "area": int(m.get("area", 0)),
                    "bbox_xywh": m.get("bbox"),  # SAM returns [x, y, w, h]
                    "mask_png_base64": png_b64,
                }
            )

        return JSONResponse(
            {
                "mode": "auto",
                "image_width": w,
                "image_height": h,
                "count": len(items),
                "masks": items,
            }
        )

    # Point mode
    points_list = parse_json_field(points, "points")
    neg_points_list = parse_json_field(neg_points, "neg_points")
    labels_list = parse_json_field(labels, "labels")

    if points_list is None and neg_points_list is None:
        raise HTTPException(
            status_code=400,
            detail="Point mode requires 'points' (foreground) and/or 'neg_points' (background), or use auto=true.",
        )

    pos = np.array(points_list or [], dtype=np.float32).reshape(-1, 2)
    neg = np.array(neg_points_list or [], dtype=np.float32).reshape(-1, 2)

    # Build combined points + labels
    all_points = np.vstack([pos, neg]) if (len(pos) + len(neg)) > 0 else np.zeros((0, 2), dtype=np.float32)

    if labels_list is not None:
        lab = np.array(labels_list, dtype=np.int32).reshape(-1)
        if lab.shape[0] != all_points.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"labels length ({lab.shape[0]}) must match total points ({all_points.shape[0]}).",
            )
        all_labels = lab
    else:
        # Default: all 'points' are foreground=1, all 'neg_points' are background=0
        all_labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    try:
        result = sam_service.segment_points(
            image_rgb=image_rgb,
            points_xy=all_points,
            labels=all_labels,
            multimask_output=multimask or return_all_candidates,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    best_mask = result.masks[result.best_index]
    best_score = float(result.scores[result.best_index])

    # Raw PNG response: best mask only
    if response_type.lower() == "png":
        png_bytes = mask_to_png_bytes(best_mask)
        # You can read this header from curl with -i if you want.
        headers = {"X-SAM-Score": f"{best_score:.6f}"}
        return Response(content=png_bytes, media_type="image/png", headers=headers)

    # JSON response
    best_png_b64 = base64.b64encode(mask_to_png_bytes(best_mask)).decode("utf-8")

    payload: Dict[str, Any] = {
        "mode": "points",
        "image_width": w,
        "image_height": h,
        "best_index": int(result.best_index),
        "best_score": best_score,
        "best_mask_png_base64": best_png_b64,
        "points_xy": all_points.tolist(),
        "labels": all_labels.tolist(),
    }

    if return_all_candidates:
        candidates = []
        for i in range(result.masks.shape[0]):
            candidates.append(
                {
                    "index": i,
                    "score": float(result.scores[i]),
                    "mask_png_base64": base64.b64encode(mask_to_png_bytes(result.masks[i])).decode("utf-8"),
                }
            )
        payload["candidates"] = candidates

    return JSONResponse(payload)