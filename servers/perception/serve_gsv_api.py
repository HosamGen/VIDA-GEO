"""
Human Perception Place Pulse — FastAPI Scoring Server
=====================================================
Wraps inference_perception.py as a REST API for use in the RSGSV pipeline.

Endpoints:
  POST /score         — Upload an image, get perception scores (0–10)
  GET  /health        — Health check / readiness probe
  GET  /categories    — List available perception categories

Startup:
  uvicorn perception_server:app --host 0.0.0.0 --port 8111
"""

import os
import sys
import io
import types
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_DIR = "./model"
DEVICE = "cuda"

PERCEPTION_CATEGORIES = ["safety", "lively", "beautiful", "wealthy", "boring", "depressing"]

HF_REPO = "Jiani11/human-perception-place-pulse"
MODEL_FILES = {
    "safety":     "safety.pth",
    "lively":     "lively.pth",
    "beautiful":  "beautiful.pth",
    "wealthy":    "wealthy.pth",
    "boring":     "boring.pth",
    "depressing": "depressing.pth",
}


# ─── Model Definition ───────────────────────────────────────────────────────

class Net(nn.Module):
    """Exact reproduction of Model_01.Net from the repo."""
    def __init__(self, num_class: int = 10):
        super(Net, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_class, bias=True),
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

    def forward(self, x):
        return self.model(x)


# ─── Preprocessing ──────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── Model Loading ──────────────────────────────────────────────────────────

def download_models(model_dir: str) -> dict:
    from huggingface_hub import hf_hub_download
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for category, filename in MODEL_FILES.items():
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            logger.info(f"Downloading {filename} from HuggingFace...")
            downloaded = hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=model_dir)
            paths[category] = downloaded
        else:
            paths[category] = local_path
    return paths


def load_models(model_dir: str, device: str) -> dict:
    # Register fake Model_01 module for unpickling
    model_01_module = types.ModuleType("Model_01")
    model_01_module.Net = Net
    sys.modules["Model_01"] = model_01_module

    model_paths = download_models(model_dir)
    models = {}
    for category in PERCEPTION_CATEGORIES:
        model = torch.load(model_paths[category], map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        models[category] = model
        logger.info(f"  Loaded {category} model")
    return models


# ─── Global State ────────────────────────────────────────────────────────────

_models: dict = {}
_transform = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all 6 models once at startup."""
    global _models, _transform
    logger.info(f"Loading perception models from {MODEL_DIR} on {DEVICE}...")
    _models = load_models(MODEL_DIR, DEVICE)
    _transform = get_transform()
    logger.info("All models loaded. Server ready.")
    yield
    logger.info("Shutting down perception server.")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Human Perception Scoring Server",
    description="Scores street-level images on 6 perceptual dimensions (Place Pulse 2.0)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Readiness check — returns 200 only after models are loaded."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"status": "ready", "device": DEVICE, "categories": PERCEPTION_CATEGORIES}


@app.get("/categories")
async def categories():
    """List available perception categories and their semantics."""
    return {
        "categories": PERCEPTION_CATEGORIES,
        "positive_high": ["safety", "lively", "beautiful", "wealthy"],
        "negative_high": ["boring", "depressing"],
        "score_range": [0, 10],
    }


@app.post("/score")
async def score_image(
    image: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    categories: Optional[str] = Query(
        default=None,
        description="Comma-separated list of categories to score (default: all 6). "
                    "e.g. 'safety,beautiful'"
    ),
):
    """
    Score an uploaded image on perceptual dimensions.

    Returns dict of {category: score} where score is in [0, 10].
    Optionally filter to specific categories via query param.
    """
    # Parse requested categories
    if categories:
        requested = [c.strip().lower() for c in categories.split(",")]
        invalid = [c for c in requested if c not in PERCEPTION_CATEGORIES]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown categories: {invalid}. Valid: {PERCEPTION_CATEGORIES}"
            )
    else:
        requested = PERCEPTION_CATEGORIES

    # Read and validate image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run inference
    img_tensor = _transform(img).unsqueeze(0).to(DEVICE)
    scores = {}
    with torch.no_grad():
        for category in requested:
            model = _models[category]
            logits = model(img_tensor)           # shape: (1, 2)
            probs = torch.softmax(logits, dim=1)
            scores[category] = round(probs[0, 1].item() * 10.0, 3)

    return {
        "filename": image.filename,
        "scores": scores,
    }


@app.post("/score_batch")
async def score_batch(
    images: list[UploadFile] = File(..., description="Multiple image files"),
    categories: Optional[str] = Query(default=None, description="Comma-separated categories"),
):
    """
    Score multiple images in one request. 
    Useful for batch evaluation in the RSGSV pipeline.
    """
    if categories:
        requested = [c.strip().lower() for c in categories.split(",")]
        invalid = [c for c in requested if c not in PERCEPTION_CATEGORIES]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown categories: {invalid}")
    else:
        requested = PERCEPTION_CATEGORIES

    results = []
    for upload in images:
        try:
            contents = await upload.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            results.append({"filename": upload.filename, "error": "Invalid image"})
            continue

        img_tensor = _transform(img).unsqueeze(0).to(DEVICE)
        scores = {}
        with torch.no_grad():
            for category in requested:
                model = _models[category]
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                scores[category] = round(probs[0, 1].item() * 10.0, 3)

        results.append({"filename": upload.filename, "scores": scores})

    return {"results": results}


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_gsv_api:app", host="0.0.0.0", port=8111, reload=False)
