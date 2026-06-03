"""
Human Perception Place Pulse - Inference Script
=================================================
Scores Google Street View (GSV) images on 6 perceptual dimensions:
  safety, lively, beautiful, wealthy, boring, depressing

Scores range 0-10:
  - safety, lively, beautiful, wealthy: higher = more positive
  - boring, depressing: higher = more negative

Model: ViT-B/16 (IMAGENET1K_SWAG_E2E_V1 backbone) with 3-layer MLP head
Input: 384x384 center crop (bicubic resize), ImageNet normalization
Source: https://github.com/strawmelon11/human-perception-place-pulse

SETUP:
  pip install torch torchvision pillow huggingface_hub pandas

USAGE:
  # Score a single image
  python inference_perception.py --image path/to/gsv_image.jpg

  # Score a folder of images
  python inference_perception.py --image_dir path/to/images/

  # Score and save to CSV
  python inference_perception.py --image_dir path/to/images/ --output scores.csv

  # Use GPU
  python inference_perception.py --image path/to/image.jpg --device cuda
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image


# ─── Model Definition (exact mirror of Model_01.py from the repo) ───────────

class Net(nn.Module):
    """
    Exact reproduction of Model_01.Net from the repo.
    ViT-B/16 backbone with the default head replaced by a 3-layer MLP.
    The pickle files reference 'Model_01.Net', so this class name matters.
    """
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
        x = self.model(x)
        return x


# ─── Preprocessing ──────────────────────────────────────────────────────────

def get_transform():
    """
    Matches the original eval.py transform exactly:
      1. Resize directly to 384x384 (bilinear, the default)
      2. ToTensor
      3. ImageNet normalization

    YOU CAN PASS ANY RESOLUTION IMAGE — the transform handles it.
    The resize forces everything to 384x384 regardless of input shape.
    """
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─── Model Loading ──────────────────────────────────────────────────────────

PERCEPTION_CATEGORIES = ["safety", "lively", "beautiful", "wealthy", "boring", "depressing"]

# HuggingFace model repo and filenames (auto-downloaded on first run)
HF_REPO = "Jiani11/human-perception-place-pulse"
MODEL_FILES = {
    "safety":     "safety.pth",
    "lively":     "lively.pth",
    "beautiful":  "beautiful.pth",
    "wealthy":    "wealthy.pth",
    "boring":     "boring.pth",
    "depressing": "depressing.pth",
}


def download_models(model_dir: str = "./model") -> dict:
    """Download model weights from HuggingFace Hub if not already cached."""
    from huggingface_hub import hf_hub_download

    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for category, filename in MODEL_FILES.items():
        local_path = os.path.join(model_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from HuggingFace...")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                local_dir=model_dir,
            )
            paths[category] = downloaded
        else:
            paths[category] = local_path
    return paths


def load_models(model_dir: str = "./model", device: str = "cpu") -> dict:
    """
    Load all 6 perception models into memory.

    The HuggingFace .pth files were saved with torch.save(model) — i.e., the
    entire model object is pickled, not just the state_dict. The pickle
    references 'Model_01.Net', so we must register that class before loading.
    """
    import types

    # Create a fake 'Model_01' module so torch.load can unpickle 'Model_01.Net'
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
        print(f"  Loaded {category} model")
    return models


# ─── Inference ───────────────────────────────────────────────────────────────

def score_image(image_path: str, models: dict, transform, device: str = "cpu") -> dict:
    """
    Score a single image on all 6 perceptual dimensions.

    The models output 2 logits (binary classifier trained on Place Pulse
    pairwise comparisons). The softmax probability of class 1 (the "positive"
    class) is taken as a confidence, then scaled to 0-10 to match the
    README's stated score range.

    Returns:
        dict with keys: safety, lively, beautiful, wealthy, boring, depressing
        Each value is a float score in [0, 10].
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    scores = {}
    with torch.no_grad():
        for category, model in models.items():
            logits = model(img_tensor)  # shape: (1, 2)
            probs = torch.softmax(logits, dim=1)
            # Probability of the "positive/high" class, scaled to 0-10
            score_0_10 = probs[0, 1].item() * 10.0

            scores[category] = round(score_0_10, 3)

    return scores


def score_single_image_simple(image_path: str, models: dict, transform, device: str = "cpu") -> dict:
    """Simplified version returning scores as floats in [0, 10]."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    scores = {}
    with torch.no_grad():
        for category, model in models.items():
            logits = model(img_tensor)  # shape: (1, 2)
            probs = torch.softmax(logits, dim=1)
            scores[category] = round(probs[0, 1].item() * 10.0, 3)
    return scores


# ─── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score GSV images on 6 human perception dimensions"
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to a directory of images")
    parser.add_argument("--model_dir", type=str, default="./model", help="Directory for model weights")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device to run inference on")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image_dir")

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
        image_paths = sorted(set(image_paths))

    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} image(s)")
    print(f"Using device: {args.device}")

    # Load models
    print("Loading models...")
    models = load_models(args.model_dir, args.device)
    transform = get_transform()

    # Run inference
    all_results = []
    for path in image_paths:
        print(f"\nScoring: {os.path.basename(path)}")
        scores = score_image(path, models, transform, args.device)

        result = {"image": os.path.basename(path)}
        for cat in PERCEPTION_CATEGORIES:
            result[cat] = scores[cat]
            print(f"  {cat:12s}: {scores[cat]:.2f} / 10")

        all_results.append(result)

    # Save to CSV if requested
    if args.output and all_results:
        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        except ImportError:
            # Fallback without pandas
            import csv
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()