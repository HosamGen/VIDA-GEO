#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def save_mask_png(mask_hw: np.ndarray, out_path: str) -> None:
    # mask_hw: bool or 0/1 array of shape (H,W)
    mask_u8 = (mask_hw.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(out_path, format="PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image (png/jpg)")
    ap.add_argument("--prompt", required=True, help='Text concept prompt, e.g. "park" or "a park"')
    ap.add_argument("--out", default="sam3_mask.png", help="Output mask PNG path")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--score_thresh", type=float, default=0.25, help="Filter detections by score")
    ap.add_argument("--mask_thresh", type=float, default=0.5, help="Threshold mask probs to binary")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load image (keep original size for final mask)
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    device = args.device
    print(f"Using device: {device}")

    # Build model + processor
    # (by default, builds model and downloads weights from HuggingFace if not present)
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model)

    # Inference
    with torch.inference_mode():
        state = processor.set_image(img)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)

        masks = output["masks"]   # Tensor (N,1,H,W)
        scores = output["scores"] # Tensor (N,)

    n = int(masks.shape[0]) if torch.is_tensor(masks) else 0
    print(f"Detected instances: {n}")

    if n == 0:
        # Save an all-zero mask
        save_mask_png(np.zeros((orig_h, orig_w), dtype=bool), args.out)
        print(f"No detections. Wrote empty mask to: {args.out}")
        return

    masks = masks.detach().cpu().float()  # (N,1,H,W)
    scores = scores.detach().cpu().float()  # (N,)

    # Filter by score threshold
    keep = scores >= float(args.score_thresh)
    if keep.sum().item() == 0:
        save_mask_png(np.zeros((orig_h, orig_w), dtype=bool), args.out)
        print(f"All detections filtered by score_thresh. Wrote empty mask to: {args.out}")
        return

    masks = masks[keep]  # (K,1,H,W)
    scores = scores[keep]

    # Union all kept instance masks
    # masks are described as binary in docs, but we threshold defensively
    union = (masks[:, 0, :, :] >= float(args.mask_thresh)).any(dim=0).numpy()  # (H,W) bool

    # Resize union mask back to original image size (SAM3 internally uses 1008x1008)
    union_img = Image.fromarray((union.astype(np.uint8) * 255), mode="L")
    union_img = union_img.resize((orig_w, orig_h), resample=Image.NEAREST)
    union = (np.array(union_img) > 127)

    save_mask_png(union, args.out)
    print(f"Wrote mask to: {args.out}")
    print("Top scores:", scores.sort(descending=True).values[:10].tolist())


if __name__ == "__main__":
    main()