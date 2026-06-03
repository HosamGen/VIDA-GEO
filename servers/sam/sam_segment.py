#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def validate_points(points, w, h, name="point"):
    for (x, y) in points:
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(
                f"{name} ({x}, {y}) is outside the image bounds: width={w}, height={h}"
            )


def save_mask_png(mask_bool: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), mask_u8)


def overlay_mask_rgb(image_rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Returns an RGB image with a colored mask overlay.
    """
    overlay = image_rgb.astype(np.float32).copy()
    color = np.array([0, 255, 0], dtype=np.float32)  # green in RGB
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_points_bgr(image_bgr: np.ndarray, pos_points, neg_points) -> np.ndarray:
    """
    Draw small circles: positive points in green, negative points in red (BGR).
    """
    out = image_bgr.copy()
    for (x, y) in pos_points:
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)  # green
    for (x, y) in neg_points:
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)  # red
    return out


def main():
    parser = argparse.ArgumentParser(description="Segment Anything (SAM) point-prompt segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM .pth checkpoint")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b", "default"],
                        help="SAM backbone type (default=vit_h)")
    parser.add_argument("--device", default=None, help='cuda, cpu, or e.g. "cuda:0" (default: auto)')

    # You can pass multiple points by repeating --point / --neg-point
    parser.add_argument("--point", nargs=2, type=int, action="append",
                        help="Foreground point (x y). Repeatable.")
    parser.add_argument("--neg-point", nargs=2, type=int, action="append",
                        help="Background point (x y). Repeatable.")

    parser.add_argument("--auto", action="store_true",
                        help="If set, ignore points and run automatic mask generation.")
    parser.add_argument("--auto-topk", type=int, default=25,
                        help="For --auto: save top-K masks by area (default=25)")

    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha (default=0.55)")
    parser.add_argument("--multimask", action="store_true",
                        help="Return multiple candidate masks (we'll pick best by score).")
    parser.add_argument("--save-all-candidates", action="store_true",
                        help="If set (with point prompts), save all candidate masks too.")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_rgb = load_image_rgb(args.image)
    h, w = image_rgb.shape[:2]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    if args.auto:
        # Automatic mask generation for the whole image
        mask_generator = SamAutomaticMaskGenerator(sam)
        with torch.no_grad():
            masks = mask_generator.generate(image_rgb)

        # Sort by area, save top-K
        masks_sorted = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
        top = masks_sorted[: max(1, args.auto_topk)]

        # Make a combined overlay
        combined = image_rgb.copy()
        rng = np.random.default_rng(0)

        for i, m in enumerate(top):
            seg = m["segmentation"].astype(bool)
            save_mask_png(seg, outdir / f"auto_mask_{i:03d}.png")

            # random-ish color overlay for visualization
            color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
            tmp = combined.astype(np.float32)
            tmp[seg] = tmp[seg] * (1.0 - args.alpha) + color.astype(np.float32) * args.alpha
            combined = np.clip(tmp, 0, 255).astype(np.uint8)

        cv2.imwrite(str(outdir / "auto_overlay.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"[OK] Wrote {len(top)} masks + auto_overlay.png to: {outdir}")
        return

    # Point-prompt mode
    pos_points = args.point or []
    neg_points = args.neg_point or []
    if len(pos_points) == 0 and len(neg_points) == 0:
        raise SystemExit(
            "No points provided. Use --point x y (recommended) or run --auto for automatic masks."
        )

    validate_points(pos_points, w, h, name="--point")
    validate_points(neg_points, w, h, name="--neg-point")

    all_points = np.array(pos_points + neg_points, dtype=np.float32)
    # Labels: 1 = foreground, 0 = background
    # (SAM point prompts use (x, y) pixel coords and labels 1/0.) 
    labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=all_points,
            point_labels=labels,
            multimask_output=args.multimask or args.save_all_candidates,
        )

    # Pick best mask by score
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)

    save_mask_png(best_mask, outdir / "mask.png")

    overlay_rgb = overlay_mask_rgb(image_rgb, best_mask, alpha=args.alpha)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = draw_points_bgr(overlay_bgr, pos_points, neg_points)
    cv2.imwrite(str(outdir / "overlay.png"), overlay_bgr)

    if args.save_all_candidates:
        for i in range(masks.shape[0]):
            save_mask_png(masks[i].astype(bool), outdir / f"candidate_{i}_score_{scores[i]:.4f}.png")

    print(f"[OK] Best score={scores[best_idx]:.4f}. Wrote mask.png + overlay.png to: {outdir}")


if __name__ == "__main__":
    main()
