# vida_geo/image_utils.py
"""
Image utilities: size/mask helpers, ROI crop (for FLUX), mask-aware paste-back,
and the coordinate-prefix extractor used for run-folder naming.

The generation agent's FLUX path uses crop_roi_bbox -> flux_fill -> paste_roi_back:
  - crop_roi_bbox returns (roi_image_path, roi_mask_path, origin_xy) so the edited
    crop can be re-placed at the same pixel origin.
  - paste_roi_back composites the edited crop into the full image using the mask,
    so only masked pixels change (resizing the crop if the editor altered its size).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def load_mask_bool(mask_path: str) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    return np.array(m) > 0


def mask_area_fraction(mask_path: str) -> float:
    return float(load_mask_bool(mask_path).mean())


def _expand_to_multiple(start: int, end: int, limit: int, multiple: int) -> Tuple[int, int]:
    size = end - start
    if size <= 0:
        return 0, min(limit, multiple)
    target = int(math.ceil(size / multiple) * multiple)
    extra = target - size
    s = start - extra // 2
    e = end + (extra - extra // 2)
    if s < 0:
        e = min(limit, e - s)
        s = 0
    if e > limit:
        s = max(0, s - (e - limit))
        e = limit
    if (e - s) % multiple != 0:
        add = multiple - ((e - s) % multiple)
        e2 = min(limit, e + add)
        s2 = max(0, e2 - (e - s + add))
        s, e = s2, e2
    return s, e


def crop_roi_bbox(
    image_path: str,
    mask_path: str,
    out_dir: str,
    pad_pct: float = 0.15,
    multiple: int = 8,
) -> Tuple[str, str, Tuple[int, int]]:
    """Crop the mask's padded bbox (snapped to `multiple`) from image and mask.

    Returns (roi_image_path, roi_mask_path, origin_xy). If the mask is empty,
    returns the originals with origin (0, 0).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    m = Image.open(mask_path).convert("L")
    arr = np.array(m) > 0

    ys, xs = np.where(arr)
    if ys.size == 0:
        return image_path, mask_path, (0, 0)

    w, h = img.size
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padx, pady = int((x1 - x0) * pad_pct), int((y1 - y0) * pad_pct)
    x0p, y0p = max(0, x0 - padx), max(0, y0 - pady)
    x1p, y1p = min(w, x1 + padx), min(h, y1 + pady)
    x0p, x1p = _expand_to_multiple(x0p, x1p, w, multiple)
    y0p, y1p = _expand_to_multiple(y0p, y1p, h, multiple)

    roi_img = out / "roi_image.png"
    roi_mask = out / "roi_mask.png"
    img.crop((x0p, y0p, x1p, y1p)).save(roi_img)
    m.crop((x0p, y0p, x1p, y1p)).save(roi_mask)
    return str(roi_img), str(roi_mask), (x0p, y0p)


def paste_roi_back(
    full_image_path: str,
    roi_edited_path: str,
    mask_path: str,
    origin_xy: Tuple[int, int],
    out_path: str,
) -> str:
    """Composite an edited ROI crop back into the full image, masked.

    Only pixels where the full mask is True (within the ROI box) are replaced,
    so the edit stays inside the intended region. The edited crop is resized to
    the ROI box if the editor changed its dimensions.
    """
    full = Image.open(full_image_path).convert("RGB")
    roi = Image.open(roi_edited_path).convert("RGB")
    mask_full = load_mask_bool(mask_path)

    x0, y0 = origin_xy
    w, h = full.size
    rw, rh = roi.size
    x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
    box_w, box_h = x1 - x0, y1 - y0

    if (rw, rh) != (box_w, box_h):
        roi = roi.resize((box_w, box_h), Image.LANCZOS)

    full_arr = np.array(full)
    roi_arr = np.array(roi)
    mask_roi = mask_full[y0:y1, x0:x1]

    region = full_arr[y0:y1, x0:x1].copy()
    region[mask_roi] = roi_arr[mask_roi]
    full_arr[y0:y1, x0:x1] = region

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full_arr).save(out_path, quality=95)
    return out_path


def clip_mask_to_box(mask_path: str, frac_x: float, frac_y: float,
                     box_frac: float, out_path: str) -> Optional[str]:
    """Clip a mask to a fixed box centered on (frac_x, frac_y).

    Used to localize an over-large mask (e.g. SAM3 'road' grabbing every road)
    to a diffuse feature's neighborhood (e.g. the intersection at the planner's
    point). Keeps only mask pixels inside a box of side `box_frac` * image size.
    Returns out_path, or None if the clipped mask is empty.
    """
    m = Image.open(mask_path).convert("L")
    arr = np.array(m) > 0
    h, w = arr.shape
    half_w = max(1, int(box_frac * w / 2))
    half_h = max(1, int(box_frac * h / 2))
    cx, cy = int(frac_x * w), int(frac_y * h)
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)

    clipped = np.zeros_like(arr)
    clipped[y0:y1, x0:x1] = arr[y0:y1, x0:x1]
    if not clipped.any():
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((clipped * 255).astype("uint8")).save(out_path)
    return out_path


def extract_coordinate_prefix(image_path: str) -> str:
    """Run-folder name: coordinate stem for GSV/satellite tiles, else the file stem.

    '33.40_-112.01_130.jpg' -> '33.40_-112.01_130'. For edit outputs nested under a
    '*_epochs' folder, recover the coordinate prefix and append the image stem.
    """
    p = Path(image_path)
    stem = p.stem
    coord = re.compile(r'^-?\d+\.\d+_-?\d+\.\d+')
    if coord.match(stem):
        return stem
    for parent in p.parents:
        if "_epochs" in parent.name:
            clean = re.sub(r'_\d{8}_\d{6}$', '', parent.name.replace("_epochs", ""))
            if clean and coord.match(clean):
                return f"{clean}_{stem}"
            if clean:
                return clean
    return stem
