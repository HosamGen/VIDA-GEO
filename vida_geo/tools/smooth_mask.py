# vida_geo/tools/smooth_mask.py
"""
Mask morphology for satellite masks: small-component removal, hole filling,
morphological close, optional road completion, optional widening.

Imported as a function (process / individual ops) by agents/smoothing.py — no
subprocess, no CLI in the pipeline path. The __main__ block is kept only as a
standalone debugging convenience.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def read_mask(path: str, thresh: int = 127) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img > thresh


def write_mask(path: str, mask_bool: np.ndarray) -> None:
    out = (mask_bool.astype(np.uint8) * 255)
    if not cv2.imwrite(path, out):
        raise RuntimeError(f"cv2.imwrite failed for: {path}")


def remove_small_components(mask_bool: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_bool
    m = (mask_bool.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(mask_bool, dtype=bool)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= (labels == i)
    return keep


def fill_holes(mask_bool: np.ndarray) -> np.ndarray:
    """Flood-fill from all border pixels; whatever stays background is a hole."""
    m = (mask_bool.astype(np.uint8) * 255)
    inv = cv2.bitwise_not(m)
    h, w = inv.shape
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)

    border_points = []
    border_points += [(x, 0) for x in range(w)]
    border_points += [(x, h - 1) for x in range(w)]
    border_points += [(0, y) for y in range(h)]
    border_points += [(w - 1, y) for y in range(h)]

    for (x, y) in border_points:
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, seedPoint=(x, y), newVal=0)

    holes = flood == 255
    return mask_bool | holes


def morph_close(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_bool
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask_bool.astype(np.uint8) * 255)
    out = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return out > 127


def dilate(mask_bool: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_bool
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask_bool.astype(np.uint8) * 255)
    out = cv2.dilate(m, kernel, iterations=1)
    return out > 127


def largest_component(mask_bool: np.ndarray):
    m = (mask_bool.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return mask_bool, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    comp = labels == idx
    x, y, w, h = stats[idx, :4]
    return comp, (x, y, w, h)


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0:
        return x
    win = int(win)
    if win < 3:
        return x.astype(float)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xpad = np.pad(x.astype(float), (pad, pad), mode="edge")
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        out[i] = np.median(xpad[i: i + win])
    return out


def fit_line(xs: np.ndarray, ys: np.ndarray):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return 0.0, float(ys[0]) if len(ys) else (0.0, 0.0)
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(a), float(b)


def complete_road_like(
    mask_bool: np.ndarray,
    aspect_thresh: float = 6.0,
    good_width_q: float = 80.0,
    smooth_win: int = 51,
    require_touch_one_edge: bool = True,
) -> np.ndarray:
    """Road completion for a long/thin largest component (BetaRisk-specific)."""
    h, w = mask_bool.shape
    comp, bbox = largest_component(mask_bool)
    if bbox is None:
        return mask_bool

    x0, y0, bw, bh = bbox
    xmin, xmax = x0, x0 + bw - 1
    ymin, ymax = y0, y0 + bh - 1

    if bw / max(1, bh) >= aspect_thresh:
        orient = "h"
    elif bh / max(1, bw) >= aspect_thresh:
        orient = "v"
    else:
        return mask_bool

    touches_left = xmin <= 0
    touches_right = xmax >= w - 1
    touches_top = ymin <= 0
    touches_bottom = ymax >= h - 1

    if require_touch_one_edge:
        if orient == "h" and not (touches_left or touches_right):
            return mask_bool
        if orient == "v" and not (touches_top or touches_bottom):
            return mask_bool

    out = mask_bool.copy()

    if orient == "h":
        xs = np.where(comp.any(axis=0))[0]
        if len(xs) < 10:
            return mask_bool
        top = np.array([np.min(np.where(comp[:, xx])[0]) for xx in xs], dtype=float)
        bot = np.array([np.max(np.where(comp[:, xx])[0]) for xx in xs], dtype=float)
        top_s = rolling_median(top, smooth_win)
        bot_s = rolling_median(bot, smooth_win)
        width = bot_s - top_s + 1.0
        center = (top_s + bot_s) / 2.0
        half = width / 2.0
        thr = np.percentile(width, good_width_q)
        good = width >= thr
        if good.sum() < 30:
            k = min(200, len(xs))
            good = np.zeros_like(width, dtype=bool)
            if touches_right:
                good[-k:] = True
            else:
                good[:k] = True
        aC, bC = fit_line(xs[good], center[good])
        half0 = float(np.median(half[good]))
        fill_x0, fill_x1 = xmin, xmax
        if touches_right and not touches_left:
            fill_x0 = 0
        if touches_left and not touches_right:
            fill_x1 = w - 1
        for xx in range(fill_x0, fill_x1 + 1):
            c = aC * xx + bC
            yt = max(0, min(h - 1, int(round(c - half0))))
            yb = max(0, min(h - 1, int(round(c + half0))))
            if yt > yb:
                yt, yb = yb, yt
            out[yt: yb + 1, xx] = True
    else:
        ys = np.where(comp.any(axis=1))[0]
        if len(ys) < 10:
            return mask_bool
        left = np.array([np.min(np.where(comp[yy, :])[0]) for yy in ys], dtype=float)
        right = np.array([np.max(np.where(comp[yy, :])[0]) for yy in ys], dtype=float)
        left_s = rolling_median(left, smooth_win)
        right_s = rolling_median(right, smooth_win)
        width = right_s - left_s + 1.0
        center = (left_s + right_s) / 2.0
        half = width / 2.0
        thr = np.percentile(width, good_width_q)
        good = width >= thr
        if good.sum() < 30:
            k = min(200, len(ys))
            good = np.zeros_like(width, dtype=bool)
            if touches_bottom:
                good[-k:] = True
            else:
                good[:k] = True
        aC, bC = fit_line(ys[good], center[good])
        half0 = float(np.median(half[good]))
        fill_y0, fill_y1 = ymin, ymax
        if touches_bottom and not touches_top:
            fill_y0 = 0
        if touches_top and not touches_bottom:
            fill_y1 = h - 1
        for yy in range(fill_y0, fill_y1 + 1):
            c = aC * yy + bC
            xl = max(0, min(w - 1, int(round(c - half0))))
            xr = max(0, min(w - 1, int(round(c + half0))))
            if xl > xr:
                xl, xr = xr, xl
            out[yy, xl: xr + 1] = True

    return out


def process(
    in_path: str,
    out_path: str,
    thresh: int = 127,
    min_area: int = 200,
    close_radius: int = 3,
    do_fill_holes: bool = True,
    complete_road: bool = False,
    aspect_thresh: float = 6.0,
    good_width_q: float = 80.0,
    widen_px: int = 0,
):
    m = read_mask(in_path, thresh=thresh)
    m = remove_small_components(m, min_area=min_area)
    m = morph_close(m, radius=close_radius)
    if do_fill_holes:
        m = fill_holes(m)
    if complete_road:
        m = complete_road_like(m, aspect_thresh=aspect_thresh,
                               good_width_q=good_width_q, smooth_win=51,
                               require_touch_one_edge=True)
    m = dilate(m, px=widen_px)
    if do_fill_holes:
        m = fill_holes(m)
    write_mask(out_path, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--thresh", type=int, default=127)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--close-radius", type=int, default=3)
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--complete-road", action="store_true")
    ap.add_argument("--aspect-thresh", type=float, default=6.0)
    ap.add_argument("--good-width-q", type=float, default=80.0)
    ap.add_argument("--widen-px", type=int, default=0)
    args = ap.parse_args()
    process(
        in_path=args.in_path, out_path=args.out_path, thresh=args.thresh,
        min_area=args.min_area, close_radius=args.close_radius,
        do_fill_holes=args.fill_holes, complete_road=args.complete_road,
        aspect_thresh=args.aspect_thresh, good_width_q=args.good_width_q,
        widen_px=args.widen_px,
    )


if __name__ == "__main__":
    main()
