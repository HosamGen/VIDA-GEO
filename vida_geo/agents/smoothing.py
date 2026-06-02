# vida_geo/agents/smoothing.py
"""
Mask smoothing (satellite only).

Wraps the cv2 morphology in tools/smooth_mask.py. Two things make this more than
a filter:

  1. It is QC-GATED: smoothing runs only when the mask isn't already bad and is
     either under-covered or fragmented (the thresholds below).
  2. It is QC-VALIDATED: the smoothed mask is re-scored by the same mask-QC, and
     kept only if the score improved. So smoothing can never make a mask worse.

GSV runs skip this entirely (decided by modality at the call site). All the old
--smooth_* CLI flags collapse into SmoothConfig with these defaults; there is no
CLI surface and no subprocess — process() is called as a function.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..tools.smooth_mask import process
from .qc_base import QCResult

log = logging.getLogger(__name__)


@dataclass
class SmoothConfig:
    enabled: bool = True
    min_area: int = 200
    close_radius: int = 3
    widen_px: int = 0
    fill_holes: bool = True
    complete_road: bool = False        # road_safety may enable; greenery must not
    coverage_threshold: float = 0.6
    fragmentation_threshold: float = 0.6
    # complete_road tuning (only used when complete_road=True)
    aspect_thresh: float = 6.0
    good_width_q: float = 80.0


def _should_smooth(qc: QCResult, cfg: SmoothConfig) -> bool:
    if not cfg.enabled:
        return False
    if qc.verdict == "bad":
        return False
    # Missing fields default to 1.0 so a QC hiccup simply skips smoothing.
    coverage = float(qc.raw.get("coverage_score", 1.0))
    fragmentation = float(qc.raw.get("fragmentation_score", 1.0))
    return coverage <= cfg.coverage_threshold or fragmentation <= cfg.fragmentation_threshold


def maybe_smooth(
    mask_path: str,
    qc: QCResult,
    cfg: SmoothConfig,
    out_path: str,
    rescore: Callable[[str], QCResult],
) -> Tuple[str, QCResult]:
    """Conditionally smooth a mask and keep it only if QC improves.

    Args:
        mask_path: accepted mask to consider smoothing.
        qc:        its current mask-QC result.
        cfg:       smoothing configuration.
        out_path:  where to write the smoothed mask if produced.
        rescore:   callable that re-runs mask-QC on a mask path -> QCResult
                   (injected by the segmentation agent to avoid a circular import).

    Returns (mask_path, qc) unchanged if smoothing is skipped or doesn't help,
    otherwise (smoothed_path, smoothed_qc).
    """
    if not _should_smooth(qc, cfg):
        return mask_path, qc

    log.info("Smoothing mask (coverage=%.2f, frag=%.2f)",
             float(qc.raw.get("coverage_score", 1.0)),
             float(qc.raw.get("fragmentation_score", 1.0)))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        process(
            in_path=mask_path,
            out_path=out_path,
            thresh=127,
            min_area=cfg.min_area,
            close_radius=cfg.close_radius,
            do_fill_holes=cfg.fill_holes,
            complete_road=cfg.complete_road,
            aspect_thresh=cfg.aspect_thresh,
            good_width_q=cfg.good_width_q,
            widen_px=cfg.widen_px,
        )
    except Exception as e:
        log.info("Smoothing failed (%s); keeping original mask.", e)
        return mask_path, qc

    if not Path(out_path).is_file():
        return mask_path, qc

    smoothed_qc = rescore(out_path)
    if smoothed_qc.score > qc.score:
        log.info("Smoothing improved QC: %.3f -> %.3f", qc.score, smoothed_qc.score)
        return out_path, smoothed_qc

    log.info("Smoothing did not improve QC (%.3f -> %.3f); keeping original.",
             qc.score, smoothed_qc.score)
    return mask_path, qc
