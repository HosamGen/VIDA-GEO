# vida_geo/agents/segmentation.py
"""
Segmentation agent — one code path, modality-routed.

Per region, in order:
  1. Text-referred segmentation via the modality's text segmenter
     (LISAt for satellite, SAM3 for GSV), trying a few keyword variations.
  2. SAM point-prompt fallback around the planner's ROI center.
  3. Best-effort: keep the highest-QC mask seen if it clears a low bar.

Every candidate mask gets a cheap heuristic pre-filter (area sanity), then the
LLM mask-QC (qc_mask, domain-keyed prompt). Accepted satellite masks pass
through the QC-gated / QC-validated smoothing step; GSV masks skip it.

The agent picks its text segmenter and its smoothing policy from the registry,
so satellite vs GSV is data, not branches.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..domains.registry import DomainSpec, ModalitySpec
from ..llm.openrouter import LLMConfig
from ..tools import clients
from ..tools.clients import SamConfig, ServiceConfig
from .qc_base import QCResult
from .qc_mask import evaluate_mask
from .smoothing import SmoothConfig, maybe_smooth
from ..image_utils import load_image_size, mask_area_fraction, clip_mask_to_box

log = logging.getLogger(__name__)

# Heuristic pre-filter bounds (fraction of image covered).
_MIN_AREA_FRACTION = 0.003
_MAX_AREA_FRACTION = 0.85
_BEST_EFFORT_FLOOR = 0.3


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _heuristic_score(mask_path: str) -> float:
    """Cheap area-based sanity score in [0,1] before paying for LLM QC."""
    area = mask_area_fraction(mask_path)
    if area < _MIN_AREA_FRACTION:
        return 0.0
    if area > _MAX_AREA_FRACTION:
        return 0.1
    if area < 0.01 or area > 0.60:
        return 0.3
    return 0.8


def _keyword_variations(keyword: str) -> List[str]:
    """Short concrete-noun variants; text segmenters prefer these."""
    kw = (keyword or "object").strip().lower()
    variants = [kw]
    if kw.endswith("s") and len(kw) > 3:
        variants.append(kw[:-1])
    elif not kw.endswith("s"):
        variants.append(kw + "s")
    synonyms = {
        "vehicle": ["car", "automobile"], "automobile": ["car"],
        "pole": ["post", "utility pole"], "utility pole": ["pole"], "electric pole": ["pole"],
        "facade": ["building", "wall"], "building facade": ["building"],
        "road surface": ["road"], "sidewalk": ["pavement"], "pavement": ["sidewalk"],
        "storefront": ["shop"], "sign": ["signage"], "signage": ["sign"],
        "trash": ["garbage", "litter"], "fence": ["barrier"],
    }
    for syn in synonyms.get(kw, []):
        if syn not in variants:
            variants.append(syn)
    return variants[:4]


# Which clients.ServiceConfig + call to use for the named text segmenter.
def _text_segmenter_call(name: str):
    """Return (config_instance, callable) for a text segmenter name."""
    if name == "sam3":
        cfg = clients.Sam3Config()
        def call(image_path, prompt, out_mask_path):
            return clients.text_segment(image_path, prompt, out_mask_path, cfg, union=True)
        return cfg, call
    if name == "lisat":
        cfg = clients.LisatConfig()
        def call(image_path, prompt, out_mask_path):
            return clients.text_segment(image_path, prompt, out_mask_path, cfg)
        return cfg, call
    raise ValueError(f"Unknown text segmenter {name!r}")


class SegmentationAgent:
    def __init__(
        self,
        image_path: str,
        region: dict,
        run_dir: str,
        spec: DomainSpec,
        modality: ModalitySpec,
        *,
        sam_cfg: Optional[SamConfig] = None,
        smooth_cfg: Optional[SmoothConfig] = None,
        mask_qc_threshold: float = 0.7,
        sam_point_attempts: int = 5,
        llm_cfg: Optional[LLMConfig] = None,
    ):
        self.image_path = str(Path(image_path).expanduser().resolve())
        self.region = region
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.modality = modality
        self.sam_cfg = sam_cfg or SamConfig()
        self.smooth_cfg = smooth_cfg or SmoothConfig()
        self.mask_qc_threshold = float(mask_qc_threshold)
        self.sam_point_attempts = int(sam_point_attempts)
        self.llm_cfg = llm_cfg

        self.region_id = str(region.get("region_id", "r0"))
        self.img_w, self.img_h = load_image_size(self.image_path)
        self.frac_x = float(region.get("frac_x", 0.5))
        self.frac_y = float(region.get("frac_y", 0.5))
        self.point_x = int(self.frac_x * self.img_w)
        self.point_y = int(self.frac_y * self.img_h)
        self.events: List[dict] = []

        # Smoothing is satellite-only; the registry decides via modality.
        self.smoothing_enabled = self.modality.name != "gsv" and self.smooth_cfg.enabled
        # Ordered text-segmenter chain, e.g. [lisat, sam3] (satellite) or [sam3] (gsv).
        self._text_segs = [(n, _text_segmenter_call(n)[1]) for n in self.modality.text_segmenters]
        # bbox-clip repair: localize an over-large mask to a fixed box around the point.
        self.bbox_clip_area = 0.4    # trigger when best mask covers > 40% of the image
        self.bbox_clip_frac = 0.25   # fixed box side = 25% of image dimensions

    # -- QC + smoothing --------------------------------------------------
    def _run_mask_qc(self, mask_path: str) -> QCResult:
        return evaluate_mask(
            image_path=self.image_path,
            mask_path=mask_path,
            spec=self.spec,
            region_label=self.region.get("label", "object"),
            region_description=self.region.get("description", ""),
            threshold=self.mask_qc_threshold,
            cfg=self.llm_cfg,
        )

    def _accept(self, mask_path: str, qc: QCResult, method: str) -> dict:
        """Apply satellite smoothing (if warranted) then return an accepted result."""
        if self.smoothing_enabled:
            smooth_out = str(self.run_dir / "masks" / self.region_id / f"{method}_smooth.png")
            mask_path, qc = maybe_smooth(
                mask_path=mask_path, qc=qc, cfg=self.smooth_cfg,
                out_path=smooth_out, rescore=self._run_mask_qc,
            )
        return {
            "success": True, "mask_path": mask_path, "mask_qc": qc.score,
            "mask_qc_obj": qc.raw, "method": method,
            "region_id": self.region_id, "region": self.region,
            "events": self.events,
        }

    # -- main ------------------------------------------------------------
    def segment_region(self) -> dict:
        masks_dir = self.run_dir / "masks" / self.region_id
        masks_dir.mkdir(parents=True, exist_ok=True)
        qc_dir = self.run_dir / "qc" / self.region_id

        keyword = self.region.get("seg_keyword") or self.region.get("label", "object")
        best_mask: Optional[str] = None
        best_qc: Optional[QCResult] = None
        best_score = 0.0
        # Track the largest mask any tool produced, for the bbox-clip repair step.
        oversize_mask: Optional[str] = None
        oversize_area = 0.0

        def _consider_oversize(mask_path: str, area: float) -> None:
            nonlocal oversize_mask, oversize_area
            if area > self.bbox_clip_area and area > oversize_area:
                oversize_mask, oversize_area = mask_path, area

        # 1) Text-referred segmentation: walk the modality's chain (e.g. lisat -> sam3).
        for text_name, text_call in self._text_segs:
            for i, kw in enumerate(_keyword_variations(keyword)):
                mask_path = str(masks_dir / f"{text_name}_{i}_{kw.replace(' ', '_')}.png")
                self.events.append({"event": "seg_text_attempt", "segmenter": text_name,
                                    "attempt": i, "keyword": kw})
                t0 = time.time()
                try:
                    res = text_call(self.image_path, kw, mask_path)
                except Exception as e:
                    self.events.append({"event": "seg_text_error", "segmenter": text_name,
                                        "keyword": kw, "error": str(e)})
                    log.info("  [%s] kw='%s' -> ERROR (%.1fs): %s", text_name, kw, time.time() - t0, e)
                    continue
                if not res.get("object_present", True):
                    self.events.append({"event": "seg_text_no_object", "segmenter": text_name, "keyword": kw})
                    log.info("  [%s] kw='%s' -> no object (%.1fs)", text_name, kw, time.time() - t0)
                    continue

                h = _heuristic_score(mask_path)
                area = mask_area_fraction(mask_path)
                _consider_oversize(mask_path, area)
                if h < 0.2:
                    log.info("  [%s] kw='%s' -> rejected by heuristic (area=%.3f, %.1fs)",
                             text_name, kw, area, time.time() - t0)
                    continue
                qc = self._run_mask_qc(mask_path)
                _write_json(qc_dir / f"mask_qc_{text_name}_{i}.json",
                            {"segmenter": text_name, "keyword": kw, "mask_path": mask_path, "qc": qc.raw})
                self.events.append({"event": "seg_text_qc", "segmenter": text_name, "keyword": kw,
                                    "qc_score": qc.score, "qc_pass": qc.passed})
                log.info("  [%s] kw='%s' -> area=%.3f qc=%.3f pass=%s (%.1fs)",
                         text_name, kw, area, qc.score, qc.passed, time.time() - t0)
                if qc.score > best_score:
                    best_mask, best_qc, best_score = mask_path, qc, qc.score
                if qc.passed:
                    log.info("  text-seg ACCEPTED via %s (kw='%s', qc=%.3f)", text_name, kw, qc.score)
                    return self._accept(mask_path, qc, method=text_name)

        # 2) SAM point-prompt fallback around the ROI center
        for a in range(self.sam_point_attempts):
            mask_path = str(masks_dir / f"sam_point_{a}.png")
            jx = max(0, min(self.img_w - 1, self.point_x + (a - self.sam_point_attempts // 2) * 8))
            jy = max(0, min(self.img_h - 1, self.point_y + (a - self.sam_point_attempts // 2) * 5))
            self.events.append({"event": "seg_point_attempt", "attempt": a, "point": [jx, jy]})
            t0 = time.time()
            try:
                clients.sam_point_segment(self.image_path, [[jx, jy]], [1], mask_path, self.sam_cfg)
            except Exception as e:
                self.events.append({"event": "seg_point_error", "attempt": a, "error": str(e)})
                log.info("  [sam_point %d] (%d,%d) -> ERROR (%.1fs): %s", a, jx, jy, time.time() - t0, e)
                continue
            h = _heuristic_score(mask_path)
            area = mask_area_fraction(mask_path)
            _consider_oversize(mask_path, area)
            if h < 0.2:
                log.info("  [sam_point %d] (%d,%d) -> rejected by heuristic (area=%.3f, %.1fs)",
                         a, jx, jy, area, time.time() - t0)
                continue
            qc = self._run_mask_qc(mask_path)
            _write_json(qc_dir / f"mask_qc_point_{a}.json",
                        {"point": [jx, jy], "mask_path": mask_path, "qc": qc.raw})
            self.events.append({"event": "seg_point_qc", "attempt": a,
                                "qc_score": qc.score, "qc_pass": qc.passed})
            log.info("  [sam_point %d] (%d,%d) -> area=%.3f qc=%.3f pass=%s (%.1fs)",
                     a, jx, jy, area, qc.score, qc.passed, time.time() - t0)
            if qc.score > best_score:
                best_mask, best_qc, best_score = mask_path, qc, qc.score
            if qc.passed:
                log.info("  sam_point ACCEPTED (attempt %d, qc=%.3f)", a, qc.score)
                return self._accept(mask_path, qc, method="sam_point")

        # 3) bbox-clip repair: if nothing passed and a tool produced an over-large
        #    mask (e.g. SAM3 'road' grabbed every road), clip it to a fixed box
        #    around the planner's point to isolate a diffuse feature (e.g. the
        #    intersection at that point), then re-QC the clipped result.
        if oversize_mask:
            clipped = str(masks_dir / "bbox_clip.png")
            self.events.append({"event": "seg_bbox_clip_attempt",
                                "source_mask": oversize_mask, "source_area": oversize_area,
                                "box_frac": self.bbox_clip_frac,
                                "point": [self.frac_x, self.frac_y]})
            t0 = time.time()
            out = clip_mask_to_box(oversize_mask, self.frac_x, self.frac_y,
                                   self.bbox_clip_frac, clipped)
            if out:
                area = mask_area_fraction(out)
                qc = self._run_mask_qc(out)
                _write_json(qc_dir / "mask_qc_bbox_clip.json",
                            {"source_mask": oversize_mask, "mask_path": out, "qc": qc.raw})
                self.events.append({"event": "seg_bbox_clip_qc",
                                    "qc_score": qc.score, "qc_pass": qc.passed})
                log.info("  [bbox_clip] from over-large mask (src_area=%.3f) -> area=%.3f "
                         "qc=%.3f pass=%s (%.1fs)", oversize_area, area, qc.score, qc.passed,
                         time.time() - t0)
                if qc.score > best_score:
                    best_mask, best_qc, best_score = out, qc, qc.score
                if qc.passed:
                    log.info("  bbox_clip ACCEPTED (qc=%.3f)", qc.score)
                    return self._accept(out, qc, method="bbox_clip")
            else:
                log.info("  [bbox_clip] produced empty mask; skipping")

        # 4) Best-effort: keep the strongest sub-threshold mask if it clears the floor
        if best_mask and best_qc and best_score >= _BEST_EFFORT_FLOOR:
            self.events.append({"event": "seg_best_effort", "score": best_score})
            log.info("  best-effort mask kept (qc=%.3f, below pass threshold)", best_score)
            return self._accept(best_mask, best_qc, method="best_effort")

        self.events.append({"event": "seg_all_failed", "region_id": self.region_id})
        log.info("  no acceptable mask after all segmenters (best qc=%.3f)", best_score)
        return {"success": False, "mask_path": None, "mask_qc": None,
                "method": None, "region_id": self.region_id,
                "region": self.region, "events": self.events}
