# vida_geo/agents/planning.py
"""
Planning agent — identifies regions to edit to move a domain's score.

One code path for both modalities. The scene framing ("street-view" vs
"satellite tile") lives in the domain-keyed planner template; the per-domain
guidance (what the metric is, what improves/worsens it) comes from the registry
hints. Direction is from spec.wants_increase, never parsed from goal text.

Outputs a plan dict {description, regions:[...]}, each region carrying:
region_id, label, description, feature_type, edit_type, frac_x, frac_y,
perception_potential, seg_keyword. seg_keyword is refined by a noun heuristic so
the segmenter gets a concrete target.

describe_and_plan      -> epoch 1 / single epoch
replan_epoch           -> epoch N>1, given a summary of what was already tried
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig, chat, strip_fences
from ..prompts.loader import load_prompt

log = logging.getLogger(__name__)


def _direction_text(spec: DomainSpec, goal: str) -> str:
    return "increase" if spec.wants_increase(goal) else "reduce"


def _repair_truncated_json(raw: str) -> Optional[dict]:
    """Salvage region objects from a truncated planner reply."""
    matches = list(re.finditer(r'\{[^{}]*"region_id"[^{}]*\}', raw, re.DOTALL))
    if not matches:
        return None
    desc_m = re.search(r'"description"\s*:\s*"([^"]*)"', raw)
    desc = desc_m.group(1) if desc_m else "Scene description unavailable"
    regions = []
    for m in matches:
        try:
            regions.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    if not regions:
        return None
    log.info("planner JSON repair recovered %d/%d region(s)", len(regions), len(matches))
    return {"description": desc, "regions": regions}


def _parse_plan(raw: str, *, allow_empty_on_fail: bool) -> dict:
    cleaned = strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning("planner JSON parse failed: %s — attempting repair", e)
        plan = _repair_truncated_json(cleaned)
        if plan is not None:
            return plan
        if allow_empty_on_fail:
            log.warning("planner repair failed — returning empty plan.")
            return {"description": "Re-planning failed", "regions": []}
        log.error("planner repair failed. Raw: %s", cleaned[:500])
        raise


def _finalize_regions(plan: dict, *, max_regions: int) -> dict:
    regions = list(plan.get("regions", []))
    if len(regions) > max_regions:
        log.info(
            "planner proposed %d regions; keeping the first %d",
            len(regions),
            max_regions,
        )
        regions = regions[:max_regions]
    plan["regions"] = regions
    for r in regions:
        r.setdefault("edit_type", "replace")
        r.setdefault("perception_potential", "medium")
        r["seg_keyword"] = refine_seg_keyword(r)
    return plan


def _hints(spec: DomainSpec) -> Dict[str, str]:
    h = spec.hints or {}
    return {
        "description": h.get("description", spec.score_key),
        "improve_by": h.get("improve_by", ""),
        "worsen_by": h.get("worsen_by", ""),
    }


def describe_and_plan(
    image_path: str,
    spec: DomainSpec,
    goal: str = "improve",
    current_scores: Optional[dict] = None,
    cfg: Optional[LLMConfig] = None,
) -> dict:
    """Epoch-1 / single-epoch planning."""
    direction = _direction_text(spec, goal)
    h = _hints(spec)
    edit_examples = h["improve_by"] if goal == "improve" else h["worsen_by"]
    score_ctx = (f"\nCurrent score(s) (0-10): {json.dumps(current_scores)}\n"
                 if current_scores else "")

    system = load_prompt(
        spec.prompt_group, "planner",
        metric=spec.score_key, direction=direction,
        metric_description=h["description"], edit_examples=edit_examples,
        score_ctx=score_ctx,
    )
    user_text = (f"Identify exactly 4 regions to {direction} the perceived "
                 f"{spec.score_key}. Respond with JSON only.")

    raw = chat(system, user_text, images=[image_path],
               cfg=cfg or LLMConfig(max_tokens=2000, temperature=0.2))
    plan = _parse_plan(raw, allow_empty_on_fail=False)
    return _finalize_regions(plan, max_regions=4)


def replan_epoch(
    image_path: str,
    spec: DomainSpec,
    goal: str,
    epoch_summary: dict,
    epoch_number: int,
    current_scores: Optional[dict] = None,
    phase_context: Optional[dict] = None,
    cfg: Optional[LLMConfig] = None,
) -> dict:
    """Epoch N>1: propose NEW regions not tried before."""
    direction = _direction_text(spec, goal)
    h = _hints(spec)
    score_ctx = (f"\nCurrent score(s) (0-10): {json.dumps(current_scores)}\n"
                 if current_scores else "")

    prev_lines = []
    for r in epoch_summary.get("regions", []):
        status = "improved" if r.get("n_edits_passed_qc", 0) > 0 else "no improvement"
        seg = "segmented" if r.get("segmentation_success") else "seg failed"
        prev_lines.append(f"  - {r.get('region_id','?')}: {str(r.get('description',''))[:60]} "
                          f"[{seg}, {status}, delta={r.get('best_score_delta','N/A')}]")
    prev_ctx = "\n".join(prev_lines) if prev_lines else "  (none)"

    tried_coords = ""
    all_tried = epoch_summary.get("all_regions_tried_all_epochs", [])
    coord_lines = [f"  ({t.get('frac_x'):.2f}, {t.get('frac_y'):.2f}) — {t.get('region_id','?')}"
                   for t in all_tried if t.get("frac_x") is not None]
    if coord_lines:
        tried_coords = "\nCoordinates already tried:\n" + "\n".join(coord_lines)

    phase_ctx = ""
    if phase_context and phase_context.get("phase", 1) > 1:
        removed = phase_context.get("removed_objects", [])
        if removed:
            phase_ctx = ("\n\nObjects removed in a prior phase: "
                         + ", ".join(str(x) for x in removed)
                         + ". Do NOT re-insert them; find NEW regions.")

    system = load_prompt(
        spec.prompt_group, "planner_epoch",
        metric=spec.score_key, direction=direction,
        metric_description=h["description"], score_ctx=score_ctx,
        epoch_number=epoch_number, previous_attempts=prev_ctx,
        tried_coords=tried_coords, phase_ctx=phase_ctx,
    )
    user_text = (f"Epoch {epoch_number}: find NEW regions to {direction} "
                 f"{spec.score_key}. Respond with JSON only.")

    raw = chat(system, user_text, images=[image_path],
               cfg=cfg or LLMConfig(max_tokens=2000, temperature=0.2))
    plan = _parse_plan(raw, allow_empty_on_fail=True)
    return _finalize_regions(plan, max_regions=3)


# ── segmentation keyword refinement ─────────────────────────────────────────

_GOOD_NOUNS = {
    "car", "cars", "truck", "bus", "vehicle", "bicycle", "motorcycle",
    "building", "house", "wall", "fence", "gate", "door", "window",
    "tree", "bush", "grass", "plant", "flower",
    "road", "sidewalk", "pavement", "curb",
    "pole", "lamp", "light", "sign", "post",
    "bench", "chair", "table", "trash", "bin", "dumpster",
    "person", "pedestrian", "roof", "chimney", "balcony", "stairs", "railing",
    "wire", "cable", "pipe", "sky", "cloud",
}
_STRIP_QUALIFIERS = {
    "exterior", "surface", "area", "section", "region", "part", "existing",
    "current", "old", "new", "main", "large", "small", "entire", "whole",
    "front", "back", "left", "right", "upper", "lower", "parked", "broken",
    "damaged", "abandoned", "empty", "dirty", "clean", "dark", "bright", "ugly",
    "nice", "bare", "blank", "open", "closed", "nearby", "adjacent", "visible",
    "prominent", "concrete", "brick", "metal", "wooden", "stone", "glass",
    "steel", "overgrown", "cracked", "rusted", "faded", "peeling", "dilapidated",
    "utility", "electric", "electrical",
}
_VAGUE_NOUNS = {"building", "wall", "structure", "vegetation", "infrastructure", "object", "element"}
_MORE_SPECIFIC = {
    "house", "shop", "store", "apartment", "garage", "shed", "cottage", "villa",
    "tower", "church", "mosque", "school", "lamp", "lamppost", "streetlight",
    "bollard", "graffiti", "mural", "awning", "canopy", "gutter", "bush",
    "bushes", "shrub", "shrubs", "hedge", "hedges", "vine", "vines", "pole",
    "post", "pillar", "column", "dumpster", "barrel", "crate", "container",
}


def refine_seg_keyword(region: dict) -> str:
    """Pick a concrete 1-2 word noun for the segmenter from label/description."""
    label = str(region.get("label", "")).lower().strip()
    desc = str(region.get("description", "")).lower().strip()
    llm_kw = str(region.get("seg_keyword", region.get("sam3_keyword", ""))).lower().strip()

    words = [w for w in label.split() if w not in _STRIP_QUALIFIERS]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    best = None
    for bg in bigrams:
        if any(w in _GOOD_NOUNS for w in bg.split()):
            best = bg
            break
    if not best:
        for w in words:
            if w in _GOOD_NOUNS:
                best = w
                break
    if not best or best in _VAGUE_NOUNS:
        for w in desc.split():
            clean = w.strip(".,;:!?()\"'").lower()
            if clean in _MORE_SPECIFIC:
                if clean.endswith("shes") or clean.endswith("ges"):
                    best = clean.rstrip("es")
                elif clean.endswith("s") and len(clean) > 3:
                    best = clean.rstrip("s")
                else:
                    best = clean
                break
    if best:
        return " ".join(best.split()[:2])
    return " ".join(llm_kw.split()[:2]) if llm_kw else "object"
