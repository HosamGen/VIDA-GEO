# vida_geo/agents/policy.py
"""
Policy agent — per-region edit feasibility + constraints.

For each planned region, classify the edit into free / constrained / remove_only
/ skip, and (for constrained) emit a one-line constraint hint that gets appended
to the generation prompt and checked by edit-QC. Also provides a fast,
no-LLM feasibility gate (is the image already at target, or all regions skipped).

Transport is the shared openrouter client; direction comes from the registry.
The 'already optimal' thresholds stay as a constant here (per your call), but are
now read by objective rather than by hardcoded metric names.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig, chat_json

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    region_id: str
    feasible: bool
    edit_class: str = "free"      # free | constrained | remove_only | skip
    strategy_prompt_hint: str = ""
    rationale: str = ""


@dataclass
class FeasibilityVerdict:
    feasible: bool
    reason: str                   # ok | already_optimal | fully_constrained
    recommendation: str = ""


_SYSTEM_PROMPT = """
You are a scene-physics adviser reviewing an image a planning AI wants to edit.
For each region you are given region_id, label, description, edit_type, and the
target metric being optimized. Decide whether each edit is PHYSICALLY PLAUSIBLE
and what constraints the editor must respect.

edit_class:
"free"        — plausible, edit as described (repaint, re-facade, add planting, etc.).
"constrained" — must stay functional but can be improved; strategy_prompt_hint MUST
                state what to preserve (e.g. "keep the road surface and lane markings
                intact; only modify edges/sidewalks").
"remove_only" — can be removed and filled with background, not replaced (trash,
                clutter, graffiti, parked vehicles, temporary objects).
"skip"        — impossible/absurd (remove the sky, edit reflections, build a tower
                on a sidewalk).

For "free", "remove_only", and "skip", strategy_prompt_hint is "".
Prefer "free" when in doubt. Every input region MUST appear in the output.

Return ONLY JSON:
{
  "regions": [
    {"region_id": "r1", "feasible": true,
     "edit_class": "free|constrained|remove_only|skip",
     "strategy_prompt_hint": "string or empty",
     "rationale": "1-2 sentences"}
  ]
}
""".strip()


def _free(region: dict) -> PolicyDecision:
    return PolicyDecision(
        region_id=str(region.get("region_id", "unknown")),
        feasible=True, edit_class="free", strategy_prompt_hint="",
        rationale="Policy check unavailable — no constraint applied.",
    )


def _parse(entry: dict) -> PolicyDecision:
    return PolicyDecision(
        region_id=str(entry.get("region_id", "unknown")),
        feasible=bool(entry.get("feasible", True)),
        edit_class=str(entry.get("edit_class", "free")),
        strategy_prompt_hint=str(entry.get("strategy_prompt_hint", "")),
        rationale=str(entry.get("rationale", "")),
    )


def evaluate_regions(
    image_path: str,
    regions: List[Dict[str, Any]],
    spec: DomainSpec,
    goal: str,
    global_description: Optional[str] = None,
    cfg: Optional[LLMConfig] = None,
) -> List[PolicyDecision]:
    """Classify all regions in one LLM call; fall back to 'free' on any failure."""
    if not regions:
        return []

    blocks = "\n".join(
        f"REGION {i+1}:\n"
        f"  region_id: {r.get('region_id', '?')}\n"
        f"  label: {r.get('label', '?')}\n"
        f"  description: {r.get('description', '')}\n"
        f"  edit_type: {r.get('edit_type', 'replace')}\n"
        f"  feature_type: {r.get('feature_type', '?')}"
        for i, r in enumerate(regions)
    )
    user_text = (
        f"Target metric: {spec.score_key} (goal: {goal})\n"
        + (f"Scene: {global_description}\n" if global_description else "")
        + f"\n{len(regions)} region(s) to assess:\n\n{blocks}\n\n"
        "Assess each region's physical plausibility. JSON only."
    )

    try:
        raw = chat_json(
            system=_SYSTEM_PROMPT, user_text=user_text, images=[image_path],
            images_first=False, cfg=cfg or LLMConfig(max_tokens=1500, temperature=0),
            fallback={"regions": []},
        )
    except Exception as e:
        logger.error("Policy call failed: %s — using free decisions.", e)
        return [_free(r) for r in regions]

    by_id = {str(e.get("region_id", "")): e for e in raw.get("regions", []) if e.get("region_id")}
    out = []
    for r in regions:
        rid = str(r.get("region_id", ""))
        out.append(_parse(by_id[rid]) if rid in by_id else _free(r))
    return out


def assess_feasibility(
    base_score: float,
    spec: DomainSpec,
    goal: str,
    policy_decisions: List[PolicyDecision],
) -> FeasibilityVerdict:
    """No-LLM gate: already at target, or all regions skipped.

    The 'already optimal' threshold is per-domain (spec.already_optimal), because
    scorers have different ranges (perception 0-10, risk/greenery 0-1). If a
    domain doesn't define one, the already-optimal check is skipped — it's an
    optional compute-saver, never a reason to silently kill a run.
    """
    thresh = spec.already_optimal  # None when undefined
    if goal == "improve" and thresh is not None:
        if spec.objective == "maximize" and base_score >= thresh:
            return FeasibilityVerdict(False, "already_optimal",
                f"{spec.score_key}={base_score:.4g} already ≥ {thresh}.")
        if spec.objective == "minimize" and base_score <= thresh:
            return FeasibilityVerdict(False, "already_optimal",
                f"{spec.score_key}={base_score:.4g} already ≤ {thresh}.")

    if policy_decisions and all(d.edit_class == "skip" for d in policy_decisions):
        return FeasibilityVerdict(False, "fully_constrained",
            f"All {len(policy_decisions)} regions deemed infeasible by policy.")

    return FeasibilityVerdict(True, "ok", "")


def build_policy_index(decisions: List[PolicyDecision]) -> Dict[str, PolicyDecision]:
    return {d.region_id: d for d in decisions}


def format_hint_for_prompt(decision: Optional[PolicyDecision]) -> str:
    if decision is None or decision.edit_class == "free" or not decision.strategy_prompt_hint:
        return ""
    return decision.strategy_prompt_hint


def log_policy_decisions(decisions: List[PolicyDecision], log: logging.Logger) -> None:
    for d in decisions:
        log.info("  policy %-10s | %-12s | %s", d.region_id, d.edit_class, d.rationale[:80])
        if d.strategy_prompt_hint:
            log.info("             hint: %s", d.strategy_prompt_hint[:100])