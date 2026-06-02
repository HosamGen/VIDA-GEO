# vida_geo/agents/suggestion.py
"""
Prompt Suggestion sub-agent — one code path for every domain.

Writes editing prompt(s) for one region in the style the target editor wants:
'gemini' (concise transformation concept) or 'flux' (visual description of the
result). All domain prompts return the SAME candidate-list schema; GSV simply
tends to return a single candidate while satellite returns several. The result
is normalized to a candidate list + a recommended prompt, which feeds the
editor competition (more candidates -> more editor inputs to compete on delta).

Dynamic per-region context (failures, used prompts, phase context, policy hint)
is assembled into the user message here; the stable instructions live in the
domain template, parameterized by $metric and $direction.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig, chat_json
from ..prompts.loader import load_prompt
from .qc_base import SuggestionResult

log = logging.getLogger(__name__)


def _normalize(raw: Dict[str, Any], fallback_prompt: str) -> SuggestionResult:
    candidates: List[str] = []

    # 1) Preferred schema: {"candidate_edits": [{"edit_prompt": ...}, ...]}
    edits = raw.get("candidate_edits")
    if isinstance(edits, list):
        candidates = [str(e.get("edit_prompt", "")).strip()
                      for e in edits if isinstance(e, dict) and e.get("edit_prompt")]

    # 2) Bare edit object the model sometimes returns instead of the wrapper:
    #    {"id": "edit_1", "edit_prompt": "...", "rationale": "..."}
    if not candidates and raw.get("edit_prompt"):
        candidates = [str(raw["edit_prompt"]).strip()]

    # 3) Older single-prompt style: {"prompt": "..."}
    if not candidates and raw.get("prompt"):
        candidates = [str(raw["prompt"]).strip()]

    candidates = [c for c in candidates if c]  # drop any empties

    # recommended_prompt if given and non-empty, else the first candidate.
    recommended = str(raw.get("recommended_prompt", "")).strip()
    if not recommended:
        recommended = candidates[0] if candidates else str(fallback_prompt).strip()

    # Ensure the recommended prompt is in the candidate list and leads it.
    if recommended and recommended not in candidates:
        candidates = [recommended] + candidates
    if not candidates and recommended:
        candidates = [recommended]

    return SuggestionResult(
        candidates=candidates,
        recommended=recommended,
        reasoning=str(raw.get("reasoning", raw.get("notes", ""))),
        raw=raw,
    )


def _context_block(
    current_score: Optional[float],
    metric: str,
    attempt: int,
    previous_failures: Optional[List[Dict[str, Any]]],
    used_prompts: Optional[List[Dict[str, Any]]],
    policy_hint: str,
    phase_context: Optional[Dict[str, Any]],
) -> str:
    bits: List[str] = []
    if current_score is not None:
        bits.append(f"Current {metric} score: {current_score:.2f}/10")
    if attempt > 0:
        bits.append(f"This is attempt #{attempt + 1}. Try a DIFFERENT approach.")

    if used_prompts:
        bits.append("PROMPTS ALREADY USED (do not repeat or paraphrase):")
        bits += [f"  {i}. {u.get('prompt', '')[:120]}" for i, u in enumerate(used_prompts, 1)]

    if previous_failures:
        bits.append("PREVIOUS FAILED ATTEMPTS (avoid these approaches):")
        for i, f in enumerate(previous_failures[-5:], 1):
            line = f"  {i}. {f.get('prompt', '')[:120]}  (delta={f.get('delta', 0):+.2f})"
            issues = f.get("issues") or []
            if issues:
                line += " | issues: " + "; ".join(str(x) for x in issues[:3])
            bits.append(line)

    if policy_hint:
        bits.append(f"REGION POLICY CONSTRAINT (mandatory): {policy_hint.strip()}")

    if phase_context and phase_context.get("phase", 1) > 1:
        removed = phase_context.get("removed_objects") or []
        if removed:
            bits.append("Objects removed in a prior phase (do NOT re-insert): "
                        + ", ".join(str(r) for r in removed))
        p1 = phase_context.get("phase1_constraints") or []
        for pc in p1:
            c = pc.get("constraint") if isinstance(pc, dict) else str(pc)
            if c:
                bits.append(f"Phase-1 constraint still in effect: {c[:160]}")

    return "\n".join(bits)


def suggest_prompt(
    image_path: str,
    spec: DomainSpec,
    region_label: str,
    region_description: str = "",
    goal: str = "improve",
    prompt_style: str = "gemini",
    mask_path: Optional[str] = None,
    current_score: Optional[float] = None,
    attempt: int = 0,
    previous_failures: Optional[List[Dict[str, Any]]] = None,
    used_prompts: Optional[List[Dict[str, Any]]] = None,
    policy_hint: str = "",
    phase_context: Optional[Dict[str, Any]] = None,
    cfg: Optional[LLMConfig] = None,
) -> SuggestionResult:
    """Return candidate editing prompts for one region (normalized)."""
    style = (prompt_style or "gemini").lower().strip()
    if style not in ("gemini", "flux"):
        style = "gemini"

    metric = spec.score_key
    direction = "increase" if spec.wants_increase(goal) else "reduce"
    system = load_prompt(spec.prompt_group, f"suggestion_{style}",
                         metric=metric, direction=direction)

    lines = [
        f"Region being edited: {region_label}",
        f"Region description: {region_description}",
    ]
    ctx = _context_block(current_score, metric, attempt, previous_failures,
                         used_prompts, policy_hint, phase_context)
    if ctx:
        lines.append("")
        lines.append(ctx)
    lines.append("")
    lines.append(f"Write {style.upper()} editing prompt(s) for this region. JSON only.")

    # Mask is attached when available so the model can see the exact edit region.
    images = [image_path] + ([mask_path] if mask_path else [])

    raw = chat_json(
        system=system,
        user_text="\n".join(lines),
        images=images,
        images_first=False,
        cfg=cfg or LLMConfig(max_tokens=900, temperature=0.2),
        fallback={"candidate_edits": [], "recommended_prompt": "",
                  "reasoning": "parse error"},
    )
    return _normalize(raw, fallback_prompt="")