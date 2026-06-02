# vida_geo/agents/qc_edit.py
"""
Edit quality control — one code path for every domain.

Judges the edited image against the original + mask (three images, whole-image,
no cropping) on realism, goal achievement, prompt adherence, and constraint
compliance. Direction ('increase' vs 'reduce') comes from the registry, never
from parsing goal text. Constraint caps are enforced in qc_base after the model
replies, so a violated constraint cannot be argued back up.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig, chat_json
from ..prompts.loader import load_prompt
from .qc_base import PASS_THRESHOLD, QCResult, finalize_edit_qc

log = logging.getLogger(__name__)

_NONE = "None specified."

_FALLBACK = {"overall_score": 0.5, "reasoning": "Parse error",
             "verdict": "ok", "constraint_compliance": "not_applicable",
             "scene_integrity": "not_applicable"}


def evaluate_edit(
    original_image_path: str,
    edited_image_path: str,
    mask_path: Optional[str],
    spec: DomainSpec,
    edit_description: str,
    goal: str = "improve",
    policy_constraint: str = "",
    scene_constraints: Optional[List[str]] = None,
    threshold: float = PASS_THRESHOLD,
    cfg: Optional[LLMConfig] = None,
) -> QCResult:
    """Score one edited image. Returns a constraint-capped QCResult.

    mask_path may be None (maskless fallback edit): the QC then judges the whole
    image with no masked-region reference.
    """
    scene_constraints = scene_constraints or []
    metric = spec.score_key
    direction = "increase" if spec.wants_increase(goal) else "reduce"
    has_mask = bool(mask_path)

    system = load_prompt(spec.prompt_group, "edit_qc",
                         metric=metric, direction=direction)

    if has_mask:
        intro = ("First image: original. Second image: edited. Third image: mask "
                 "(white = the region that was allowed to change).")
    else:
        intro = ("First image: original. Second image: edited. No mask was used — "
                 "this was a full-image edit; judge the whole image.")
    lines = [
        intro,
        f"Edit description: {edit_description}",
        f"Target metric: {metric} (goal: {direction} the {metric} score).",
        f"Region policy constraint: {policy_constraint.strip() or _NONE}",
    ]
    if scene_constraints:
        lines.append("Scene elements elsewhere that must NOT be damaged:")
        lines += [f"  - {c}" for c in scene_constraints]
    else:
        lines.append(f"Scene elements to preserve: {_NONE}")
    lines.append("Score this edit. Respond with JSON only.")

    images = [original_image_path, edited_image_path]
    if has_mask:
        images.append(mask_path)

    raw = chat_json(
        system=system,
        user_text="\n".join(lines),
        images=images,
        cfg=cfg or LLMConfig(max_tokens=700, temperature=0),
        fallback=dict(_FALLBACK),
    )
    return finalize_edit_qc(raw, threshold=threshold)
