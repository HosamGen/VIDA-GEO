# vida_geo/agents/qc_mask.py
"""
Mask quality control — one code path for every domain.

The prompt text is loaded by the domain's prompt_group; the JSON schema and the
normalization are identical across domains, so there is no modality fork. The
satellite prompts additionally ask for coverage_score / fragmentation_score,
which the smoothing step reads — GSV prompts may omit them (smoothing is
satellite-only anyway).
"""
from __future__ import annotations

import logging
from typing import Optional

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig, chat_json
from ..prompts.loader import load_prompt
from .qc_base import PASS_THRESHOLD, QCResult, finalize_mask_qc

log = logging.getLogger(__name__)

_FALLBACK = {"overall_score": 0.5, "reasoning": "Parse error",
             "coverage_score": 1.0, "fragmentation_score": 1.0, "verdict": "ok"}


def evaluate_mask(
    image_path: str,
    mask_path: str,
    spec: DomainSpec,
    region_label: str,
    region_description: str = "",
    threshold: float = PASS_THRESHOLD,
    cfg: Optional[LLMConfig] = None,
) -> QCResult:
    """Judge whether `mask_path` covers the intended region in `image_path`."""
    system = load_prompt(spec.prompt_group, "mask_qc", metric=spec.score_key)

    user_text = (
        f"The mask should cover: {region_label}\n"
        f"Description: {region_description}\n"
        f"Judge how well the mask matches this target. Respond with JSON only."
    )
    raw = chat_json(
        system=system,
        user_text=user_text,
        images=[image_path, mask_path],          # original first, then mask
        cfg=cfg or LLMConfig(max_tokens=600, temperature=0),
        fallback=dict(_FALLBACK),
    )
    return finalize_mask_qc(raw, threshold=threshold)
