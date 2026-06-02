# vida_geo/prompts/loader.py
"""
Loads prompt templates from configs/prompts/<group>/<stage>.txt.

`group` is the domain's prompt_group (e.g. "gsv", "road_safety", "greenery");
`stage` is one of: mask_qc, edit_qc, suggestion_gemini, suggestion_flux.

Templates use string.Template ($metric, $direction) rather than str.format
on purpose: the prompts embed literal JSON braces { } as output examples, and
$-substitution leaves those untouched. safe_substitute means a missing key is
left as-is instead of raising, so a template can omit placeholders it doesn't use.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "prompts"

_STAGES = ("mask_qc", "edit_qc", "suggestion_gemini", "suggestion_flux",
           "planner", "planner_epoch")


@lru_cache(maxsize=None)
def _read(group: str, stage: str) -> str:
    path = _PROMPTS_DIR / group / f"{stage}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt template not found: {path} (group={group!r}, stage={stage!r})"
        )
    return path.read_text(encoding="utf-8").strip()


def load_prompt(group: str, stage: str, **subs: object) -> str:
    """Return the template for (group, stage), with $placeholders substituted."""
    if stage not in _STAGES:
        raise ValueError(f"Unknown prompt stage {stage!r}; valid: {_STAGES}")
    template = _read(group, stage)
    if subs:
        return Template(template).safe_substitute(
            {k: str(v) for k, v in subs.items()}
        )
    return template
