# vida_geo/agents/qc_base.py
"""
Shared QC scaffolding: normalized result types plus the constraint-cap math.

Every QC backend (mask or edit, any domain) returns the SAME normalized
QCResult, so the pipeline never branches on modality to read a score. The raw
model JSON is kept under .raw for logging/debugging.

The edit-QC JSON schema is standardized across all domains (one vocabulary for
constraint_compliance / scene_integrity), so the cap math here is the single
source of truth — replacing the per-file _apply_caps copies in the old
gsv/risk/green QC modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PASS_THRESHOLD = 0.7  # default; callers may pass their own threshold

# Canonical constraint vocabulary (all domain prompts emit these labels).
_REGION_CAP = {"violated": 0.3, "partially_violated": 0.5}
_SCENE_CAP = {"damaged": 0.4, "minor_damage": 0.6}


@dataclass
class QCResult:
    passed: bool
    score: float                 # normalized 0..1, after any constraint cap
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return str(self.raw.get("verdict", "ok"))


@dataclass
class SuggestionResult:
    candidates: List[str]        # candidate edit prompts (>=1)
    recommended: str             # the chosen prompt
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def constraint_cap(region_label: Optional[str], scene_label: Optional[str]) -> float:
    """Lowest score cap implied by the constraint/scene compliance labels.

    Returns 1.0 (no cap) when both are absent / not_applicable / compliant.
    """
    cap = 1.0
    if region_label in _REGION_CAP:
        cap = min(cap, _REGION_CAP[region_label])
    if scene_label in _SCENE_CAP:
        cap = min(cap, _SCENE_CAP[scene_label])
    return cap


def normalized_score(raw: Dict[str, Any]) -> float:
    """Pull the 0..1 score out of a QC payload (overall_score, else score)."""
    for key in ("overall_score", "score"):
        if key in raw:
            try:
                return float(raw[key])
            except (TypeError, ValueError):
                pass
    return 0.5  # neutral default if the model omitted a score


def finalize_edit_qc(raw: Dict[str, Any], threshold: float = PASS_THRESHOLD) -> QCResult:
    """Apply constraint caps and build a QCResult from an edit-QC payload."""
    score = normalized_score(raw)
    region = str(raw.get("constraint_compliance", "not_applicable")).lower()
    scene = str(raw.get("scene_integrity", "not_applicable")).lower()

    cap = constraint_cap(region, scene)
    if cap < 1.0 and score > cap:
        raw["score_cap_applied"] = cap
        score = cap

    raw["overall_score"] = round(score, 4)
    return QCResult(
        passed=score >= threshold,
        score=score,
        reason=str(raw.get("reasoning", raw.get("observed_change_summary", ""))),
        raw=raw,
    )


def finalize_mask_qc(raw: Dict[str, Any], threshold: float = PASS_THRESHOLD) -> QCResult:
    """Build a QCResult from a mask-QC payload (no constraint caps here)."""
    score = normalized_score(raw)
    raw["overall_score"] = round(score, 4)
    return QCResult(
        passed=score >= threshold,
        score=score,
        reason=str(raw.get("reasoning", raw.get("summary", ""))),
        raw=raw,
    )
