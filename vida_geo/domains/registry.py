# vida_geo/domains/registry.py
"""
Domain registry: the single place that knows what a --domain means.

Resolves a domain name into its modality, black-box scorer, score key, and
optimization direction, and owns all "is this edit an improvement?" math so no
other module ever special-cases a metric again.

    reg = load_registry()
    spec = reg.domain("depressing")
    spec.modality          # "gsv"
    spec.scorer            # "perception"   (-> services.yaml)
    spec.score_key         # "depressing"
    spec.objective         # "minimize"
    spec.output_root       # "outputs/depressing_runs"
    spec.is_better(4.0, 5.0)            # True  (lower depressing = better)
    spec.progress(4.0, 5.0)            # +1.0  (signed so positive = improvement)

    tools = reg.tools_for("depressing")
    tools.text_segmenter   # "sam3"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# configs/domains.yaml lives at the repo root, two levels up from this file.
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "domains.yaml"

_VALID_OBJECTIVES = ("maximize", "minimize")
_VALID_GOALS = ("improve", "worsen")


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    text_segmenters: tuple    # ordered chain, e.g. ("lisat","sam3") satellite, ("sam3",) gsv
    fallback_segmenter: str   # point/auto fallback, e.g. "sam"
    editors: tuple            # available editors, e.g. ("flux", "gemini")

    @property
    def text_segmenter(self) -> str:
        """First text segmenter (back-compat for single-value callers)."""
        return self.text_segmenters[0] if self.text_segmenters else ""


@dataclass(frozen=True)
class DomainSpec:
    name: str
    modality: str
    scorer: str               # service name; resolve via services.yaml
    score_key: str            # field to read from the scorer response
    objective: str            # "maximize" | "minimize"
    hints: Dict[str, str] = field(default_factory=dict)  # planner guidance text
    already_optimal: Optional[float] = None   # skip-if-already-at-target threshold (per scorer scale)

    @property
    def output_root(self) -> str:
        return f"outputs/{self.name}_runs"

    @property
    def prompt_group(self) -> str:
        """Folder under configs/prompts/ holding this domain's templates.

        The GSV perception domains share one templated group (they differ only
        by metric name); satellite domains each get their own folder because
        their wording differs substantively (road-network vs vegetation).
        """
        return "gsv" if self.modality == "gsv" else self.name

    def _gain_sign(self, goal: str) -> float:
        """+1 if we want the score to go UP, -1 if we want it to go DOWN."""
        if goal not in _VALID_GOALS:
            raise ValueError(f"goal must be one of {_VALID_GOALS}, got {goal!r}")
        want_up = (self.objective == "maximize") == (goal == "improve")
        return 1.0 if want_up else -1.0

    def progress(self, new_score: float, base_score: float, goal: str = "improve") -> float:
        """Signed progress toward the goal. Positive always means 'better'."""
        return self._gain_sign(goal) * (new_score - base_score)

    def is_better(self, new_score: float, base_score: float, goal: str = "improve") -> bool:
        return self.progress(new_score, base_score, goal) > 0.0

    def wants_increase(self, goal: str = "improve") -> bool:
        """True if reaching the goal means the score should go UP.

        Lets prompt/QC agents phrase 'increase' vs 'reduce' from one source
        instead of each special-casing boring/depressing/risk.
        """
        return self._gain_sign(goal) > 0.0

    @staticmethod
    def raw_delta(new_score: float, base_score: float) -> float:
        """Unsigned change, for display (matches the figure's S_cand - S_base)."""
        return new_score - base_score


class DomainRegistry:
    def __init__(self, domains: Dict[str, DomainSpec], modalities: Dict[str, ModalitySpec]):
        self._domains = domains
        self._modalities = modalities

    @property
    def names(self) -> List[str]:
        """All valid --domain values (use as argparse choices)."""
        return list(self._domains.keys())

    @property
    def modalities(self) -> List[str]:
        return list(self._modalities.keys())

    def domain(self, name: str) -> DomainSpec:
        try:
            return self._domains[name]
        except KeyError:
            raise KeyError(f"Unknown domain {name!r}. Valid: {self.names}") from None

    def modality(self, name: str) -> ModalitySpec:
        try:
            return self._modalities[name]
        except KeyError:
            raise KeyError(f"Unknown modality {name!r}. Valid: {self.modalities}") from None

    def tools_for(self, domain_name: str) -> ModalitySpec:
        """Segmenters/editors for a domain, via its modality."""
        return self.modality(self.domain(domain_name).modality)


def _parse(raw: dict) -> DomainRegistry:
    modalities: Dict[str, ModalitySpec] = {}
    for name, m in (raw.get("modalities") or {}).items():
        seg = m.get("text_segmenters")
        if seg is None:
            scalar = m.get("text_segmenter")        # legacy single-value support
            seg = [scalar] if scalar else []
        modalities[name] = ModalitySpec(
            name=name,
            text_segmenters=tuple(seg),
            fallback_segmenter=m["fallback_segmenter"],
            editors=tuple(m.get("editors", ())),
        )

    domains: Dict[str, DomainSpec] = {}
    for name, d in (raw.get("domains") or {}).items():
        modality = d["modality"]
        if modality not in modalities:
            raise ValueError(f"Domain {name!r} references unknown modality {modality!r}")
        objective = d["objective"]
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"Domain {name!r} objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
            )
        domains[name] = DomainSpec(
            name=name,
            modality=modality,
            scorer=d["scorer"],
            score_key=d["score_key"],
            objective=objective,

            hints=dict(d.get("hints", {}) or {}),


            already_optimal=d.get("already_optimal"),
        )

    if not domains:
        raise ValueError("Domain registry is empty.")
    return DomainRegistry(domains, modalities)


@lru_cache(maxsize=None)
def load_registry(path: Optional[str] = None) -> DomainRegistry:
    cfg_path = Path(path).expanduser().resolve() if path else _DEFAULT_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Domain registry not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _parse(raw)
