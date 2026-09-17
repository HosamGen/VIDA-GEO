# vida_geo/config.py
"""
Run configuration.

AgentConfig holds the experiment knobs (thresholds, attempts, editor set, ROI
params). Service URLs are NOT here — they load from configs/services.yaml (or env
vars) via load_services(), so deployment facts stay out of the experiment surface.

The CLI builds an AgentConfig from a small flag set; everything else takes the
dataclass defaults below (override in code or YAML, not as CLI flags).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml

from .agents.smoothing import SmoothConfig
from .llm.openrouter import LLMConfig
from .tools.clients import (
    FluxConfig, SamConfig, Sam3Config, LisatConfig,
    PerceptionConfig, GreeneryConfig, RiskConfig, ServiceConfig,
)

_SERVICES_PATH = Path(__file__).resolve().parents[1] / "configs" / "services.yaml"


@lru_cache(maxsize=None)
def load_services(path: Optional[str] = None) -> dict:
    p = Path(path).expanduser().resolve() if path else _SERVICES_PATH
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _svc(cls, name: str, services: dict):
    """Build a client config with explicit environment URLs overriding YAML."""
    entry = (services.get("services") or {}).get(name, {}) or {}
    cfg = cls()
    # The client dataclass already reads <SERVICE>_URL. Keep that value when
    # explicitly set; services.yaml supplies the deployment default otherwise.
    if entry.get("url") and not os.getenv(f"{name.upper()}_URL"):
        cfg.url = entry["url"]
    if entry.get("timeout_s"):
        cfg.timeout_s = int(entry["timeout_s"])
    return cfg


@dataclass
class AgentConfig:
    # experiment knobs (CLI-exposed)
    goal: str = "improve"                       # improve | worsen
    output_root: Optional[str] = None           # derived from domain if None
    mask_qc_threshold: float = 0.7
    edit_qc_threshold: float = 0.7
    min_score_delta: float = 0.1
    editors: List[str] = field(default_factory=lambda: ["flux", "gemini"])

    # rarely-touched defaults (no CLI flag; override in code/yaml)
    max_mask_attempts_per_region: int = 3
    max_edit_attempts_per_region: int = 1       # 1 = single attempt (edit_retries=0)
    sam_point_attempts: int = 5
    roi_pad_pct: float = 0.15
    roi_multiple: int = 8

    # sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)

    # service client configs (filled by from_services)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    greenery: GreeneryConfig = field(default_factory=GreeneryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    sam3: Sam3Config = field(default_factory=Sam3Config)
    lisat: LisatConfig = field(default_factory=LisatConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)

    def with_services(self, services: Optional[dict] = None) -> "AgentConfig":
        s = services if services is not None else load_services()
        self.perception = _svc(PerceptionConfig, "perception", s)
        self.greenery = _svc(GreeneryConfig, "greenery", s)
        self.risk = _svc(RiskConfig, "risk", s)
        self.sam3 = _svc(Sam3Config, "sam3", s)
        self.lisat = _svc(LisatConfig, "lisat", s)
        self.sam = _svc(SamConfig, "sam", s)
        self.flux = _svc(FluxConfig, "flux", s)
        fd = (s.get("flux_defaults") or {})
        if "guidance" in fd:
            self.flux.guidance = float(fd["guidance"])
        if "num_steps" in fd:
            self.flux.num_steps = int(fd["num_steps"])
        return self

    def scorer_for(self, scorer_name: str) -> ServiceConfig:
        return {"perception": self.perception, "greenery": self.greenery,
                "risk": self.risk}[scorer_name]
