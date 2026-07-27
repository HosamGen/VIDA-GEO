"""Expose BetaRisk without downloading redundant ImageNet initialization weights."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torchvision.models as torchvision_models


VIDA_GEO_ROOT = Path(__file__).resolve().parents[2]
BETARISK_REPO = Path(
    os.environ.get("BETARISK_REPO", VIDA_GEO_ROOT.parent / "BetaRisk")
).expanduser().resolve()
if not (BETARISK_REPO / "serve_risk_api.py").is_file():
    raise FileNotFoundError(
        "BetaRisk API checkout not found. Set BETARISK_REPO to a directory "
        "containing serve_risk_api.py."
    )
sys.path.insert(0, str(BETARISK_REPO))

_original_resnet50 = torchvision_models.resnet50


def _offline_resnet50(*args, **kwargs):
    # The full BetaRisk checkpoint replaces these weights immediately.
    kwargs["weights"] = None
    return _original_resnet50(*args, **kwargs)


torchvision_models.resnet50 = _offline_resnet50

from serve_risk_api import app  # noqa: E402,F401
