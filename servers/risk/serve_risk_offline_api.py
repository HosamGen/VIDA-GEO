"""Expose BetaRisk without downloading redundant ImageNet initialization weights."""
from __future__ import annotations

import sys
from pathlib import Path

import torchvision.models as torchvision_models


BETARISK_REPO = Path(
    "/l/users/hosam.elgendy/BetaRisk"
).resolve()
sys.path.insert(0, str(BETARISK_REPO))

_original_resnet50 = torchvision_models.resnet50


def _offline_resnet50(*args, **kwargs):
    # The full BetaRisk checkpoint replaces these weights immediately.
    kwargs["weights"] = None
    return _original_resnet50(*args, **kwargs)


torchvision_models.resnet50 = _offline_resnet50

from serve_risk_api import app  # noqa: E402,F401
