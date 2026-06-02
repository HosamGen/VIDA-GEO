# vida_geo/runio.py
"""
Shared run I/O: logger, JSON writers, event stream, and per-stage timing.

All three orchestrators (full_pipeline, epoch_runner, phase_runner) use these so
logging and result formatting are identical everywhere. result.json is written
through the NaN/inf-safe serializer; run.log is concise (stage banners + timings,
no raw API payloads anywhere).
"""
from __future__ import annotations

import json
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Attach handlers to the TOP-LEVEL package logger so that every module
    # logger (vida_geo.agents.segmentation, .generation, ...) propagates here
    # and lands in run.log. Previously handlers were on a run-specific child
    # logger, so agent log.info() calls had no handler and silently vanished.
    root = logging.getLogger("vida_geo")
    root.setLevel(logging.INFO)
    root.propagate = False

    # Drop handlers from any previous run so logs don't fan out to old files.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

    # The run logger is a child; it inherits the handlers via propagation.
    log = logging.getLogger(f"vida_geo.{run_dir.name}")
    log.setLevel(logging.INFO)
    log.propagate = True
    return log


def serialisable(obj: Any, _seen: Optional[set] = None) -> Any:
    """Recursively make an object JSON-safe (NaN/inf -> None, Path -> str, cycles -> None)."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, (dict, list, tuple, set)):
        oid = id(obj)
        if oid in _seen:
            return None
        _seen.add(oid)
    if isinstance(obj, dict):
        return {k: serialisable(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serialisable(v, _seen) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialisable(obj), indent=2, ensure_ascii=False),
                    encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(serialisable(obj), ensure_ascii=False) + "\n")


class Timings:
    """Accumulates per-stage durations for the result.json `timings` block."""

    def __init__(self) -> None:
        self._t: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str, log: Optional[logging.Logger] = None):
        t0 = time.time()
        try:
            yield
        finally:
            dt = round(time.time() - t0, 3)
            self._t[name] = round(self._t.get(name, 0.0) + dt, 3)
            if log is not None:
                log.info("TIMING %s: %.2fs", name, dt)

    def add(self, name: str, seconds: float) -> None:
        self._t[name] = round(self._t.get(name, 0.0) + float(seconds), 3)

    def as_dict(self) -> Dict[str, float]:
        return dict(self._t)


def banner(log: logging.Logger, title: str) -> None:
    """One concise stage banner line (replaces the old 3-line === blocks)."""
    log.info("──── %s ────", title)
