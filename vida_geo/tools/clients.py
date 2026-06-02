# vida_geo/tools/clients.py
"""
Transport-only HTTP clients for VIDA-GEO's model servers.

Tools (image -> mask / image -> image):
    text_segment      LISAt (satellite) or SAM3 (GSV) — text-referred segmentation
    sam_point_segment SAM point-prompted fallback
    sam_auto_segment  SAM automatic fallback
    flux_fill         FLUX inpainting editor

Black-box scorers (image -> score):
    score_image       POST an image, return the raw JSON payload
    read_score        pull one value out of a payload by key
    score_target      convenience: score_image + read_score in one call

Each function does exactly one thing: call a service and return a result.
No decisions, no direction math, no prompt logic — that lives in the agents.
Defaults match the local server ports; override via env var or services.yaml.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ═══════════════════════════════════════════════════════════════════════════
# Configs — one ServiceConfig base, one field name (`url`) everywhere
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ServiceConfig:
    url: str = "http://127.0.0.1:8000"
    timeout_s: int = 120
    # Scorer endpoint shape (ignored by non-scorer services):
    #   score_route  : base route; upload posts to it, path posts to <route>/path
    #   prefer_path   : if True and the file is locally readable by the server,
    #                   send the path (no upload) — faster on a shared filesystem.
    score_route: str = "/score"
    prefer_path: bool = False


@dataclass
class PerceptionConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("PERCEPTION_URL", "http://127.0.0.1:8111"))


@dataclass
class GreeneryConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("GREENERY_URL", "http://127.0.0.1:8006"))
    # OEM-lightweight server exposes /score/upload and /score/path; path is faster
    # on a shared filesystem (server reads the file directly, no upload).
    score_route: str = "/score"
    prefer_path: bool = True


@dataclass
class RiskConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("RISK_URL", "http://127.0.0.1:8003"))
    # BetaRisk server exposes /score/upload and /score/path; path is faster on a
    # shared filesystem (no image upload — the server reads the file directly).
    score_route: str = "/score"
    prefer_path: bool = True


@dataclass
class Sam3Config(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("SAM3_URL", "http://127.0.0.1:8005"))


@dataclass
class LisatConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("LISAT_URL", "http://127.0.0.1:8001"))


@dataclass
class SamConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("SAM_URL", "http://127.0.0.1:8004"))
    auto_topk: int = 3


@dataclass
class FluxConfig(ServiceConfig):
    url: str = field(default_factory=lambda: os.getenv("FLUX_URL", "http://127.0.0.1:8002"))
    timeout_s: int = 900            # FLUX is slow; give it room
    guidance: float = 40.0
    num_steps: int = 50


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
}


def _mime(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")


def _resolve_in(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def _prepare_out(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _base(cfg: ServiceConfig) -> str:
    return cfg.url.rstrip("/")


def _raise_for_bad(resp: requests.Response, prefix: str) -> None:
    if 200 <= resp.status_code < 300:
        return
    try:
        body = resp.json()
        detail = body.get("detail", body)
    except Exception:
        detail = resp.text[:500]
    raise RuntimeError(f"{prefix} HTTP {resp.status_code}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# Black-box scorers (perception / greenery / risk)
# ═══════════════════════════════════════════════════════════════════════════

def score_image(image_path: str, cfg: ServiceConfig, params: Optional[dict] = None) -> dict:
    """POST an image to a scorer and return its raw JSON payload.

    Honors cfg.score_route and cfg.prefer_path. When prefer_path is set, sends
    the filepath as JSON to <route>/path (fast: no upload, server reads the file).
    Falls back to multipart upload at <route>/upload (or <route>) if the server
    can't read the path (404) or path mode isn't configured.
    """
    p = _resolve_in(image_path)
    base = _base(cfg)
    route = getattr(cfg, "score_route", "/score")

    # Fast path: send the filepath (server and agent share a filesystem).
    if getattr(cfg, "prefer_path", False):
        resp = requests.post(f"{base}{route}/path",
                             json={"image_path": str(p)}, timeout=cfg.timeout_s)
        if resp.status_code != 404:           # 404 => server couldn't see the file; fall back
            _raise_for_bad(resp, "Scorer")
            return resp.json()

    # Upload path: multipart. Use /upload when a route is set, else the bare route.
    upload_url = f"{base}{route}/upload" if getattr(cfg, "prefer_path", False) else f"{base}{route}"
    with open(p, "rb") as f:
        resp = requests.post(
            upload_url,
            files={"image": (p.name, f, _mime(p))},
            params=params or {},
            timeout=cfg.timeout_s,
        )
    _raise_for_bad(resp, "Scorer")
    return resp.json()


def read_score(payload: dict, score_key: str) -> float:
    """
    Extract one score by key, tolerating both conventions:
      flat   {"greenery_score": 0.42}        -> read_score(p, "greenery_score")
      nested {"scores": {"safety": 7.1}}     -> read_score(p, "safety")
    """
    if score_key in payload:
        return float(payload[score_key])
    scores = payload.get("scores")
    if isinstance(scores, dict) and score_key in scores:
        return float(scores[score_key])
    raise KeyError(f"score_key {score_key!r} not found in payload keys {list(payload)}")


def score_target(image_path: str, cfg: ServiceConfig, score_key: str,
                 params: Optional[dict] = None) -> float:
    """Score an image and return just the target value."""
    return read_score(score_image(image_path, cfg, params=params), score_key)


# ═══════════════════════════════════════════════════════════════════════════
# Text-referred segmentation — unified for SAM3 (GSV) and LISAt (satellite)
# ═══════════════════════════════════════════════════════════════════════════

def text_segment(
    image_path: str,
    prompt: str,
    out_mask_path: str,
    cfg: ServiceConfig,
    *,
    union: Optional[bool] = None,         # SAM3 only
    score_thresh: Optional[float] = None, # SAM3 only
    mask_thresh: Optional[float] = None,  # SAM3 only
    max_new_tokens: Optional[int] = None, # LISAt only
) -> Dict[str, Any]:
    """
    Text-prompted segmentation via POST /segment/path.

    Works against either backend — both return a base64 PNG mask plus an
    `object_present` flag. Backend-specific knobs are sent only when provided,
    so the same call serves LISAt and SAM3; the caller picks `cfg` by modality.
    """
    img = _resolve_in(image_path)
    outp = _prepare_out(out_mask_path)

    payload: Dict[str, Any] = {
        "image_path": str(img),
        "prompt": prompt,
        "return_mask_base64": True,
    }
    if union is not None:
        payload["union"] = bool(union)
    if score_thresh is not None:
        payload["score_thresh"] = float(score_thresh)
    if mask_thresh is not None:
        payload["mask_thresh"] = float(mask_thresh)
    if max_new_tokens is not None:
        payload["max_new_tokens"] = int(max_new_tokens)

    resp = requests.post(f"{_base(cfg)}/segment/path", json=payload, timeout=cfg.timeout_s)
    _raise_for_bad(resp, "TextSeg")
    data = resp.json()

    b64 = data.get("mask_png_base64")
    if not b64:
        raise RuntimeError("Segmenter did not return mask_png_base64")
    outp.write_bytes(base64.b64decode(b64.encode("ascii")))

    return {
        "mask_path": str(outp),
        "object_present": bool(data.get("object_present", False)),
        "num_instances": int(data.get("num_instances", 0) or 0),
        "scores": list(data.get("scores", []) or []),
        "text": str(data.get("text", "")),       # LISAt's generated text, if any
        "prompt": str(data.get("prompt", prompt)),
        "meta": data,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SAM fallbacks (point-prompted + automatic) — POST /segment, multipart
# ═══════════════════════════════════════════════════════════════════════════

def sam_point_segment(
    image_path: str, points: List[List[int]], labels: List[int],
    out_path: str, cfg: SamConfig, response_type: str = "png",
) -> Dict[str, Any]:
    p = _resolve_in(image_path)
    outp = _prepare_out(out_path)
    form = {
        "points": json.dumps(points),
        "labels": json.dumps(labels),
        "response_type": response_type,
    }
    with open(p, "rb") as f:
        resp = requests.post(
            f"{_base(cfg)}/segment",
            files={"image": (p.name, f, _mime(p))},
            data=form, timeout=cfg.timeout_s,
        )
    _raise_for_bad(resp, "SAM")

    if response_type.lower() == "png":
        outp.write_bytes(resp.content)
        score = None
        hdr = resp.headers.get("X-SAM-Score")
        if hdr:
            try:
                score = float(hdr)
            except ValueError:
                pass
        return {"mask_path": str(outp), "score": score}

    data = resp.json()
    b64 = data.get("best_mask_png_base64") or data.get("best_mask")
    if not b64:
        raise RuntimeError("SAM returned no mask")
    outp.write_bytes(base64.b64decode(b64))
    return {"mask_path": str(outp), "meta": data}


def sam_auto_segment(image_path: str, out_dir: str, cfg: SamConfig,
                     topk: Optional[int] = None) -> Dict[str, Any]:
    p = _resolve_in(image_path)
    outd = Path(out_dir).expanduser().resolve()
    outd.mkdir(parents=True, exist_ok=True)
    form = {
        "auto": "true",
        "auto_topk": str(int(topk or cfg.auto_topk)),
        "response_type": "json",
    }
    with open(p, "rb") as f:
        resp = requests.post(
            f"{_base(cfg)}/segment",
            files={"image": (p.name, f, _mime(p))},
            data=form, timeout=cfg.timeout_s,
        )
    _raise_for_bad(resp, "SAM")
    data = resp.json()

    masks = []
    for i, m in enumerate(data.get("masks", [])):
        b64 = m.get("mask_png_base64")
        if not b64:
            continue
        mask_path = outd / f"sam_mask_{i:02d}.png"
        mask_path.write_bytes(base64.b64decode(b64.encode("ascii")))
        masks.append({"path": str(mask_path), "area": m.get("area"), "bbox": m.get("bbox")})
    return {"masks": masks, "count": len(masks)}


# ═══════════════════════════════════════════════════════════════════════════
# FLUX fill editor — POST /fill/upload, multipart
# ═══════════════════════════════════════════════════════════════════════════

def flux_fill(
    img_cond_path: str, img_mask_path: str, prompt: str, out_path: str,
    cfg: FluxConfig, guidance: Optional[float] = None,
    num_steps: Optional[int] = None, seed: Optional[int] = None,
) -> Dict[str, Any]:
    img = _resolve_in(img_cond_path)
    msk = _resolve_in(img_mask_path)
    outp = _prepare_out(out_path)
    g = float(guidance if guidance is not None else cfg.guidance)
    steps = int(num_steps if num_steps is not None else cfg.num_steps)

    form = {"prompt": prompt, "guidance": str(g), "num_steps": str(steps)}
    if seed is not None:
        form["seed"] = str(int(seed))

    with open(img, "rb") as f_img, open(msk, "rb") as f_msk:
        resp = requests.post(
            f"{_base(cfg)}/fill/upload",
            files={
                "image": (img.name, f_img, "image/png"),
                "mask": (msk.name, f_msk, "image/png"),
            },
            data=form, timeout=cfg.timeout_s,
        )
    _raise_for_bad(resp, "FLUX")
    outp.write_bytes(resp.content)

    used_seed = resp.headers.get("X-FLUX-Seed")
    return {
        "out_path": str(outp),
        "seed": int(used_seed) if used_seed else seed,
        "guidance": g, "num_steps": steps,
    }


def flux_warmup(cfg: FluxConfig, size: int = 256, steps: int = 1) -> dict:
    """Pay FLUX's cold-start before a run so it fails fast if the server is down."""
    resp = requests.post(f"{_base(cfg)}/warmup", json={"size": size, "steps": steps},
                         timeout=cfg.timeout_s)
    _raise_for_bad(resp, "FLUX warmup")
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════
# Scorer resolution — registry scorer name -> (config, callable)
# ═══════════════════════════════════════════════════════════════════════════

_SCORER_CONFIGS = {
    "perception": PerceptionConfig,
    "greenery": GreeneryConfig,
    "risk": RiskConfig,
}


def scorer_config(scorer_name: str) -> ServiceConfig:
    """Instantiate the ServiceConfig for a registry scorer name."""
    try:
        return _SCORER_CONFIGS[scorer_name]()
    except KeyError:
        raise ValueError(
            f"Unknown scorer {scorer_name!r}; valid: {list(_SCORER_CONFIGS)}"
        ) from None