# vida_geo/tools/gemini.py
"""
Gemini image-editing client (OpenRouter).

Replaces the nano_banana_cli.py subprocess with a direct API call, preserving
the exact request format that worked: text instruction first, then base image,
then optional mask; modalities=["image","text"]; image read back from
message.images[0].image_url.url.

Editor role only — produces an edited image file. Mask-constrained when a mask
is given, full-image otherwise.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from ..accounting import record_openrouter_usage
from ..llm.openrouter import post_with_retry

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
}


@dataclass
class GeminiConfig:
    model: str = field(default_factory=lambda: os.getenv("GEMINI_EDIT_MODEL",
                                                          "google/gemini-2.5-flash-image"))
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: int = 300
    max_tokens: int = 1024
    referer: str = "https://vida-geo"
    title: str = "VIDA-GEO Image Editor"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _mime(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")


def _instruction(prompt: str, has_mask: bool) -> str:
    """The two prompt templates carried over verbatim from nano_banana_cli."""
    if has_mask:
        return (
            "You will receive two images:\n"
            "1) The original image to edit.\n"
            "2) A mask. White pixels indicate the ONLY region you are allowed to change. "
            "Black pixels must remain EXACTLY the same.\n\n"
            f"Task: {prompt}\n\n"
            "Rules:\n"
            "- ONLY modify pixels inside the white mask region.\n"
            "- Do not change anything outside the white region.\n"
            "- Blend naturally with the scene.\n"
            "- Do not add text, labels, watermarks, or borders."
        )
    return (
        "Edit this image according to the following instruction:\n\n"
        f"{prompt}\n\n"
        "Rules:\n"
        "- Keep everything else the same.\n"
        "- Blend naturally with the scene.\n"
        "- Do not add text, labels, watermarks, or borders."
    )


def gemini_edit(
    image_path: str,
    prompt: str,
    out_image_path: str,
    cfg: GeminiConfig = GeminiConfig(),
    mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Edit an image with Gemini via OpenRouter and write the result to disk.

    Returns {out_path, method, model, prompt_used}.
    """
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {cfg.api_key_env}")

    base = Path(image_path).expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(str(base))

    has_mask = bool(mask_path) and Path(mask_path).expanduser().resolve().is_file()

    content: list = [
        {"type": "text", "text": _instruction(prompt, has_mask)},
        {"type": "image_url",
         "image_url": {"url": f"data:{_mime(base)};base64,{_b64(base)}"}},
    ]
    if has_mask:
        m = Path(mask_path).expanduser().resolve()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{_mime(m)};base64,{_b64(m)}"}})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.referer,
        "X-Title": cfg.title,
    }
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "max_tokens": cfg.max_tokens,
    }

    resp = post_with_retry(OPENROUTER_URL, headers=headers, json=payload,
                           timeout=cfg.timeout_s, label="Gemini")
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:500])
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {detail}")

    data = resp.json()
    record_openrouter_usage("nanobanana", cfg.model, data)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices.")
    message = choices[0].get("message", {})
    images = message.get("images", [])
    if not images:
        raise RuntimeError(
            "No images in response — model may not support image output. "
            f"Content: {str(message.get('content', ''))[:200]}"
        )

    data_url = images[0].get("image_url", {}).get("url", "")
    if not data_url:
        raise RuntimeError("Image entry has no URL.")
    b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url

    outp = Path(out_image_path).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(base64.b64decode(b64))

    return {"out_path": str(outp), "method": "gemini",
            "model": cfg.model, "prompt_used": prompt}
