# vida_geo/llm/openrouter.py
"""
Shared OpenRouter chat client for VIDA-GEO's reasoning agents
(planner, policy, prompt-suggestion, mask-QC, edit-QC).

One place that knows how to build the request, attach images, call the API,
strip ```json fences, and parse JSON. Prompt CONTENT stays in the agents
(and soon in configs/prompts/*.yaml) — this module is transport only.

Typical use:
    from vida_geo.llm.openrouter import chat_json, LLMConfig

    result = chat_json(
        system=system_prompt,
        user_text="Score this edit. JSON only.",
        images=[original_path, edited_path],   # order preserved
        cfg=LLMConfig(max_tokens=500),
        fallback={"score": 0.5, "reasoning": "parse error"},
    )

For callers that need custom recovery (e.g. the planner's truncated-JSON
repair), call chat() for the raw string and parse it yourself.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

from ..accounting import record_openrouter_usage

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry policy for transient rate-limit / unavailability responses.
_RETRY_STATUS = (429, 503)
_MAX_RETRIES = 5
_BASE_DELAY = 2.0   # seconds; grows exponentially per attempt
_MAX_DELAY = 60.0


def post_with_retry(url: str, *, headers: dict, json: dict, timeout: float,
                    label: str = "OpenRouter") -> "requests.Response":
    """POST that retries on 429/503 with Retry-After + exponential backoff + jitter.

    Returns the final Response (which the caller still checks for non-2xx). Only
    transient statuses are retried; other errors return immediately for the
    caller to handle. Raises requests exceptions only after exhausting retries.
    """
    import random
    import time as _time

    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=json, timeout=timeout)
        except requests.RequestException as e:
            last_exc = e
            resp = None

        if resp is not None and resp.status_code not in _RETRY_STATUS:
            return resp
        if attempt == _MAX_RETRIES:
            if resp is not None:
                return resp           # let caller surface the final 429/503 body
            raise last_exc

        # Honor Retry-After when present, else exponential backoff with jitter.
        delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1.0)
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    delay = max(delay, float(ra))
                except ValueError:
                    pass
            log.warning("%s %s — retrying in %.1fs (attempt %d/%d)",
                        label, resp.status_code, delay, attempt + 1, _MAX_RETRIES)
        else:
            log.warning("%s request error %s — retrying in %.1fs (attempt %d/%d)",
                        label, last_exc, delay, attempt + 1, _MAX_RETRIES)
        _time.sleep(delay)
    return resp  # unreachable, but keeps type checkers happy

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
}


@dataclass
class LLMConfig:
    """Reasoning-model settings. `model` defaults to the OPENROUTER_MODEL env var.

    Per-step overrides (e.g. a stronger model for edit-QC) are passed at call
    time; later these can be driven per modality/metric from the registry.
    """
    model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL",
                                                          "openai/gpt-5.1"))
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: int = 60
    max_tokens: int = 800
    temperature: Optional[float] = None   # None -> omit (provider default)


def _mime(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")


def _image_part(image_path: str) -> Dict[str, Any]:
    p = Path(image_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{_mime(p)};base64,{b64}"}}


def strip_fences(raw: str) -> str:
    """Remove a leading ```/```json fence and trailing ``` if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        nl = raw.find("\n")
        raw = raw[nl + 1:] if nl != -1 else raw[3:]
        raw = raw.rstrip()
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return raw


def extract_json_object(text: str):
    """Best-effort recovery of a JSON object from a messy reply.

    Reasoning models sometimes wrap the JSON in prose or a reasoning preamble.
    Try, in order: direct parse, fenced parse, then the first balanced {...} block
    (respecting strings/escapes). Returns the parsed dict/list or None.
    """
    for candidate in (text, strip_fences(text)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    s = strip_fences(text)
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = s.find("{", start + 1)
    return None


def chat(
    system: str,
    user_text: str,
    images: Optional[Sequence[str]] = None,
    *,
    images_first: bool = True,
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout_s: Optional[int] = None,
) -> str:
    """Single-turn system+user call. Returns the raw assistant text.

    images: file paths attached to the user message, order preserved.
    images_first: image parts before the text part (True) or after (False).
    Raises RuntimeError on a missing key or a non-2xx response.
    """
    cfg = cfg or LLMConfig()
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {cfg.api_key_env}")

    text_part = {"type": "text", "text": user_text}
    image_parts = [_image_part(p) for p in (images or [])]
    user_content = (image_parts + [text_part]) if images_first else ([text_part] + image_parts)

    payload: Dict[str, Any] = {
        "model": model or cfg.model,
        "max_tokens": max_tokens if max_tokens is not None else cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    temp = temperature if temperature is not None else cfg.temperature
    if temp is not None:
        payload["temperature"] = temp

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = post_with_retry(
        OPENROUTER_URL, headers=headers, json=payload,
        timeout=timeout_s if timeout_s is not None else cfg.timeout_s,
        label="OpenRouter",
    )
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:500])
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {detail}")

    data = resp.json()
    record_openrouter_usage("chatgpt", model or cfg.model, data)
    return data["choices"][0]["message"]["content"]


def chat_json(
    system: str,
    user_text: str,
    images: Optional[Sequence[str]] = None,
    *,
    images_first: bool = True,
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout_s: Optional[int] = None,
    fallback: Optional[dict] = None,
) -> dict:
    """chat() + fence-strip + json.loads. Returns `fallback` on parse failure.

    If `fallback` is None and parsing fails, raises json.JSONDecodeError so the
    caller can decide. Transport/HTTP errors always propagate from chat().
    """
    raw = chat(
        system, user_text, images,
        images_first=images_first, cfg=cfg, model=model,
        max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s,
    )
    parsed = extract_json_object(raw)
    if parsed is not None:
        return parsed
    # Hard failure: surface the actual reply (not truncated to 200) so a bad
    # template or a model that won't emit JSON is diagnosable, then fall back.
    log.warning("LLM returned unparseable JSON. Raw reply follows:\n%s", raw)
    if fallback is not None:
        return dict(fallback)
    raise json.JSONDecodeError("no JSON object found in reply", raw or "", 0)
