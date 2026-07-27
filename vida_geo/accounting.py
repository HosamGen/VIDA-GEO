"""
Process-local OpenRouter usage accounting.

The VIDA-GEO CLI handles one image per process. Reasoning and image-edit calls
both go through OpenRouter, whose non-streaming responses include a ``usage``
object with the billed ``cost``. This module keeps a small, prompt-free ledger
so the CLI can write exact per-role costs at the end of the job.
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Optional


_ROLES = ("chatgpt", "nanobanana")
_LOCK = threading.Lock()


def _new_bucket() -> Dict[str, Any]:
    return {
        "requests": 0,
        "priced_requests": 0,
        "unpriced_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "models": {},
        "records": [],
    }


_STATE: Dict[str, Dict[str, Any]] = {role: _new_bucket() for role in _ROLES}


def reset_usage() -> None:
    """Reset the process-local ledger before starting one image job."""
    with _LOCK:
        for role in _ROLES:
            _STATE[role] = _new_bucket()


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_openrouter_usage(
    role: str,
    requested_model: str,
    response_payload: Dict[str, Any],
) -> None:
    """Record one successful OpenRouter response without storing prompts/images."""
    if role not in _ROLES:
        raise ValueError(f"Unknown accounting role: {role}")

    usage = response_payload.get("usage") or {}
    model = str(response_payload.get("model") or requested_model or "unknown")
    cost = _number(usage.get("cost"))
    prompt_tokens = _integer(usage.get("prompt_tokens"))
    completion_tokens = _integer(usage.get("completion_tokens"))
    total_tokens = _integer(usage.get("total_tokens"))

    record = {
        "generation_id": response_payload.get("id"),
        "model": model,
        "cost_usd": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    with _LOCK:
        bucket = _STATE[role]
        bucket["requests"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["total_tokens"] += total_tokens
        bucket["models"][model] = bucket["models"].get(model, 0) + 1
        bucket["records"].append(record)
        if cost is None:
            bucket["unpriced_requests"] += 1
        else:
            bucket["priced_requests"] += 1
            bucket["cost_usd"] += cost


def usage_summary() -> Dict[str, Any]:
    """Return an isolated summary safe to write into JSON results."""
    with _LOCK:
        roles = copy.deepcopy(_STATE)

    for bucket in roles.values():
        bucket["cost_usd"] = round(float(bucket["cost_usd"]), 12)

    chatgpt_cost = roles["chatgpt"]["cost_usd"]
    nanobanana_cost = roles["nanobanana"]["cost_usd"]
    unpriced = sum(v["unpriced_requests"] for v in roles.values())
    return {
        "currency": "USD",
        "source": "OpenRouter response usage.cost",
        "chatgpt_cost_usd": chatgpt_cost,
        "nanobanana_cost_usd": nanobanana_cost,
        "total_cost_usd": round(chatgpt_cost + nanobanana_cost, 12),
        "cost_complete": unpriced == 0,
        "unpriced_requests": unpriced,
        "chatgpt": roles["chatgpt"],
        "nanobanana": roles["nanobanana"],
    }
