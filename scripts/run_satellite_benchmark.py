#!/usr/bin/env python3
"""Serial, resumable VIDA-GEO runner for the final satellite benchmarks.

The public benchmark folders are named ``greenery`` and ``road_risk``. VIDA-GEO
calls the latter domain ``road_safety`` because its improvement objective is to
lower BetaRisk's ``risk_mean``. Files marked ``forced-reference`` are retained
in the 100-image benchmark but deliberately excluded from new experiments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path("/l/users/hosam.elgendy/benchmark_images")
METRICS = ("greenery", "road_risk")
DOMAIN_BY_METRIC = {"greenery": "greenery", "road_risk": "road_safety"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
CSV_FIELDS = [
    "run_signature", "dataset_number", "metric", "domain", "source",
    "input_image", "output_image", "manifest_score",
    "input_score", "output_score", "raw_delta", "goal_delta",
    "pipeline_success", "status", "error", "elapsed_seconds",
    "chatgpt_model", "chatgpt_requests", "chatgpt_cost_usd",
    "nanobanana_model", "nanobanana_requests", "nanobanana_cost_usd",
    "total_cost_usd", "cost_complete", "run_dir", "job_summary_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one final satellite metric through VIDA-GEO.")
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument(
        "--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs" / "satellite_benchmark")
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Default: <output-root>/benchmark_results_<metric>.csv")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Smoke-test limit after exclusions; 0 means every eligible image.")
    parser.add_argument(
        "--include-forced-reference", action="store_true",
        help="Include the two old/reference images; off by default.")

    parser.add_argument("--goal", choices=("improve", "worsen"), default="improve")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument(
        "--new-regions-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--mask-qc-threshold", type=float, default=0.7)
    parser.add_argument("--edit-qc-threshold", type=float, default=0.7)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--max-edit-attempts", type=int, default=1)
    parser.add_argument("--reasoning-model", default="openai/gpt-5.1")
    parser.add_argument(
        "--editor-model", default="google/gemini-2.5-flash-image")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    return parser


def parse_number(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def parse_manifest_score(metric: str, path: Path) -> Optional[float]:
    pattern = r"_green-(-?\d+(?:\.\d+)?)_" if metric == "greenery" \
        else r"_risk-(-?\d+(?:\.\d+)?)_"
    match = re.search(pattern, f"_{path.stem}_")
    return float(match.group(1)) if match else None


def load_inputs(args) -> List[Dict[str, Any]]:
    metric_dir = args.benchmark_root / args.metric
    if not metric_dir.is_dir():
        raise FileNotFoundError(f"Metric directory not found: {metric_dir}")

    paths = sorted(
        path.resolve()
        for path in metric_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not args.include_forced_reference:
        paths = [path for path in paths if "forced-reference" not in path.name]
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be 0 or a positive integer")
        paths = paths[:args.limit]
    if not paths:
        raise ValueError(f"No eligible images found in {metric_dir}")

    rows: List[Dict[str, Any]] = []
    seen_coordinates = set()
    for path in paths:
        match = COORD_RE.search(path.stem)
        if not match:
            raise ValueError(f"Cannot parse terminal coordinates: {path.name}")
        coordinates = (
            round(float(match.group(1)), 6),
            round(float(match.group(2)), 6),
        )
        if coordinates in seen_coordinates:
            raise ValueError(f"Duplicate coordinates in {metric_dir}: {coordinates}")
        seen_coordinates.add(coordinates)
        rows.append({
            "number": parse_number(path),
            "metric": args.metric,
            "domain": DOMAIN_BY_METRIC[args.metric],
            "path": path,
            "score": parse_manifest_score(args.metric, path),
            "source": "benchmark_images",
        })
    return rows


def expected_config(args, domain: str) -> Dict[str, Any]:
    return {
        "domain": domain,
        "goal": args.goal,
        "max_epochs": args.max_epochs,
        "new_regions_only": bool(args.new_regions_only),
        "two_phase": bool(args.two_phase),
        "mask_qc_threshold": args.mask_qc_threshold,
        "edit_qc_threshold": args.edit_qc_threshold,
        "min_delta": args.min_delta,
        "max_edit_attempts": max(1, args.max_edit_attempts),
        "editors": ["gemini"],
        "reasoning_model": args.reasoning_model,
        "editor_model": args.editor_model,
    }


def run_signature(args) -> str:
    import hashlib

    payload = {
        "metric": args.metric,
        "goal": args.goal,
        "max_epochs": args.max_epochs,
        "new_regions_only": bool(args.new_regions_only),
        "two_phase": bool(args.two_phase),
        "mask_qc_threshold": args.mask_qc_threshold,
        "edit_qc_threshold": args.edit_qc_threshold,
        "min_delta": args.min_delta,
        "max_edit_attempts": max(1, args.max_edit_attempts),
        "editors": ["gemini"],
        "reasoning_model": args.reasoning_model,
        "editor_model": args.editor_model,
        "include_forced_reference": bool(args.include_forced_reference),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def discover_summaries(args) -> Dict[str, tuple]:
    found: Dict[str, tuple] = {}
    if not args.output_root.is_dir():
        return found
    domain = DOMAIN_BY_METRIC[args.metric]
    expected = expected_config(args, domain)
    for path in args.output_root.rglob("job_summary.json"):
        summary = load_json(path)
        if not summary or summary.get("domain") != domain:
            continue
        input_image = summary.get("input_image")
        if not input_image or summary.get("config") != expected:
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        previous = found.get(input_image)
        if previous is None or stamp > previous[0]:
            found[input_image] = (stamp, path, summary)
    return found


def model_string(bucket: dict) -> str:
    models = bucket.get("models") or {}
    return ";".join(
        f"{name}:{count}" for name, count in sorted(models.items()))


def csv_row(item: Dict[str, Any], signature: str,
            summary_path: Path, summary: dict) -> Dict[str, Any]:
    costs = summary.get("costs") or {}
    chatgpt = costs.get("chatgpt") or {}
    nanobanana = costs.get("nanobanana") or {}
    return {
        "run_signature": signature,
        "dataset_number": item["number"],
        "metric": item["metric"],
        "domain": item["domain"],
        "source": item["source"],
        "input_image": str(item["path"]),
        "output_image": summary.get("output_image"),
        "manifest_score": item["score"],
        "input_score": summary.get("input_score"),
        "output_score": summary.get("output_score"),
        "raw_delta": summary.get("raw_delta"),
        "goal_delta": summary.get("goal_delta"),
        "pipeline_success": summary.get("pipeline_success"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "chatgpt_model": model_string(chatgpt),
        "chatgpt_requests": chatgpt.get("requests"),
        "chatgpt_cost_usd": costs.get("chatgpt_cost_usd"),
        "nanobanana_model": model_string(nanobanana),
        "nanobanana_requests": nanobanana.get("requests"),
        "nanobanana_cost_usd": costs.get("nanobanana_cost_usd"),
        "total_cost_usd": costs.get("total_cost_usd"),
        "cost_complete": costs.get("cost_complete"),
        "run_dir": summary.get("run_dir"),
        "job_summary_path": str(summary_path),
    }


def error_row(item: Dict[str, Any], signature: str, error: str) -> Dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update({
        "run_signature": signature,
        "dataset_number": item["number"],
        "metric": item["metric"],
        "domain": item["domain"],
        "source": item["source"],
        "input_image": str(item["path"]),
        "manifest_score": item["score"],
        "pipeline_success": False,
        "status": "process_error",
        "error": error,
        "cost_complete": False,
    })
    return row


def load_csv_rows(path: Path) -> Dict[tuple, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("run_signature", ""), row.get("input_image", "")): row
            for row in csv.DictReader(handle)
        }


def write_csv_atomic(path: Path, rows: Dict[tuple, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row.get("run_signature") or "",
            int(row["dataset_number"])
            if str(row.get("dataset_number", "")).isdigit() else 10**9,
            row.get("input_image") or "",
        ),
    )
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def health_check(metric: str) -> None:
    scorer = (
        ("greenery", "http://127.0.0.1:8006/health")
        if metric == "greenery"
        else ("risk", "http://127.0.0.1:8003/health")
    )
    endpoints = (
        scorer,
        ("lisat", "http://127.0.0.1:8001/health"),
        ("sam3", "http://127.0.0.1:8005/health"),
        ("sam", "http://127.0.0.1:8004/health"),
    )
    failures = []
    for name, url in endpoints:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status != 200:
                    failures.append(f"{name} ({url}): HTTP {response.status}")
        except Exception as error:
            failures.append(f"{name} ({url}): {error}")
    if failures:
        raise RuntimeError(
            "Required VIDA-GEO services are unavailable:\n  "
            + "\n  ".join(failures))


def command_for(args, item: Dict[str, Any]) -> List[str]:
    command = [
        sys.executable, "-m", "vida_geo.cli",
        "--image", str(item["path"]),
        "--domain", item["domain"],
        "--goal", args.goal,
        "--max_epochs", str(args.max_epochs),
        "--mask_qc_threshold", str(args.mask_qc_threshold),
        "--edit_qc_threshold", str(args.edit_qc_threshold),
        "--min_delta", str(args.min_delta),
        "--max_edit_attempts", str(max(1, args.max_edit_attempts)),
        "--reasoning_model", args.reasoning_model,
        "--editor_model", args.editor_model,
        "--disable_flux",
        "--output_root", str(
            (args.output_root / item["metric"]).resolve()),
    ]
    if args.new_regions_only:
        command.append("--new_regions_only")
    if args.two_phase:
        command.append("--two_phase")
    return command


def run_and_tee(command: List[str], batch_log) -> int:
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            batch_log.write(line)
            batch_log.flush()
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
        raise


def log_line(handle, text: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {text}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def main() -> int:
    args = build_parser().parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv
        else args.output_root / f"benchmark_results_{args.metric}.csv"
    )
    items = load_inputs(args)
    signature = run_signature(args)
    print(
        f"Validated {len(items)} unique {args.metric} inputs "
        f"(VIDA-GEO domain={DOMAIN_BY_METRIC[args.metric]}).")
    print("Forced-reference images included:", bool(args.include_forced_reference))
    print("Output root:", args.output_root)
    print("Summary CSV:", args.summary_csv)
    print("Editors: Nano Banana only (FLUX disabled)")
    if args.dry_run:
        for item in items:
            print(f"{item['metric']:10s} {item['number']:>4s} {item['path']}")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not exported in this job.")
    if not args.skip_health_check:
        health_check(args.metric)

    args.output_root.mkdir(parents=True, exist_ok=True)
    existing = discover_summaries(args)
    csv_rows = load_csv_rows(args.summary_csv)
    batch_log_path = args.output_root / f"batch_{args.metric}.log"
    started = time.perf_counter()
    processed = skipped = errors = 0

    with batch_log_path.open("a", encoding="utf-8", buffering=1) as batch_log:
        log_line(
            batch_log,
            f"BATCH START signature={signature} metric={args.metric} "
            f"jobs={len(items)} config="
            f"{json.dumps(expected_config(args, DOMAIN_BY_METRIC[args.metric]), sort_keys=True)}",
        )
        try:
            for index, item in enumerate(items, 1):
                input_key = str(item["path"])
                previous = existing.get(input_key)
                if (
                    not args.no_resume
                    and previous
                    and previous[2].get("status")
                    in {"success", "no_improving_edit"}
                ):
                    skipped += 1
                    csv_rows[(signature, input_key)] = csv_row(
                        item, signature, previous[1], previous[2])
                    write_csv_atomic(args.summary_csv, csv_rows)
                    log_line(
                        batch_log,
                        f"SKIP [{index}/{len(items)}] {item['path'].name} "
                        f"status={previous[2].get('status')}",
                    )
                    continue

                log_line(
                    batch_log,
                    f"RUN [{index}/{len(items)}] {item['path'].name}")
                command = command_for(args, item)
                batch_log.write("COMMAND " + json.dumps(command) + "\n")
                batch_log.flush()
                returncode = run_and_tee(command, batch_log)

                refreshed = discover_summaries(args)
                current = refreshed.get(input_key)
                if current:
                    existing[input_key] = current
                    csv_rows[(signature, input_key)] = csv_row(
                        item, signature, current[1], current[2])
                    write_csv_atomic(args.summary_csv, csv_rows)
                    processed += 1
                    log_line(
                        batch_log,
                        f"DONE [{index}/{len(items)}] returncode={returncode} "
                        f"status={current[2].get('status')} "
                        f"cost=${(current[2].get('costs') or {}).get('total_cost_usd', 0):.6f}",
                    )
                    if current[2].get("status") == "failed":
                        errors += 1
                        if args.stop_on_error:
                            return 1
                else:
                    errors += 1
                    error = (
                        f"VIDA-GEO exited {returncode} without a matching "
                        "job_summary.json")
                    csv_rows[(signature, input_key)] = error_row(
                        item, signature, error)
                    write_csv_atomic(args.summary_csv, csv_rows)
                    log_line(batch_log, f"ERROR [{index}/{len(items)}] {error}")
                    if args.stop_on_error:
                        return 1
        except KeyboardInterrupt:
            log_line(batch_log, "INTERRUPTED - rerun the same command to resume.")
            return 130

        current_rows = [
            row for (sig, _), row in csv_rows.items() if sig == signature
        ]
        total_cost = sum(
            float(row.get("total_cost_usd") or 0) for row in current_rows)
        chatgpt_cost = sum(
            float(row.get("chatgpt_cost_usd") or 0) for row in current_rows)
        nanobanana_cost = sum(
            float(row.get("nanobanana_cost_usd") or 0) for row in current_rows)
        elapsed = time.perf_counter() - started
        log_line(
            batch_log,
            f"BATCH COMPLETE processed={processed} skipped={skipped} "
            f"errors={errors} wall_seconds={elapsed:.1f} "
            f"chatgpt_cost=${chatgpt_cost:.6f} "
            f"nanobanana_cost=${nanobanana_cost:.6f} "
            f"total_cost=${total_cost:.6f}",
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
