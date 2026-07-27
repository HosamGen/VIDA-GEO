#!/usr/bin/env python3
"""
Serial, resumable VIDA-GEO runner for the 600-image GSV experiment set.

Inputs are the union of:
  1. experiment_manifest.csv (old/reference images already excluded)
  2. benchmark_images/extra_images/<metric>/ (the 58 replacements)

Each image is run in a fresh Python process so accounting is isolated. Completed
jobs are discovered from job_summary.json, making the same command safe to
resume after interruption.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path("/l/users/hosam.elgendy/benchmark_images")
DEFAULT_METRICS = (
    "safety", "lively", "beautiful", "wealthy", "boring", "depressing",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")

CSV_FIELDS = [
    "run_signature", "dataset_number", "metric", "source",
    "input_image", "output_image", "manifest_score",
    "input_score", "output_score", "raw_delta", "goal_delta",
    "pipeline_success", "status", "error", "elapsed_seconds",
    "chatgpt_model", "chatgpt_requests", "chatgpt_cost_usd",
    "nanobanana_model", "nanobanana_requests", "nanobanana_cost_usd",
    "total_cost_usd", "cost_complete", "run_dir", "job_summary_path",
]


@dataclass(frozen=True)
class InputItem:
    number: str
    metric: str
    path: Path
    score: Optional[float]
    source: str


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run the 600-image GSV benchmark through VIDA-GEO serially and resumably.")
    ap.add_argument(
        "--manifest", type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "experiment_manifest.csv",
        help="Experiment manifest containing non-reference benchmark images.")
    ap.add_argument(
        "--extra-root", type=Path,
        default=DEFAULT_BENCHMARK_ROOT / "extra_images",
        help="Replacement images, arranged as <extra-root>/<metric>/.")
    ap.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs" / "gsv_benchmark",
        help="Root for per-metric run directories, batch log, and CSV.")
    ap.add_argument(
        "--summary-csv", type=Path, default=None,
        help="CSV path (default: <output-root>/benchmark_results.csv).")
    ap.add_argument(
        "--metrics", nargs="+", choices=DEFAULT_METRICS, default=list(DEFAULT_METRICS),
        help="Metrics to run, in the requested order.")
    ap.add_argument(
        "--limit-per-metric", type=int, default=100,
        help="Run the first N experiment images per metric; 100 is the full set.")

    ap.add_argument("--goal", choices=("improve", "worsen"), default="improve")
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--new-regions-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--two-phase", action="store_true",
                    help="Optional; off by default to reproduce the old gsv_runs workflow.")
    ap.add_argument("--mask-qc-threshold", type=float, default=0.7)
    ap.add_argument("--edit-qc-threshold", type=float, default=0.7)
    ap.add_argument("--min-delta", type=float, default=0.1)
    ap.add_argument("--max-edit-attempts", type=int, default=1)
    ap.add_argument("--reasoning-model", default="openai/gpt-5.1")
    ap.add_argument("--editor-model", default="google/gemini-2.5-flash-image")

    ap.add_argument("--dry-run", action="store_true",
                    help="Validate and print the planned jobs without calling models.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Rerun even when an equivalent completed job_summary.json exists.")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Stop the batch after a process/server failure instead of continuing.")
    ap.add_argument("--skip-health-check", action="store_true")
    return ap


def parse_score_from_name(path: Path) -> Optional[float]:
    parts = path.stem.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def parse_number(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def load_inputs(manifest: Path, extra_root: Path,
                metrics: Iterable[str], limit: int) -> List[InputItem]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not extra_root.is_dir():
        raise FileNotFoundError(f"Extra-image root not found: {extra_root}")
    if limit < 1:
        raise ValueError("--limit-per-metric must be at least 1")

    wanted = set(metrics)
    by_metric: Dict[str, List[InputItem]] = {m: [] for m in metrics}
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row.get("metric", "")
            if metric not in wanted:
                continue
            path = Path(row["image_path"]).expanduser().resolve()
            score = float(row["metric_score"]) if row.get("metric_score") else None
            by_metric[metric].append(InputItem(
                number=row.get("number") or parse_number(path),
                metric=metric, path=path, score=score, source="manifest"))

    for metric in metrics:
        extra_dir = extra_root / metric
        if not extra_dir.is_dir():
            raise FileNotFoundError(f"Extra-image metric directory not found: {extra_dir}")
        for path in sorted(extra_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                by_metric[metric].append(InputItem(
                    number=parse_number(path), metric=metric, path=path.resolve(),
                    score=parse_score_from_name(path), source="extra_images"))

    selected: List[InputItem] = []
    seen_paths = set()
    seen_coords: Dict[tuple, Path] = {}
    for metric in metrics:
        items = sorted(
            by_metric[metric],
            key=lambda x: (int(x.number) if x.number.isdigit() else 10**9, x.path.name))
        if len(items) < limit:
            raise ValueError(
                f"{metric}: requested {limit}, but only {len(items)} experiment images are available")
        items = items[:limit]
        for item in items:
            if not item.path.is_file():
                raise FileNotFoundError(f"Input image not found: {item.path}")
            if item.path in seen_paths:
                raise ValueError(f"Duplicate input path: {item.path}")
            seen_paths.add(item.path)
            match = COORD_RE.search(item.path.stem)
            if not match:
                raise ValueError(f"Cannot parse terminal coordinates from: {item.path.name}")
            coord = (round(float(match.group(1)), 6), round(float(match.group(2)), 6))
            if coord in seen_coords:
                raise ValueError(
                    f"Duplicate coordinates {coord}: {seen_coords[coord]} and {item.path}")
            seen_coords[coord] = item.path
        selected.extend(items)
    return selected


def expected_config(args, metric: str) -> Dict[str, Any]:
    return {
        "domain": metric,
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
    payload = {
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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def discover_summaries(output_root: Path, args) -> Dict[str, tuple]:
    """Return newest equivalent summary keyed by absolute input path."""
    found: Dict[str, tuple] = {}
    if not output_root.is_dir():
        return found
    for path in output_root.rglob("job_summary.json"):
        summary = load_json(path)
        if not summary:
            continue
        input_image = summary.get("input_image")
        metric = summary.get("domain")
        if not input_image or metric not in args.metrics:
            continue
        if summary.get("config") != expected_config(args, metric):
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        old = found.get(input_image)
        if old is None or stamp > old[0]:
            found[input_image] = (stamp, path, summary)
    return found


def model_string(bucket: dict) -> str:
    models = bucket.get("models") or {}
    return ";".join(f"{name}:{count}" for name, count in sorted(models.items()))


def csv_row(item: InputItem, signature: str,
            summary_path: Path, summary: dict) -> Dict[str, Any]:
    costs = summary.get("costs") or {}
    chatgpt = costs.get("chatgpt") or {}
    nanobanana = costs.get("nanobanana") or {}
    return {
        "run_signature": signature,
        "dataset_number": item.number,
        "metric": item.metric,
        "source": item.source,
        "input_image": str(item.path),
        "output_image": summary.get("output_image"),
        "manifest_score": item.score,
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


def process_error_row(item: InputItem, signature: str, error: str) -> Dict[str, Any]:
    row = {field: None for field in CSV_FIELDS}
    row.update({
        "run_signature": signature,
        "dataset_number": item.number,
        "metric": item.metric,
        "source": item.source,
        "input_image": str(item.path),
        "manifest_score": item.score,
        "pipeline_success": False,
        "status": "process_error",
        "error": error,
        "cost_complete": False,
    })
    return row


def load_csv_rows(path: Path) -> Dict[tuple, Dict[str, Any]]:
    rows: Dict[tuple, Dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row.get("run_signature", ""), row.get("input_image", ""))] = row
    return rows


def write_csv_atomic(path: Path, rows: Dict[tuple, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda r: (
            r.get("run_signature") or "",
            DEFAULT_METRICS.index(r["metric"]) if r.get("metric") in DEFAULT_METRICS else 999,
            int(r["dataset_number"]) if str(r.get("dataset_number", "")).isdigit() else 10**9,
            r.get("input_image") or "",
        ))
    with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
        tmp = Path(f.name)
    os.replace(tmp, path)


def health_check() -> None:
    endpoints = {
        "perception": "http://127.0.0.1:8111/health",
        "sam3": "http://127.0.0.1:8005/health",
        "sam": "http://127.0.0.1:8004/health",
    }
    failures = []
    for name, url in endpoints.items():
        try:
            with urlopen(url, timeout=5) as response:
                if response.status != 200:
                    failures.append(f"{name} ({url}): HTTP {response.status}")
        except Exception as exc:
            failures.append(f"{name} ({url}): {exc}")
    if failures:
        raise RuntimeError("Required VIDA-GEO services are unavailable:\n  " + "\n  ".join(failures))


def command_for(args, item: InputItem) -> List[str]:
    command = [
        sys.executable, "-m", "vida_geo.cli",
        "--image", str(item.path),
        "--domain", item.metric,
        "--goal", args.goal,
        "--max_epochs", str(args.max_epochs),
        "--mask_qc_threshold", str(args.mask_qc_threshold),
        "--edit_qc_threshold", str(args.edit_qc_threshold),
        "--min_delta", str(args.min_delta),
        "--max_edit_attempts", str(max(1, args.max_edit_attempts)),
        "--reasoning_model", args.reasoning_model,
        "--editor_model", args.editor_model,
        "--disable_flux",
        "--output_root", str((args.output_root / item.metric).resolve()),
    ]
    if args.new_regions_only:
        command.append("--new_regions_only")
    if args.two_phase:
        command.append("--two_phase")
    return command


def run_and_tee(command: List[str], batch_log) -> int:
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
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
    args.manifest = args.manifest.expanduser().resolve()
    args.extra_root = args.extra_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv else args.output_root / "benchmark_results.csv")

    items = load_inputs(
        args.manifest, args.extra_root, args.metrics, args.limit_per_metric)
    signature = run_signature(args)
    counts = {metric: sum(1 for x in items if x.metric == metric) for metric in args.metrics}

    print(f"Validated {len(items)} unique inputs: {counts}")
    print(f"Run signature: {signature}")
    print(f"Output root: {args.output_root}")
    print(f"Summary CSV: {args.summary_csv}")
    print("Editors: Nano Banana only (FLUX disabled)")
    if args.dry_run:
        for item in items:
            print(f"{item.metric:10s} {item.number:>4s} {item.path}")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not exported in this shell.")
    if not args.skip_health_check:
        health_check()

    args.output_root.mkdir(parents=True, exist_ok=True)
    batch_log_path = args.output_root / "batch.log"
    existing = discover_summaries(args.output_root, args)
    csv_rows = load_csv_rows(args.summary_csv)
    batch_started = time.perf_counter()
    processed = skipped = errors = 0

    with open(batch_log_path, "a", encoding="utf-8", buffering=1) as batch_log:
        log_line(batch_log, f"BATCH START signature={signature} jobs={len(items)} config="
                 f"{json.dumps(expected_config(args, args.metrics[0]), sort_keys=True)}")
        try:
            for index, item in enumerate(items, 1):
                key = str(item.path)
                prior = existing.get(key)
                if (not args.no_resume and prior
                        and prior[2].get("status") in {"success", "no_improving_edit"}):
                    skipped += 1
                    row = csv_row(item, signature, prior[1], prior[2])
                    csv_rows[(signature, key)] = row
                    write_csv_atomic(args.summary_csv, csv_rows)
                    log_line(batch_log, f"SKIP [{index}/{len(items)}] {item.metric}/{item.path.name} "
                             f"status={prior[2].get('status')}")
                    continue

                log_line(batch_log, f"RUN [{index}/{len(items)}] {item.metric}/{item.path.name}")
                command = command_for(args, item)
                batch_log.write("COMMAND " + json.dumps(command) + "\n")
                batch_log.flush()
                returncode = run_and_tee(command, batch_log)

                refreshed = discover_summaries(args.output_root, args)
                current = refreshed.get(key)
                if current:
                    existing[key] = current
                    row = csv_row(item, signature, current[1], current[2])
                    csv_rows[(signature, key)] = row
                    write_csv_atomic(args.summary_csv, csv_rows)
                    processed += 1
                    log_line(batch_log, f"DONE [{index}/{len(items)}] returncode={returncode} "
                             f"status={current[2].get('status')} "
                             f"cost=${(current[2].get('costs') or {}).get('total_cost_usd', 0):.6f}")
                    if current[2].get("status") == "failed":
                        errors += 1
                        if args.stop_on_error:
                            return 1
                else:
                    errors += 1
                    error = f"VIDA-GEO exited {returncode} without a matching job_summary.json"
                    csv_rows[(signature, key)] = process_error_row(item, signature, error)
                    write_csv_atomic(args.summary_csv, csv_rows)
                    log_line(batch_log, f"ERROR [{index}/{len(items)}] {error}")
                    if args.stop_on_error:
                        return 1
        except KeyboardInterrupt:
            log_line(batch_log, "INTERRUPTED — rerun the same command to resume.")
            return 130

        current_rows = [
            row for (sig, _), row in csv_rows.items() if sig == signature]
        total_cost = sum(float(row.get("total_cost_usd") or 0) for row in current_rows)
        chatgpt_cost = sum(float(row.get("chatgpt_cost_usd") or 0) for row in current_rows)
        nanobanana_cost = sum(
            float(row.get("nanobanana_cost_usd") or 0) for row in current_rows)
        wall = time.perf_counter() - batch_started
        log_line(
            batch_log,
            f"BATCH COMPLETE processed={processed} skipped={skipped} errors={errors} "
            f"wall_seconds={wall:.1f} chatgpt_cost=${chatgpt_cost:.6f} "
            f"nanobanana_cost=${nanobanana_cost:.6f} total_cost=${total_cost:.6f}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
