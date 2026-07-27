# vida_geo/cli.py
"""
VIDA-GEO command-line entry point.

  python -m vida_geo.cli --image img.jpg --domain safety
  python -m vida_geo.cli --image tile.png --domain greenery --max_epochs 3
  python -m vida_geo.cli --image img.jpg --domain boring --goal worsen --disable_flux
  python -m vida_geo.cli --image img.jpg --domain safety --max_epochs 2 --two_phase

Flag set is deliberately small: image, domain, goal, epochs/phase, the QC/delta
thresholds you actually sweep, and the editor toggles. Service URLs live in
configs/services.yaml; ROI/attempt knobs are AgentConfig defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

from .accounting import reset_usage, usage_summary
from .config import AgentConfig
from .domains.registry import load_registry
from .orchestration.full_pipeline import run_full_agent
from .orchestration.epoch_runner import EpochConfig, EpochRunner
from .orchestration.phase_runner import PhaseRunner
from .runio import write_json


def build_parser(domain_choices) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="VIDA-GEO agentic geo-editing pipeline")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--domain", required=True, choices=domain_choices,
                    help="Target domain (sets modality, scorer, and improvement direction)")
    ap.add_argument("--goal", default="improve", choices=["improve", "worsen"],
                    help="improve (default) moves the score the beneficial way; worsen flips it")

    ap.add_argument("--max_epochs", type=int, default=1, help=">1 runs the multi-epoch loop")
    ap.add_argument("--new_regions_only", action="store_true",
                    help="In epochs 2+, only try newly proposed regions")
    ap.add_argument("--min_improvement", type=float, default=0.0,
                    help="Stop early if an epoch improves less than this (0 = disabled)")
    ap.add_argument("--two_phase", action="store_true",
                    help="After phase 1, run a phase-2 epoch on each successful edit")

    ap.add_argument("--mask_qc_threshold", type=float, default=0.7)
    ap.add_argument("--edit_qc_threshold", type=float, default=0.7)
    ap.add_argument("--min_delta", type=float, default=0.1)

    ap.add_argument("--disable_flux", action="store_true", help="Drop FLUX from the editor set")
    ap.add_argument("--disable_gemini", action="store_true", help="Drop Gemini from the editor set")
    ap.add_argument("--reasoning_model", default=None,
                    help="OpenRouter reasoning model (default: OPENROUTER_MODEL or openai/gpt-5.1)")
    ap.add_argument("--editor_model", default=None,
                    help="OpenRouter image editor (default: GEMINI_EDIT_MODEL or google/gemini-2.5-flash-image)")
    ap.add_argument("--max_edit_attempts", type=int, default=1,
                    help="Generation attempts per editor/region (default: 1)")

    ap.add_argument("--output_root", default=None, help="Override output dir (default: outputs/<domain>_runs)")
    ap.add_argument("--run_id", default=None)
    return ap


def cfg_from_args(args) -> AgentConfig:
    editors = ["flux", "gemini"]
    if args.disable_flux:
        editors.remove("flux")
    if args.disable_gemini:
        editors.remove("gemini")
    if not editors:
        raise SystemExit("Cannot disable both editors — at least one of FLUX/Gemini must stay enabled.")
    return AgentConfig(
        goal=args.goal, output_root=args.output_root,
        mask_qc_threshold=args.mask_qc_threshold, edit_qc_threshold=args.edit_qc_threshold,
        min_score_delta=args.min_delta, editors=editors,
        max_edit_attempts_per_region=max(1, args.max_edit_attempts),
    )


def _finalize_job(args, result: dict, phase_result: dict | None,
                  registry, started_at: str, started_clock: float) -> dict:
    """Persist wall time, exact OpenRouter costs, and score lineage for one image."""
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed = round(time.perf_counter() - started_clock, 3)
    costs = usage_summary()

    run_loc = result.get("run_root") or result.get("run_dir")
    if not run_loc:
        return {}
    run_dir = Path(run_loc).expanduser().resolve()

    baseline = result.get("baseline_score")
    phase_best = (phase_result or {}).get("best_candidate") or {}
    phase_success = bool(phase_result and phase_result.get("success") and phase_best)
    best = phase_best if phase_success else (result.get("best_candidate") or {})
    output_image = (best.get("out_image") or result.get("final_image")
                    or result.get("image_in") or result.get("image") or args.image)
    output_score = ((phase_result or {}).get("best_score") if phase_success
                    else result.get("final_score"))
    if output_score is None:
        output_score = baseline

    raw_delta = None
    goal_delta = None
    if isinstance(baseline, (int, float)) and isinstance(output_score, (int, float)):
        raw_delta = float(output_score) - float(baseline)
        goal_delta = registry.domain(args.domain).progress(
            float(output_score), float(baseline), args.goal)

    error = (phase_result or {}).get("error") if phase_result else None
    error = error or result.get("error")
    pipeline_success = bool(phase_success or result.get("success"))
    status = "success" if pipeline_success else ("failed" if error else "no_improving_edit")
    config = {
        "domain": args.domain,
        "goal": args.goal,
        "max_epochs": args.max_epochs,
        "new_regions_only": bool(args.new_regions_only),
        "two_phase": bool(args.two_phase),
        "mask_qc_threshold": args.mask_qc_threshold,
        "edit_qc_threshold": args.edit_qc_threshold,
        "min_delta": args.min_delta,
        "max_edit_attempts": max(1, args.max_edit_attempts),
        "editors": [e for e in ("flux", "gemini")
                    if not ((e == "flux" and args.disable_flux)
                            or (e == "gemini" and args.disable_gemini))],
        "reasoning_model": args.reasoning_model or os.getenv(
            "OPENROUTER_MODEL", "openai/gpt-5.1"),
        "editor_model": args.editor_model or os.getenv(
            "GEMINI_EDIT_MODEL", "google/gemini-2.5-flash-image"),
    }
    summary = {
        "status": status,
        "pipeline_success": pipeline_success,
        "error": error,
        "input_image": str(Path(args.image).expanduser().resolve()),
        "output_image": str(Path(output_image).expanduser().resolve()) if output_image else None,
        "domain": args.domain,
        "goal": args.goal,
        "input_score": baseline,
        "output_score": output_score,
        "raw_delta": raw_delta,
        "goal_delta": goal_delta,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed,
        "costs": costs,
        "config": config,
        "run_dir": str(run_dir),
    }

    write_json(run_dir / "usage.json", costs)
    write_json(run_dir / "job_summary.json", summary)
    result["elapsed_seconds"] = elapsed
    result["costs"] = costs
    result["job_summary"] = summary
    write_json(run_dir / "result.json", result)

    line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},000 | INFO | "
        f"JOB TOTAL elapsed={elapsed:.1f}s "
        f"chatgpt_cost=${costs['chatgpt_cost_usd']:.6f} "
        f"nanobanana_cost=${costs['nanobanana_cost_usd']:.6f} "
        f"total_cost=${costs['total_cost_usd']:.6f} "
        f"unpriced_requests={costs['unpriced_requests']}"
    )
    with open(run_dir / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return summary


def main() -> int:
    reg = load_registry()
    args = build_parser(reg.names).parse_args()
    if args.reasoning_model:
        os.environ["OPENROUTER_MODEL"] = args.reasoning_model
    if args.editor_model:
        os.environ["GEMINI_EDIT_MODEL"] = args.editor_model
    cfg = cfg_from_args(args)
    reset_usage()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_clock = time.perf_counter()

    multi = args.max_epochs > 1
    if multi:
        runner = EpochRunner(args.image, args.domain, cfg,
                             EpochConfig(max_epochs=args.max_epochs,
                                         min_improvement_per_epoch=args.min_improvement,
                                         new_regions_only=args.new_regions_only),
                             registry=reg)
        result = runner.run()
        prompt_history = runner.get_prompt_history()
    else:
        result = run_full_agent(args.image, args.domain, cfg, run_id=args.run_id, registry=reg)
        prompt_history = result.get("prompt_history", {})

    phase_result = None
    if args.two_phase and result.get("success"):
        phase_result = PhaseRunner(
            result, args.domain, cfg, prompt_history=prompt_history, registry=reg).run()

    run_loc = result.get("run_root") or result.get("run_dir") or "?"
    _finalize_job(args, result, phase_result, reg, started_at, started_clock)
    if not result.get("success"):
        err = result.get("error")
        print(f"[{args.domain}/{args.goal}] success=False"
              + (f" — {err}" if err else " — no improving edit found")
              + f" | run={run_loc}")
        return 1

    baseline = result.get("baseline_score")
    final = result.get("final_score")
    base_s = f"{baseline:.3f}" if isinstance(baseline, (int, float)) else "n/a"
    final_s = f"{final:.3f}" if isinstance(final, (int, float)) else "n/a"
    print(f"[{args.domain}/{args.goal}] success=True "
          f"baseline={base_s} final={final_s} | run={run_loc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
