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
import sys

from .config import AgentConfig
from .domains.registry import load_registry
from .orchestration.full_pipeline import run_full_agent
from .orchestration.epoch_runner import EpochConfig, EpochRunner
from .orchestration.phase_runner import PhaseRunner


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
    )


def main() -> int:
    reg = load_registry()
    args = build_parser(reg.names).parse_args()
    cfg = cfg_from_args(args)

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

    if args.two_phase and result.get("success"):
        PhaseRunner(result, args.domain, cfg, prompt_history=prompt_history, registry=reg).run()

    run_loc = result.get("run_root") or result.get("run_dir") or "?"
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