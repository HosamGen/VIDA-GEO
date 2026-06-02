# vida_geo/orchestration/full_pipeline.py
"""
Single-pass orchestrator: baseline -> plan -> policy -> per-region segment+generate.

_run_epoch is the shared core reused by EpochRunner and PhaseRunner. run_full_agent
is the single-epoch entry point. Direction/scoring/output-root all come from the
domain registry; there is no _is_better here (it's spec.is_better).
"""
from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..domains.registry import DomainRegistry, DomainSpec, load_registry
from ..image_utils import extract_coordinate_prefix
from ..runio import Timings, append_jsonl, banner, setup_logger, write_json
from ..tools import clients
from ..tools.gemini import GeminiConfig
from ..agents.planning import describe_and_plan
from ..agents.policy import (
    assess_feasibility, build_policy_index, evaluate_regions,
    format_hint_for_prompt, log_policy_decisions,
)
from ..agents.segmentation import SegmentationAgent
from ..agents.generation import GenerationAgent


def _baseline_score(image_path: str, spec: DomainSpec, cfg: AgentConfig) -> tuple:
    """Return (target_score, full_payload) from the domain's scorer."""
    scorer_cfg = cfg.scorer_for(spec.scorer)
    payload = clients.score_image(image_path, scorer_cfg)
    return clients.read_score(payload, spec.score_key), payload


def _run_epoch(
    image_path: str,
    cfg: AgentConfig,
    spec: DomainSpec,
    reg: DomainRegistry,
    run_dir: Path,
    log,
    base_score: float,
    regions: List[dict],
    timings: Timings,
    precomputed_masks: Optional[Dict[str, dict]] = None,
    previous_prompt_history: Optional[Dict[str, List[dict]]] = None,
    phase_context: Optional[dict] = None,
) -> Dict[str, Any]:
    """One epoch: policy -> segmentation -> generation. Returns candidates + state."""
    events = run_dir / "events.jsonl"
    precomputed_masks = precomputed_masks or {}
    previous_prompt_history = previous_prompt_history or {}
    phase_context = phase_context or {}
    modality = reg.tools_for(spec.name)

    # ── Policy ──────────────────────────────────────────────────────────
    banner(log, "POLICY")
    with timings.stage("policy", log):
        decisions = evaluate_regions(image_path, regions, spec, cfg.goal, cfg=cfg.llm)
    log_policy_decisions(decisions, log)
    policy_index = build_policy_index(decisions)
    policy_data = [{"region_id": d.region_id, "edit_class": d.edit_class,
                    "feasible": d.feasible, "rationale": d.rationale,
                    "strategy_prompt_hint": d.strategy_prompt_hint} for d in decisions]
    write_json(run_dir / "policy.json", policy_data)
    append_jsonl(events, {"event": "policy_check", "decisions": policy_data})

    skip_ids = {d.region_id for d in decisions if d.edit_class == "skip"}
    if skip_ids:
        log.info("Policy: skipping %d region(s): %s", len(skip_ids), sorted(skip_ids))
        regions = [r for r in regions if r.get("region_id") not in skip_ids]

    feas = assess_feasibility(base_score, spec, cfg.goal, decisions)
    if not feas.feasible:
        log.info("FEASIBILITY GATE: %s — %s", feas.reason, feas.recommendation)
        return {"all_candidates": [], "best_candidate": None, "success": False,
                "failure_history_by_region": {}, "accepted_masks_by_region": {},
                "policy_data": policy_data,
                "feasibility_verdict": {"feasible": False, "reason": feas.reason,
                                        "recommendation": feas.recommendation}}
    if not regions:
        log.info("No regions left after policy filter.")
        return {"all_candidates": [], "best_candidate": None, "success": False,
                "failure_history_by_region": {}, "accepted_masks_by_region": {},
                "policy_data": policy_data}

    # ── Segmentation ────────────────────────────────────────────────────
    banner(log, "SEGMENTATION")
    seg_results: Dict[str, dict] = {}
    accepted_masks_by_region: Dict[str, List[dict]] = {}
    for i, region in enumerate(regions):
        rid = region.get("region_id", f"r{i}")
        if rid in precomputed_masks:
            log.info("Region %s: reusing precomputed mask.", rid)
            pre = precomputed_masks[rid]
            seg_results[rid] = {"success": True, "mask_path": pre.get("mask_path"),
                                "mask_qc": pre.get("mask_qc"),
                                "method": pre.get("method", "precomputed"),
                                "region": region, "region_id": rid, "events": []}
            accepted_masks_by_region[rid] = [pre]
            continue

        with timings.stage("segmentation", log):
            agent = SegmentationAgent(
                image_path=image_path, region=region, run_dir=str(run_dir),
                spec=spec, modality=modality, sam_cfg=cfg.sam,
                smooth_cfg=cfg.smooth, mask_qc_threshold=cfg.mask_qc_threshold,
                sam_point_attempts=cfg.sam_point_attempts, llm_cfg=cfg.llm,
            )
            mask_result = agent.segment_region()
        for ev in mask_result["events"]:
            append_jsonl(events, {**ev, "region": rid})
        seg_results[rid] = mask_result
        if mask_result["success"]:
            log.info("Region %s: mask via %s (qc=%.3f)", rid, mask_result["method"],
                     mask_result.get("mask_qc") or 0.0)
            accepted_masks_by_region[rid] = [{"mask_path": mask_result["mask_path"],
                                              "mask_qc": mask_result["mask_qc"],
                                              "method": mask_result["method"]}]
        else:
            log.info("Region %s: no acceptable mask.", rid)
            accepted_masks_by_region[rid] = []

    # ── Generation ──────────────────────────────────────────────────────
    banner(log, "GENERATION")
    constrained_hints = {d.region_id: d.strategy_prompt_hint for d in decisions
                         if d.edit_class == "constrained" and d.strategy_prompt_hint}

    all_candidates: List[dict] = []
    best_candidate: Optional[dict] = None
    best_score = base_score
    failure_history_by_region: Dict[str, list] = {}

    gemini_cfg = GeminiConfig()
    for i, region in enumerate(regions):
        rid = region.get("region_id", f"r{i}")
        mask_result = seg_results.get(rid)
        # Always hand the region to the generation agent — it decides whether to
        # do a masked edit or the maskless Gemini fallback (which it skips itself
        # if Gemini is disabled). Ensure the region object is carried through.
        if not mask_result:
            mask_result = {"success": False, "mask_path": None, "mask_qc": None,
                           "method": None, "region_id": rid, "region": region}
        elif not mask_result.get("region"):
            mask_result["region"] = region

        policy_decision = policy_index.get(rid)
        region_failures = (previous_prompt_history.get(rid, [])
                           + failure_history_by_region.get(rid, []))
        scene_constraints = [h for r2, h in constrained_hints.items() if r2 != rid]

        with timings.stage("generation", log):
            gen = GenerationAgent(
                image_path=image_path, mask_result=mask_result, spec=spec,
                baseline_score=base_score, run_dir=str(run_dir),
                editors=cfg.editors, goal=cfg.goal, policy_decision=policy_decision,
                scene_constraints=scene_constraints, phase_context=phase_context,
                flux_cfg=cfg.flux, gemini_cfg=gemini_cfg, llm_cfg=cfg.llm,
                edit_qc_threshold=cfg.edit_qc_threshold, min_delta=cfg.min_score_delta,
                max_attempts_per_region=cfg.max_edit_attempts_per_region,
                roi_pad_pct=cfg.roi_pad_pct, roi_multiple=cfg.roi_multiple,
            )
            gen_result = gen.generate(previous_prompt_history=region_failures)

        for ev in gen_result["events"]:
            append_jsonl(events, {**ev, "region": rid})

        for c in gen_result["all_attempts"]:
            all_candidates.append(c)
            if not c.get("is_improvement"):
                failure_history_by_region.setdefault(rid, []).append(
                    {"prompt": c.get("edit_prompt", ""), "delta": c.get("delta"),
                     "edit_qc": c.get("edit_qc", 0.0)})

        bc = gen_result.get("best_candidate")
        if bc and (best_candidate is None or spec.is_better(bc["new_score"], best_score, cfg.goal)):
            best_score, best_candidate = bc["new_score"], bc

    return {"all_candidates": all_candidates, "best_candidate": best_candidate,
            "success": best_candidate is not None,
            "failure_history_by_region": failure_history_by_region,
            "accepted_masks_by_region": accepted_masks_by_region,
            "policy_data": policy_data}


def run_full_agent(image_path: str, domain: str, cfg: Optional[AgentConfig] = None,
                   run_id: Optional[str] = None, registry: Optional[DomainRegistry] = None) -> Dict[str, Any]:
    reg = registry or load_registry()
    spec = reg.domain(domain)
    cfg = (cfg or AgentConfig()).with_services()
    timings = Timings()

    img = Path(image_path).expanduser().resolve()
    if not img.is_file():
        raise FileNotFoundError(str(img))

    output_root = Path(cfg.output_root or spec.output_root).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / (run_id or f"{extract_coordinate_prefix(str(img))}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(run_dir)
    events = run_dir / "events.jsonl"
    shutil.copy2(str(img), str(run_dir / f"input_{img.name}"))

    banner(log, f"VIDA-GEO RUN — domain={domain} goal={cfg.goal} editors={cfg.editors}")

    # Baseline
    with timings.stage("baseline", log):
        try:
            base_score, payload = _baseline_score(str(img), spec, cfg)
        except Exception as e:
            log.error("Baseline scoring failed: %s", e)
            return {"image": str(img), "domain": domain, "success": False,
                    "error": f"Baseline scoring failed: {e}", "run_dir": str(run_dir)}
    write_json(run_dir / "baseline.json", {"score": base_score, "score_key": spec.score_key,
                                           "payload": payload})
    append_jsonl(events, {"event": "baseline", "score": base_score})
    log.info("Baseline %s=%.4f", spec.score_key, base_score)

    # Plan
    banner(log, "PLANNING")
    with timings.stage("planning", log):
        try:
            plan = describe_and_plan(str(img), spec, goal=cfg.goal,
                                     current_scores={spec.score_key: base_score}, cfg=cfg.llm)
        except Exception as e:
            log.error("Planning failed: %s", e)
            return {"image": str(img), "domain": domain, "baseline_score": base_score,
                    "success": False, "error": f"Planning failed: {e}", "run_dir": str(run_dir)}
    write_json(run_dir / "plan.json", plan)
    append_jsonl(events, {"event": "plan", "n_regions": len(plan.get("regions", []))})
    regions = plan.get("regions", [])
    log.info("Planning done | regions=%d", len(regions))
    if not regions:
        log.warning("Planner returned 0 regions; using fallback center region.")
        regions = [{"region_id": "r_fallback", "label": "center region",
                    "description": "fallback center region", "feature_type": "other",
                    "edit_type": "replace", "frac_x": 0.5, "frac_y": 0.5,
                    "perception_potential": "medium", "seg_keyword": "building"}]

    epoch = _run_epoch(str(img), cfg, spec, reg, run_dir, log, base_score, regions, timings)

    # Final result
    banner(log, "RESULTS")
    best = epoch.get("best_candidate")
    successful = sorted(
        [{"out_image": c.get("out_image"), "edit_qc": c.get("edit_qc", 0.0),
          "score": c.get("new_score"), "delta": c.get("delta"),
          "edit_prompt": c.get("edit_prompt"), "edit_type": c.get("edit_type"),
          "editor": c.get("editor"), "region_id": c.get("region_id")}
         for c in epoch["all_candidates"] if c.get("is_improvement")],
        key=lambda x: -(x.get("delta") or 0))

    result = {
        "image": str(img), "domain": domain, "score_key": spec.score_key, "goal": cfg.goal,
        "baseline_score": base_score, "plan": plan,
        "best_candidate": best,
        "final_score": best["new_score"] if best else base_score,
        "final_delta": best["delta"] if best else 0.0,
        "all_successful_edits": successful,
        "all_candidates": epoch["all_candidates"],
        "failure_history_by_region": epoch["failure_history_by_region"],
        "policy_data": epoch.get("policy_data", []),
        "success": best is not None,
        "timings": timings.as_dict(),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "result.json", result)
    append_jsonl(events, {"event": "run_complete", "success": result["success"]})
    if best:
        log.info("Best: region %s via %s | %s %.4f→%.4f (Δ=%+.4f)", best["region_id"],
                 best.get("editor"), spec.score_key, base_score, best["new_score"], best["delta"])
    else:
        log.info("No improving edit found.")
    return result
