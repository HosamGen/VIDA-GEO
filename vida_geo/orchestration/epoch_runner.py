# vida_geo/orchestration/epoch_runner.py
"""
Multi-epoch orchestrator. Each epoch: (re)plan -> _run_epoch. Successful masks are
reused across epochs; failed prompts accumulate in history to avoid repetition; the
planner proposes new regions in later epochs from the accumulated summary.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..domains.registry import DomainRegistry, load_registry
from ..image_utils import extract_coordinate_prefix
from ..runio import Timings, banner, setup_logger, write_json
from ..agents.planning import describe_and_plan, replan_epoch
from .full_pipeline import _baseline_score, _run_epoch


@dataclass
class EpochConfig:
    max_epochs: int = 3
    min_improvement_per_epoch: float = 0.0
    reuse_masks_across_epochs: bool = True
    new_regions_only: bool = False
    max_prompt_history_per_region: int = 10


class EpochRunner:
    def __init__(self, image_path: str, domain: str, agent_cfg: AgentConfig,
                 epoch_cfg: Optional[EpochConfig] = None,
                 registry: Optional[DomainRegistry] = None):
        self.image_path = str(Path(image_path).expanduser().resolve())
        self.domain = domain
        self.reg = registry or load_registry()
        self.spec = self.reg.domain(domain)
        self.cfg = agent_cfg.with_services()
        self.epoch_cfg = epoch_cfg or EpochConfig()
        self.timings = Timings()

        self._accepted_masks: Dict[str, dict] = {}
        self._prompt_history: Dict[str, List[dict]] = {}
        self._all_candidates: List[dict] = []
        self._region_summaries: List[dict] = []
        self._tried_ids: set = set()

        img = Path(self.image_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(self.cfg.output_root or self.spec.output_root).expanduser().resolve()
        self.run_root = root / f"{extract_coordinate_prefix(str(img))}_{ts}_epochs"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log = setup_logger(self.run_root)

    # accessors for PhaseRunner
    def get_prompt_history(self) -> Dict[str, List[dict]]:
        return dict(self._prompt_history)

    def get_all_candidates(self) -> List[dict]:
        return list(self._all_candidates)

    def run(self) -> Dict[str, Any]:
        banner(self.log, f"EPOCH RUNNER — domain={self.domain} max_epochs={self.epoch_cfg.max_epochs}")
        img = Path(self.image_path)
        shutil.copy2(self.image_path, str(self.run_root / f"input_{img.name}"))

        with self.timings.stage("baseline", self.log):
            try:
                base_score, payload = _baseline_score(self.image_path, self.spec, self.cfg)
            except Exception as e:
                self.log.error("Baseline scoring failed: %s", e)
                return {"run_root": str(self.run_root), "image_in": self.image_path,
                        "success": False, "error": f"Baseline scoring failed: {e}"}
        write_json(self.run_root / "baseline.json",
                   {"score": base_score, "score_key": self.spec.score_key, "payload": payload})
        self.log.info("Baseline %s=%.4f", self.spec.score_key, base_score)
        best_overall = base_score

        for epoch in range(1, self.epoch_cfg.max_epochs + 1):
            banner(self.log, f"EPOCH {epoch}/{self.epoch_cfg.max_epochs}")
            epoch_dir = self.run_root / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            with self.timings.stage("planning", self.log):
                if epoch == 1:
                    plan = describe_and_plan(self.image_path, self.spec, goal=self.cfg.goal,
                                             current_scores={self.spec.score_key: base_score}, cfg=self.cfg.llm)
                else:
                    summary = self._build_summary(epoch - 1, base_score)
                    plan = replan_epoch(self.image_path, self.spec, goal=self.cfg.goal,
                                        epoch_summary=summary, epoch_number=epoch,
                                        current_scores={self.spec.score_key: base_score}, cfg=self.cfg.llm)
            write_json(epoch_dir / "plan.json", plan)

            new_regions = [r for r in plan.get("regions", [])
                           if r.get("region_id") not in self._tried_ids]
            precomputed = dict(self._accepted_masks) if self.epoch_cfg.reuse_masks_across_epochs else {}

            regions_this = self._regions_for_epoch(epoch, new_regions)
            if regions_this is None:
                break

            epoch_result = _run_epoch(
                self.image_path, self.cfg, self.spec, self.reg, epoch_dir, self.log,
                base_score, regions_this, self.timings,
                precomputed_masks=precomputed, previous_prompt_history=self._prompt_history)
            epoch_result["epoch_num"] = epoch
            write_json(epoch_dir / "epoch_result.json", epoch_result)

            fv = epoch_result.get("feasibility_verdict")
            if fv and not fv.get("feasible", True):
                self.log.warning("Epoch %d infeasible (%s). Stopping.", epoch, fv.get("reason"))
                break

            self._update_state(epoch_result, epoch, regions_this)
            for r in regions_this:
                self._tried_ids.add(r.get("region_id"))
            for c in epoch_result.get("all_candidates", []):
                c["epoch"] = epoch
            self._all_candidates.extend(epoch_result.get("all_candidates", []))

            bc = epoch_result.get("best_candidate")
            best_this = None
            if bc and bc.get("new_score") is not None and not math.isnan(float(bc["new_score"])):
                best_this = bc["new_score"]
                if self.spec.is_better(best_this, best_overall, self.cfg.goal):
                    best_overall = best_this
            self.log.info("Epoch %d done. best_this=%s overall=%.4f", epoch,
                          f"{best_this:.4f}" if best_this is not None else "none", best_overall)
            write_json(epoch_dir / "epoch_summary.json", self._build_summary(epoch, base_score))

            if (self.epoch_cfg.min_improvement_per_epoch > 0 and epoch > 1
                    and (best_this is None
                         or abs(best_this - base_score) < self.epoch_cfg.min_improvement_per_epoch)):
                self.log.info("Epoch %d below improvement threshold. Stopping.", epoch)
                break

        return self._final_result(base_score, payload)

    def _regions_for_epoch(self, epoch: int, new_regions: List[dict]) -> Optional[List[dict]]:
        if not new_regions and epoch == 1:
            self.log.info("Epoch 1: planner proposed no regions.")
            return None
        if not new_regions and epoch > 1:
            if self.epoch_cfg.new_regions_only:
                self.log.info("Epoch %d: no new regions (new_regions_only). Stopping.", epoch)
                return None
            rids = list(self._accepted_masks.keys())
            if not rids:
                self.log.info("Epoch %d: no new regions and no prior masks. Stopping.", epoch)
                return None
            seen, regions = set(), []
            for s in reversed(self._region_summaries):
                if s["region_id"] in rids and s["region_id"] not in seen:
                    regions.append(s["region_obj"]); seen.add(s["region_id"])
            self.log.info("Epoch %d: re-running %d prior region(s).", epoch, len(regions))
            return regions
        if self.epoch_cfg.new_regions_only:
            self.log.info("Epoch %d: %d new region(s).", epoch, len(new_regions))
            return new_regions
        rids = list(self._accepted_masks.keys())
        seen = {r.get("region_id") for r in new_regions}
        rerun = []
        for s in reversed(self._region_summaries):
            if s["region_id"] in rids and s["region_id"] not in seen:
                rerun.append(s["region_obj"]); seen.add(s["region_id"])
        if rerun:
            self.log.info("Epoch %d: %d new + %d re-run region(s).", epoch, len(new_regions), len(rerun))
        return new_regions + rerun

    def _update_state(self, epoch_result: dict, epoch: int, regions: List[dict]) -> None:
        outcomes: Dict[str, dict] = {}
        for r in regions:
            rid = r.get("region_id", "?")
            outcomes[rid] = {"region_id": rid, "epoch": epoch, "segmentation_success": False,
                             "n_edits_passed_qc": 0, "best_score_delta": None, "best_prompt": None,
                             "region_obj": r, "seg_keyword": r.get("seg_keyword", "")}
        for c in epoch_result.get("all_candidates", []):
            rid = c.get("region_id", "?")
            if rid not in outcomes:
                continue
            o = outcomes[rid]
            o["segmentation_success"] = True
            if c.get("is_improvement"):
                o["n_edits_passed_qc"] += 1
            d = c.get("delta")
            if d is not None and not math.isnan(float(d)):
                if o["best_score_delta"] is None or abs(float(d)) > abs(float(o["best_score_delta"])):
                    o["best_score_delta"], o["best_prompt"] = float(d), c.get("edit_prompt", "")
        for rid, masks in epoch_result.get("accepted_masks_by_region", {}).items():
            if masks:
                self._accepted_masks[rid] = masks[0]
                if rid in outcomes:
                    outcomes[rid]["segmentation_success"] = True
        for c in epoch_result.get("all_candidates", []):
            rid = c.get("region_id", "?")
            self._prompt_history.setdefault(rid, []).append(
                {"prompt": c.get("edit_prompt", ""), "qc_score": c.get("edit_qc", 0.0),
                 "score_delta": c.get("delta"), "passed": c.get("is_improvement", False)})
        for rid in self._prompt_history:
            cap = self.epoch_cfg.max_prompt_history_per_region
            if len(self._prompt_history[rid]) > cap:
                self._prompt_history[rid] = self._prompt_history[rid][-cap:]
        for rid, o in outcomes.items():
            o["prompt_history"] = self._prompt_history.get(rid, [])
            self._region_summaries.append(o)

    def _build_summary(self, epoch: int, base_score: float) -> dict:
        this = [s for s in self._region_summaries if s.get("epoch") == epoch]
        regions = [{"region_id": s["region_id"],
                    "frac_x": s["region_obj"].get("frac_x"), "frac_y": s["region_obj"].get("frac_y"),
                    "description": s["region_obj"].get("description", ""),
                    "edit_type": s["region_obj"].get("edit_type", "replace"),
                    "segmentation_success": s["segmentation_success"],
                    "n_edits_passed_qc": s["n_edits_passed_qc"],
                    "best_score_delta": s.get("best_score_delta"),
                    "best_prompt": s.get("best_prompt")} for s in this]
        all_tried = [{"region_id": s["region_id"], "epoch": s.get("epoch"),
                      "frac_x": s["region_obj"].get("frac_x"),
                      "frac_y": s["region_obj"].get("frac_y")} for s in self._region_summaries]
        return {"epoch": epoch, "baseline_score": base_score,
                "regions": regions, "all_regions_tried_all_epochs": all_tried}

    def _final_result(self, base_score: float, payload: dict) -> dict:
        best, best_score = None, base_score
        for c in self._all_candidates:
            ns = c.get("new_score")
            if ns is None or math.isnan(float(ns)):
                continue
            if self.spec.is_better(ns, best_score, self.cfg.goal):
                best, best_score = c, ns
        successful = [{"out_image": c.get("out_image"), "edit_qc": c.get("edit_qc", 0.0),
                       "score": c.get("new_score"), "delta": c.get("delta"),
                       "edit_prompt": c.get("edit_prompt"), "editor": c.get("editor"),
                       "region_id": c.get("region_id"), "epoch": c.get("epoch"),
                       "is_improvement": c.get("is_improvement", False)}
                      for c in self._all_candidates]
        result = {"run_root": str(self.run_root), "image_in": self.image_path,
                  "domain": self.domain, "score_key": self.spec.score_key, "goal": self.cfg.goal,
                  "n_epochs_run": len({c.get("epoch") for c in self._all_candidates}),
                  "baseline_score": base_score, "best_candidate": best,
                  "final_image": best.get("out_image") if best else self.image_path,
                  "final_score": best_score, "score_delta": best_score - base_score,
                  "all_successful_edits": successful, "total_candidates": len(self._all_candidates),
                  "success": best is not None, "prompt_history": self._prompt_history,
                  "accepted_masks": dict(self._accepted_masks),
                  "policy_data": [],  # last epoch's policy is in epoch dirs
                  "timings": self.timings.as_dict()}
        write_json(self.run_root / "result.json", result)
        banner(self.log, "EPOCH RUNNER FINISHED")
        self.log.info("Best %s=%.4f (Δ%+.4f) over %d candidate(s)", self.spec.score_key,
                      best_score, best_score - base_score, len(self._all_candidates))
        return result
