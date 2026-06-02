# vida_geo/orchestration/phase_runner.py
"""
Phase-2 runner: take each successful Phase-1 region edit, treat that edited image
as a fresh input, and run one epoch on it — carrying forward removed objects,
prompt history, and Phase-1 constraints so nothing is re-inserted or repeated.
One Phase-2 sub-run per successful Phase-1 region; outputs under <run>/phase_2/region_<id>/.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..domains.registry import DomainRegistry, load_registry
from ..runio import Timings, banner, setup_logger, write_json
from ..agents.planning import describe_and_plan
from .full_pipeline import _baseline_score, _run_epoch


class PhaseRunner:
    def __init__(self, phase1_result: dict, domain: str, agent_cfg: AgentConfig,
                 prompt_history: Optional[Dict[str, List[dict]]] = None,
                 run_root: Optional[str] = None,
                 registry: Optional[DomainRegistry] = None):
        self.phase1_result = phase1_result
        self.domain = domain
        self.reg = registry or load_registry()
        self.spec = self.reg.domain(domain)
        self.cfg = agent_cfg.with_services()
        self.prompt_history = prompt_history or {}
        self.timings = Timings()
        self.phase1_edits = self._collect_phase1_edits()

        if run_root:
            self.run_root = Path(run_root).expanduser().resolve()
        else:
            p1 = self.phase1_result.get("run_root") or self.phase1_result.get("run_dir") \
                or str(Path(self.spec.output_root) / "unknown")
            self.run_root = Path(p1) / "phase_2"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log = setup_logger(self.run_root)

    def _collect_phase1_edits(self) -> List[dict]:
        best_by_region: Dict[str, dict] = {}
        for c in self.phase1_result.get("all_candidates", []):
            if not c.get("is_improvement"):
                continue
            out = c.get("out_image", "")
            if not out or not Path(out).is_file():
                continue
            rid = c.get("region_id", "?")
            if rid not in best_by_region or abs(c.get("delta", 0)) > abs(best_by_region[rid].get("delta", 0)):
                best_by_region[rid] = c
        labels = {r.get("region_id", ""): r.get("label", "")
                  for r in self.phase1_result.get("plan", {}).get("regions", [])}
        return [{"region_id": rid, "out_image": c["out_image"],
                 "edit_type": c.get("edit_type", "replace"), "prompt": c.get("edit_prompt", ""),
                 "label": labels.get(rid, rid), "delta": c.get("delta", 0),
                 "new_score": c.get("new_score")} for rid, c in best_by_region.items()]

    def _phase_context(self, edit: dict) -> dict:
        removed, prev = [], []
        if edit.get("edit_type") == "remove":
            removed.append(edit.get("label", edit.get("region_id", "unknown")))
        prev.append({"region_id": edit["region_id"], "label": edit.get("label", ""),
                     "edit_type": edit.get("edit_type", "replace"),
                     "prompt": edit.get("prompt", "")[:100], "was_improvement": True})
        for r in self.phase1_result.get("plan", {}).get("regions", []):
            if r.get("edit_type") == "remove" and r.get("label") and r["label"] not in removed:
                removed.append(r["label"])
        constraints = [{"region_id": pd["region_id"], "constraint": pd["strategy_prompt_hint"],
                        "rationale": pd.get("rationale", "")}
                       for pd in self.phase1_result.get("policy_data", [])
                       if pd.get("edit_class") == "constrained" and pd.get("strategy_prompt_hint")]
        return {"phase": 2, "source_region_id": edit["region_id"], "removed_objects": removed,
                "previous_edits": prev, "phase1_edit_type": edit.get("edit_type", "replace"),
                "phase1_constraints": constraints}

    def _run_single(self, edit: dict, sub_dir: Path) -> Dict[str, Any]:
        rid = edit["region_id"]
        image_path = edit["out_image"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        self.log.info("Phase 2 sub-run: region %s (input=%s)", rid, image_path)
        shutil.copy2(image_path, str(sub_dir / f"phase2_input_{Path(image_path).name}"))

        with self.timings.stage("baseline", self.log):
            try:
                base_score, payload = _baseline_score(image_path, self.spec, self.cfg)
            except Exception as e:
                return {"region_id": rid, "phase1_image": image_path, "success": False,
                        "error": f"Baseline scoring failed: {e}"}
        write_json(sub_dir / "baseline.json", {"score": base_score, "score_key": self.spec.score_key,
                                              "phase": 2, "source_region": rid})

        with self.timings.stage("planning", self.log):
            try:
                plan = describe_and_plan(image_path, self.spec, goal=self.cfg.goal,
                                         current_scores={self.spec.score_key: base_score}, cfg=self.cfg.llm)
            except Exception as e:
                return {"region_id": rid, "phase1_image": image_path, "baseline_score": base_score,
                        "success": False, "error": f"Planning failed: {e}"}
        write_json(sub_dir / "plan.json", plan)
        regions = plan.get("regions", [])
        if not regions:
            return {"region_id": rid, "phase1_image": image_path, "baseline_score": base_score,
                    "plan": plan, "success": False, "error": "No regions proposed"}

        region_history = {rid: self.prompt_history[rid]} if self.prompt_history.get(rid) else {}
        epoch_result = _run_epoch(image_path, self.cfg, self.spec, self.reg, sub_dir, self.log,
                                  base_score, regions, self.timings,
                                  previous_prompt_history=region_history,
                                  phase_context=self._phase_context(edit))

        best = epoch_result.get("best_candidate")
        sub = {"region_id": rid, "phase1_image": image_path,
               "phase1_edit_type": edit.get("edit_type"), "phase1_delta": edit.get("delta"),
               "baseline_score": base_score, "plan": plan,
               "all_candidates": epoch_result.get("all_candidates", []), "best_candidate": best,
               "final_image": best.get("out_image") if best else image_path,
               "final_score": best["new_score"] if best else base_score,
               "score_delta": best["delta"] if best else 0.0, "success": best is not None}
        write_json(sub_dir / "phase2_result.json", sub)
        return sub

    def run(self) -> Dict[str, Any]:
        banner(self.log, f"PHASE 2 RUNNER — {len(self.phase1_edits)} successful Phase-1 edit(s)")
        if not self.phase1_edits:
            return {"run_root": str(self.run_root), "phase": 2, "n_inputs": 0,
                    "sub_results": [], "best_candidate": None, "success": False,
                    "error": "No successful Phase 1 edits"}

        sub_results, best, best_score = [], None, None
        for edit in self.phase1_edits:
            sub = self._run_single(edit, self.run_root / f"region_{edit['region_id']}")
            sub_results.append(sub)
            bc = sub.get("best_candidate")
            if bc and bc.get("new_score") is not None and not math.isnan(float(bc["new_score"])):
                if best is None or self.spec.is_better(bc["new_score"], best_score, self.cfg.goal):
                    best_score = bc["new_score"]
                    best = {**bc, "phase1_region_id": edit["region_id"], "phase1_image": edit["out_image"]}

        n_ok = sum(1 for s in sub_results if s.get("success"))
        result = {"run_root": str(self.run_root), "phase": 2, "domain": self.domain,
                  "score_key": self.spec.score_key, "goal": self.cfg.goal,
                  "n_inputs": len(self.phase1_edits), "n_successful": n_ok,
                  "sub_results": sub_results, "best_candidate": best, "best_score": best_score,
                  "success": best is not None, "timings": self.timings.as_dict()}
        write_json(self.run_root / "phase2_result.json", result)
        banner(self.log, "PHASE 2 COMPLETE")
        self.log.info("Sub-runs: %d total, %d successful", len(sub_results), n_ok)
        return result
