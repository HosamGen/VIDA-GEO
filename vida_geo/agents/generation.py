# vida_geo/agents/generation.py
"""
Generation agent — the editor competition.

For one region with an accepted mask, produce candidate edits with the enabled
editors, score every candidate against the black-box scorer, and keep the best
by goal-direction delta.

Editor contract (the key asymmetry):
  - FLUX is a masked inpainter -> crop the ROI, fill the crop, paste back into
    the full image. Internal-only; the artifact handed to QC/scorer is full-frame.
  - Gemini is a full-image generator -> always edits the whole image with the
    mask as guidance; output is already full-frame.
So regardless of editor, edit-QC and the scorer always receive
(full_original, full_result, full_mask). The crop is invisible past paste-back.

Per-style prompts: FLUX competes with flux-style prompts, Gemini with
gemini-style prompts (suggestion agent produces both). Best delta across all
candidates wins.

Removal (GSV only): edit_type='remove' -> Gemini only, full image, a direct
removal prompt (no LLM suggestion). If Gemini is disabled, the region is skipped
(FLUX is never used for removal). Satellite never removes.

Direction and scoring come from the registry; nothing here special-cases a metric.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..domains.registry import DomainSpec
from ..llm.openrouter import LLMConfig
from ..tools import clients
from ..tools.clients import FluxConfig
from ..tools.gemini import GeminiConfig, gemini_edit
from ..image_utils import crop_roi_bbox, paste_roi_back
from .qc_base import QCResult
from .qc_edit import evaluate_edit
from .suggestion import suggest_prompt

log = logging.getLogger(__name__)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class GenerationAgent:
    def __init__(
        self,
        image_path: str,
        mask_result: Dict[str, Any],
        spec: DomainSpec,
        baseline_score: float,
        run_dir: str,
        *,
        editors: Optional[List[str]] = None,         # subset of ["flux","gemini"]
        goal: str = "improve",
        plan: Optional[Dict[str, Any]] = None,
        policy_decision: Optional[Any] = None,
        scene_constraints: Optional[List[str]] = None,
        phase_context: Optional[Dict[str, Any]] = None,
        flux_cfg: Optional[FluxConfig] = None,
        gemini_cfg: Optional[GeminiConfig] = None,
        llm_cfg: Optional[LLMConfig] = None,
        edit_qc_threshold: float = 0.7,
        min_delta: float = 0.1,
        max_attempts_per_region: int = 1,
        roi_pad_pct: float = 0.15,
        roi_multiple: int = 8,
    ):
        self.image_path = str(Path(image_path).expanduser().resolve())
        self.mask_result = mask_result or {}
        self.spec = spec
        self.baseline = float(baseline_score)
        self.run_dir = Path(run_dir)
        self.goal = goal
        self.plan = plan or {}
        self.policy_decision = policy_decision
        self.scene_constraints = list(scene_constraints or [])
        self.phase_context = dict(phase_context or {})

        self.editors = [e for e in (editors or ["flux", "gemini"]) if e in ("flux", "gemini")]
        self.flux_cfg = flux_cfg or FluxConfig()
        self.gemini_cfg = gemini_cfg or GeminiConfig()
        self.llm_cfg = llm_cfg
        self.scorer_cfg = clients.scorer_config(spec.scorer)

        self.edit_qc_threshold = float(edit_qc_threshold)
        self.min_delta = float(min_delta)
        self.max_attempts = max(1, int(max_attempts_per_region))
        self.roi_pad_pct = float(roi_pad_pct)
        self.roi_multiple = int(roi_multiple)

        self.region = self.mask_result.get("region") or {}
        self.region_id = str(self.mask_result.get("region_id") or "region")
        _mp = self.mask_result.get("mask_path")
        self.mask_path = str(_mp) if _mp else None      # real None when absent (not "None")
        self.edit_type = str(self.region.get("edit_type", "replace"))
        self.has_mask = bool(self.mask_result.get("success") and _mp)
        self.img_stem = Path(self.image_path).stem

        self.edits_dir = self.run_dir / "edits" / self.region_id
        self.roi_dir = self.run_dir / "roi" / self.region_id
        self.qc_dir = self.run_dir / "qc" / self.region_id
        self.prompts_dir = self.run_dir / "prompts"
        for d in (self.edits_dir, self.roi_dir, self.qc_dir, self.prompts_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.events: List[Dict[str, Any]] = []
        self.used_prompts: List[Dict[str, Any]] = []
        self.all_attempts: List[Dict[str, Any]] = []
        self.good_candidates: List[Dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------
    @property
    def gemini_enabled(self) -> bool:
        return "gemini" in self.editors

    @property
    def flux_enabled(self) -> bool:
        return "flux" in self.editors

    def _policy_hint(self) -> str:
        if self.policy_decision is None:
            return ""
        return str(getattr(self.policy_decision, "strategy_prompt_hint", "") or "")

    def _build_remove_prompt(self) -> str:
        label = self.region.get("label", "object")
        return (
            f"Remove the {label} from the highlighted/masked area. "
            f"Fill naturally with the surrounding background — match textures, "
            f"lighting, and perspective seamlessly."
        )

    # -- editors ---------------------------------------------------------
    def _run_flux(self, prompt: str, attempt_id: int) -> Optional[str]:
        """Crop ROI -> fill crop -> paste back. Returns full-frame image path."""
        out_full = self.edits_dir / f"{self.img_stem}_{self.region_id}_flux_{attempt_id}.png"
        work = self.roi_dir / f"flux_attempt_{attempt_id}"
        t0 = time.time()
        try:
            roi_img, roi_mask, origin = crop_roi_bbox(
                self.image_path, self.mask_path, str(work),
                pad_pct=self.roi_pad_pct, multiple=self.roi_multiple,
            )
            roi_out = str(work / "roi_flux_edit.png")
            clients.flux_fill(roi_img, roi_mask, prompt, roi_out, self.flux_cfg)
            paste_roi_back(self.image_path, roi_out, self.mask_path, origin, str(out_full))
            log.info("    [flux] edit done (%.1fs)", time.time() - t0)
            return str(out_full)
        except Exception as e:
            self.log_event("flux_error", attempt=attempt_id, error=str(e))
            log.warning("    [flux] edit FAILED (%.1fs): %s", time.time() - t0, e)
            return None

    def _run_gemini(self, prompt: str, attempt_id: int) -> Optional[str]:
        """Edit the full image directly (mask as guidance). Returns image path."""
        out_full = self.edits_dir / f"{self.img_stem}_{self.region_id}_gemini_{attempt_id}.png"
        t0 = time.time()
        try:
            res = gemini_edit(self.image_path, prompt, str(out_full),
                              cfg=self.gemini_cfg, mask_path=self.mask_path)
            log.info("    [gemini] edit done (%.1fs)", time.time() - t0)
            return res["out_path"]
        except Exception as e:
            self.log_event("gemini_error", attempt=attempt_id, error=str(e))
            log.warning("    [gemini] edit FAILED (%.1fs): %s", time.time() - t0, e)
            return None

    # -- scoring + QC of one produced image ------------------------------
    def _evaluate(self, out_image: str, editor: str, prompt: str, attempt_id: int,
                  mask_path: Optional[str] = None, edit_type: Optional[str] = None) -> Dict[str, Any]:
        qc_mask = mask_path if mask_path is not None else (self.mask_path if self.has_mask else None)
        t_qc = time.time()
        qc = evaluate_edit(
            original_image_path=self.image_path,
            edited_image_path=out_image,
            mask_path=qc_mask,
            spec=self.spec,
            edit_description=prompt,
            goal=self.goal,
            policy_constraint=self._policy_hint(),
            scene_constraints=self.scene_constraints,
            threshold=self.edit_qc_threshold,
            cfg=self.llm_cfg,
        )
        qc_dt = time.time() - t_qc
        _write_json(self.qc_dir / f"edit_qc_{editor}_{attempt_id}.json", qc.raw)

        # Score every candidate, even QC failures — they may surface as
        # "possible changes". One scorer call is cheap relative to a generation.
        t_sc = time.time()
        try:
            new_score = clients.score_target(out_image, self.scorer_cfg, self.spec.score_key)
            score_err = None
        except Exception as e:
            new_score = self.baseline
            score_err = str(e)
            log.warning("    [%s] scoring failed: %s", editor, e)
        sc_dt = time.time() - t_sc

        delta = self.spec.progress(new_score, self.baseline, self.goal)   # +ve = better
        raw_delta = self.spec.raw_delta(new_score, self.baseline)
        qc_passed = qc.passed
        is_improvement = bool(qc_passed and delta >= self.min_delta)

        log.info("    [%s] qc=%.3f (%.1fs) score=%.3f delta=%+.3f (%.1fs) improve=%s",
                 editor, qc.score, qc_dt, new_score, delta, sc_dt, is_improvement)

        attempt = {
            "attempt": attempt_id, "editor": editor, "edit_prompt": prompt,
            "region_id": self.region_id, "edit_type": edit_type or self.edit_type,
            "mask_path": qc_mask, "out_image": out_image,
            "edit_qc": qc.score, "edit_qc_obj": qc.raw,
            "new_score": new_score, "delta": delta, "raw_delta": raw_delta,
            "score_error": score_err,
            "qc_passed": qc_passed, "is_improvement": is_improvement,
            "policy_constraint": self._policy_hint(),
            "scene_constraints": self.scene_constraints,
        }
        self.log_event("edit_attempt", attempt=attempt_id, editor=editor,
                       edit_qc=qc.score, delta=delta, qc_passed=qc_passed,
                       is_improvement=is_improvement)
        return attempt

    def _failed_attempt(self, editor: str, prompt: str, attempt_id: int) -> Dict[str, Any]:
        self.log_event("edit_failed", attempt=attempt_id, editor=editor)
        return {
            "attempt": attempt_id, "editor": editor, "edit_prompt": prompt,
            "region_id": self.region_id, "out_image": None,
            "edit_qc": 0.0, "new_score": self.baseline, "delta": 0.0,
            "qc_passed": False, "is_improvement": False,
        }

    # -- prompt selection per editor -------------------------------------
    def _prompts_for(self, editor: str, prev_history: Optional[List[Dict[str, Any]]]) -> List[str]:
        style = "flux" if editor == "flux" else "gemini"
        t0 = time.time()
        try:
            sug = suggest_prompt(
                image_path=self.image_path, spec=self.spec,
                region_label=self.region.get("label", "object"),
                region_description=self.region.get("description", ""),
                goal=self.goal, prompt_style=style, mask_path=self.mask_path,
                current_score=self.baseline, previous_failures=prev_history,
                used_prompts=self.used_prompts, policy_hint=self._policy_hint(),
                phase_context=self.phase_context, cfg=self.llm_cfg,
            )
            _write_json(self.prompts_dir / f"suggestion_{self.region_id}_{style}.json", sug.raw)
            # recommended first, then the rest, de-duped, capped at the budget
            ordered, seen = [], set()
            for p in [sug.recommended, *sug.candidates]:
                p = (p or "").strip()
                if p and p not in seen:
                    ordered.append(p); seen.add(p)
            log.info("    [%s] suggestion -> %d prompt(s) (%.1fs)", editor, len(ordered[:self.max_attempts]), time.time() - t0)
            return ordered[: self.max_attempts]
        except Exception as e:
            log.warning("  prompt suggestion failed (%s); skipping %s", e, editor)
            return []

    # -- main ------------------------------------------------------------
    def generate(self, previous_prompt_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # No usable mask -> maskless Gemini fallback (ON by default).  <<< MASKLESS FALLBACK
        # Gemini is the only editor that can edit without a mask; FLUX is an
        # inpainter and needs one. So if Gemini is disabled, we skip the region.
        if not self.has_mask:
            return self._generate_maskless(previous_prompt_history)

        # Removal path: GSV + Gemini only.
        if self.edit_type == "remove":
            return self._generate_removal()

        attempt_id = 0
        for editor in self.editors:
            log.info("  Region %s — editor [%s]", self.region_id, editor)
            for prompt in self._prompts_for(editor, previous_prompt_history):
                attempt_id += 1
                self.used_prompts.append({"prompt": prompt, "editor": editor})
                runner = self._run_flux if editor == "flux" else self._run_gemini
                out = runner(prompt, attempt_id)
                attempt = (self._evaluate(out, editor, prompt, attempt_id)
                           if out else self._failed_attempt(editor, prompt, attempt_id))
                self.all_attempts.append(attempt)
                if attempt.get("is_improvement"):
                    self.good_candidates.append(attempt)

        return self._result()

    def _generate_maskless(self, previous_prompt_history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """No mask was found for this region. Do one full-image Gemini edit guided by
        the region description. Labeled edit_type='maskless_fallback' so these can be
        filtered out of strict per-region analysis. FLUX cannot run maskless, so if
        Gemini is disabled we skip the region entirely."""
        if not self.gemini_enabled:
            self.log_event("maskless_skipped",
                           reason="no mask and Gemini disabled; FLUX requires a mask")
            log.info("  Region %s: no mask and Gemini disabled — skipped "
                     "(FLUX cannot edit without a mask).", self.region_id)
            return self._result(note="no mask; FLUX-only run cannot edit maskless")

        log.info("  Region %s: no mask — maskless Gemini fallback.", self.region_id)
        prompts = self._prompts_for("gemini", previous_prompt_history) or [
            self.region.get("description", "")]
        attempt_id = 0
        for prompt in prompts:
            attempt_id += 1
            self.used_prompts.append({"prompt": prompt, "editor": "gemini",
                                      "edit_type": "maskless_fallback"})
            out = self._run_gemini_maskless(prompt, attempt_id)
            attempt = (self._evaluate(out, "gemini", prompt, attempt_id,
                                      mask_path=None, edit_type="maskless_fallback")
                       if out else self._failed_attempt("gemini", prompt, attempt_id))
            self.all_attempts.append(attempt)
            if attempt.get("is_improvement"):
                self.good_candidates.append(attempt)
        return self._result()

    def _run_gemini_maskless(self, prompt: str, attempt_id: int) -> Optional[str]:
        """Full-image Gemini edit with no mask."""
        out_full = self.edits_dir / f"{self.img_stem}_{self.region_id}_gemini_nomask_{attempt_id}.png"
        try:
            res = gemini_edit(self.image_path, prompt, str(out_full),
                              cfg=self.gemini_cfg, mask_path=None)
            return res["out_path"]
        except Exception as e:
            self.log_event("gemini_error", attempt=attempt_id, error=str(e))
            log.warning("  maskless Gemini edit failed: %s", e)
            return None

    def _generate_removal(self) -> Dict[str, Any]:
        if self.spec.modality != "gsv":
            self.log_event("remove_skipped", reason="removal not supported for satellite")
            return self._result(note="removal not supported for this modality")
        if not self.gemini_enabled:
            self.log_event("remove_skipped",
                           reason="removal requires Gemini; FLUX cannot perform clean removal")
            log.info("  removal requires Gemini — region skipped (Gemini disabled).")
            return self._result(note="removal requires Gemini (Gemini disabled)")

        prompt = self._build_remove_prompt()
        self.used_prompts.append({"prompt": prompt, "editor": "gemini", "edit_type": "remove"})
        out = self._run_gemini(prompt, attempt_id=1)
        attempt = (self._evaluate(out, "gemini", prompt, 1)
                   if out else self._failed_attempt("gemini", prompt, 1))
        self.all_attempts.append(attempt)
        if attempt.get("is_improvement"):
            self.good_candidates.append(attempt)
        return self._result()

    def _pick_best(self) -> Optional[Dict[str, Any]]:
        if not self.good_candidates:
            return None
        # Highest goal-direction delta; QC score breaks ties.
        return max(self.good_candidates,
                   key=lambda c: (float(c["delta"]), float(c["edit_qc"])))

    def _result(self, note: str = "") -> Dict[str, Any]:
        best = self._pick_best()
        return {
            "success": bool(best),
            "region_id": self.region_id,
            "editors": self.editors,
            "best_candidate": best,
            "good_candidates": list(self.good_candidates),
            "all_attempts": list(self.all_attempts),
            "events": self.events,
            "note": note,
        }

    def log_event(self, event: str, **kw: Any) -> None:
        self.events.append({"event": event, "region_id": self.region_id, **kw})