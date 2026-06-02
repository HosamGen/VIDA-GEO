# check_structure.py — run from VIDA-GEO/ root: python check_structure.py
# Confirms the registry loads and every domain resolves all six prompt templates.
# Needs only pyyaml (no servers, no API key, no numpy/cv2).
from vida_geo.domains.registry import load_registry
from vida_geo.prompts.loader import load_prompt

STAGES = ["mask_qc", "edit_qc", "suggestion_gemini", "suggestion_flux",
          "planner", "planner_epoch"]

reg = load_registry()
ok = True
for d in reg.names:
    spec = reg.domain(d)
    for st in STAGES:
        try:
            load_prompt(spec.prompt_group, st, metric=d, direction="increase",
                        metric_description="x", edit_examples="y", score_ctx="",
                        epoch_number=2, previous_attempts="", tried_coords="", phase_ctx="")
        except FileNotFoundError as e:
            ok = False
            print("MISSING:", e)

if ok:
    print("OK — all %d domains resolve all prompts" % len(reg.names))
else:
    print("FIX MISSING ABOVE (check configs/prompts/<group>/ folders)")
