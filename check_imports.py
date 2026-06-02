# check_imports.py — run from VIDA-GEO/ root. No servers or API key needed.
# Confirms every module imports and the registry/config/CLI wiring resolves.
import importlib

MODS = [
    "vida_geo.domains.registry", "vida_geo.prompts.loader", "vida_geo.llm.openrouter",
    "vida_geo.tools.clients", "vida_geo.tools.gemini", "vida_geo.tools.smooth_mask",
    "vida_geo.image_utils", "vida_geo.runio",
    "vida_geo.agents.qc_base", "vida_geo.agents.qc_mask", "vida_geo.agents.qc_edit",
    "vida_geo.agents.suggestion", "vida_geo.agents.smoothing", "vida_geo.agents.segmentation",
    "vida_geo.agents.generation", "vida_geo.agents.planning", "vida_geo.agents.policy",
    "vida_geo.config", "vida_geo.orchestration.full_pipeline",
    "vida_geo.orchestration.epoch_runner", "vida_geo.orchestration.phase_runner", "vida_geo.cli",
]

bad = False
for m in MODS:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad = True
        print("FAIL", m, "->", repr(e))
print("IMPORTS:", "FAILED — see above" if bad else f"all {len(MODS)} modules import")

if not bad:
    from vida_geo.config import AgentConfig
    from vida_geo.domains.registry import load_registry
    reg = load_registry()
    cfg = AgentConfig().with_services()
    print("domains:", reg.names)
    print("perception_url:", cfg.perception.url, "| flux_url:", cfg.flux.url,
          "| flux_timeout:", cfg.flux.timeout_s)
    for d in reg.names:
        assert cfg.scorer_for(reg.domain(d).scorer).url
    print("scorer routing OK for all", len(reg.names), "domains")
    print("\nREADY — structure + wiring verified. Next: start servers, then a real run.")
