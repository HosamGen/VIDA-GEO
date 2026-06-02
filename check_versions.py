# check_versions.py — run from VIDA-GEO/ root. Tells you whether the file Python
# actually loads is the UPDATED version, by checking for a marker unique to each fix.
import importlib
checks = [
    ("vida_geo.agents.segmentation", "sam_point %d",            "per-model segmentation logging"),
    ("vida_geo.llm.openrouter",      "extract_json_object",     "robust JSON extractor"),
    ("vida_geo.agents.generation",   "_generate_maskless",      "maskless fallback"),
    ("vida_geo.agents.qc_edit",      "No mask was used",        "maskless-safe edit QC"),
    ("vida_geo.agents.suggestion",   "SuggestionResult",        "normalized suggestion"),
]
import inspect
for mod, marker, label in checks:
    m = importlib.import_module(mod)
    src = inspect.getsource(m)
    status = "UPDATED" if marker in src else "*** STALE ***"
    print(f"  {status:14} {label:32} <- {m.__file__}")
print("\nIf any line says STALE, that file on disk is the OLD version (or a .pyc is shadowing it).")
