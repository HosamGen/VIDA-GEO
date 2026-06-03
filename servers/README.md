# VIDA-GEO model servers

The VIDA-GEO agent is environment-agnostic: it only sends HTTP requests to model
servers. Each server runs the actual model and exposes a small FastAPI app. Servers
are **independent processes in their own conda environments** (their dependencies
conflict, so they cannot share one env) and may sit on different GPUs.

You only need to start the servers a given run uses:

| domain        | scorer            | text-segmenter | also needs            |
|---------------|-------------------|----------------|-----------------------|
| GSV metrics   | perception (8111) | sam3 (8005)    | sam (8004), flux (8002) |
| road_safety   | risk (8003)       | lisat (8001)   | sam (8004), flux (8002) |
| greenery      | greenery (8006)   | lisat (8001)   | sam (8004), flux (8002) |

`sam` (point-prompt fallback) is used by every domain. `flux` is only needed when
FLUX is enabled (omit it if you always run `--disable_flux`).

Ports must match `configs/services.yaml`.

## Where the server files live

Each server's `serve_*_api.py` wrapper (and its `*_predictor.py` / `*_score.py`
module) must run where the model's code is importable. Two patterns are used:

- **Fork (preferred when upstream files were modified):** the wrapper + any
  modified model files live in a fork of the upstream repo. Just clone the fork
  and launch — nothing to copy. BetaRisk uses this (single-scale `mymodels.py`
  only exists in the fork).
- **Copy-in (when upstream is unmodified):** the wrapper lives in this folder
  under `servers/<model>/`; clone the upstream repo, copy the wrapper in, launch.

Each section below says which pattern applies. This folder documents launch
commands and checkpoints; ports must match `configs/services.yaml`.

## Launch commands (edit GPUs/paths to your machine)

### Perception scorer — Place Pulse ViT (GSV) — port 8111
```bash
conda activate <perception-env>
cd /path/to/perception_repo
CUDA_VISIBLE_DEVICES=4 python -m uvicorn serve_gsv_api:app \
  --host 127.0.0.1 --port 8111 --workers 1
```

### Risk scorer — BetaRisk (satellite) — port 8003
Use the fork (it contains the single-scale `utils/mymodels.py`, the training/inference
scripts, and the server wrappers — no file-copying needed):
**https://github.com/HosamGen/BetaRisk**  (forked from https://github.com/FOURM-LAB/BetaRisk)

Checkpoint: this project uses a single-scale retrain (`epoch_15`).
Download: <ADD CHECKPOINT LINK HERE>
```bash
git clone https://github.com/HosamGen/BetaRisk.git
cd BetaRisk
conda activate <risk-env>                          # see the fork's README for env setup
export RISK_CHECKPOINT=/abs/path/to/epoch_15.pth   # REQUIRED (no default)
CUDA_VISIBLE_DEVICES=1 python -m uvicorn serve_risk_api:app \
  --host 127.0.0.1 --port 8003 --workers 1
```

### Greenery scorer — OEM-Lightweight (satellite) — port 8006
Upstream: https://github.com/cliffbb/oem-lightweight  (repo name is case-insensitive
when you clone — name the local folder whatever you like)
Files in `servers/greenery/` (`serve_greenery_api.py`, `greenery_score.py`) are the
wrappers added on top of oem-lightweight — copy them into your clone so its
`sparsemask_api` / `fasterseg_api` modules are importable.

No separate checkpoint to host: it uses the default SparseMask files that ship
with the oem-lightweight repo (`models/SparseMask/...`).
```bash
conda activate <greenery-env>
cd /path/to/oem-lightweight
export OEM_REPO=$(pwd)
export OEM_MODEL=sparsemask
export OEM_ARCH=models/SparseMask/mask_thres_0.001.npy
export OEM_WEIGHTS=models/SparseMask/checkpoint_63750.pth.tar
CUDA_VISIBLE_DEVICES=1 python -m uvicorn serve_greenery_api:app \
  --host 127.0.0.1 --port 8006 --workers 1
```

### LISAt — text-referred segmentation (satellite) — port 8001
```bash
conda activate <lisat-env>
cd /path/to/LISAt
CUDA_VISIBLE_DEVICES=1 python -m uvicorn serve_lisat_api:app \
  --host 127.0.0.1 --port 8001 --workers 1
```

### SAM3 — text-referred segmentation (GSV) — port 8005
```bash
conda activate <sam3-env>
cd /path/to/sam3
# requires access to facebook/sam3 weights (gated) or an ungated mirror; set HF_TOKEN
CUDA_VISIBLE_DEVICES=1 python -m uvicorn serve_sam3_api:app \
  --host 127.0.0.1 --port 8005 --workers 1
```

### SAM — point-prompt fallback (all domains) — port 8004
```bash
conda activate <sam-env>
cd /path/to/sam
CUDA_VISIBLE_DEVICES=1 python -m uvicorn serve_sam_api:app \
  --host 127.0.0.1 --port 8004 --workers 1
```

### FLUX Fill — masked inpainting editor — port 8002
```bash
conda activate <flux-env>
cd /path/to/flux_repo
CUDA_VISIBLE_DEVICES=2 python -m uvicorn serve_flux_fill_api:app \
  --host 127.0.0.1 --port 8002 --workers 1
```

## Health checks

Every server exposes `GET /health`. Confirm a server is up before a run:
```bash
curl -s http://127.0.0.1:8002/health        # flux
curl -s http://127.0.0.1:8111/health        # perception
```

## Scorer endpoints (for reference)

- perception (8111): `POST /score`  (multipart upload)
- risk (8003), greenery (8006): `POST /score/path` (JSON `{"image_path": ...}`,
  fast — server reads the file) with `POST /score/upload` as fallback.
The client (`vida_geo/tools/clients.py`) picks the right one automatically.