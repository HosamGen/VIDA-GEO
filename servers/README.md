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

> Note: any `*_cli.py` / `infer_*.py` / `inference_*.py` file in a server folder is
> an **optional** standalone command-line tool for testing that model on a single
> image. The servers do not import them — you can delete them if you want a leaner repo.

## Launch commands (edit GPUs/paths to your machine)

### Perception scorer — Place Pulse (GSV) — port 8111
Upstream model: https://github.com/strawmelon11/human-perception-place-pulse

Weights: auto-downloaded from HuggingFace (`Jiani11/human-perception-place-pulse`) on first run — no manual checkpoint needed.

`servers/perception/serve_gsv_api.py` is **self-contained**, just install deps and run.

(`inference_perception.py` is an optional standalone CLI for scoring without the server.)
```bash
conda activate <perception-env>     # needs: torch torchvision pillow fastapi huggingface_hub
cd servers/perception
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_gsv_api:app \
  --host 127.0.0.1 --port 8111 --workers 1
```

### Risk scorer — BetaRisk (satellite) — port 8003
Use this fork (no file-copying needed): **https://github.com/HosamGen/BetaRisk**  (forked from https://github.com/FOURM-LAB/BetaRisk)

Checkpoint: this project uses a single-scale retrain (`epoch_15`).
Download: [checkpoint](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hosam_elgendy_mbzuai_ac_ae/IQCPZMkvD6qPTr6RXlyYVgOZAdhDhsaDa6MUdbjuPwYAgq4?e=hhKKn7)
```bash
git clone https://github.com/HosamGen/BetaRisk.git
cd BetaRisk
conda activate <risk-env>                          # see the fork's README for env setup
export RISK_CHECKPOINT=/abs/path/to/epoch_15.pth   # REQUIRED (no default)
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_risk_api:app \
  --host 127.0.0.1 --port 8003 --workers 1
```

### Greenery scorer — OEM-Lightweight (satellite) — port 8006
Upstream: https://github.com/cliffbb/oem-lightweight
Files in `servers/greenery/` (`serve_greenery_api.py`, `greenery_score.py`) are the
wrappers added on top of oem-lightweight — copy them into your clone so its
`sparsemask_api` / `fasterseg_api` modules are importable.

```bash
conda activate <greenery-env>
cd /path/to/oem-lightweight
export OEM_REPO=$(pwd)
export OEM_MODEL=sparsemask
export OEM_ARCH=models/SparseMask/mask_thres_0.001.npy
export OEM_WEIGHTS=models/SparseMask/checkpoint_63750.pth.tar
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_greenery_api:app \
  --host 127.0.0.1 --port 8006 --workers 1
```

### LISAt — text-referred segmentation (satellite) — port 8001
Upstream: https://github.com/lisat-bair/LISAt_code
Files in `servers/lisat/` (`serve_lisat_api.py`, `lisat_predictor.py`, `infer_lisat.py`)
are the wrappers — copy them into your LISAt_code clone so its `model/`, `dataloaders/`,
and `utils` modules are importable. 

(`infer_lisat.py` is an optional standalone CLI.)

Model: set `LISAT_MODEL_PATH` to the LISAt-7b checkpoint dir (local path, or the HF
id `jquenum/LISAt-7b`).
```bash
conda activate <lisat-env>
cd /path/to/LISAt_code               # has model/, dataloaders/, utils
export LISAT_MODEL_PATH=checkpoints/LISAt-7b    # or jquenum/LISAt-7b
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_lisat_api:app \
  --host 127.0.0.1 --port 8001 --workers 1
```

### SAM3 — text-referred segmentation (GSV) — port 8005
Upstream: https://github.com/facebookresearch/sam3
Files in `servers/sam3/` (`serve_sam3_api.py`, `sam3_predictor.py`,
`sam3_text_segment_cli.py`) are the wrappers — copy them where the `sam3` package
is importable (install SAM3 per its repo). 
Weights download from HuggingFace on
first build (the SAM3 model is gated — accept its license and set `HF_TOKEN`).

```bash
conda activate <sam3-env>
cd /path/to/sam3                     # where the sam3 package is importable
export HF_TOKEN=hf_...               # for gated SAM3 weights
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_sam3_api:app \
  --host 127.0.0.1 --port 8005 --workers 1
```

### SAM — point-prompt fallback (all domains) — port 8004
Upstream: https://github.com/facebookresearch/segment-anything
Files in `servers/sam/` (`serve_sam_api.py`, `sam_predictor.py`, `sam_segment.py`)
are standalone — they only need the `segment_anything` package importable, NOT the
repo cloned. 

Either:
- `pip install git+https://github.com/facebookresearch/segment-anything.git`, or
- clone the repo and `pip install -e .`

Checkpoint: The server auto-downloads the public checkpoint on first start if it's missing 
(set `SAM_MODEL_TYPE` to pick vit_h/l/b; default vit_h). 

```bash
conda activate <sam-env>            # needs: segment-anything torch torchvision opencv-python fastapi
cd servers/sam
export SAM_MODEL_TYPE=vit_h          # vit_h | vit_l | vit_b
export SAM_CHECKPOINT=checkpoints/sam_vit_h_4b8939.pth   # auto-downloaded here if absent
CUDA_VISIBLE_DEVICES=X python -m uvicorn serve_sam_api:app \
  --host 127.0.0.1 --port 8004 --workers 1
```

### FLUX Fill — masked inpainting editor — port 8002
Upstream model: https://github.com/black-forest-labs/flux  (FLUX.1-Fill-dev)
Files in `servers/flux/` (`serve_flux_fill_api.py`, `flux_fill_predictor.py`) are the
wrappers. The predictor has **two backends** (set `FLUX_BACKEND`):
- `diffusers` (easiest): uses HF `diffusers.FluxFillPipeline` — just
  `pip install diffusers transformers accelerate`, weights auto-download from HF.
  No upstream repo clone needed.
- `flux` (default): uses the official black-forest-labs/flux internals — clone that
  repo and put it on `PYTHONPATH`.

Weights: `black-forest-labs/FLUX.1-Fill-dev` (gated on HF — accept its license and
set `HF_TOKEN` so the download works). 
FLUX is only needed when FLUX editing is enabled (skip it entirely if you always run the agent with `--disable_flux`).

Recommended to run this on a different GPU than the other tools.

```bash
conda activate <flux-env>           # diffusers backend: torch diffusers transformers accelerate fastapi
cd servers/flux
export HF_TOKEN=hf_...               # for the gated FLUX.1-Fill-dev download
export FLUX_BACKEND=diffusers        # or 'flux' for the official repo backend
# optional: export FLUX_QUANT=nf4    # lower VRAM via pre-quantized NF4
CUDA_VISIBLE_DEVICES=Y python -m uvicorn serve_flux_fill_api:app \
  --host 127.0.0.1 --port 8002 --workers 1
```

Optional warmup to pay the cold start once (the first real edit is otherwise slow):
`export FLUX_EAGER_LOAD=1 FLUX_WARMUP=1`, or `POST /warmup` after startup.

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
