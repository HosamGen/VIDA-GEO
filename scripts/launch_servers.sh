#!/usr/bin/env bash
# Launch the model servers VIDA-GEO talks to. Edit GPU/port assignment to your box.
# Only start the services your domain needs:
#   GSV domains:       perception, sam3, sam, (flux unless --disable_flux)
#   satellite domains: risk OR greenery, lisat, sam, (flux unless --disable_flux)
#
# Each server runs in its own conda env (the upstream model's env), pinned to a GPU.
# Ports here must match configs/services.yaml.
set -euo pipefail

# --- edit these to match your machine -------------------------------------
GPU_PERCEPTION=${GPU_PERCEPTION:-4}
GPU_RISK=${GPU_RISK:-1}
GPU_GREENERY=${GPU_GREENERY:-1}
GPU_LISAT=${GPU_LISAT:-1}
GPU_SAM=${GPU_SAM:-1}
GPU_SAM3=${GPU_SAM3:-1}
GPU_FLUX=${GPU_FLUX:-2}        # FLUX wants a GPU to itself
SERVERS_DIR=${SERVERS_DIR:-servers}
# --------------------------------------------------------------------------

launch () {  # name gpu port module env
  local name=$1 gpu=$2 port=$3 module=$4 env=$5
  echo "starting $name on GPU $gpu :$port (env=$env)"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n "$env" \
    python -m uvicorn "$SERVERS_DIR.$module:app" \
    --host 127.0.0.1 --port "$port" --workers 1 &
}

# name           gpu              port  module                env
launch perception $GPU_PERCEPTION 8111  serve_gsv_api         perception
launch sam3       $GPU_SAM3       8005  serve_sam3_api        sam3
launch sam        $GPU_SAM        8004  serve_sam_api         sam
launch flux       $GPU_FLUX       8002  serve_flux_fill_api   flux
# satellite-only (start when running road_safety / greenery):
# launch risk     $GPU_RISK       8003  serve_risk_api        beta
# launch greenery $GPU_GREENERY   8006  serve_greenery_api    greenery
# launch lisat    $GPU_LISAT      8001  serve_lisat_api       lisat

echo "servers launching in background; check /health on each port."
wait
