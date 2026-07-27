#!/usr/bin/env bash
#SBATCH --job-name=vida_satellite
#SBATCH --output=/l/users/hosam.elgendy/VIDA-GEO/slurm_logs/%x_%j.out
#SBATCH --error=/l/users/hosam.elgendy/VIDA-GEO/slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120

# One self-contained satellite metric job. It starts the metric scorer, LISAt,
# SAM3, and SAM on its allocated GPU, then runs the resumable VIDA-GEO batch.

set -Eeuo pipefail
umask 077

REPO="/l/users/hosam.elgendy/VIDA-GEO"
RUNNER_PYTHON="/home/hosam.elgendy/miniconda3/envs/earth/bin/python"

LISAT_DIR="/l/users/hosam.elgendy/LISAt_code"
LISAT_PYTHON="/home/hosam.elgendy/miniconda3/envs/lisat/bin/python"
SAM3_DIR="${REPO}/servers/sam3"
SAM3_PYTHON="/home/hosam.elgendy/miniconda3/envs/sam3/bin/python"
SAM_DIR="${REPO}/servers/sam"
SAM_PYTHON="/home/hosam.elgendy/miniconda3/envs/sam/bin/python"

BETARISK_DIR="/l/users/hosam.elgendy/BetaRisk"
RISK_PYTHON="/home/hosam.elgendy/miniconda3/envs/beta/bin/python"
GREENERY_DIR="/l/users/hosam.elgendy/oem-lightweight"
GREENERY_PYTHON="/home/hosam.elgendy/miniconda3/envs/green/bin/python"

METRIC="${METRIC:?METRIC was not passed by the submission wrapper}"
case "${METRIC}" in
  greenery|road_risk) ;;
  *)
    echo "ERROR: unsupported satellite metric: ${METRIC}" >&2
    exit 2
    ;;
esac

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY was not inherited by the Slurm job." >&2
  exit 2
fi
export OPENROUTER_API_KEY

for executable in \
  "${RUNNER_PYTHON}" "${LISAT_PYTHON}" "${SAM3_PYTHON}" "${SAM_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: required Python executable is missing: ${executable}" >&2
    exit 2
  fi
done
if [[ "${METRIC}" == "greenery" && ! -x "${GREENERY_PYTHON}" ]]; then
  echo "ERROR: greenery Python is missing: ${GREENERY_PYTHON}" >&2
  exit 2
fi
if [[ "${METRIC}" == "road_risk" && ! -x "${RISK_PYTHON}" ]]; then
  echo "ERROR: BetaRisk Python is missing: ${RISK_PYTHON}" >&2
  exit 2
fi

GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
if [[ -z "${GPU_LIST}" ]]; then
  echo "ERROR: Slurm did not expose an allocated GPU." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
GPU_ID="${GPU_IDS[0]}"

OUTPUT_ROOT="${SATELLITE_OUTPUT_ROOT:-${REPO}/outputs/satellite_benchmark}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/benchmark_results_${METRIC}.csv}"
JOB_LOG_DIR="${REPO}/slurm_logs/satellite_${METRIC}_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_ROOT}" "${JOB_LOG_DIR}"

export HF_HOME="${HF_HOME:-/l/users/hosam.elgendy/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SERVER_PIDS=()

cleanup() {
  local status=$?
  trap - EXIT
  echo "Stopping local satellite model servers..."
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Metric: ${METRIC}"
echo "Allocated GPUs: ${GPU_LIST}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "LISAt model: ${LISAT_MODEL_PATH:-jquenum/LISAt-7b}"

if [[ "${METRIC}" == "greenery" ]]; then
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export OEM_REPO="${GREENERY_DIR}"
    export OEM_MODEL="sparsemask"
    export OEM_ARCH="models/SparseMask/mask_thres_0.001.npy"
    export OEM_WEIGHTS="models/SparseMask/checkpoint_63750.pth.tar"
    export OEM_DEVICE="cuda"
    export OEM_INPUT_SIZE="1024"
    export OEM_USE_TTA="0"
    export OEM_USE_FP16="0"
    cd "${GREENERY_DIR}"
    exec "${GREENERY_PYTHON}" -m uvicorn serve_greenery_api:app \
      --host 127.0.0.1 --port 8006 --workers 1
  ) >"${JOB_LOG_DIR}/scorer.log" 2>&1 &
else
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    export RISK_CHECKPOINT="${RISK_CHECKPOINT:-${BETARISK_DIR}/old_ckpts/epoch_15.pth}"
    export RISK_MODEL_CLASS="ResNet50_MultiScale_Probabilistic"
    export RISK_DEVICE="cuda"
    cd "${REPO}/servers/risk"
    exec "${RISK_PYTHON}" -m uvicorn serve_risk_offline_api:app \
      --host 127.0.0.1 --port 8003 --workers 1
  ) >"${JOB_LOG_DIR}/scorer.log" 2>&1 &
fi
SERVER_PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export LISAT_MODEL_PATH="${LISAT_MODEL_PATH:-jquenum/LISAt-7b}"
  export LISAT_DEVICE="${LISAT_DEVICE:-cuda:0}"
  export LISAT_DEVICE_MAP="${LISAT_DEVICE_MAP:-auto}"
  export LISAT_DTYPE="${LISAT_DTYPE:-float16}"
  export LISAT_IMAGE_SIZE="${LISAT_IMAGE_SIZE:-1024}"
  export LISAT_AUTOCAST="${LISAT_AUTOCAST:-1}"
  # Leave room on a 32 GiB RTX 5000 Ada for SAM3, SAM, and the scorer.
  export LISAT_MAX_GPU_MEMORY="${LISAT_MAX_GPU_MEMORY:-18GiB}"
  export LISAT_MAX_CPU_MEMORY="${LISAT_MAX_CPU_MEMORY:-64GiB}"
  export LISAT_OFFLOAD_FOLDER="${LISAT_OFFLOAD_FOLDER:-${SLURM_TMPDIR:-/tmp}/vida_lisat_${SLURM_JOB_ID}}"
  mkdir -p "${LISAT_OFFLOAD_FOLDER}"
  cd "${LISAT_DIR}"
  exec "${LISAT_PYTHON}" -m uvicorn serve_lisat_api:app \
    --host 127.0.0.1 --port 8001 --workers 1
) >"${JOB_LOG_DIR}/lisat.log" 2>&1 &
SERVER_PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
  export SAM3_EAGER_LOAD="${SAM3_EAGER_LOAD:-1}"
  cd "${SAM3_DIR}"
  exec "${SAM3_PYTHON}" -m uvicorn serve_sam3_api:app \
    --host 127.0.0.1 --port 8005 --workers 1
) >"${JOB_LOG_DIR}/sam3.log" 2>&1 &
SERVER_PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export SAM_DEVICE="${SAM_DEVICE:-cuda}"
  export SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
  export SAM_CHECKPOINT="${SAM_CHECKPOINT:-${SAM_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
  cd "${SAM_DIR}"
  exec "${SAM_PYTHON}" -m uvicorn serve_sam_api:app \
    --host 127.0.0.1 --port 8004 --workers 1
) >"${JOB_LOG_DIR}/sam.log" 2>&1 &
SERVER_PIDS+=("$!")

wait_for_health() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local log_file="$4"
  local health_file="$5"

  echo "Waiting for ${name}: ${url}"
  for _ in $(seq 1 720); do
    if curl -fsS --connect-timeout 2 --max-time 5 \
      "${url}" >"${health_file}" 2>/dev/null; then
      echo "${name} is ready: $(tr -d '\n' <"${health_file}")"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${name} exited before becoming healthy." >&2
      tail -n 100 "${log_file}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "ERROR: ${name} did not become healthy within 60 minutes." >&2
  tail -n 100 "${log_file}" >&2 || true
  return 1
}

if [[ "${METRIC}" == "greenery" ]]; then
  SCORER_URL="http://127.0.0.1:8006/health"
  SCORER_NAME="greenery"
else
  SCORER_URL="http://127.0.0.1:8003/health"
  SCORER_NAME="risk"
fi

wait_for_health \
  "${SCORER_NAME}" "${SCORER_URL}" \
  "${SERVER_PIDS[0]}" "${JOB_LOG_DIR}/scorer.log" \
  "${JOB_LOG_DIR}/scorer_health.json"
wait_for_health \
  "LISAt" "http://127.0.0.1:8001/health" \
  "${SERVER_PIDS[1]}" "${JOB_LOG_DIR}/lisat.log" \
  "${JOB_LOG_DIR}/lisat_health.json"
wait_for_health \
  "SAM3" "http://127.0.0.1:8005/health" \
  "${SERVER_PIDS[2]}" "${JOB_LOG_DIR}/sam3.log" \
  "${JOB_LOG_DIR}/sam3_health.json"
wait_for_health \
  "SAM" "http://127.0.0.1:8004/health" \
  "${SERVER_PIDS[3]}" "${JOB_LOG_DIR}/sam.log" \
  "${JOB_LOG_DIR}/sam_health.json"

echo "All satellite services are healthy. Starting ${METRIC}."
cd "${REPO}"
export PYTHON_BIN="${RUNNER_PYTHON}"

"${REPO}/run_satellite_benchmark.sh" \
  --metric "${METRIC}" \
  --output-root "${OUTPUT_ROOT}" \
  --summary-csv "${SUMMARY_CSV}"
