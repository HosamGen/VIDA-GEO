#!/usr/bin/env bash
#SBATCH --job-name=vida_gsv_metric
#SBATCH --output=/l/users/hosam.elgendy/VIDA-GEO/slurm_logs/%x_%j.out
#SBATCH --error=/l/users/hosam.elgendy/VIDA-GEO/slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120

# One self-contained GSV metric job:
#   One allocated GPU shared by the human-perception scorer, SAM3, and SAM.
#   If the submission is explicitly overridden to allocate at least 3 GPUs,
#   each service is assigned a separate GPU instead.
# followed by the resumable VIDA-GEO metric runner.
#
# Submit this file through submit_gsv_metric.sh so METRIC and the already
# exported OPENROUTER_API_KEY are captured safely by Slurm.

set -Eeuo pipefail
umask 077

REPO="/l/users/hosam.elgendy/VIDA-GEO"
PERCEPTION_DIR="/l/users/hosam.elgendy/human-perception-place-pulse"
PERCEPTION_PYTHON="/home/hosam.elgendy/miniconda3/envs/gsv/bin/python"
SAM3_DIR="${REPO}/servers/sam3"
SAM3_PYTHON="/home/hosam.elgendy/miniconda3/envs/sam3/bin/python"
SAM_DIR="${REPO}/servers/sam"
SAM_PYTHON="/home/hosam.elgendy/miniconda3/envs/sam/bin/python"
RUNNER_PYTHON="/home/hosam.elgendy/miniconda3/envs/earth/bin/python"

METRIC="${METRIC:?METRIC was not passed by the submission wrapper}"
case "${METRIC}" in
  safety|lively|beautiful|wealthy|boring|depressing) ;;
  *)
    echo "ERROR: unsupported GSV metric: ${METRIC}" >&2
    exit 2
    ;;
esac

LIMIT_PER_METRIC="${LIMIT_PER_METRIC:-}"
if [[ -z "${LIMIT_PER_METRIC}" ]]; then
  # The final 100 for each metric includes the prior-reference outputs:
  # 9 prior + 91 new for safety/depressing; 10 prior + 90 new otherwise.
  case "${METRIC}" in
    safety|depressing) LIMIT_PER_METRIC=91 ;;
    *) LIMIT_PER_METRIC=90 ;;
  esac
fi
if [[ ! "${LIMIT_PER_METRIC}" =~ ^[1-9][0-9]*$ ]] \
    || (( LIMIT_PER_METRIC > 100 )); then
  echo "ERROR: LIMIT_PER_METRIC must be an integer from 1 to 100." >&2
  exit 2
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY was not inherited by the Slurm job." >&2
  exit 2
fi
export OPENROUTER_API_KEY

for executable in \
  "${PERCEPTION_PYTHON}" "${SAM3_PYTHON}" "${SAM_PYTHON}" "${RUNNER_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "ERROR: required Python executable is missing: ${executable}" >&2
    exit 2
  fi
done

GSV_OUTPUT_ROOT="${GSV_OUTPUT_ROOT:-${REPO}/outputs/gsv_benchmark}"
SUMMARY_CSV="${SUMMARY_CSV:-${GSV_OUTPUT_ROOT}/benchmark_results_${METRIC}.csv}"
JOB_LOG_DIR="${REPO}/slurm_logs/${METRIC}_${SLURM_JOB_ID}"
mkdir -p "${JOB_LOG_DIR}" "${GSV_OUTPUT_ROOT}"

# Slurm normally provides the job's allocated devices here. Preserve the
# actual identifiers rather than assuming the allocation is physical GPUs 0-2.
GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
if [[ -z "${GPU_LIST}" ]]; then
  echo "ERROR: Slurm did not expose CUDA_VISIBLE_DEVICES or SLURM_JOB_GPUS." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
if (( ${#GPU_IDS[@]} < 1 )); then
  echo "ERROR: this job needs at least 1 GPU, but Slurm exposed: ${GPU_LIST}" >&2
  exit 2
fi

# The default ws-ia partition exposes one GPU per node. All three lightweight
# services share it. A manual 3-GPU allocation still gets one GPU per service.
PERCEPTION_GPU="${GPU_IDS[0]}"
SAM3_GPU="${GPU_IDS[0]}"
SAM_GPU="${GPU_IDS[0]}"
if (( ${#GPU_IDS[@]} >= 3 )); then
  SAM3_GPU="${GPU_IDS[1]}"
  SAM_GPU="${GPU_IDS[2]}"
fi

SERVER_PIDS=()

cleanup() {
  local status=$?
  trap - EXIT
  echo "Stopping local model servers..."
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
echo "Input limit: ${LIMIT_PER_METRIC}"
echo "Allocated GPUs: ${GPU_LIST}"
echo "Service GPUs: perception=${PERCEPTION_GPU} sam3=${SAM3_GPU} sam=${SAM_GPU}"
echo "Output root: ${GSV_OUTPUT_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Server logs: ${JOB_LOG_DIR}"

(
  export CUDA_VISIBLE_DEVICES="${PERCEPTION_GPU}"
  cd "${PERCEPTION_DIR}"
  exec "${PERCEPTION_PYTHON}" -m uvicorn serve_gsv_api:app \
    --host 127.0.0.1 --port 8111 --workers 1
) >"${JOB_LOG_DIR}/perception.log" 2>&1 &
SERVER_PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${SAM3_GPU}"
  export SAM3_DEVICE="${SAM3_DEVICE:-cuda}"
  export SAM3_EAGER_LOAD="${SAM3_EAGER_LOAD:-1}"
  cd "${SAM3_DIR}"
  exec "${SAM3_PYTHON}" -m uvicorn serve_sam3_api:app \
    --host 127.0.0.1 --port 8005 --workers 1
) >"${JOB_LOG_DIR}/sam3.log" 2>&1 &
SERVER_PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${SAM_GPU}"
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
  for _ in $(seq 1 180); do
    if curl -fsS --connect-timeout 2 --max-time 5 \
      "${url}" >"${health_file}" 2>/dev/null; then
      echo "${name} is ready: $(tr -d '\n' <"${health_file}")"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: ${name} exited before becoming healthy." >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi
    sleep 5
  done

  echo "ERROR: ${name} did not become healthy within 15 minutes." >&2
  tail -n 80 "${log_file}" >&2 || true
  return 1
}

wait_for_health \
  "human-perception" "http://127.0.0.1:8111/health" \
  "${SERVER_PIDS[0]}" "${JOB_LOG_DIR}/perception.log" \
  "${JOB_LOG_DIR}/perception_health.json"
wait_for_health \
  "SAM3" "http://127.0.0.1:8005/health" \
  "${SERVER_PIDS[1]}" "${JOB_LOG_DIR}/sam3.log" \
  "${JOB_LOG_DIR}/sam3_health.json"
wait_for_health \
  "SAM" "http://127.0.0.1:8004/health" \
  "${SERVER_PIDS[2]}" "${JOB_LOG_DIR}/sam.log" \
  "${JOB_LOG_DIR}/sam_health.json"

echo "All services are healthy. Starting VIDA-GEO metric ${METRIC}."
cd "${REPO}"
export PYTHON_BIN="${RUNNER_PYTHON}"

"${REPO}/run_gsv_benchmark.sh" \
  --metrics "${METRIC}" \
  --limit-per-metric "${LIMIT_PER_METRIC}" \
  --output-root "${GSV_OUTPUT_ROOT}" \
  --summary-csv "${SUMMARY_CSV}"
