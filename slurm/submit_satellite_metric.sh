#!/usr/bin/env bash
# Submit one satellite metric job, optionally with an existing Slurm dependency.
#
# Usage:
#   ./slurm/submit_satellite_metric.sh greenery afterany:152406:152407

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/run_satellite_metric.sbatch"
METRIC="${1:-}"
DEPENDENCY="${2:-}"

case "${METRIC}" in
  greenery|road_risk) ;;
  *)
    echo "Usage: $0 {greenery|road_risk} [SLURM_DEPENDENCY]" >&2
    exit 2
    ;;
esac
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: export OPENROUTER_API_KEY before submitting." >&2
  exit 2
fi
if [[ ! -f "${BATCH_SCRIPT}" ]]; then
  echo "ERROR: missing batch script: ${BATCH_SCRIPT}" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available in this shell." >&2
  exit 2
fi

mkdir -p "${REPO}/slurm_logs"
SBATCH_ARGS=(
  --parsable
  "--job-name=vida_sat_${METRIC}"
  "--export=ALL,METRIC=${METRIC}"
)
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=("--dependency=${DEPENDENCY}")
fi
if [[ -n "${VIDA_SLURM_PARTITION:-}" ]]; then
  SBATCH_ARGS+=("--partition=${VIDA_SLURM_PARTITION}")
fi
if [[ -n "${VIDA_SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=("--account=${VIDA_SLURM_ACCOUNT}")
fi
if [[ -n "${VIDA_SLURM_QOS:-}" ]]; then
  SBATCH_ARGS+=("--qos=${VIDA_SLURM_QOS}")
fi
if [[ -n "${VIDA_SLURM_CONSTRAINT:-}" ]]; then
  SBATCH_ARGS+=("--constraint=${VIDA_SLURM_CONSTRAINT}")
fi
if [[ -n "${VIDA_SLURM_EXCLUDE:-}" ]]; then
  SBATCH_ARGS+=("--exclude=${VIDA_SLURM_EXCLUDE}")
fi

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${BATCH_SCRIPT}")"
echo "Submitted metric=${METRIC} job_id=${JOB_ID} dependency=${DEPENDENCY:-none}"
echo "Slurm log: ${REPO}/slurm_logs/vida_sat_${METRIC}_${JOB_ID}.out"
echo "CSV: ${REPO}/outputs/satellite_benchmark/benchmark_results_${METRIC}.csv"
