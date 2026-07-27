#!/usr/bin/env bash
# Submit one immediate, one-image GSV smoke test.
#
# Usage:
#   export OPENROUTER_API_KEY='...'
#   ./slurm/submit_gsv_smoke.sh
#   ./slurm/submit_gsv_smoke.sh boring

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/run_gsv_metric.sbatch"

METRIC="${1:-boring}"
case "${METRIC}" in
  safety|lively|beautiful|wealthy|boring|depressing) ;;
  *)
    echo "Usage: $0 [safety|lively|beautiful|wealthy|boring|depressing]" >&2
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

SMOKE_OUTPUT_ROOT="${REPO}/outputs/gsv_smoke"
SMOKE_SUMMARY_CSV="${SMOKE_OUTPUT_ROOT}/smoke_results_${METRIC}.csv"
SBATCH_ARGS=(
  --parsable
  "--job-name=vida_smoke_${METRIC}"
  "--export=ALL,METRIC=${METRIC},LIMIT_PER_METRIC=1,GSV_OUTPUT_ROOT=${SMOKE_OUTPUT_ROOT},SUMMARY_CSV=${SMOKE_SUMMARY_CSV}"
)

# Optional cluster-specific overrides without editing the scripts.
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

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${BATCH_SCRIPT}")"
echo "Submitted one-image smoke test: metric=${METRIC} job_id=${JOB_ID}"
echo "Slurm log: ${REPO}/slurm_logs/vida_smoke_${METRIC}_${JOB_ID}.out"
echo "Smoke CSV: ${SMOKE_SUMMARY_CSV}"
