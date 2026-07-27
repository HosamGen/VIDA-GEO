#!/usr/bin/env bash
# Submit one self-contained GSV metric job.
#
# Usage:
#   export OPENROUTER_API_KEY='...'
#   ./slurm/submit_gsv_metric.sh beautiful now+9hours
#
# The second argument is any Slurm --begin value. It defaults to now+9hours.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
BATCH_SCRIPT="${SCRIPT_DIR}/run_gsv_metric.sbatch"

METRIC="${1:-}"
BEGIN_TIME="${2:-now+9hours}"

case "${METRIC}" in
  safety|lively|beautiful|wealthy|boring|depressing) ;;
  *)
    echo "Usage: $0 {safety|lively|beautiful|wealthy|boring|depressing} [SLURM_BEGIN]" >&2
    exit 2
    ;;
esac

# The old reference inputs are not in experiment_manifest.csv. Limit each job
# to the exact number of new inputs needed to reach 100 after legacy reuse.
case "${METRIC}" in
  safety|depressing) LIMIT_PER_METRIC=91 ;;
  *) LIMIT_PER_METRIC=90 ;;
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
  "--job-name=vida_${METRIC}"
  "--begin=${BEGIN_TIME}"
  "--export=ALL,METRIC=${METRIC},LIMIT_PER_METRIC=${LIMIT_PER_METRIC}"
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
echo "Submitted metric=${METRIC} new_inputs=${LIMIT_PER_METRIC} job_id=${JOB_ID} eligible_at=${BEGIN_TIME}"
echo "Slurm log: ${REPO}/slurm_logs/vida_${METRIC}_${JOB_ID}.out"
echo "Metric CSV: ${REPO}/outputs/gsv_benchmark/benchmark_results_${METRIC}.csv"
