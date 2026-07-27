#!/usr/bin/env bash
# Submit two independent metric jobs with the same delayed start.
#
# Defaults:
#   metric 1: beautiful
#   metric 2: wealthy
#   eligible: now+9hours
#
# Usage:
#   export OPENROUTER_API_KEY='...'
#   ./slurm/submit_two_gsv_metrics.sh
#   ./slurm/submit_two_gsv_metrics.sh boring depressing now+9hours

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRIC_ONE="${1:-beautiful}"
METRIC_TWO="${2:-wealthy}"
BEGIN_TIME="${3:-now+4hours}"

if [[ "${METRIC_ONE}" == "${METRIC_TWO}" ]]; then
  echo "ERROR: choose two different metrics." >&2
  exit 2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: export OPENROUTER_API_KEY before submitting." >&2
  exit 2
fi

"${SCRIPT_DIR}/submit_gsv_metric.sh" "${METRIC_ONE}" "${BEGIN_TIME}"
"${SCRIPT_DIR}/submit_gsv_metric.sh" "${METRIC_TWO}" "${BEGIN_TIME}"

echo "Both jobs were submitted. --begin makes them eligible at ${BEGIN_TIME};"
echo "their actual start still depends on Slurm resource availability."
