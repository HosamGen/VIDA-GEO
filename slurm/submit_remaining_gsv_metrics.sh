#!/usr/bin/env bash
# Submit one pair of GSV metrics. This wrapper never submits more than two jobs.
#
# Usage:
#   export OPENROUTER_API_KEY='...'
#   ./slurm/submit_remaining_gsv_metrics.sh
#   ./slurm/submit_remaining_gsv_metrics.sh beautiful wealthy
#   ./slurm/submit_remaining_gsv_metrics.sh beautiful wealthy now+9hours
#
# The default pair is boring + depressing so the promoted boring smoke result
# can be resumed as part of the full boring run. The default delay is 9 hours.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRIC_ONE="${1:-boring}"
METRIC_TWO="${2:-depressing}"
BEGIN_TIME="${3:-now+4hours}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: export OPENROUTER_API_KEY before submitting." >&2
  exit 2
fi

"${SCRIPT_DIR}/submit_two_gsv_metrics.sh" \
  "${METRIC_ONE}" "${METRIC_TWO}" "${BEGIN_TIME}"
