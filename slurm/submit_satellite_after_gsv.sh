#!/usr/bin/env bash
# Submit greenery and road-risk so both wait for Beautiful and Wealthy.
#
# Defaults are the current Beautiful/Wealthy jobs discovered on 2026-07-27:
#   Beautiful 152406, Wealthy 152407
#
# Usage:
#   export OPENROUTER_API_KEY='...'
#   ./slurm/submit_satellite_after_gsv.sh
#   ./slurm/submit_satellite_after_gsv.sh 152406 152407 afterany

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAUTIFUL_JOB_ID="${1:-152406}"
WEALTHY_JOB_ID="${2:-152407}"
DEPENDENCY_TYPE="${3:-afterany}"

if [[ ! "${BEAUTIFUL_JOB_ID}" =~ ^[0-9]+$ ]] \
    || [[ ! "${WEALTHY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: parent job IDs must be numeric." >&2
  exit 2
fi
case "${DEPENDENCY_TYPE}" in
  afterany|afterok) ;;
  *)
    echo "ERROR: dependency type must be afterany or afterok." >&2
    exit 2
    ;;
esac
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: export OPENROUTER_API_KEY before submitting." >&2
  exit 2
fi

DEPENDENCY="${DEPENDENCY_TYPE}:${BEAUTIFUL_JOB_ID}:${WEALTHY_JOB_ID}"
echo "Both satellite jobs will use dependency ${DEPENDENCY}."
"${SCRIPT_DIR}/submit_satellite_metric.sh" greenery "${DEPENDENCY}"
"${SCRIPT_DIR}/submit_satellite_metric.sh" road_risk "${DEPENDENCY}"
echo "Both satellite jobs were submitted and may run concurrently after the parents."
