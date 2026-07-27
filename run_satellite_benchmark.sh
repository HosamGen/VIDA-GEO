#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/l/users/hosam.elgendy/VIDA-GEO"
PYTHON_BIN="${PYTHON_BIN:-/home/hosam.elgendy/miniconda3/envs/earth/bin/python}"

cd "${REPO}"
exec "${PYTHON_BIN}" -u "${REPO}/scripts/run_satellite_benchmark.py" "$@"
