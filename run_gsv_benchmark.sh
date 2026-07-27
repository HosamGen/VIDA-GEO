#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hosam.elgendy/miniconda3/envs/earth/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to a VIDA-GEO environment with requirements.txt installed." >&2
    exit 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/scripts/run_gsv_benchmark.py" "$@"
