#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${CHECK_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u data/check.py "$@")

echo "[check] starting: ${CMD[*]}"
"${CMD[@]}"
