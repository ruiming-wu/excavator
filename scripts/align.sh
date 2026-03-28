#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${ALIGN_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u data/align.py "$@")

echo "[align] starting: ${CMD[*]}"
"${CMD[@]}"
