#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${TRAIN_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u -m excavator_policy.train --config src/excavator_policy/config.yaml "$@")

echo "[train] starting: ${CMD[*]}"
"${CMD[@]}" &
TRAIN_PID=$!

cleanup() {
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[train] stopping training..."
    kill -INT "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[train] press 'q' to stop"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${TRAIN_PID}" || true
