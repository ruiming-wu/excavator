#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load shared sim env vars (source mode, no auto run).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${RECORDER_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"
_setup_internal_rclpy_env "record" "${PYTHON_BIN}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u -m excavator_sim.record "$@")

echo "[record] starting: ${CMD[*]}"
"${CMD[@]}" &
RECORD_PID=$!

cleanup() {
  if kill -0 "${RECORD_PID}" 2>/dev/null; then
    echo "[record] stopping recorder..."
    kill -INT "${RECORD_PID}" 2>/dev/null || true
    wait "${RECORD_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[record] press 'q' to stop"
while kill -0 "${RECORD_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${RECORD_PID}" || true
