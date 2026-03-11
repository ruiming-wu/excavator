#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${REPLAY_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u data/replay.py "$@")

echo "[replay] starting: ${CMD[*]}"
"${CMD[@]}" &
REPLAY_PID=$!

cleanup() {
  if kill -0 "${REPLAY_PID}" 2>/dev/null; then
    echo "[replay] stopping replay..."
    kill -INT "${REPLAY_PID}" 2>/dev/null || true
    wait "${REPLAY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[replay] press 'q' in window or terminal to stop"
while kill -0 "${REPLAY_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${REPLAY_PID}" || true
