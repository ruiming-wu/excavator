#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${EVAL_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"
_setup_internal_rclpy_env "eval" "${PYTHON_BIN}"

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u -m excavator_policy.eval "$@")

echo "[eval] starting: ${CMD[*]}"
"${CMD[@]}" &
EVAL_PID=$!

cleanup() {
  if kill -0 "${EVAL_PID}" 2>/dev/null; then
    echo "[eval] stopping evaluator..."
    kill -INT "${EVAL_PID}" 2>/dev/null || true
    wait "${EVAL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[eval] press 'q' to stop"
while kill -0 "${EVAL_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${EVAL_PID}" || true
