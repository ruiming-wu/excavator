#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load sim env vars (source mode, no auto run).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${TELEOP_PYTHON_BIN:-${EXCAVATOR_PYTHON_BIN}}"
_setup_internal_rclpy_env "teleop" "${PYTHON_BIN}"

cd "${EXCAVATOR_ROOT}"
URDF_PATH="${EXCAVATOR_ASSET_PATH:-${EXCAVATOR_ROOT}/assets/excavator/excavator_4dof.urdf}"
CMD=("${PYTHON_BIN}" -u -m excavator_sim.teleop --urdf "${URDF_PATH}" "$@")

echo "[teleop] starting: ${CMD[*]}"
"${CMD[@]}" &
TELEOP_PID=$!

cleanup() {
  if kill -0 "${TELEOP_PID}" 2>/dev/null; then
    echo "[teleop] stopping teleop..."
    kill -INT "${TELEOP_PID}" 2>/dev/null || true
    wait "${TELEOP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[teleop] press 'q' to stop"
while kill -0 "${TELEOP_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${TELEOP_PID}" || true
