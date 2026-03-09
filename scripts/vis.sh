#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load shared sim env vars (source mode, no auto run).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${VIS_PYTHON_BIN:-python}"
ISAAC_SIM_PKG="$("${PYTHON_BIN}" -c "import os,isaacsim; print(os.path.dirname(isaacsim.__file__))" 2>/dev/null || true)"
if [ -z "${ISAAC_SIM_PKG}" ]; then
  echo "[vis] Failed to locate isaacsim package; cannot configure internal rclpy."
  exit 1
fi

INTERNAL_RCLPY_DIR="${ISAAC_SIM_PKG}/exts/isaacsim.ros2.bridge/jazzy/rclpy"
if [ ! -d "${INTERNAL_RCLPY_DIR}" ]; then
  echo "[vis] Missing internal rclpy path: ${INTERNAL_RCLPY_DIR}"
  exit 1
fi
export PYTHONPATH="${INTERNAL_RCLPY_DIR}:${PYTHONPATH:-}"
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

cd "${EXCAVATOR_ROOT}"
CMD=("${PYTHON_BIN}" -u -m excavator_sim.vis "$@")

echo "[vis] starting: ${CMD[*]}"
"${CMD[@]}" &
VIS_PID=$!

cleanup() {
  if kill -0 "${VIS_PID}" 2>/dev/null; then
    echo "[vis] stopping visualization..."
    kill -INT "${VIS_PID}" 2>/dev/null || true
    wait "${VIS_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[vis] press 'q' to stop"
while kill -0 "${VIS_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${VIS_PID}" || true

