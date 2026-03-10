#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load sim env vars (source mode, no auto run).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sim.sh"

PYTHON_BIN="${TELEOP_PYTHON_BIN:-python}"
ISAAC_SIM_PKG="$("${PYTHON_BIN}" -c "import os,isaacsim; print(os.path.dirname(isaacsim.__file__))" 2>/dev/null || true)"
if [ -z "${ISAAC_SIM_PKG}" ]; then
  echo "[teleop] Failed to locate isaacsim package; cannot configure internal rclpy."
  exit 1
fi

INTERNAL_RCLPY_DIR="${ISAAC_SIM_PKG}/exts/isaacsim.ros2.bridge/jazzy/rclpy"
if [ ! -d "${INTERNAL_RCLPY_DIR}" ]; then
  echo "[teleop] Missing internal rclpy path: ${INTERNAL_RCLPY_DIR}"
  exit 1
fi
export PYTHONPATH="${INTERNAL_RCLPY_DIR}:${PYTHONPATH:-}"
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH

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
