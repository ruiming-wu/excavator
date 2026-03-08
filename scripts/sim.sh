#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

_setup_sim_env() {
  local script_path
  script_path="${BASH_SOURCE[0]}"

  unset AMENT_PREFIX_PATH
  unset COLCON_PREFIX_PATH
  export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

  export EXCAVATOR_ROOT="$(cd "$(dirname "${script_path}")/.." && pwd)"
  export PYTHONPATH="${EXCAVATOR_ROOT}/src:${PYTHONPATH:-}"
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

  local isaac_sim_pkg
  isaac_sim_pkg="$(python -c "import os,isaacsim; print(os.path.dirname(isaacsim.__file__))" 2>/dev/null || true)"
  if [ -n "${isaac_sim_pkg}" ] && [ -d "${isaac_sim_pkg}/exts/isaacsim.ros2.bridge/jazzy/lib" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${isaac_sim_pkg}/exts/isaacsim.ros2.bridge/jazzy/lib"
  fi

  export EXCAVATOR_ASSETS_DIR="${EXCAVATOR_ASSETS_DIR:-${EXCAVATOR_ROOT}/assets}"
  export EXCAVATOR_CONFIGS_DIR="${EXCAVATOR_CONFIGS_DIR:-${EXCAVATOR_ROOT}/configs}"
  export EXCAVATOR_DATA_RAW_DIR="${EXCAVATOR_DATA_RAW_DIR:-${EXCAVATOR_ROOT}/data/raw}"
  export EXCAVATOR_DATA_PROCESSED_DIR="${EXCAVATOR_DATA_PROCESSED_DIR:-${EXCAVATOR_ROOT}/data/processed}"
  export EXCAVATOR_RUNS_DIR="${EXCAVATOR_RUNS_DIR:-${EXCAVATOR_ROOT}/runs}"
  export EXCAVATOR_LOGS_DIR="${EXCAVATOR_LOGS_DIR:-${EXCAVATOR_ROOT}/logs}"

  export EXCAVATOR_ASSET_PATH="${EXCAVATOR_ASSET_PATH:-${EXCAVATOR_ASSETS_DIR}/excavator_4dof/excavator_4dof.urdf}"
  export EXCAVATOR_SCENE_USD="${EXCAVATOR_SCENE_USD:-}"
  export EXCAVATOR_SEED="${EXCAVATOR_SEED:-}"

  mkdir -p \
    "${EXCAVATOR_DATA_RAW_DIR}" \
    "${EXCAVATOR_DATA_PROCESSED_DIR}" \
    "${EXCAVATOR_RUNS_DIR}" \
    "${EXCAVATOR_LOGS_DIR}"

  echo "[sim] EXCAVATOR_ROOT=${EXCAVATOR_ROOT}"
  echo "[sim] EXCAVATOR_ASSET_PATH=${EXCAVATOR_ASSET_PATH}"
}

# If sourced: only export env vars.
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _setup_sim_env
  return 0
fi

_setup_sim_env

cd "${EXCAVATOR_ROOT}"
PYTHON_BIN="${SIM_PYTHON_BIN:-python}"
CMD=("${PYTHON_BIN}" -u -m excavator_sim.run_sim "$@")

echo "[sim] starting: ${CMD[*]}"
"${CMD[@]}" &
SIM_PID=$!

cleanup() {
  if kill -0 "${SIM_PID}" 2>/dev/null; then
    echo "[sim] stopping simulator..."
    kill -INT "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[sim] press 'q' to stop"
while kill -0 "${SIM_PID}" 2>/dev/null; do
  if read -r -s -n 1 -t 0.2 key; then
    if [ "${key}" = "q" ] || [ "${key}" = "Q" ]; then
      cleanup
      exit 0
    fi
  fi
done

wait "${SIM_PID}" || true
