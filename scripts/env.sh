#!/usr/bin/env bash
set -e
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

# Usage:
#   source scripts/env.sh
# Assumes your conda env is already activated externally.

if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "[env] This file should be sourced, not executed."
  echo "[env] Correct usage: source scripts/env.sh"
fi

# Isaac Sim internal ROS2 bridge runtime (avoid loading system ROS Python into conda).
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# Project root and python import path
SCRIPT_PATH="${0}"
if [ -n "${BASH_VERSION:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
export EXCAVATOR_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
export PYTHONPATH="${EXCAVATOR_ROOT}/src:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Add Isaac Sim bundled ROS2 bridge libs (required for internal rclpy load in many setups).
ISAAC_SIM_PKG="$(python -c "import os,isaacsim; print(os.path.dirname(isaacsim.__file__))" 2>/dev/null || true)"
if [ -n "${ISAAC_SIM_PKG}" ] && [ -d "${ISAAC_SIM_PKG}/exts/isaacsim.ros2.bridge/jazzy/lib" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ISAAC_SIM_PKG}/exts/isaacsim.ros2.bridge/jazzy/lib"
fi

# Data / outputs (all overridable from outside)
export EXCAVATOR_ASSETS_DIR="${EXCAVATOR_ASSETS_DIR:-${EXCAVATOR_ROOT}/assets}"
export EXCAVATOR_CONFIGS_DIR="${EXCAVATOR_CONFIGS_DIR:-${EXCAVATOR_ROOT}/configs}"
export EXCAVATOR_DATA_RAW_DIR="${EXCAVATOR_DATA_RAW_DIR:-${EXCAVATOR_ROOT}/data/raw}"
export EXCAVATOR_DATA_PROCESSED_DIR="${EXCAVATOR_DATA_PROCESSED_DIR:-${EXCAVATOR_ROOT}/data/processed}"
export EXCAVATOR_RUNS_DIR="${EXCAVATOR_RUNS_DIR:-${EXCAVATOR_ROOT}/runs}"
export EXCAVATOR_LOGS_DIR="${EXCAVATOR_LOGS_DIR:-${EXCAVATOR_ROOT}/logs}"

# Optional runtime parameters
export EXCAVATOR_ASSET_PATH="${EXCAVATOR_ASSET_PATH:-${EXCAVATOR_ASSETS_DIR}/excavator_4dof/excavator_4dof.urdf}"
export EXCAVATOR_SCENE_USD="${EXCAVATOR_SCENE_USD:-}"
# Leave empty by default so each run can randomize scene unless user pins a seed.
export EXCAVATOR_SEED="${EXCAVATOR_SEED:-}"

mkdir -p \
  "${EXCAVATOR_DATA_RAW_DIR}" \
  "${EXCAVATOR_DATA_PROCESSED_DIR}" \
  "${EXCAVATOR_RUNS_DIR}" \
  "${EXCAVATOR_LOGS_DIR}"

echo "[env] EXCAVATOR_ROOT=${EXCAVATOR_ROOT}"
echo "[env] EXCAVATOR_ASSET_PATH=${EXCAVATOR_ASSET_PATH}"
