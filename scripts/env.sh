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

# ROS2 Jazzy
ROS_SETUP="/opt/ros/jazzy/setup.sh"
if [ -n "${BASH_VERSION:-}" ]; then
  ROS_SETUP="/opt/ros/jazzy/setup.bash"
fi

if [ -f "${ROS_SETUP}" ]; then
  . "${ROS_SETUP}"
else
  echo "[env] Missing ROS2 setup: ${ROS_SETUP}"
  return 1 2>/dev/null || exit 1
fi

# Project root and python import path
SCRIPT_PATH="${0}"
if [ -n "${BASH_VERSION:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
export EXCAVATOR_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
export PYTHONPATH="${EXCAVATOR_ROOT}/src:${PYTHONPATH:-}"

# Data / outputs (all overridable from outside)
export EXCAVATOR_ASSETS_DIR="${EXCAVATOR_ASSETS_DIR:-${EXCAVATOR_ROOT}/assets}"
export EXCAVATOR_CONFIGS_DIR="${EXCAVATOR_CONFIGS_DIR:-${EXCAVATOR_ROOT}/configs}"
export EXCAVATOR_DATA_RAW_DIR="${EXCAVATOR_DATA_RAW_DIR:-${EXCAVATOR_ROOT}/data/raw}"
export EXCAVATOR_DATA_PROCESSED_DIR="${EXCAVATOR_DATA_PROCESSED_DIR:-${EXCAVATOR_ROOT}/data/processed}"
export EXCAVATOR_RUNS_DIR="${EXCAVATOR_RUNS_DIR:-${EXCAVATOR_ROOT}/runs}"
export EXCAVATOR_LOGS_DIR="${EXCAVATOR_LOGS_DIR:-${EXCAVATOR_ROOT}/logs}"

# Optional runtime parameters
export EXCAVATOR_ASSET_PATH="${EXCAVATOR_ASSET_PATH:-${EXCAVATOR_ASSETS_DIR}/excavator.usd}"
export EXCAVATOR_SCENE_USD="${EXCAVATOR_SCENE_USD:-}"
export EXCAVATOR_SEED="${EXCAVATOR_SEED:-42}"

mkdir -p \
  "${EXCAVATOR_DATA_RAW_DIR}" \
  "${EXCAVATOR_DATA_PROCESSED_DIR}" \
  "${EXCAVATOR_RUNS_DIR}" \
  "${EXCAVATOR_LOGS_DIR}"

echo "[env] EXCAVATOR_ROOT=${EXCAVATOR_ROOT}"
