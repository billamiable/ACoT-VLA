#!/bin/bash
# Wrapper: put Conda FFmpeg on PATH/LD_LIBRARY_PATH for torchcodec, then run compute_norm_stats.
#
# Example:
#   ./scripts/run_compute_norm_stats.sh pi05_genie_sim_10_mini_task_20260312 \
#     --output-path /home/billyw/iDataset/simulation/genie_sim/dataset/instruction/norm_stats/ten_mini_task_merge
#
# Usage: ./scripts/run_compute_norm_stats.sh <CONFIG_NAME> [--output-path DIR] [...]
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${CONDA_PREFIX:-/home/billyw/miniconda3}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${PREFIX}/bin:${PATH}"

CONFIG="${1:?usage: $0 <config_name> [--output-path DIR ...]}"
shift
exec uv run python scripts/compute_norm_stats.py --config-name "$CONFIG" "$@"
