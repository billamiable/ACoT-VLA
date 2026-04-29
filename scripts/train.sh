export DEBUG_MODE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

CONFIG_NAME=${1}
EXP_NAME=${2}

export WANDB_MODE=${WANDB_MODE:-online}

# Ensure torchcodec can find FFmpeg shared libs.
PREFIX="${CONDA_PREFIX:-/home/billyw/miniconda3}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${PREFIX}/bin:${PATH}"

env | sort
uv run python scripts/train.py $CONFIG_NAME --exp-name=$EXP_NAME