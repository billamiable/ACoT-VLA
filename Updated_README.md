# Internal replication notes

Short path to reproduce GenieSim / π0.5 fine-tuning. Full docs: [README.md](README.md).

---

## Installation

```bash
git clone https://github.com/AgibotTech/ACoT-VLA.git
cd ACoT-VLA
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## Norm stats (`scripts/run_compute_norm_stats.sh`)

This script currently exports Conda FFmpeg (`PATH` / `LD_LIBRARY_PATH`) because my environment could not find FFmpeg shared libs for `torchcodec`. If your environment has no FFmpeg issue, you can remove that part from the bash script. From repo root:

```bash
./scripts/run_compute_norm_stats.sh pi05_genie_sim_10_mini_task_20260312 \
  --output-path /home/billyw/iDataset/simulation/genie_sim/dataset/instruction/norm_stats/ten_mini_task_merge
```

Writes `.../ten_mini_task_merge/norm_stats.json`. Match `AssetsConfig` / paths in `src/openpi/training/config.py` for that config.

---

## Finetune

Before training (online W&B), login in the same `uv` environment:

```bash
uv run wandb login
```

```bash
bash scripts/train.sh pi05_genie_sim_10_mini_task_20260312 <EXP_NAME>
```

(`scripts/train.sh` currently also exports Conda FFmpeg path for the same reason above; you can delete that block if your environment works without it.)

Require `batch_size % jax.device_count() == 0`. Optional: `DEBUG_MODE`, `WANDB_MODE`, `XLA_PYTHON_CLIENT_MEM_FRACTION` as in `scripts/train.sh`.
