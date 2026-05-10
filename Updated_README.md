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

---

## GenieSim task-suite benchmark (π0.5)

Official batch-run doc: [Genie Sim — Batch Run Task Suite](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_315-batch-run-task-suite).

**Same environment as** Isaac-GR00T `scripts/preprocess_agibot/Updated_User_Guide.md` §8: clone `genie_sim`, image, assets, GUI, container, **`run_batch_tasks.sh`**. The **only** difference is **how you start inference** (here: OpenPI `serve_policy.py` on π0.5 export; GR00T uses `serve_gr00t_websocket.py`).

1. Clone simulator and build image:

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
docker build -f ./scripts/dockerfile -t registry.agibot.com/genie-sim/open_source:latest .
```

2. Download GenieSim assets under `source/geniesim/assets`:

```bash
git clone https://modelscope.cn/datasets/agibot_world/GenieSimAssets.git -b rolling
```

3. Start simulator GUI (repo root):

```bash
cd genie_sim
./scripts/start_gui.sh
```

4. New terminal — enter container:

```bash
cd genie_sim
./scripts/into.sh
```

5. **On the policy machine** (host with GPU for JAX), from `ACoT-VLA` root, start the OpenPI websocket server and note `host:port` reachable from the benchmark container:

```bash
cd ACoT-VLA
uv run scripts/serve_policy.py \
  --host=0.0.0.0 \
  --port=8999 \
  policy:checkpoint \
  --policy.config=pi05_genie_sim_10_mini_task_20260312 \
  --policy.dir=/home/yujie/workspace/yujie/iDataset/VLA/openpi/checkpoint/pi05_genie_sim_10_mini_task_20260312/pi05_genie_10task_run1/29999_export
```

(Adjust `--policy.dir` if your export lives elsewhere.)

6. **Inside the genie_sim benchmark container**, run IF suite (use the host/IP the container can reach, not necessarily `127.0.0.1`):

```bash
./scripts/run_batch_tasks.sh --num-episode 3 --type if --infer-host {ip:port}
```

Example if policy listens on host `127.0.1.1:8999`:

```bash
./scripts/run_batch_tasks.sh --num-episode 3 --type if --infer-host 127.0.1.1:8999
```

Notes:

- Full-quality runs use `--num-episode 3`; `--num-episode 1` is ok for smoke tests. Runtime is long (on the order of hours for multi-episode runs, as in the GR00T guide).
- **Protocol:** `serve_policy.py` is OpenPI `WebsocketPolicyServer` (`docs/remote_inference.md`), not GR00T msgpack. If IF fails or scores are nonsense, confirm GenieSim IF targets an OpenPI-compatible client.
- **`config.py` `repo_id` paths:** if your machine does not have those LeRobot roots, you may see index-mapping warnings when loading the policy unless paths are updated locally.
