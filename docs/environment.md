# Environment setup

ReSAIL uses the included Slime fork with Ray, SGLang, Megatron, and Qwen3. The reference configuration uses Python 3.12.3, eight H800 GPUs, data parallelism 8, and tensor, pipeline, and context parallelism 1 during training. The Megatron backend requires NumPy 1.x.

## Build the container

The recommended setup is the Docker environment below, with experiments run from the mounted source checkout. Its pinned versions provide a repeatable reference installation.

The host needs Linux, Python 3.10 or newer for the launch guard, Docker, an NVIDIA driver, and NVIDIA Container Toolkit. Run these commands from the ReSAIL source root on the host:

```bash
docker build -f docker/Dockerfile.resail -t resail:paper .

export RESAIL_MODEL_DIR=/absolute/path/to/models
export RESAIL_CHECKPOINT_DIR=/absolute/path/to/base-checkpoints
export RESAIL_DATASET_DIR=/absolute/path/to/datasets
export RESAIL_CONTAINER=resail-paper
bash scripts/prepare/start_container.sh

bash scripts/run_in_container.sh python3 scripts/prepare/check_environment.py --gpus
```

The three host directories must exist. The container mounts them read-only as `/root/models`, `/root/checkpoints`, and `/root/datasets`; it mounts this checkout as `/workspace/slime` for code, task caches, and new runs. The image starts from `slimerl/slime:nightly-dev-20260530a` and installs the task requirements in [`docker/tasks/resail.requirements.txt`](../docker/tasks/resail.requirements.txt). The check reports runtime versions, dependency issues, and visible GPUs. The experiment launchers require the resources listed below.

For a three-cycle 4B run, plan for roughly 320 GiB of free working space in addition to downloaded models and task data. The current launcher requires eight idle GPUs, a container CPU quota of 160, a PID limit of 16384, at least 128 GiB of available host memory, and the intended checkout mount. Run one GPU job at a time in this container.

## Environment check results

`check_environment.py --gpus` reports runtime versions, visible GPUs, and the result of `pip check`. Version differences (`different_environment`) and dependency conflicts (`dependency_conflicts`) are advisory by default. The pinned base image can report conflicts involving SGLang, CUDA packages, cuDNN, and Megatron Bridge extras; inspect the package details in the output before changing dependencies.

Add `--strict` to require the reference versions and GPU count. Add `--strict-dependencies` to fail on dependency conflicts. These flags can be combined.

## Prepare models and tasks

Place the Qwen3-4B or Qwen3-8B Hugging Face weights and tokenizer under `/root/models/Qwen/`, as described in [models and task data](inputs.md). If you do not have a matching Megatron base checkpoint, convert the HF model into the writable checkout:

```bash
bash scripts/launch_experiment.sh prepare-qwen3-4b -- \
  bash scripts/prepare/convert_text_checkpoint.sh 4b \
    /root/models/Qwen/Qwen3-4B \
    /workspace/slime/data/base-checkpoints/Qwen3-4B
```

Use `8b` and the matching Qwen3-8B paths for the larger model. The run examples use the default `/root/checkpoints/Qwen3-4B_torch_dist` or 8B counterpart. If you converted a base checkpoint into the checkout, add `--base-checkpoint /workspace/slime/data/base-checkpoints/Qwen3-4B` to both `text_c1.sh` and `text_cycles.sh`; use the matching 8B path for 8B. Prepare the [ALFWorld](../slime_plugins/agent_tasks/alfworld/README.md) or [TextCraft](../slime_plugins/agent_tasks/textcraft/README.md) cache in this same checkout before collecting C1.

Experiment logging defaults to W&B **offline** mode and needs no W&B account or network connection. To upload metrics, export `WANDB_API_KEY` on the host before starting the container and add `--wandb-mode online` to `text_c1.sh` or `text_cycles.sh`. Use `--wandb-mode disabled` to disable W&B. The start script passes an existing key into the container; do not put credentials in source files or run directories.

Set `CONTAINER_HTTP_PROXY`, `CONTAINER_HTTPS_PROXY`, and `CONTAINER_NO_PROXY` only when your network requires them; the start script does not assume a proxy.

## CPU checks

After preparing the task caches, run the public configuration checks without GPU visibility:

```bash
bash scripts/run_in_container.sh env CUDA_VISIBLE_DEVICES= \
  bash scripts/check_public_contracts.sh
```

Additional tests for code changes are described in [Contributing](../CONTRIBUTING.md).

See the [reproduction guide](reproduction.md) for C1 collection, training, and evaluation commands.
