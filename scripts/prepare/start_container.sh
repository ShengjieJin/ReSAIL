#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RESAIL_MODEL_DIR:?Set RESAIL_MODEL_DIR to the model directory}"
: "${RESAIL_CHECKPOINT_DIR:?Set RESAIL_CHECKPOINT_DIR to the base Megatron checkpoint directory}"
: "${RESAIL_DATASET_DIR:?Set RESAIL_DATASET_DIR to the external dataset directory}"
container=${RESAIL_CONTAINER:-resail-paper}
image=${RESAIL_IMAGE:-resail:paper}

for directory in "$RESAIL_MODEL_DIR" "$RESAIL_CHECKPOINT_DIR" "$RESAIL_DATASET_DIR"; do
  [[ -d "$directory" ]] || { echo "Input directory does not exist: $directory" >&2; exit 2; }
done
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container already exists: $container. Inspect it or choose another RESAIL_CONTAINER." >&2
  exit 2
fi

runtime_env=()
if [[ -n ${CONTAINER_HTTP_PROXY:-} ]]; then
  runtime_env+=(--env "HTTP_PROXY=$CONTAINER_HTTP_PROXY")
fi
if [[ -n ${CONTAINER_HTTPS_PROXY:-} ]]; then
  runtime_env+=(--env "HTTPS_PROXY=$CONTAINER_HTTPS_PROXY")
fi
if [[ -n ${CONTAINER_NO_PROXY:-} ]]; then
  runtime_env+=(--env "NO_PROXY=$CONTAINER_NO_PROXY")
fi
if [[ -n ${WANDB_API_KEY:-} ]]; then
  runtime_env+=(--env WANDB_API_KEY)
fi

exec docker run -d --name "$container" --gpus all --cpus 160 --pids-limit 16384 \
  --shm-size 16g --ulimit nofile=524288:524288 \
  --mount "type=bind,src=$repo_root,dst=/workspace/slime" \
  --mount "type=bind,src=$repo_root,dst=/root/slime" \
  --mount "type=bind,src=$RESAIL_MODEL_DIR,dst=/root/models,readonly" \
  --mount "type=bind,src=$RESAIL_CHECKPOINT_DIR,dst=/root/checkpoints,readonly" \
  --mount "type=bind,src=$RESAIL_DATASET_DIR,dst=/root/datasets,readonly" \
  -e PYTHONPATH=/root/Megatron-LM:/workspace/slime \
  -e PYTHONPYCACHEPREFIX=/tmp/resail_pycache \
  "${runtime_env[@]}" \
  --workdir /workspace/slime "$image" sleep infinity
