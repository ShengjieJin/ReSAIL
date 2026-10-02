#!/usr/bin/env bash
# Run inside the dedicated container, through scripts/launch_experiment.sh.
set -euo pipefail

if (( $# != 3 )) || [[ "$1" != 4b && "$1" != 8b ]]; then
  echo "usage: $0 {4b|8b} /absolute/HF/model /absolute/new/output" >&2
  exit 2
fi
model_size=$1
hf_model=$2
checkpoint_output=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ "$hf_model" == /* && "$checkpoint_output" == /* ]] || {
  echo "Model and output paths must be absolute." >&2; exit 2;
}
[[ -f "$hf_model/config.json" ]] || { echo "Missing HF config: $hf_model" >&2; exit 2; }
[[ ! -e "$checkpoint_output" ]] || { echo "Refusing to overwrite: $checkpoint_output" >&2; exit 2; }
python3 scripts/prepare/verify_text_checkpoint_roundtrip.py inspect-hf \
  --model-size "$model_size" --original-hf "$hf_model"
python3 scripts/prepare/check_environment.py --gpus
source "scripts/models/qwen3-${model_size%b}B.sh"
export PYTHONPATH="/root/Megatron-LM:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p "$checkpoint_output"
torchrun --standalone --nproc-per-node=8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint "$hf_model" --save "$checkpoint_output" \
  --bf16 --use-cpu-initialization --ckpt-format torch_dist \
  > "$checkpoint_output/conversion.log" 2>&1
python3 scripts/prepare/verify_text_checkpoint_roundtrip.py inspect-checkpoint \
  --model-size "$model_size" --original-hf "$hf_model" \
  --checkpoint-root "$checkpoint_output"
