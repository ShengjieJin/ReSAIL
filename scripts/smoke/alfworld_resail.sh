#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
EXP_NAME=alfworld_qwen3_4b_resail_lifecycle_smoke
RUN_ID=${RUN_ID:-${EXP_NAME}_${RUN_TIMESTAMP}}
RUN_DIR=${RUN_DIR:-${repo_root}/runs/paper/${RUN_ID}}
CONFIG_PATH=${RUN_DIR}/alfworld/4b/resail/c1/bundle/expanded_config.json
SEED=1234
ROLLOUT_SEED=42
export RUN_TIMESTAMP EXP_NAME RUN_ID RUN_DIR CONFIG_PATH SEED ROLLOUT_SEED

# This smoke intentionally saves and reloads one final checkpoint to test the
# complete lifecycle. It uses two updates with the formal 32-trajectory batch.
exec python3 -m exp.paper.text run \
  --task alfworld --model 4b --method resail --cycle 1 \
  --mode smoke --smoke-updates 2 \
  --run-root "$RUN_DIR" --input-root "${RESAIL_INPUT_ROOT:-${repo_root}/data/paper}"
