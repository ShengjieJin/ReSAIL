#!/usr/bin/env bash
# CPU-only main-text configuration contracts. Run in Docker.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/resail_pycache}
python3 -m compileall -q \
  exp/paper/text.py exp/paper/text_epd_alfworld.py \
  exp/paper/text_epd_textcraft.py exp/paper/text_guidance.py \
  scripts/prepare/check_environment.py \
  slime slime_plugins/agent_tasks/alfworld slime_plugins/agent_tasks/textcraft \
  slime_plugins/agent_tasks/common
for script in \
  scripts/experiments/text_cycles.sh \
  scripts/experiments/text_epd_alfworld.sh \
  scripts/experiments/text_epd_textcraft.sh \
  scripts/experiments/text_guidance.sh \
  scripts/prepare/text_c1.sh \
  scripts/smoke/alfworld_resail.sh \
  scripts/prepare/convert_text_checkpoint.sh \
  scripts/prepare/start_container.sh \
  scripts/launch_experiment.sh \
  scripts/run_in_container.sh; do
  bash -n "$script"
done
python3 -m pytest tests/paper/test_text.py tests/paper/test_check_environment.py "$@"
