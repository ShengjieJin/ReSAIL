#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/slime}
task=alfworld
model=4b
method=resail
mode=paper
run_root=
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --task|--model|--method|--mode|--run-root)
      ((i + 1 < ${#args[@]})) || { echo "missing value for ${args[i]}" >&2; exit 2; }
      case "${args[i]}" in
        --task) task=${args[i + 1]} ;;
        --model) model=${args[i + 1]} ;;
        --method) method=${args[i + 1]} ;;
        --mode) mode=${args[i + 1]} ;;
        --run-root) run_root=${args[i + 1]} ;;
      esac
      ((i += 1))
      ;;
    --cycle)
      echo "text_cycles.sh controls cycles 1-3; omit --cycle" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$run_root" ]]; then
  stamp=$(date -u +%Y%m%d_%H%M%S)
  run_root="$REPO_DIR/runs/paper/${task}_${model}_${method}_${mode}_${stamp}_$$"
  args+=(--run-root "$run_root")
fi
printf 'Paper text run: %s\n' "$run_root"
cd "$REPO_DIR"
if [[ "$method" == react ]]; then
  python3 -m exp.paper.text run --cycle 1 "${args[@]}"
  exit 0
fi
for cycle in 1 2 3; do
  python3 -m exp.paper.text run --cycle "$cycle" "${args[@]}"
done
