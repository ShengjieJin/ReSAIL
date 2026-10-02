#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )) || [[ "$2" != -- ]]; then
  echo "usage: $0 <audit-name> -- <command> [args...]" >&2
  exit 2
fi
audit_name=$1
shift 2
[[ "$audit_name" != /* && "$audit_name" != */ && "$audit_name" =~ ^[A-Za-z0-9._/-]+$ ]] || {
  echo "invalid audit name: $audit_name" >&2
  exit 2
}
IFS=/ read -r -a audit_segments <<<"$audit_name"
for segment in "${audit_segments[@]}"; do
  [[ -n "$segment" && "$segment" != . && "$segment" != .. && "$segment" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "invalid audit name: $audit_name" >&2
    exit 2
  }
done

CONTAINER="${RESAIL_CONTAINER:-${CONTAINER:-resail-paper}}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mounted_root=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/slime"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")
[[ "$mounted_root" == "$repo_root" ]] || {
  echo "Refusing to run: container must mount this ReSAIL checkout at /workspace/slime." >&2
  exit 2
}
guard_dir="runs/host_guard"
mkdir -p "$guard_dir/$(dirname "$audit_name")"
chmod 700 "$guard_dir" "$guard_dir/$(dirname "$audit_name")"

exec python3 tools/experiment_guard.py \
  --container "$CONTAINER" \
  --audit-log "$guard_dir/$audit_name.jsonl" \
  --lock-file "$guard_dir/container.lock" \
  --expected-gpus 8 \
  --expected-cpu-limit 160 \
  --expected-pids-limit 16384 \
  --warn-pids 12000 \
  --degraded-pids 13500 \
  --stop-pids 15000 \
  --min-available-memory-gib 128 \
  --stop-load 176 \
  --load-strikes 4 \
  --interval-seconds 10 \
  --restart-zombies 1000 \
  -- "$@"
