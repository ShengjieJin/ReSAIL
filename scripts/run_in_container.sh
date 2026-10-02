#!/usr/bin/env bash
set -euo pipefail

(( $# > 0 )) || { echo "usage: $0 <command> [arguments...]" >&2; exit 2; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
container=${RESAIL_CONTAINER:-resail-paper}

# Refuse a container mounted to a different checkout, including the source project.
mounted_root=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/slime"}}{{.Source}}{{end}}{{end}}' "$container")
[[ "$mounted_root" == "$repo_root" ]] || {
  echo "Container must mount this ReSAIL checkout at /workspace/slime." >&2
  exit 2
}
exec docker exec -i --workdir /workspace/slime "$container" "$@"
