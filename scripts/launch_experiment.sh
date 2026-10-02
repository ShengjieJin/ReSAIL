#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )) || [[ "$2" != -- ]]; then
  echo "usage: $0 <run-id> -- <container-command> [arguments...]" >&2
  exit 2
fi
run_id=$1
shift 2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RESAIL_CONTAINER=${RESAIL_CONTAINER:-resail-paper}
cd "$repo_root"

# The guard checks the checkout mount before it can start or clean up a job.
exec tools/run_guarded_experiment.sh "paper/$run_id" -- \
  docker exec -i --workdir /workspace/slime "$RESAIL_CONTAINER" "$@"
