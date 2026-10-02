#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/slime}
cd "$REPO_DIR"
exec python3 -m exp.paper.text prepare-c1 "$@"
