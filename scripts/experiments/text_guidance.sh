#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/slime}
CORPUS_DIR=${PAPER_CORPUS_DIR:?set PAPER_CORPUS_DIR}
OUTPUT_DIR=${PAPER_GUIDANCE_DIR:?set PAPER_GUIDANCE_DIR}
MODEL_PATH=${PAPER_GUIDANCE_MODEL_PATH:?set PAPER_GUIDANCE_MODEL_PATH}
PORT_BASE=${PAPER_GUIDANCE_PORT_BASE:-29800}
ENGINE_COUNT=${PAPER_GUIDANCE_ENGINE_COUNT:-8}
GLOBAL_CONCURRENCY=${PAPER_GUIDANCE_GLOBAL_CONCURRENCY:-256}
METADATA_PROFILE=${PAPER_GUIDANCE_METADATA_PROFILE:-textcraft}
EXPECTED_TRAJECTORIES=${PAPER_GUIDANCE_EXPECTED_TRAJECTORIES:-960}
TRAJECTORY_SCHEDULE=${PAPER_GUIDANCE_TRAJECTORY_SCHEDULE:-}
MAX_NEW_TOKENS=${PAPER_GUIDANCE_MAX_NEW_TOKENS:-1024}
SUMMARY_SCHEMA=${PAPER_GUIDANCE_SUMMARY_SCHEMA:-own_outcome_detailed}
TEMPERATURE=${PAPER_GUIDANCE_TEMPERATURE:-1.0}
ATTEMPT_SEED_OFFSET=${PAPER_GUIDANCE_ATTEMPT_SEED_OFFSET:-0}
PRIOR_ATTEMPT_SEED_OFFSET=${PAPER_GUIDANCE_PRIOR_ATTEMPT_SEED_OFFSET:-0}
EXISTING_MAX_NEW_TOKENS=${PAPER_GUIDANCE_EXISTING_MAX_NEW_TOKENS:-}
PRIOR_FAILURE_DIR=${PAPER_GUIDANCE_PRIOR_FAILURE_DIR:-}
EMPTY_ON_RECOVERY_EXHAUSTION=${PAPER_GUIDANCE_EMPTY_ON_RECOVERY_EXHAUSTION:-0}
RUNTIME_PROFILE=${PAPER_GUIDANCE_RUNTIME_PROFILE:-}
EXECUTION_MODE=${PAPER_GUIDANCE_EXECUTION_MODE:-}

cd "${REPO_DIR}"
[[ -d "${CORPUS_DIR}" ]] || { echo "missing paper corpus: ${CORPUS_DIR}" >&2; exit 2; }
[[ -d "${MODEL_PATH}" ]] || { echo "missing paper guidance model: ${MODEL_PATH}" >&2; exit 2; }
mkdir -p "${OUTPUT_DIR}/logs"

port_is_unused() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
try: sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) else 1)
finally: s.close()
PY
}

PORTS=()
PIDS=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${PIDS[@]:-}"; do kill -0 -- "-${pid}" 2>/dev/null && alive=1; done
    (( alive == 0 )) && break
    sleep 1
  done
  for pid in "${PIDS[@]:-}"; do kill -KILL -- "-${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  for port in "${PORTS[@]:-}"; do port_is_unused "${port}" || status=1; done
  exit "${status}"
}
trap cleanup EXIT INT TERM

ENDPOINT_FILE="${OUTPUT_DIR}/logs/endpoints.txt"
: >"${ENDPOINT_FILE}"
for gpu in $(seq 0 $((ENGINE_COUNT - 1))); do
  port=$((PORT_BASE + gpu))
  port_is_unused "${port}" || { echo "paper guidance port already in use: ${port}" >&2; exit 2; }
  PORTS+=("${port}")
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${port}" --tp-size 1 --mem-fraction-static 0.85 \
    >"${OUTPUT_DIR}/logs/server_${gpu}.log" 2>&1 &
  PIDS+=("$!")
  echo "http://127.0.0.1:${port}" >>"${ENDPOINT_FILE}"
done

for port in "${PORTS[@]}"; do
  deadline=$((SECONDS + 300))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; do
    (( SECONDS < deadline )) || { echo "paper guidance server failed readiness on ${port}" >&2; exit 2; }
    sleep 2
  done
done

MATERIALIZER_ARGS=(
  --corpus-dir "${CORPUS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model-path "${MODEL_PATH}"
  --endpoint-file "${ENDPOINT_FILE}"
  --engine-count "${ENGINE_COUNT}"
  --global-concurrency "${GLOBAL_CONCURRENCY}"
  --metadata-profile "${METADATA_PROFILE}"
  --expected-trajectories "${EXPECTED_TRAJECTORIES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --summary-schema "${SUMMARY_SCHEMA}"
  --temperature "${TEMPERATURE}"
  --attempt-seed-offset "${ATTEMPT_SEED_OFFSET}"
  --prior-attempt-seed-offset "${PRIOR_ATTEMPT_SEED_OFFSET}"
)
if [[ -n "${TRAJECTORY_SCHEDULE}" ]]; then
  MATERIALIZER_ARGS+=(--trajectory-schedule "${TRAJECTORY_SCHEDULE}")
fi
if [[ -n "${EXISTING_MAX_NEW_TOKENS}" ]]; then
  MATERIALIZER_ARGS+=(--existing-max-new-tokens "${EXISTING_MAX_NEW_TOKENS}")
fi
if [[ -n "${PRIOR_FAILURE_DIR}" ]]; then
  MATERIALIZER_ARGS+=(--prior-failure-dir "${PRIOR_FAILURE_DIR}")
fi
if [[ "${EMPTY_ON_RECOVERY_EXHAUSTION}" == 1 ]]; then
  MATERIALIZER_ARGS+=(--empty-on-recovery-exhaustion)
fi
if [[ -n "${RUNTIME_PROFILE}" ]]; then
  MATERIALIZER_ARGS+=(--runtime-profile "${RUNTIME_PROFILE}")
fi
if [[ -n "${EXECUTION_MODE}" ]]; then
  MATERIALIZER_ARGS+=(--execution-mode "${EXECUTION_MODE}")
fi

python3 -m exp.paper.text_guidance \
  "${MATERIALIZER_ARGS[@]}" \
  | tee "${OUTPUT_DIR}/logs/materializer.log"
