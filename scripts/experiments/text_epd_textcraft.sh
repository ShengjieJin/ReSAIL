#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/slime}
CORPUS_DIR=${RESAIL_EPD_CORPUS_DIR:?set RESAIL_EPD_CORPUS_DIR}
TARGET_DIR=${RESAIL_EPD_TARGET_DIR:?set RESAIL_EPD_TARGET_DIR}
MODEL_PATH=${RESAIL_EPD_MODEL_PATH:?set RESAIL_EPD_MODEL_PATH}
IDENTITY=${RESAIL_EPD_IDENTITY:?set RESAIL_EPD_IDENTITY}
TEACHER_ITERATION=${RESAIL_EPD_TEACHER_ITERATION:?set RESAIL_EPD_TEACHER_ITERATION}
GUIDANCE_SUMMARY_DIR=${RESAIL_EPD_GUIDANCE_SUMMARY_DIR:-}
PORT_BASE=${RESAIL_EPD_PORT_BASE:-29700}
ENGINE_COUNT=${RESAIL_EPD_ENGINE_COUNT:-8}
GLOBAL_CONCURRENCY=${RESAIL_EPD_GLOBAL_CONCURRENCY:-512}
METADATA_PROFILE=${RESAIL_EPD_METADATA_PROFILE:-textcraft}
ATTEMPT_SEED_OFFSET=${RESAIL_EPD_ATTEMPT_SEED_OFFSET:-0}
EXPECTED_TRAJECTORIES=${RESAIL_EPD_EXPECTED_TRAJECTORIES:-240}

cd "${REPO_DIR}"
[[ -d "${CORPUS_DIR}" ]] || { echo "missing TextCraft corpus: ${CORPUS_DIR}" >&2; exit 2; }
[[ -d "${MODEL_PATH}" ]] || { echo "missing TextCraft EPD teacher: ${MODEL_PATH}" >&2; exit 2; }
mkdir -p "${TARGET_DIR}/logs"

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

ENDPOINT_FILE="${TARGET_DIR}/logs/endpoints.txt"
: >"${ENDPOINT_FILE}"
for gpu in $(seq 0 $((ENGINE_COUNT - 1))); do
  port=$((PORT_BASE + gpu))
  port_is_unused "${port}" || { echo "TextCraft EPD port already in use: ${port}" >&2; exit 2; }
  PORTS+=("${port}")
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    >"${TARGET_DIR}/logs/server_${gpu}.log" 2>&1 &
  PIDS+=("$!")
  echo "http://127.0.0.1:${port}" >>"${ENDPOINT_FILE}"
done

for port in "${PORTS[@]}"; do
  deadline=$((SECONDS + 300))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; do
    (( SECONDS < deadline )) || { echo "TextCraft EPD server failed readiness on ${port}" >&2; exit 2; }
    sleep 2
  done
done

EXTRA_ARGS=()
if [[ -n "${GUIDANCE_SUMMARY_DIR}" ]]; then
  [[ -d "${GUIDANCE_SUMMARY_DIR}" ]] || { echo "missing guidance summaries: ${GUIDANCE_SUMMARY_DIR}" >&2; exit 2; }
  EXTRA_ARGS+=(--guidance-summary-dir "${GUIDANCE_SUMMARY_DIR}")
fi

python3 -m exp.paper.text_epd_textcraft \
  --corpus-dir "${CORPUS_DIR}" \
  --output-dir "${TARGET_DIR}" \
  --model-path "${MODEL_PATH}" \
  --materialization-identity "${IDENTITY}" \
  --teacher-iteration "${TEACHER_ITERATION}" \
  --endpoint-file "${ENDPOINT_FILE}" \
  --engine-count "${ENGINE_COUNT}" \
  --global-concurrency "${GLOBAL_CONCURRENCY}" \
  --metadata-profile "${METADATA_PROFILE}" \
  --attempt-seed-offset "${ATTEMPT_SEED_OFFSET}" \
  --expected-trajectories "${EXPECTED_TRAJECTORIES}" \
  "${EXTRA_ARGS[@]}" \
  | tee "${TARGET_DIR}/logs/materializer.log"
