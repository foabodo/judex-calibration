#!/bin/bash
# run_local_f16_pilot.sh FAMILY BASE_GGUF BASE_LABEL POST_GGUF POST_LABEL OUT_DIR [CTX] [PORT]
# One family of the LOCAL fp16 science pilot (docs/local_smoke_quickstart.md §2·Mac-D):
# serve the fp16 BASE gguf on llama.cpp -> run the full 120-cell reasoning-ON base leg ->
# swap to the fp16 POST gguf -> post leg -> the driver writes the per-family report.
# Legs checkpoint per cell, so re-running this script resumes a crashed leg.
#
# FAMILY must be a NON-PANEL tag (e.g. gemma3-4b, qwen3-4b) so the report can never be
# mistaken for panel output. Run detached (nohup) — legs take ~1.5-4 h each.
#
# Env: LLAMA_SERVER = llama-server binary (default: on PATH); JUDEX_PY = driver python
#      (default: judex-arm); PILOT_LOG_DIR = server logs (default: $TMPDIR);
#      BUDGET = CoT token budget (default 2048 — the live default).
set -euo pipefail
FAMILY="$1"; BASE_GGUF="$2"; BASE_LABEL="$3"; POST_GGUF="$4"; POST_LABEL="$5"; OUT="$6"
CTX="${7:-24576}"; PORT="${8:-8123}"
SERVER_BIN="${LLAMA_SERVER:-llama-server}"
PY="${JUDEX_PY:-/Users/fabodo/anaconda3/envs/judex-arm/bin/python}"
LOGS="${PILOT_LOG_DIR:-${TMPDIR:-/tmp}}"
CAL="$(cd "$(dirname "$0")/.." && pwd)"
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true' EXIT

serve() { # gguf logfile
  "$SERVER_BIN" -m "$1" --host 127.0.0.1 --port "$PORT" --ctx-size "$CTX" --n-gpu-layers 999 \
    > "$2" 2>&1 &
  SRV=$!
  for _ in $(seq 1 90); do
    # -f: a 503 while the model is still loading must NOT count as healthy
    curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "!! server died during load — see $2" >&2; exit 1; }
    sleep 2
  done
  echo "!! server not healthy after 180s — see $2" >&2; exit 1
}

echo "== [$FAMILY] BASE leg: $BASE_GGUF (ctx $CTX)"
serve "$BASE_GGUF" "$LOGS/pilot_${FAMILY}_base_server.log"
(cd "$CAL" && "$PY" scripts/run_qwen_phase1.py --base-url "http://127.0.0.1:$PORT" \
  --base-model "$BASE_LABEL" --family "$FAMILY" --out "$OUT" --budget "${BUDGET:-2048}")
kill "$SRV"; SRV=""; sleep 3

echo "== [$FAMILY] POST leg: $POST_GGUF (ctx $CTX)"
serve "$POST_GGUF" "$LOGS/pilot_${FAMILY}_post_server.log"
(cd "$CAL" && "$PY" scripts/run_qwen_phase1.py --post-url "http://127.0.0.1:$PORT" \
  --post-model "$POST_LABEL" --family "$FAMILY" --out "$OUT" --budget "${BUDGET:-2048}")
kill "$SRV"; SRV=""

echo "== [$FAMILY] DONE — report: $CAL/$OUT/study_a_report.json"
