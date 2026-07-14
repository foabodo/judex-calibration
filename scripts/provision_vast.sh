#!/usr/bin/env bash
# provision_vast.sh — one-command Mode C vLLM serving on vast.ai for Study A.
#
#   up   <hf_model_repo> [post]   provision an on-demand box serving the model (bf16, logprobs),
#                                 wait for the vLLM endpoint, print the OpenAI base URL.
#   url  <instance_id>            print the base URL of a running instance.
#   down <instance_id>            destroy the instance (cost is wall-clock — do this promptly).
#
# Keys from Keychain: vastai-api-key, hf-token. Base leg uses [base] (default); the `post` arg
# adds the family's reasoning parser where one exists (qwen3 for Qwen; Gemma 4 has no separate
# reasoning control, so its post leg is served plain). Examples:
#   ID_URL=$(scripts/provision_vast.sh up Qwen/Qwen3.5-35B-A3B-Base)
#   .../python scripts/run_qwen_phase1.py --base-url "$ID_URL" --base-model Qwen/Qwen3.5-35B-A3B-Base --out runs/qwen
#   ID_URL=$(scripts/provision_vast.sh up google/gemma-4-26B-A4B)          # Gemma base leg
#   ID_URL=$(scripts/provision_vast.sh up google/gemma-4-26B-A4B-it post)  # Gemma post leg
#   scripts/provision_vast.sh down <instance_id>
#
# See docs/vast_quickstart.md for the full account->shutdown walkthrough. The vllm/vllm-openai
# image takes the model + vLLM flags as container --args (confirmed by the vast vLLM guide);
# vast CLI flags can still evolve — verify with `vastai --help` if create/show output shifts.
set -euo pipefail

# static_ip=true + direct_port_count>1 are REQUIRED for the public IP:port. Weights pull from HF
# every launch (vast hosts no base), so inet_down (bandwidth) + disk_space gate the wall-clock cost.
# Cheap-trial defaults (override via env for giants): single H200, sized for the ~18.3k-token
# corpus-v2 prompts + the 2048 CoT budget (32k-context pin; >=24576 required — measured by
# scripts/measure_prompt_budget.py) — gpu_ram>=140, not 80. reliability/inet_down_cost guard the pull.
# Both trial families use these defaults: Qwen 35B-A3B (~70 GB bf16) needs the H200; Gemma
# 26B-A4B (~50 GB bf16) might squeeze onto 80 GB but the same H200 query removes OOM risk for
# pennies (cost is dominated by the download, not the card).
OFFER_QUERY=${VAST_OFFER_QUERY:-'gpu_ram>=140 num_gpus=1 static_ip=true direct_port_count>1 inet_down>1000 inet_down_cost<0.05 reliability>0.98 disk_space>192 cuda_vers>=12.4 rentable=true'}
DISK=${VAST_DISK:-192}
PORT=8000
MAXLEN=${VLLM_MAX_MODEL_LEN:-32768}
GPU_UTIL=${VLLM_GPU_UTIL:-0.92}
POLL_TRIES=${VAST_POLL_TRIES:-120}   # x30s ≈ 60 min for download+load

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }; }
need vastai; need python3; need curl
export VAST_API_KEY=${VAST_API_KEY:-$(security find-generic-password -s vastai-api-key -w)}

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    MODEL=${1:?usage: up <hf_model_repo> [post]}; VARIANT=${2:-base}
    HF_TOKEN=${HF_TOKEN:-$(security find-generic-password -s hf-token -w 2>/dev/null || true)}
    # The vllm/vllm-openai image takes the model + vLLM flags as container --args (NOT an
    # on-start command); --raw must precede --args (which consumes the rest of the line).
    # The reasoning parser is FAMILY-SPECIFIC: Qwen's post thinks natively (qwen3 parser);
    # Gemma 4 has no separate reasoning control, so its post leg is served plain.
    EXTRA=""
    if [ "$VARIANT" = "post" ]; then
      case "$MODEL" in
        Qwen/*|qwen/*)             EXTRA="--reasoning-parser qwen3" ;;
        google/gemma-*|*/gemma-*)  EXTRA="" ;;
        *) echo ">> NOTE: no reasoning parser configured for $MODEL post leg; serving plain" >&2 ;;
      esac
    fi

    echo ">> searching offers: $OFFER_QUERY" >&2
    OFFER=$(vastai search offers "$OFFER_QUERY" --order dph --raw \
            | python3 -c 'import sys,json; d=json.load(sys.stdin); print((d if isinstance(d,list) else d.get("offers",[]))[0]["id"])')
    echo ">> creating instance on offer $OFFER serving $MODEL ($VARIANT)" >&2
    IID=$(vastai create instance "$OFFER" --image vllm/vllm-openai:latest --disk "$DISK" \
            --env "-p $PORT:$PORT -e HF_TOKEN=$HF_TOKEN" --raw \
            --args --model "$MODEL" --dtype bfloat16 --max-model-len "$MAXLEN" \
                   --gpu-memory-utilization "$GPU_UTIL" $EXTRA \
          | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("new_contract") or d.get("id") or "")')
    [ -n "$IID" ] || { echo "!! could not parse instance id from create" >&2; exit 1; }
    echo ">> instance=$IID — waiting for the vLLM endpoint (model download+load can take many minutes)..." >&2
    for _ in $(seq 1 "$POLL_TRIES"); do
      URL=$("$0" url "$IID" 2>/dev/null || true)
      if [ -n "$URL" ] && curl -fsS --max-time 8 "$URL/v1/models" >/dev/null 2>&1; then
        echo ">> READY  instance=$IID  url=$URL  (destroy with: $0 down $IID)" >&2
        echo "$URL"; exit 0
      fi
      sleep 30
    done
    echo "!! timed out; inspect with: vastai logs $IID   (instance=$IID still running — destroy if unwanted)" >&2
    exit 1
    ;;
  url)
    IID=${1:?usage: url <instance_id>}
    vastai show instance "$IID" --raw | python3 -c '
import sys, json
d = json.load(sys.stdin)
ip = (d.get("public_ipaddr") or "").strip()
ports = d.get("ports") or {}
m = ports.get("8000/tcp") or []
hp = m[0].get("HostPort", "") if m else ""
print(f"http://{ip}:{hp}" if ip and hp else "", end="")
'
    ;;
  down)
    IID=${1:?usage: down <instance_id>}; vastai destroy instance "$IID"; echo "destroyed $IID" >&2
    ;;
  *)
    echo "usage: $0 {up <hf_model_repo> [post] | url <instance_id> | down <instance_id>}" >&2; exit 2
    ;;
esac
