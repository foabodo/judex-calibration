#!/usr/bin/env bash
# provision_vast.sh — one-command Mode C vLLM serving on vast.ai for Study A.
#
#   up   <hf_model_repo> [post]   provision an on-demand box serving the model (bf16, logprobs),
#                                 wait for the vLLM endpoint, print the OpenAI base URL.
#   url  <instance_id>            print the base URL of a running instance.
#   down <instance_id>            destroy the instance (cost is wall-clock — do this promptly).
#
# Keys from Keychain: vastai-api-key, hf-token. Base leg uses [base] (default); post leg adds
# the qwen3 reasoning parser via the `post` arg. Example:
#   ID_URL=$(scripts/provision_vast.sh up Qwen/Qwen3.5-35B-A3B-Base)
#   .../python scripts/run_qwen_phase1.py --base-url "$ID_URL" --base-model Qwen/Qwen3.5-35B-A3B-Base --out runs/phase1_qwen
#   scripts/provision_vast.sh down <instance_id>
#
# See docs/vast_quickstart.md for the full account->shutdown walkthrough. The vllm/vllm-openai
# image takes the model + vLLM flags as container --args (confirmed by the vast vLLM guide);
# vast CLI flags can still evolve — verify with `vastai --help` if create/show output shifts.
set -euo pipefail

# static_ip=true + direct_port_count>1 are REQUIRED for the public IP:port to work.
OFFER_QUERY=${VAST_OFFER_QUERY:-'compute_cap>=800 gpu_ram>=80 num_gpus=1 static_ip=true direct_port_count>1 cuda_vers>=12.4 disk_space>200 rentable=true'}
DISK=${VAST_DISK:-200}
PORT=8000
MAXLEN=${VLLM_MAX_MODEL_LEN:-8192}
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
    EXTRA=""; [ "$VARIANT" = "post" ] && EXTRA="--reasoning-parser qwen3"

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
