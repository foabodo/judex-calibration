# Serving a base/post model on vast.ai vLLM (Phase 1: Qwen)

One model per server. Serve base, run elicitation, tear down; repeat for post on the
same box. Token-slicing needs **logprobs on /v1/completions** — vLLM provides this.
Use **bf16** (calibration is logit-sensitive; fp8 perturbs it).

## 1. Provision (Qwen3.5-35B-A3B ~70 GB bf16 -> 1×H100-80)
```bash
export VAST_API_KEY=$(security find-generic-password -s vastai-api-key -w)   # add to Keychain
vastai search offers 'gpu_name=H100_SXM num_gpus=1 disk_space>200' --order 'dph'
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 200 \
  --env '-e HF_TOKEN=<hf>' --ssh
vastai ssh-url <INSTANCE_ID>     # -> ssh root@<host> -p <port>
```

## 2. Serve the base model (on the box)
```bash
# base:
vllm serve Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.92
# health:
curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'
```
Giants (later): add `--tensor-parallel-size 8 [--pipeline-parallel-size 2] --enable-expert-parallel`.

## 3. Tunnel + elicit (on the Mac, via Claude Code)
```bash
ssh -N -L 8000:localhost:8000 root@<host> -p <port> &      # background tunnel
# base leg:
.../judex-evaluator/.venv/bin/python scripts/run_qwen_phase1.py \
  --base-url http://localhost:8000 --base-model Qwen/Qwen3.5-35B-A3B-Base \
  --out runs/phase1_qwen
```

## 4. Swap to post on the same box, re-run
```bash
# kill base vllm, then:
vllm serve Qwen/Qwen3.5-35B-A3B --dtype bfloat16 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --reasoning-parser qwen3   # native thinking ON (live methodology)
# Mac:
.../python scripts/run_qwen_phase1.py --post-url http://localhost:8000 \
  --post-model Qwen/Qwen3.5-35B-A3B --out runs/phase1_qwen
```

## 5. Analyse + tear down
```bash
.../python scripts/run_qwen_phase1.py --analyze-only --out runs/phase1_qwen
vastai destroy instance <INSTANCE_ID>     # cost is wall-clock — destroy promptly
```

Notes: per-cell elicitation is 2 calls (reason + answer) + up to 5 echo calls if a
letter misses top-K; 120 cells is minutes of inference. The cost is download + load,
so minimize box wall-clock. Add `vastai`/`ssh`/`curl localhost:8000/*` to
`.claude/settings.local.json` to cut permission prompts.
