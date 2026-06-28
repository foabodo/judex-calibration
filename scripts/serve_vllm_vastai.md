# Serving Qwen base/post on vast.ai for Study A

> For the linear account→data→shutdown walkthrough (custom template, exact commands, troubleshooting)
> see **`docs/vast_quickstart.md`**. This file is the *modes reference* it draws on.


**You call the model as an OpenAI-compatible HTTP API from your Mac.** You do not drive the
server's CLI for inference, and SSH is *optional* (a private alternative to exposing the port).
The only "CLI" is the one `vllm serve` launch — and that can be the instance's on-start command,
so the box boots already serving. Our `elicit_post.py` / `elicit_base.py` speak OpenAI-compatible
HTTP, so they work against any endpoint below by setting `--base-url`.

## The four options (and which fits Study A)

| Mode | What | Fits Study A? |
|---|---|---|
| **A. Serverless** (`openai.vast.ai` proxy, autoscaling, scale-to-zero) | easiest, pay-per-use | **Post leg only** — instruct/chat; usually no base weights and no `/v1/completions`+logprobs |
| **B. One-click model template** (the vast.ai/model page) | rent a preconfigured instance, send requests | **Post leg only** — preconfigured for the *instruct* model |
| **C. Custom on-demand + your own `vllm serve`** ← **PRIMARY** | rent raw GPU, launch vLLM with the **base** repo + flags, hit `IP:port` | **Yes** — the only path that gives base weights + `/v1/completions` logprobs + bf16 + full generation control for the reason→answer two-stage |
| **D. Offline in-process** (no server) | run a script on the box: `vllm.LLM(...).generate(..., prompt_logprobs=...)`, copy results back | **Fallback** — most direct logit access if HTTP logprobs misbehave |

Base token-slicing (the core of Study A) needs base weights **and** logprobs **and** bf16 **and**
two-stage control — A/B give none of these, so **use Mode C** (Mode D as fallback). The post leg
can use A/B/C interchangeably.

---

## Mode C — primary runbook (one model per box; bf16; logprobs on by default)

> **One-command wrapper:** `scripts/provision_vast.sh up <hf_repo> [post]` does steps 1–2 (search →
> create with `--onstart-cmd` → poll health) and prints the base URL; `… down <id>` tears it down.
> The manual steps below are the reference it automates.
>
> vast CLI flags evolve — verify against `vastai --help` / docs. Cost is dominated by weight
> download + load, not the (few-minute) 120-cell inference, so minimize wall-clock.

### 1. Provision an on-demand instance (Qwen 35B/3B ~70 GB bf16 → 1×H100-80)
```bash
export VAST_API_KEY=$(security find-generic-password -s vastai-api-key -w)   # add to Keychain
export HF_TOKEN=$(security find-generic-password -s hf-token -w)
vastai search offers 'gpu_name=H100_SXM num_gpus=1 disk_space>200 inet_down>500' --order 'dph'
# Boot already serving the BASE model; expose container :8000 to a public port.
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 200 \
  --env "-p 8000:8000 -e HF_TOKEN=$HF_TOKEN" \
  --onstart-cmd 'vllm serve Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --port 8000 \
                 --max-model-len 8192 --gpu-memory-utilization 0.92'
vastai show instance <INSTANCE_ID>     # read the public host:port mapped to 8000
```
Giants later: add `--tensor-parallel-size 8 [--pipeline-parallel-size 2] --enable-expert-parallel`.

### 2. Reach the endpoint from the Mac — pick ONE
- **Direct (default):** use the public `http://<host>:<port>` from `show instance`. No SSH.
- **Private (optional):** `ssh -N -L 8000:localhost:8000 root@<host> -p <ssh_port> &` then use `http://localhost:8000`.

```bash
curl -s http://<host>:<port>/v1/models   # health: lists the served model id
```

### 3. Elicit the base leg (from the Mac)
```bash
.../judex-evaluator/.venv/bin/python scripts/run_qwen_phase1.py \
  --base-url http://<host>:<port> --base-model Qwen/Qwen3.5-35B-A3B-Base \
  --out runs/phase1_qwen
```

### 4. Swap to the post model on the same box, re-run
```bash
# stop the base server, then relaunch (post reasons natively — live methodology):
vllm serve Qwen/Qwen3.5-35B-A3B --dtype bfloat16 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --reasoning-parser qwen3
.../python scripts/run_qwen_phase1.py --post-url http://<host>:<port> \
  --post-model Qwen/Qwen3.5-35B-A3B --out runs/phase1_qwen
```

### 5. Analyse + tear down
```bash
.../python scripts/run_qwen_phase1.py --analyze-only --out runs/phase1_qwen   # Q1–Q4 report
vastai destroy instance <INSTANCE_ID>     # cost is wall-clock — destroy promptly
```

---

## Mode D — offline fallback (run on the box; no HTTP)
Use if the served `/v1/completions` logprobs/echo prove unreliable. Reads token-level logprobs
directly in-process, then copy the JSON back with `scp`.
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3.5-35B-A3B-Base", dtype="bfloat16", max_model_len=8192)
# stage 2 answer prompts already end in the ANSWER_SCAFFOLD; read top logprobs at the answer token:
out = llm.generate(answer_prompts, SamplingParams(temperature=0, max_tokens=1, logprobs=20))
# out[i].outputs[0].logprobs[0] -> {token_id: Logprob}; map A..E, softmax (same as elicit_base).
```
Mirror `elicit_base`'s A–E mapping + softmax so the artifacts match the HTTP path.

---

## Notes
- **Version:** the vast.ai *model page* is Qwen**3.6**-35B-A3B (instruct); Study A pins Qwen**3.5**-35B-A3B **Base + post**. Pick one version and confirm its **base** sibling exists on HF for whichever you serve.
- Per-cell elicitation = 2 calls (reason + answer) + up to 5 echo calls if a letter misses top-K; 120 cells is minutes.
- Add `vastai`, `curl http://*/v1/*` (+ `ssh` only if tunneling) to `.claude/settings.local.json` to cut permission prompts.
