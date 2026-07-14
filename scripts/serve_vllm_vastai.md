# Serving the Study A base/post pairs on vast.ai (cheap trial: Qwen + Gemma)

> For the linear account→data→shutdown walkthrough (custom template, exact commands, troubleshooting)
> see **`docs/vast_quickstart.md`**. This file is the *modes reference* it draws on.


**You call the model as an OpenAI-compatible HTTP API from your Mac.** You do not drive the
server's CLI for inference, and SSH is *optional* (a private alternative to exposing the port).
The model + vLLM flags are passed as the container's `--args`, so the box boots already serving (no
hand-run CLI). Our `elicit_post.py` / `elicit_base.py` speak OpenAI-compatible HTTP, so they work
against any endpoint below by setting `--base-url`.

## The four options (and which fits Study A)

| Mode | What | Fits Study A? |
|---|---|---|
| **A. Serverless** (`openai.vast.ai` proxy, autoscaling, scale-to-zero) | easiest, pay-per-use | **Post leg only** — instruct/chat; usually no base weights and no `/v1/completions`+logprobs |
| **B. One-click model template** (the vast.ai/model marketplace) | rent a preconfigured instance | **Neither leg** — the marketplace is **instruct-only; there is NO base model hosted** (e.g. `vast.ai/model/qwen35-35b-a3b` = instruct, no `-base`) |
| **C. Custom on-demand, generic `vllm/vllm-openai` image, `--model <HF repo>`** ← **PRIMARY (both legs)** | rent raw GPU; vLLM pulls the repo from **Hugging Face** at container start; hit `IP:port` | **Yes** — base weights + `/v1/completions` logprobs + bf16 + reason→answer control; same channel for base AND post |
| **D. Offline in-process** (no server) | run a script on the box: `vllm.LLM(...).generate(..., logprobs=20)`, copy results back | **Fallback** — most direct logit access if HTTP logprobs misbehave |

**vast.ai hosts no base checkpoint**, so we do not use the model marketplace (A/B) at all — **both
legs use Mode C** with `--model <HF repo>` (base AND post pulled from Hugging Face, keeping them in
the same token-slice channel). Consequences for provisioning:
- The **HF download is the dominant wall-clock/cost** (the box pulls weights at start, not your Mac) — filter offers for **`inet_down`** (bandwidth) and **`disk_space`**, and set a generous health-poll timeout.
- Needs a working **`HF_TOKEN`** and an **accepted license** for each gated repo (a gated 401 fails the in-container pull silently).
- **Avoid re-downloading:** reuse one box for both legs (SSH-restart vLLM on the post repo), and for the **giants** mount a **persistent volume as the HF cache** (`HF_HOME=/data` on the volume) so weights survive teardown/relaunch.
- (Optional cheaper post leg: `elicit_post.py` can still hit the post model via OpenRouter *verbalized* — no GPU — but that is a different channel from the base token-slice; only use it if you accept the cross-channel confound.)

---

## Mode C — primary runbook (one model per box; bf16; logprobs on by default)

> **One-command wrapper:** `scripts/provision_vast.sh up <hf_repo> [post]` does steps 1–2 (search →
> create with `--onstart-cmd` → poll health) and prints the base URL; `… down <id>` tears it down.
> The manual steps below are the reference it automates.
>
> vast CLI flags evolve — verify against `vastai --help` / docs. Cost is dominated by weight
> download + load, not the (few-minute) 120-cell inference, so minimize wall-clock.

### 1. Provision an on-demand instance (Qwen 35B-A3B: ~70 GB weights, but 32k context → 1×H200)
```bash
export VAST_API_KEY=$(security find-generic-password -s vastai-api-key -w)   # add to Keychain
export HF_TOKEN=$(security find-generic-password -s hf-token -w)
# gpu_ram>=140 (H200): 70 GB weights + a 32k-context KV cache for the ~18.3k-token corpus-v2 prompts
# (+2048 CoT budget => >=24576 required, pin 32768 — scripts/measure_prompt_budget.py) won't fit 80 GB.
# static_ip+direct_port_count = reachable endpoint; inet_down/reliability/inet_down_cost
# guard the HF pull (the cost driver).
vastai search offers \
  'gpu_ram>=140 num_gpus=1 static_ip=true direct_port_count>1 inet_down>1000 inet_down_cost<0.05 reliability>0.98 disk_space>192 cuda_vers>=12.4 rentable=true' \
  --order dph
# The vllm/vllm-openai image takes the model + flags as container --args; vLLM pulls the BASE repo
# from HF at start. --args must be LAST. Expose container :8000 to a public port.
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 192 \
  --env "-p 8000:8000 -e HF_TOKEN=$HF_TOKEN -e HF_HUB_ENABLE_HF_TRANSFER=1" \
  --args --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92
vastai show instance <INSTANCE_ID>     # read the public host:port mapped to 8000
```
Giants later: append `--tensor-parallel-size 8 [--pipeline-parallel-size 2] --enable-expert-parallel`
to the `--args`, raise `--disk`, and mount a persistent volume as the HF cache (`-e HF_HOME=/data`).

**Family serving quirks (verified 2026-07-14, `scripts/measure_prompt_budget.py`):**
- **Mistral-Large-3** (both legs) ships Mistral-native format — `params.json`, NO `config.json` —
  so vLLM needs `--config-format mistral --load-format mistral --tokenizer-mode mistral` appended
  to the `--args`. (Its HF `tokenizer.json` also over-counts ~21% vs the tekken tokenizer vLLM
  then actually serves with; the measured 18.3k worst prompt is the tekken count.)
- **Llama-4-Maverick** (both legs) is HF-**gated** and the `hf-token` account is NOT yet in the
  authorized list (403 as of 2026-07-14) — request/accept access on both repos BEFORE provisioning,
  or the in-container pull fails; then re-run the budget script for the llama rows.
- **Kimi-K2** tokenizer/config need `trust_remote_code` + `tiktoken` only for LOCAL measurement
  tooling; vLLM serves it natively — no extra flags.

### 2. Reach the endpoint from the Mac — pick ONE
- **Direct (default):** use the public `http://<host>:<port>` from `show instance`. No SSH.
- **Private (optional):** `ssh -N -L 8000:localhost:8000 root@<host> -p <ssh_port> &` then use `http://localhost:8000`.

```bash
curl -s http://<host>:<port>/v1/models   # health: lists the served model id
```

### 3. Elicit the base leg (from the Mac)
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py \
  --base-url http://<host>:<port> --base-model Qwen/Qwen3.5-35B-A3B-Base \
  --family qwen --out runs/qwen
```

### 4. Swap to the post model on the same box, re-run
```bash
# stop the base server, then relaunch (post reasons natively — live methodology).
# The reasoning parser is FAMILY-SPECIFIC: qwen3 for Qwen; Gemma's post (-it) takes NO parser.
vllm serve Qwen/Qwen3.5-35B-A3B --dtype bfloat16 --port 8000 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --reasoning-parser qwen3
conda run -n judex-arm python scripts/run_qwen_phase1.py --post-url http://<host>:<port> \
  --post-model Qwen/Qwen3.5-35B-A3B --family qwen --out runs/qwen
```

### 5. Analyse + tear down
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py --analyze-only --family qwen --out runs/qwen   # Q1–Q4
vastai destroy instance <INSTANCE_ID>     # cost is wall-clock — destroy promptly
```

### 6. Second trial family — Gemma 4 26B-A4B (quickstart §12)
Same Mode-C flow with base `google/gemma-4-26B-A4B` / post `google/gemma-4-26B-A4B-it`
(~50 GB bf16; the same offer query works), `--family gemma --out runs/gemma`, and **no reasoning
parser** on the post leg. Then merge the two families for the cross-family Q3 report:
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py \
  --merge qwen=runs/qwen gemma=runs/gemma --out runs/trial_cheap
```

---

## Mode D — offline fallback (run on the box; no HTTP)
Use if the served `/v1/completions` logprobs/echo prove unreliable. Reads token-level logprobs
directly in-process, then copy the JSON back with `scp`.
```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3.5-35B-A3B-Base", dtype="bfloat16", max_model_len=32768)
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
