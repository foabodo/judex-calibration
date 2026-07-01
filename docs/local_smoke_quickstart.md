# Local smoke quick start — validate the Study A pipeline on your own GPU (no spend)

Goal: prove the whole pipeline works end to end — **serve a base+instruct pair on vLLM →
token-slice logits → `study_a` report → drop-in calibration block** — on hardware you own,
*before* paying for the remote vast.ai run. This is a **plumbing test**, not science.

> **Smoke models are stand-ins, not panel models.** The six scientific models are fixed; a smaller
> Qwen (or Gemma) only validates the *machinery*, so its τ_oc numbers are **discarded**. Prefer a
> **Qwen** pre/post pair: it's **non-Google** (no evaluator-firewall issue), the **same lineage** as
> the panel's `qwen3.5-35b-a3b`, and `Qwen3-30B-A3B` even matches its **MoE / A3B** shape. (Gemma
> works too but is **Google-family** — a reserved *evaluator* — so it's a less-clean choice; see §9.)

## 0. Which machine

- **Your Mac (16 GB RAM) cannot serve this.** The pipeline drives a **vLLM** `/v1/completions`
  endpoint with logprobs; vLLM needs **CUDA**, and these models won't fit 16 GB RAM anyway.
- **Your Linux RTX 3090-Ti box is the target.** It is the "local vLLM server" — the *exact* Mode-C
  path the vast runbook uses, so the smoke exercises the real code. Two topologies:
  - **(A) All-on-Linux (simplest):** clone the repos on the Linux box, serve vLLM on `localhost`,
    and run the driver there too. No networking. **Recommended for the first smoke.**
  - **(B) Mac drives, Linux serves:** keep the repos on the Mac, serve vLLM on the Linux box, and
    point the driver at `http://<linux-lan-ip>:8000` (or an SSH tunnel). Reuses the Mac checkout.

## 1. Pick a smoke pair (all verified base+instruct on HF; fit a 24 GB 3090-Ti)

| Pair (base + instruct) | Arch | Total | On 24 GB | When |
|---|---|---|---|---|
| **`Qwen/Qwen3-4B-Base` + `Qwen/Qwen3-4B`** | dense | 4 B | **bf16, easy** | **default — first green-light smoke** |
| `Qwen/Qwen2.5-7B` + `Qwen/Qwen2.5-7B-Instruct` | dense | 7.6 B | bf16, comfortable | a slightly bigger clean bf16 run |
| `Qwen/Qwen3-14B-Base` + `Qwen/Qwen3-14B` | dense | 14.8 B | int4 (bf16 no) | dense mid, Qwen3-gen |
| **`Qwen/Qwen3-30B-A3B-Base` + `Qwen/Qwen3-30B-A3B`** | **MoE** (3.3 B active) | 30.5 B | int4, tight | **MoE serving fidelity** (matches the panel shape) |

Notes: your **3090-Ti is Ampere → no FP8 hardware**; the only quant path is **INT4**
(bitsandbytes on-the-fly, or an AWQ/GPTQ repo). **`Qwen3-4B` needs no quant** (bf16 fits, so no
logit confound — the cleanest smoke). **`Qwen3-32B` is unusable** — Qwen never released a 32B *base*
checkpoint, so there's no pre/post pair. Keep the **KV cache at default (bf16)** for the 4B/7B; for
the 30B int4 you'll need `int8` KV (below). vLLM **≥ 0.8.5** serves Qwen3 dense + MoE.

## 2. One-time setup on the Linux box

```bash
# repos (topology A) — recursive submodules
git clone --recurse-submodules git@github.com:foabodo/judex.git && cd judex
# analysis/driver env (needs numpy + judex, same as the Mac driver):
python -m venv .venv-cal && source .venv-cal/bin/activate
pip install -e judex-evaluator            # brings numpy, POT, pyyaml, judex
pip install -e judex-calibration
# vLLM server env (separate):
python -m venv .venv-vllm && ./.venv-vllm/bin/pip install -U 'vllm>=0.8.5' bitsandbytes
export HF_TOKEN=...        # Qwen is Apache-2.0/ungated, but a token avoids rate limits
```

## 3. Serve the BASE leg (Linux box, terminal 1)

**Default — Qwen3-4B, bf16 (no quant):**
```bash
./.venv-vllm/bin/vllm serve Qwen/Qwen3-4B-Base \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000
# wait for "Application startup complete", then:
curl -s http://localhost:8000/v1/models
# confirm logprobs on /v1/completions (the channel elicit_base uses):
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Base","prompt":"Answer:","max_tokens":1,"logprobs":20}' | head -c 400
```

**MoE-fidelity variant — Qwen3-30B-A3B, int4** (do this *after* the 4B smoke passes; tighter):
```bash
./.venv-vllm/bin/vllm serve Qwen/Qwen3-30B-A3B-Base \
  --quantization bitsandbytes --load-format bitsandbytes --dtype bfloat16 \
  --max-model-len 24576 --gpu-memory-utilization 0.90 \
  --kv-cache-dtype int8_per_token_head \
  --host 0.0.0.0 --port 8000
```
This exercises vLLM's **MoE routing / fused-MoE kernels** (the model loads all 128 experts on the one
card). True **expert-parallel** *sharding* needs multiple GPUs (`--tensor-parallel-size N
--enable-expert-parallel`) — that's what the vast giants use; not applicable single-card. If vLLM's
bnb-on-MoE path misbehaves, fall back to `Qwen3-14B` (dense int4) or the `Qwen3-4B` default. If OOM:
lower `--max-model-len` / `--gpu-memory-utilization`.

## 4. Run the BASE leg (driver — terminal 2 on Linux, or the Mac for topology B)

Start tiny: **6 cells** (one per Article) with `--no-reason` (skips CoT → fastest; validates the
answer-logit path). `URL` = `http://localhost:8000` (A) or `http://<linux-lan-ip>:8000` (B).

```bash
cd judex/judex-calibration
../.venv-cal/bin/python scripts/run_qwen_phase1.py \
  --base-url "$URL" --base-model Qwen/Qwen3-4B-Base \
  --out runs/smoke_qwen --limit 6 --no-reason
# -> writes runs/smoke_qwen/pre.json (6 token-sliced distributions)
```

## 5. Swap to the POST leg (same box) and run it

```bash
# terminal 1: Ctrl-C the base server, then serve the INSTRUCT variant (same channel):
./.venv-vllm/bin/vllm serve Qwen/Qwen3-4B \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000
# terminal 2: same --limit so pre/post cover the same cells:
../.venv-cal/bin/python scripts/run_qwen_phase1.py \
  --post-url "$URL" --post-model Qwen/Qwen3-4B \
  --out runs/smoke_qwen --limit 6 --no-reason
# -> writes runs/smoke_qwen/post.json
```
(For the 30B-A3B variant, swap the two model ids to `Qwen/Qwen3-30B-A3B-Base` / `Qwen/Qwen3-30B-A3B`
and reuse the int4 serve command from §3.)

## 6. Analyse (no server needed) → the Q1–Q4 report + calibration block

```bash
../.venv-cal/bin/python scripts/run_qwen_phase1.py --analyze-only --out runs/smoke_qwen
# reads pre.json/post.json, joins to CANONICAL AIReg GT, writes:
#   runs/smoke_qwen/study_a_report.json          (Q1 T*_pre, Q2 τ_oc, Q3 tau_oc_summary, Q4 closed-side)
#   runs/smoke_qwen/pipeline_calibration_block.json  (drop-in evaluator block)
```

## 7. What "the plumbing works" looks like

You are testing the **machinery**, not the calibration. Success =:
- `pre.json` / `post.json` each hold N entries of 5 finite probabilities that sum to 1;
- the elicitation logged `method: topk` (or `echo` fallback) with `covered: 5` — logprobs over A–E
  were actually read;
- `study_a_report.json` has a `qwen` family row with numeric `T_rps`, `T_rel`, `murphy{}`, a finite
  `tau_oc`, and a `closed_side_check_Q4` block;
- `pipeline_calibration_block.json` was written (`mode: temperature`).

**Ignore the actual values** (a small/quantized Qwen on 6 cells is scientifically meaningless). If all
of the above appear, the vLLM→token-slice→analysis→integration path is proven and you can commit to the
paid vast run with the six real (bf16) panel models.

Then, optionally, a fuller local dry-run: drop `--no-reason` (CoT; keep `--max-model-len 32768`) and
raise `--limit` (e.g. `24`) to exercise the reasoning path and per-Article few-shot at scale.

## 8. Teardown / notes
- Stop vLLM (Ctrl-C in terminal 1). No standing cost — it's your hardware.
- `runs/` is gitignored; keep `study_a_report.json` if you want a record of the smoke.
- Cross-checks with the vast path: same code (`elicit_base`, `run_qwen_phase1`), same
  `/v1/completions` logprobs channel — only the box and (bf16 vs int4) dtype differ.
- Reachability (topology B): open port 8000 on the Linux box, or tunnel from the Mac with
  `ssh -N -L 8000:localhost:8000 <user>@<linux-host>` and use `http://localhost:8000`.

## 9. Other options (why Qwen is preferred)
- **Gemma 4** (`google/gemma-4-12B`(+`-it`), or the MoE `26B-A4B`, or dense `31B`) also fits int4 on
  24 GB, but Gemma is **Google-family** — a reserved JUDEX *evaluator* — so even as a smoke it's a
  less-clean choice than a Qwen pair. Use only if you specifically want a non-Qwen sanity check.
- **`Qwen3-8B`** (`-Base` + plain) is fine too but bf16 is tight on 24 GB (needs int8 KV); the `4B`
  (clean bf16) or `2.5-7B` (comfortable bf16) are easier.
- Any pair must have **both** a base *and* an instruct repo — that's why `Qwen3-32B` (instruct-only)
  and most "chat-only" models are out.
