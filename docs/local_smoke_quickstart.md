# Local smoke quick start — validate the Study A pipeline on your own GPU (no spend)

> **SMOKE (plumbing only) — this is NOT the study.** int4/Q4 + `--limit` + `--no-reason` + a small
> stand-in model ⇒ **τ_oc is MEANINGLESS; discard it, never paste its calibration block into
> `pipeline.yaml`.** For the real bf16 / all-120-cell / reasoning-ON measurement, see
> `docs/vast_quickstart.md` + `docs/vast_claude_code_orchestration.md`.

Goal: prove the whole pipeline works end to end — **serve a base+instruct pair (llama.cpp on the Mac,
or vLLM on a Linux/CUDA box) → token-slice logits → `study_a` report → calibration block** — on
hardware you own, *before* paying for the remote vast.ai run. It exercises the **machinery, not the
science**; the recommended first smoke runs on your Mac via **llama.cpp** (only topology B/C uses
vLLM, the exact server the paid vast run uses).

> **Smoke models are stand-ins, not panel models.** The six scientific models are fixed; a smaller
> Qwen (or Gemma) only validates the *machinery*, so its τ_oc numbers are **discarded**. Prefer a
> **Qwen** pre/post pair: it's **non-Google** (no evaluator-firewall issue), the **same lineage** as
> the panel's `qwen3.5-35b-a3b`, and `Qwen3-30B-A3B` even matches its **MoE / A3B** shape. (Gemma
> works too but is **Google-family** — a reserved *evaluator* — so it's a less-clean choice; see §9.)

## 0. Which machine

The driver/analysis (`run_qwen_phase1.py`, `study_a`) is pure CPU/Python — it runs anywhere the
repo + evaluator venv are. Only the **model server** needs a GPU. Three topologies:

- **(A) All-on-Mac — Apple Silicon (Metal).** For the **small default (`Qwen3-4B`)** the Mac is
  enough: a 4 B model is ~2.5 GB (Q4) / ~8 GB (fp16), fits 16 GB unified memory, and runs fast on
  Metal. **vLLM does *not* run on Apple Silicon**, so serve with a **Metal-native** OpenAI-compatible
  server (**llama.cpp**, below); `elicit_base` parses its logprobs shape too. Single machine, reuses
  the checkout you already have, no networking. **Recommended for the first `Qwen3-4B` smoke.** (Not
  for the 30B-A3B / 14B tiers — those need the 24 GB CUDA card.)
- **(B) All-on-Linux — RTX 3090-Ti (vLLM).** The *exact* Mode-C path the vast runbook uses (highest
  fidelity, zero code differences); required for the larger/MoE tiers. Clone the repos on the Linux
  box, serve vLLM on `localhost`, run the driver there too.
- **(C) Mac drives, Linux serves.** Keep the repos on the Mac, serve vLLM on the Linux box, point the
  driver at `http://<linux-lan-ip>:8000` (or an SSH tunnel).

> **Fidelity note:** topology A serves via llama.cpp, not vLLM, so it validates everything except
> vLLM's *exact* logprobs wire-shape — which is already covered by `tests/test_elicit_base.py` and
> re-confirmed with one `curl` when you spin up vast. For a plumbing smoke that's a fine trade for
> running entirely on the machine you're already set up on.

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

## 2·Mac. Run the smoke on Apple Silicon (Metal) — `Qwen3-4B` (topology A)

> **SMOKE ONLY.** The Q4/int4 weights, `--limit 6`, and `--no-reason` below all perturb/subsample the
> logits — the τ_oc this produces is **meaningless, discard it**. These settings must **never** be
> used on the paid vast run (bf16, all 120 cells, reasoning ON).

Everything on the Mac; no networking, no Linux box. Serve with **llama.cpp** (Metal); the driver
hits it on `localhost`. `elicit_base` parses llama.cpp's logprobs shape (server-agnostic).

**Setup (once):**
```bash
brew install llama.cpp                     # provides `llama-server` (Metal build)
# driver/analysis env (you likely already have this — it's the evaluator venv used all along):
python3 -m venv .venv-cal && source .venv-cal/bin/activate
pip install -e judex-evaluator && pip install -e judex-calibration
```

**Weights (GGUF).** Instruct is a ready GGUF; the base needs a one-time convert:
```bash
# POST (instruct): pulled straight from HF by llama-server (no manual download)
#   -> Qwen3-4B instruct GGUF: unsloth/Qwen3-4B-GGUF (e.g. :Q4_K_M, ~2.5 GB)
# PRE (base): convert once (no base GGUF is published)
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-4B-Base --local-dir ~/models/qwen3-4b-base
python "$(brew --prefix llama.cpp)/libexec/convert_hf_to_gguf.py" \
  ~/models/qwen3-4b-base --outfile ~/models/qwen3-4b-base-Q4_K_M.gguf --outtype q4_k_m
```
(Shortcut for a *pure* plumbing check: skip the base and run the **instruct through both legs** —
τ_oc ≈ 1 is meaningless but it still proves serve→token-slice→analysis→block. Use the real base
when you want a non-trivial number.)

**Serve the BASE leg (terminal 1):**
```bash
llama-server -m ~/models/qwen3-4b-base-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8000 --ctx-size 32768 --n-gpu-layers 999
# health + confirm logprobs on /v1/completions (the channel elicit_base uses):
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Answer:","n_predict":1,"max_tokens":1,"logprobs":20,"n_probs":20}' | head -c 400
```
`--n-gpu-layers 999` offloads all layers to the Metal GPU. `llama-server` serves `/v1/completions`
raw (no chat template — correct for the base leg). If the health curl shows no logprobs, update
llama.cpp (`brew upgrade llama.cpp`).

**Run BASE → swap to POST → analyse (terminal 2):**
```bash
cd judex/judex-calibration
../.venv-cal/bin/python scripts/run_qwen_phase1.py \
  --base-url http://127.0.0.1:8000 --base-model qwen3-4b-base \
  --out runs/smoke_qwen_mac --limit 6 --no-reason        # -> pre.json

# terminal 1: Ctrl-C, then serve the instruct straight from HF:
llama-server -hf unsloth/Qwen3-4B-GGUF:Q4_K_M --host 127.0.0.1 --port 8000 --ctx-size 32768 --n-gpu-layers 999
# terminal 2:
../.venv-cal/bin/python scripts/run_qwen_phase1.py \
  --post-url http://127.0.0.1:8000 --post-model qwen3-4b-instruct \
  --out runs/smoke_qwen_mac --limit 6 --no-reason        # -> post.json

../.venv-cal/bin/python scripts/run_qwen_phase1.py --analyze-only --out runs/smoke_qwen_mac
```
(`--base-model` / `--post-model` are just labels for llama.cpp — the served model is whatever `-m` /
`-hf` loaded.) Then read the success checklist in §7. For anything bigger than `Qwen3-4B`, use the
Linux/vLLM path below.

## 2. One-time setup on the Linux box (topology B/C — vLLM)

```bash
# repos (topology B, all-on-Linux) — recursive submodules
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

**Ignore the actual values** (a small/quantized Qwen on 6 cells is scientifically meaningless — the run
auto-flags `smoke` in the report + calibration block). If all of the above appear, the
serve→token-slice→analysis→integration path is proven (on the Mac/topology-A path this covers everything
**except** vLLM's exact logprobs wire-shape — re-confirm that with one `curl` when you spin up vast, per
the §0 fidelity note). You can then commit to the paid vast run with the six real (bf16) panel models.

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
