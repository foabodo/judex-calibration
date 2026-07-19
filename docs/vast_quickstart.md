# Vast.ai quick start — Study A Phase 1 (the two-family cheap trial: Qwen + Gemma), account → data → shutdown

End-to-end walkthrough for the **pre-Study A trial**: collecting the TWO cheap-tier Study A
families — **Qwen 3.5-35B-A3B** (§5–§10) and **Gemma 4-26B-A4B** (§12, same flow) — on rented GPUs
using a **custom template** + the `vllm/vllm-openai` server, then tearing down. Two families from
distinct lineages validate the methodology **cross-family** (a first Q3 τ_oc spread, §13) before any
mid/giant spend, for ~$13–18 total. You serve each model as an OpenAI-compatible HTTP endpoint and
call it from your Mac; the box's CLI is not used for inference.

> **LIVE experiment** — vast.ai GPU, **vLLM, bf16, all 120 cells, reasoning ON**: this run produces
> the **real τ_oc**. Never use `--limit`/`--no-reason`/int4 here — those belong to the *free* Mac
> plumbing smoke (`docs/local_smoke_quickstart.md`), whose τ_oc is meaningless. Run that smoke first if
> you haven't.

> **Prefer to let Claude Code drive the whole run on the box** (serve → elicit → swap → analyse →
> teardown, troubleshooting autonomously)? See `docs/vast_claude_code_orchestration.md` — it installs
> Claude Code on this instance and hands it a ready-made orchestration brief. This quick start is the
> manual, from-the-Mac version it builds on.

**What we're collecting (Study A context):** per-cell 5-way compliance distributions for the
**base** (pretrained) and **post** (instruct) variants of BOTH cheap families on the 120 AIReg
cells, via token-slicing logits over A–E after a reasoning span (`elicit_base.py`). The base leg is
the reason for going to vLLM at all — it needs **base weights + `/v1/completions` logprobs + bf16**,
which the one-click/serverless paths don't give. Phase 0 (the free accuracy gate) is already done;
this is the first paid step (~$5–10 Qwen + ~$8 Gemma). Each family gets its own run dir
(`--family qwen --out runs/qwen`, `--family gemma --out runs/gemma`); §13 merges them into the
cross-family report.

> **vast.ai hosts NO base model — we download BOTH variants from Hugging Face.** The vast "Models"
> marketplace/templates are instruct-only (`vast.ai/model/qwen35-35b-a3b` = the instruct/thinking
> model; there is no `-base`). So we do **not** use the model marketplace at all: we rent a generic
> GPU with the vanilla `vllm/vllm-openai` image and let vLLM pull each repo from HF via `--model`:
> - post: `Qwen/Qwen3.5-35B-A3B` — https://huggingface.co/Qwen/Qwen3.5-35B-A3B
> - base: `Qwen/Qwen3.5-35B-A3B-Base` — https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base
>
> Consequences (baked into the steps below): the **HF download is now the dominant wall-clock/cost**,
> so filter offers for **bandwidth + disk**; ensure the **HF token works and the model license is
> accepted**; prefer **reusing one box for both legs** (or a persistent volume for the giants) to
> avoid re-downloading. Use **bf16** throughout (fp8 perturbs the very logits we measure). The base
> (non-instruct) model has no chat template — serve + call it on **`/v1/completions` only** (which is
> exactly the token-slice channel we use).

---

## 1. Create the account + add credit
1. Sign up at https://cloud.vast.ai and verify email.
2. **Billing → add credit** (vast is prepaid; ~$25 is plenty for Qwen). No credit = instances won't start.

## 2. Install the CLI + set the API key (store in Keychain)
```bash
pip install --upgrade vastai
# API key from https://cloud.vast.ai/cli/
security add-generic-password -U -a "$USER" -s vastai-api-key -w '<YOUR_KEY>'      # one-time: stash in Keychain
vastai set api-key "$(security find-generic-password -s vastai-api-key -w)"
```

## 3. Hugging Face token + accept the model license
1. On https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base **and** the post repo — plus the Gemma pair
   (`google/gemma-4-26B-A4B` / `-it`) for §12 — click through / accept the license if the repo is
   gated (Qwen and Gemma 4 are Apache-2.0/ungated, but confirm — a gated repo makes the
   in-container download fail silently with a 401).
2. Verify your read token can fetch them, then store it in Keychain:
```bash
security add-generic-password -U -a "$USER" -s hf-token -w '<HF_READ_TOKEN>'        # one-time
HF_TOKEN=$(security find-generic-password -s hf-token -w) \
  hf download Qwen/Qwen3.5-35B-A3B-Base --revision main --dry-run   # confirms access
```
(Optional: also add it under **Account → Environment Variables** in the console so every instance
gets it regardless of template.)

## 4. Create a custom template (the Mode C config, once)
Console: **Templates → + New Template** and set:
- **Image:** `vllm/vllm-openai:latest`
- **Docker options / ports:** expose `-p 8000:8000` (add port **8000 TCP**)
- **Environment variables:** `HF_TOKEN = <your token>` (or rely on the account-level env var)
- **Launch arguments** (the container args appended to the vLLM server) — **base** leg:
  ```
  --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92
  ```
  (`--max-model-len 32768`: under **corpus v2** the live prompt is ≈ **18.3 k tokens** worst-case —
  k=4 v2 few-shot ≈ 7 k + the full TechOps evidence ≈ 10 k — plus the 2048-token CoT `--budget`,
  so **≥ 24576 is required** (measured 2026-07-14, `scripts/measure_prompt_budget.py`); 32768 is
  the standard pin with headroom for a raised budget. **Disk Space: 192 GB** — image + ~70 GB
  weights + HF cache/headroom; this also fits both base+post if you reuse the box. **Launch mode:
  Docker ENTRYPOINT** — the image entrypoint serves these args; leave the **on-start script
  empty**.)
- Save as e.g. `study-a-vllm-base`.

CLI equivalent (or just use `scripts/provision_vast.sh`, which does search→create→poll→print-URL):
```bash
vastai create template --name study-a-vllm-base --image vllm/vllm-openai:latest \
  --env '-p 8000:8000 -e HF_TOKEN=<token>' \
  --args --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92
```

## 5. Find a suitable GPU offer
Qwen 35B-A3B bf16 ≈ 70 GB weights, **but our corpus-v2 prompts (~18.3 k tokens + the 2048 CoT
budget) need the 32 k-context KV cache too** (only ~1.6–2.6 GB/seq for Qwen — weights, not KV,
are the binding constraint) — so size by VRAM: **1×H200 (141 GB), `num_gpus=1`**, not an 80 GB card. `static_ip=true`
+ `direct_port_count>1` give the public `IP:port`; `inet_down` gates the ~70 GB HF pull;
`reliability>0.98` keeps a host from dropping mid-download; `inet_down_cost` low avoids per-GB
bandwidth charges; `disk_space>192` holds weights + cache (192 GB also fits both legs on one box).
```bash
vastai search offers \
  'gpu_ram>=140 num_gpus=1 static_ip=true direct_port_count>1 inet_down>1000 inet_down_cost<0.05 reliability>0.98 disk_space>192 cuda_vers>=12.4 rentable=true' \
  --order dph    # cheapest $/hr first; note the OFFER_ID
```

## 6. Launch the base instance (from the template / CLI)
Console: open the template → pick the offer → **Rent**. Or CLI:
```bash
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 192 \
  --env "-p 8000:8000 -e HF_TOKEN=$(security find-generic-password -s hf-token -w)" \
  --args --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92
# note the INSTANCE_ID it prints
```

## 7. Get the endpoint + wait until healthy
```bash
vastai show instance <INSTANCE_ID>           # read public IP + the host port mapped to 8000
URL=http://<IP>:<PORT>
curl -s "$URL/v1/models"                      # 200 + the model id once download+load finish (minutes)
vastai logs <INSTANCE_ID>                      # watch download/load progress if not ready
```
(`scripts/provision_vast.sh up Qwen/Qwen3.5-35B-A3B-Base` automates steps 5–7 and prints `$URL`.)

> **Fork here — you now have a healthy endpoint.** Continue manually with §8–§11 below, **or** hand
> the collect → swap → analyse → teardown loop to **Claude Code running on the box** — it automates
> §8–§11 and troubleshoots OOM/context/logprobs for you. Bootstrap `scripts/provision_claude_code.sh`
> and give it the brief: see `docs/vast_claude_code_orchestration.md`.

## 8. Collect the BASE leg (from the Mac)
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py \
  --base-url "$URL" --base-model Qwen/Qwen3.5-35B-A3B-Base --family qwen --out runs/qwen
# writes runs/qwen/pre.json (120 token-sliced distributions)
```
Quick sanity before the full run: `curl -s "$URL/v1/completions" -H 'Content-Type: application/json' \
-d '{"model":"Qwen/Qwen3.5-35B-A3B-Base","prompt":"Answer:","max_tokens":1,"logprobs":20}'` should
return `logprobs` — if it doesn't, fall back to Mode D (offline `LLM.generate`, see serve_vllm_vastai.md).

## 9. Switch to the POST leg
Both repos come from HF, so the post weights download regardless (reusing a box saves only
re-provisioning, not the download). Two options:
- **Fresh instance (recommended with ENTRYPOINT mode):** destroy the base box (step 11), set the
  template `--model` → `Qwen/Qwen3.5-35B-A3B` and add `--reasoning-parser qwen3`, launch, repeat 6–7.
- **Reuse one box (needs the SSH launch mode, not ENTRYPOINT):** the 192 GB disk holds both models;
  `ssh` in, stop the base server, relaunch on the post repo:
  ```bash
  ssh root@<host> -p <ssh_port>
  pkill -f 'vllm serve' ; sleep 3
  vllm serve Qwen/Qwen3.5-35B-A3B --dtype bfloat16 --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 --reasoning-parser qwen3   # post reasons natively (live methodology)
  ```

Then collect the post leg:
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py --post-url "$URL" \
  --post-model Qwen/Qwen3.5-35B-A3B --family qwen --out runs/qwen      # writes post.json
```

## 10. Analyse — the Q1–Q4 report
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py --analyze-only --family qwen --out runs/qwen
```
Reads `pre.json`/`post.json`, joins to the AIReg human GT, and emits `study_a_report.json`:
- **Q1** `T*_pre` (≈1 if the base is well-calibrated), **Q2** post `T*`/`τ_oc` + argmax retention,
- **Q3** `tau_oc_summary` (cross-family stability — meaningful once ≥2 families are in; the trial's
  second family (§12) + the merge (§13) give the first real spread),
- **Q4** `closed_side_check` (the median `τ_oc` applied to the closed pair's (Claude/GPT) AIReg outputs: does Murphy
  Reliability drop without hurting Resolution/RPS? — needs a Claude+GPT run; the legacy gemini-gpt run is
  Gemini/GPT, and **as of 2026-07-18 no 120-cell on-pair run exists** (guide §4.8). If the run dir is
  absent — `runs/` is gitignored, so that is the default on a fresh box — the report records
  `closed_side_check_Q4: {skipped: true, …}` and prints a `[Q4] SKIPPED` warning rather than silently
  omitting the question).
**Accuracy gate:** if `argmax_acc` is low and both `T*` peg at the search bound, Qwen failed the gate
the same way the closed evaluators did — record it and reconsider before serving the giants. The
report makes pegs explicit: `T_rps_saturated`/`T_rel_saturated`/`tau_oc_saturated` (a boundary value
is an artefact, not a fit), plus `tau_oc_reference_degenerate` when the `pre` leg itself pegged so
`τ_oc` aligns to a reference that could not be fit; the driver prints `[PEGGED]` instead of the
`[integrate]` paste banner in either case.

## 11. Shut down (do this promptly — cost is wall-clock)
```bash
vastai destroy instance <INSTANCE_ID>        # or scripts/provision_vast.sh down <INSTANCE_ID>
vastai show instances                         # confirm nothing is still running/billing
```
Cost is dominated by weight download + load, not the few-minute inference. Two sequential
instances (base then post) for Qwen ≈ **$5–10** total.

## 12. Second family — Gemma 4 26B-A4B (same flow, different repos)

The trial's second cheap family, from a different lineage (Google vs Alibaba), turns the analysis
cross-family. Repeat §5–§11 with only these substitutions:

- **Repos:** base `google/gemma-4-26B-A4B`, post `google/gemma-4-26B-A4B-it` (both ungated
  Apache-2.0 on HF; ~50 GB bf16 MoE, 25.2B total / 3.8B active — *smaller* than Qwen).
- **Offers/template:** the SAME §5 query and §4 template work (Gemma might squeeze onto an 80 GB
  card, but the H200 query removes OOM risk and the card is not the cost driver). Same
  `--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92`; disk 192 GB holds both legs.
- **NO reasoning parser on the post leg** — Gemma 4 has no separate reasoning control (unlike Qwen's
  `--reasoning-parser qwen3`). `scripts/provision_vast.sh up google/gemma-4-26B-A4B-it post` handles
  this automatically; if relaunching by hand, just omit the flag. The base leg still reasons via the
  few-shot CoT scaffold (matched condition, same as Qwen).
- **Driver calls:** use the family's own run dir —
  ```bash
  conda run -n judex-arm python scripts/run_qwen_phase1.py \
    --base-url "$URL" --base-model google/gemma-4-26B-A4B --family gemma --out runs/gemma
  # ... swap to the post leg (§9), then:
  conda run -n judex-arm python scripts/run_qwen_phase1.py --post-url "$URL" \
    --post-model google/gemma-4-26B-A4B-it --family gemma --out runs/gemma
  ```
- **Token check:** before the full base run, re-run the §8 `curl` logprobs sanity against the Gemma
  endpoint — the A–E answer tokens must come back in the top-K under the **Gemma tokenizer**
  (`configs/models.yaml` `scale.answer_tokens` is verify-per-tokenizer; the echo fallback covers
  stragglers, but 5/5 in top-K is the healthy signal).
- **Cost:** ≈ **$8** (smaller weights than Qwen; same wall-clock shape).

Note: `configs/models.yaml` pins Gemma's post leg to `vllm_hf` (self-hosted, same stack as the
base) — per the user (2026-07-03) this IS the plan for every family: both legs from HF weights on
vast vLLM. The OpenRouter route (verified available, Novita bf16) is a just-in-case fallback only,
to be invoked solely if unquantized-model logits cannot feasibly be obtained from the HF weights;
it is not planned to be used.

## 13. Merge — the first cross-family report
With both family dirs complete (each leg 120 cells):
```bash
conda run -n judex-arm python scripts/run_qwen_phase1.py \
  --merge qwen=runs/qwen gemma=runs/gemma --out runs/trial_cheap
```
`runs/trial_cheap/study_a_report.json` then carries both families plus a 2-family
`tau_oc_summary` (median/spread — the first Q3 signal) and the Q4 closed-side check at the
cross-family median. The merge inherits `smoke` from any source dir with a `.smoke` sentinel or a
<120-cell leg — a partial leg cannot launder into a clean trial report. **Gate to the mid tier
(guide §7):** both families clear the accuracy gate with finite, plausibly-clustered τ_oc.

---

## Troubleshooting / notes
- **No logprobs returned:** the routed server must be `vllm/vllm-openai` on `/v1/completions` (not a
  chat-only proxy). If absent, use **Mode D** (offline `vllm.LLM(...).generate(..., logprobs=20)`).
- **OOM / won't load:** raise `--disk`, lower `--gpu-memory-utilization` or `--max-model-len`, or pick
  a bigger-VRAM offer. Giants (later families) need `--tensor-parallel-size 8 [--pipeline-parallel-size 2] --enable-expert-parallel`.
- **Endpoint unreachable:** the offer lacked `static_ip=true` / `direct_port_count>1`; re-pick.
- **Weights download from HF every launch (vast caches nothing for us):** size `--disk` for the repo
  (~70 GB Qwen per leg; the standard **192 GB** holds both cheap-trial legs plus image + cache, as
  §4/§5 say — size up per `configs/models.yaml` for the mid/giant families), prefer
  high-`inet_down` offers, and gate the
  HF token/license (step 3). vLLM downloads into the container HF cache; for very large repos set
  `-e HF_HOME=/workspace/hf` (a big mounted path) so the cache lands on the rented disk, not `/`.
- **Giants (later families) need a caching strategy:** re-pulling 0.7–2 TB from HF per launch is slow
  and costly. Use a vast **persistent volume**: download each base/post once onto the volume, then
  attach it to instances (point `HF_HOME`/`--download-dir` at it) so subsequent launches skip the
  download. For one-off Qwen this isn't worth it — just reuse one box (step 9).
- **Keys:** never echo them; always `security find-generic-password -s <name> -w` inline.
- **Permissions:** add `vastai`, `curl http://*/v1/*` (and `ssh` only if you tunnel) to
  `.claude/settings.local.json` to cut prompts.

Sources: [vast.ai CLI/quickstart + vLLM serving](https://vast.ai/article/serving-online-inference-with-vllm-api-on-vast),
[Creating a custom template](https://docs.vast.ai/creating-a-custom-template),
[Template settings](https://docs.vast.ai/documentation/templates/template-settings),
[OpenAI-compatible API](https://docs.vast.ai/guides/serverless/openai-compatible-api).
