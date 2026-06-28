# Vast.ai quick start — Study A Phase 1 (Qwen), account → data → shutdown

End-to-end walkthrough for collecting one Study A family (Qwen) on a rented GPU using a **custom
template** + the `vllm/vllm-openai` server, then tearing it down. You serve each model as an
OpenAI-compatible HTTP endpoint and call it from your Mac; the box's CLI is not used for inference.

**What we're collecting (Study A context):** per-cell 5-way compliance distributions for the
**base** (pretrained) and **post** (instruct) Qwen variants on the 120 AIReg cells, via
token-slicing logits over A–E after a reasoning span (`elicit_base.py`). The base leg is the
reason for going to vLLM at all — it needs **base weights + `/v1/completions` logprobs + bf16**,
which the one-click/serverless paths don't give. Phase 0 (the free accuracy gate) is already done;
this is the first paid step (~$5–10 for Qwen).

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
security add-generic-password -s vastai-api-key -w '<YOUR_KEY>'      # one-time: stash in Keychain
vastai set api-key "$(security find-generic-password -s vastai-api-key -w)"
```

## 3. Hugging Face token + accept the model license
1. On https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base **and** the post repo, click through / accept
   the license if the repo is gated (Qwen is usually Apache-2.0/ungated, but confirm — a gated repo
   makes the in-container download fail silently with a 401).
2. Verify your read token can fetch them, then store it in Keychain:
```bash
security add-generic-password -s hf-token -w '<HF_READ_TOKEN>'        # one-time
HF_TOKEN=$(security find-generic-password -s hf-token -w) \
  huggingface-cli download Qwen/Qwen3.5-35B-A3B-Base --revision main --dry-run   # confirms access
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
  --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92
  ```
- Save as e.g. `study-a-vllm-base`.

CLI equivalent (or just use `scripts/provision_vast.sh`, which does search→create→poll→print-URL):
```bash
vastai create template --name study-a-vllm-base --image vllm/vllm-openai:latest \
  --env '-p 8000:8000 -e HF_TOKEN=<token>' \
  --args --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92
```

## 5. Find a suitable GPU offer
Qwen 35B/3B bf16 ≈ 70 GB → 1×H100-80. **`static_ip=true` and `direct_port_count>1` are required**
for the public `IP:port`; **`inet_down`** matters because the ~70 GB HF pull is now the gating cost;
**`disk_space>200`** to hold the weights (≥400 if you reuse one box for both legs — step 9).
```bash
vastai search offers \
  'compute_cap>=800 gpu_ram>=80 num_gpus=1 static_ip=true direct_port_count>1 inet_down>1000 disk_space>200 cuda_vers>=12.4 rentable=true' \
  --order dph    # cheapest $/hr first; note the OFFER_ID
```

## 6. Launch the base instance (from the template / CLI)
Console: open the template → pick the offer → **Rent**. Or CLI:
```bash
vastai create instance <OFFER_ID> --image vllm/vllm-openai:latest --disk 200 \
  --env "-p 8000:8000 -e HF_TOKEN=$(security find-generic-password -s hf-token -w)" \
  --args --model Qwen/Qwen3.5-35B-A3B-Base --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.92
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

## 8. Collect the BASE leg (from the Mac)
```bash
.../judex-evaluator/.venv/bin/python scripts/run_qwen_phase1.py \
  --base-url "$URL" --base-model Qwen/Qwen3.5-35B-A3B-Base --out runs/phase1_qwen
# writes runs/phase1_qwen/pre.json (120 token-sliced distributions)
```
Quick sanity before the full run: `curl -s "$URL/v1/completions" -H 'Content-Type: application/json' \
-d '{"model":"Qwen/Qwen3.5-35B-A3B-Base","prompt":"Answer:","max_tokens":1,"logprobs":20}'` should
return `logprobs` — if it doesn't, fall back to Mode D (offline `LLM.generate`, see serve_vllm_vastai.md).

## 9. Switch to the POST leg
Because both repos come from HF, re-provisioning means a second ~70 GB download. Two options:
- **Reuse one box (recommended — avoids the second download):** keep the instance (size it
  `disk_space>400` in step 5), `ssh` in, stop the base vLLM server, and relaunch on the post repo:
  ```bash
  ssh root@<host> -p <ssh_port>
  pkill -f 'vllm serve' ; sleep 3
  vllm serve Qwen/Qwen3.5-35B-A3B --dtype bfloat16 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.92 --reasoning-parser qwen3   # post reasons natively (live methodology)
  ```
- **Fresh instance:** destroy the base box (step 11), edit the template `--model` →
  `Qwen/Qwen3.5-35B-A3B` + add `--reasoning-parser qwen3`, launch, repeat steps 6–7.

Then collect the post leg:
```bash
.../python scripts/run_qwen_phase1.py --post-url "$URL" \
  --post-model Qwen/Qwen3.5-35B-A3B --out runs/phase1_qwen      # writes post.json
```

## 10. Analyse — the Q1–Q4 report
```bash
.../python scripts/run_qwen_phase1.py --analyze-only --out runs/phase1_qwen
```
Reads `pre.json`/`post.json`, joins to the AIReg human GT, and emits `study_a_report.json`:
- **Q1** `T*_pre` (≈1 if the base is well-calibrated), **Q2** post `T*`/`τ_oc` + argmax retention,
- **Q3** `tau_oc_summary` (cross-family stability — meaningful once ≥2 families are in),
- **Q4** `closed_side_check` (the median `τ_oc` applied to the Gemini/GPT AIReg outputs: does Murphy
  Reliability drop without hurting Resolution/RPS?).
**Accuracy gate:** if `argmax_acc` is low and both `T*` peg at the search bound, Qwen failed the gate
the same way the closed evaluators did — record it and reconsider before serving the giants.

## 11. Shut down (do this promptly — cost is wall-clock)
```bash
vastai destroy instance <INSTANCE_ID>        # or scripts/provision_vast.sh down <INSTANCE_ID>
vastai show instances                         # confirm nothing is still running/billing
```
Cost is dominated by weight download + load, not the few-minute inference. Two sequential
instances (base then post) for Qwen ≈ **$5–10** total.

---

## Troubleshooting / notes
- **No logprobs returned:** the routed server must be `vllm/vllm-openai` on `/v1/completions` (not a
  chat-only proxy). If absent, use **Mode D** (offline `vllm.LLM(...).generate(..., logprobs=20)`).
- **OOM / won't load:** raise `--disk`, lower `--gpu-memory-utilization` or `--max-model-len`, or pick
  a bigger-VRAM offer. Giants (later families) need `--tensor-parallel-size 8 [--pipeline-parallel-size 2] --enable-expert-parallel`.
- **Endpoint unreachable:** the offer lacked `static_ip=true` / `direct_port_count>1`; re-pick.
- **Weights download from HF every launch (vast caches nothing for us):** size `--disk` for the repo
  (~70 GB Qwen; ≥400 GB to hold base+post on one box), prefer high-`inet_down` offers, and gate the
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
