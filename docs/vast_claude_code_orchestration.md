# Orchestrating the Study A vast.ai run with Claude Code (on the box)

Instead of driving the vast.ai GPU box step by step from your Mac, install **Claude Code on the
rented box** and let it orchestrate the run end to end — serve vLLM, run the driver, **troubleshoot
autonomously** (OOM, ports, context, logprobs), swap base→post, analyse, return results, and tear
down. On an ephemeral box this is faster than hand-driving and survives SSH drops.

> Read `docs/vast_quickstart.md` first (provisioning a Mode-C box, HF token, offers). This doc
> adds the **Claude-Code-on-the-box** layer on top of a provisioned instance.

---

## 0. Prereqs

- A provisioned vast.ai instance you can `ssh` into (Mode C; the `vllm/vllm-openai` image is
  Debian-based, runs as **root**, has Python/pip, and usually **no Node.js**).
- Your `HF_TOKEN` (for weight pulls) and a Claude credential (below).
- The box is **ephemeral and third-party** — treat any secret you put on it as exposed; tear the box
  down as soon as the run finishes.

## 1. Install Claude Code on the box

Use the **native installer** (single binary, no Node.js — right for this container):

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"          # the binary lands in ~/.local/bin
claude --version
```

(npm alternative, only if you want the Node toolchain: install Node 20 via
`curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs`, then
`npm install -g @anthropic-ai/claude-code`.)

## 2. Authenticate — headless, no browser

Claude Code reads a credential straight from the environment; **no `claude login`** needed in
non-interactive mode. Two options:

```bash
# Option A — API key (simplest for an ephemeral box):
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B — subscription OAuth token (mint ONCE on your Mac where a browser exists):
#   (on the Mac)  claude setup-token        # prints a ~1-year token
export CLAUDE_CODE_OAUTH_TOKEN="claude_code_oauth_token_..."
```

**Security (do this):** pass the secret at connect time, not via the vast **template/console env**
(those persist across the instance's life and logs). E.g. from the Mac, pulling from Keychain:
```bash
ssh root@<host> -p <ssh_port> \
  "export ANTHROPIC_API_KEY='$(security find-generic-password -s anthropic-api-key -w)'; bash -l"
```
Prefer a **key you can rotate** (or the OAuth token over a raw API key), and **destroy the box**
right after. Never echo the secret into a log.

## 3. Repo + Python env on the box

The box is not your Mac — a fresh env here is fine (the `judex-arm`/no-venv rule is Mac-only):

```bash
cd /workspace
git clone --recurse-submodules https://github.com/foabodo/judex.git && cd judex
python -m venv .venv && . .venv/bin/activate         # or conda, if the image has it
pip install -e judex-evaluator -e judex-calibration  # judex + numpy + POT + pyyaml
export HF_TOKEN="hf_..."                              # for the base/post weight pulls
```
The driver reaches vLLM on `http://127.0.0.1:8000`; the canonical AIReg GT + corpus few-shot load
from the git-tracked sibling repos (no gitignored-run dependency — see `aireg.py`).

## 4. Run Claude Code to orchestrate

vLLM is a **long-running** server that must outlive individual Claude commands. Claude Code kills its
*own* tracked background shells ~5 s after a `-p` run ends, so **do not** let Claude background vLLM as
a normal tool task — run vLLM **detached** (`nohup` + PID file, or its own `tmux` window) so Claude can
start it, query it, swap it, and kill it by PID across turns.

Two ways to run, both inside `tmux` so they survive SSH disconnects:

**A. Supervised (recommended first) — you attach and watch, and can intervene:**
```bash
tmux new -s exp
. /workspace/judex/.venv/bin/activate
export ANTHROPIC_API_KEY=...   HF_TOKEN=...
cd /workspace/judex/judex-calibration
claude          # interactive; paste the brief in §5, watch it work, steer as needed
# detach: Ctrl-b d   |   reattach from the Mac: ssh -t root@<host> -p <port> "tmux attach -t exp"
```

**B. Autonomous — fire-and-forget on the throwaway box:**
```bash
tmux new -d -s exp "cd /workspace/judex/judex-calibration && \
  claude -p \"\$(cat /workspace/brief.md)\" \
    --dangerously-skip-permissions \
    --max-turns 60 \
    --output-format stream-json --verbose \
    > /workspace/claude-run.log 2>&1"
tail -f /workspace/claude-run.log       # or attach the tmux session
```
`--dangerously-skip-permissions` lets Claude run every bash step without prompts — appropriate on an
**isolated, ephemeral** box (the API key is the only real exposure, and it exists regardless).
If you want tighter control, drop that flag and pass an allow-list instead, e.g.
`--allowedTools "Bash(vllm *)" "Bash(nohup *)" "Bash(python *)" "Bash(curl *)" "Bash(kill *)" "Bash(tmux *)" "Read" "Edit"`
— in `-p` mode any tool not on the list is auto-denied (no hang). Use `--max-turns` to bound the run;
the real cost control is **wall-clock on the box**, so tell Claude to tear down the moment it's done.

## 5. The orchestration brief (paste this to the on-box Claude)

Save as `/workspace/brief.md` (autonomous mode) or paste into interactive `claude`. Parameterised for
**one family**; repeat per family, or extend the brief to loop the `configs/models.yaml` roster.

```md
You are orchestrating a Study A calibration run on THIS rented GPU box. Work in
/workspace/judex/judex-calibration with the .venv active. vLLM is your model server; the driver is
scripts/run_qwen_phase1.py. Be autonomous and fast; diagnose failures yourself.

FAMILY: Qwen  (BASE repo Qwen/Qwen3.5-35B-A3B-Base, POST repo Qwen/Qwen3.5-35B-A3B)
Do it in this order and REPORT after each step:

1. SERVE THE BASE (detached so it outlives your commands — use its own tmux window, NOT a tracked
   background task): `tmux new-window -d -n vllm "vllm serve Qwen/Qwen3.5-35B-A3B-Base \
   --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --port 8000 \
   > /workspace/vllm.log 2>&1"`. Poll `curl -s http://127.0.0.1:8000/v1/models` until 200
   (weights download first — watch /workspace/vllm.log). Then confirm logprobs actually come back:
   `curl -s http://127.0.0.1:8000/v1/completions -d '{"model":"x","prompt":"Answer:","max_tokens":1,"logprobs":20}' -H 'Content-Type: application/json'`.

2. ELICIT THE BASE LEG:
   `python scripts/run_qwen_phase1.py --base-url http://127.0.0.1:8000 --base-model Qwen/Qwen3.5-35B-A3B-Base --out runs/qwen`
   (add `--limit 6 --no-reason` for a fast pipeline check first; drop them for the real 120-cell,
   reasoning run once the check is green). Verify runs/qwen/pre.json has 120 (or N) sum-to-1 dists.

3. SWAP TO THE POST: kill the vLLM window (`tmux kill-window -t vllm`), relaunch the same command on
   Qwen/Qwen3.5-35B-A3B (add `--reasoning-parser qwen3` for the post's native reasoning), wait healthy,
   then `python scripts/run_qwen_phase1.py --post-url http://127.0.0.1:8000 --post-model Qwen/Qwen3.5-35B-A3B --out runs/qwen`.

4. ANALYSE: `python scripts/run_qwen_phase1.py --analyze-only --out runs/qwen`. Read
   runs/qwen/study_a_report.json (Q1 T*_pre, Q2 tau_oc, Q3 tau_oc_summary, Q4 closed_side_check) and
   runs/qwen/pipeline_calibration_block.json. Summarise the numbers.

TROUBLESHOOTING (fix these yourself, don't wait):
- CUDA OOM / won't load: lower `--gpu-memory-utilization` (0.90→0.85) or `--max-model-len`, or the
  offer is too small — report the VRAM gap. MoE giants: add `--tensor-parallel-size N --enable-expert-parallel`.
- Prompt exceeds context: raise `--max-model-len` (AIReg prompts are ~14k tokens; keep >= 16384).
- No logprobs on /v1/completions: ensure it's `vllm serve` (not a chat-only proxy); the base has no
  chat template so use /v1/completions only. elicit_base is server-agnostic but needs logprobs.
- Endpoint unreachable from the driver: it's localhost on this box, so check the vLLM window is alive
  (`tmux list-windows`) and the port (`ss -ltnp | grep 8000`).
- Weights slow/failing: it's the HF pull (dominant cost) — check HF_TOKEN + license; watch vllm.log.

RETURN RESULTS (the box is ephemeral — persist before teardown): copy the outputs somewhere durable —
either `tar czf /workspace/qwen_results.tgz runs/qwen && echo "scp root@<host>:/workspace/qwen_results.tgz"`
for me to pull, or commit them to a branch: from /workspace/judex/judex-calibration,
`git checkout -b vast-run-qwen && git add -f runs/qwen/study_a_report.json runs/qwen/pipeline_calibration_block.json && git commit -m "Study A Qwen vast run" && git push -u origin vast-run-qwen`
(runs/ is gitignored, so force-add just the two small JSONs). Also note run-id + tau_oc for the
handoff, per the guide's cadence.

TEARDOWN: after results are safe, stop vLLM (`tmux kill-window -t vllm`). Tell me it's safe to
`vastai destroy instance <id>` — do NOT destroy the box yourself.
```

## 6. After it finishes

- Pull the artifacts (`scp …/qwen_results.tgz .`) or `git fetch` the `vast-run-*` branch; the
  `pipeline_calibration_block.json` is the drop-in for `judex-evaluator/pipeline.yaml → calibration`
  (see the guide §4.6 seam note).
- **Destroy the box yourself** (`vastai destroy instance <id>`) — cost is wall-clock. The brief tells
  the on-box Claude *not* to destroy it, so you keep the kill switch.
- Rotate the API key/token if you used a rotatable one.

## Caveats
- **Secret exposure:** the credential lives in the box's env during the run. Ephemeral box + prompt
  teardown + a rotatable key/OAuth token is the mitigation; don't bake it into the vast template.
- **`--dangerously-skip-permissions`** is reasonable on an isolated throwaway box but skips *all*
  approval gates — use the `--allowedTools` allow-list (§4) if you want Claude constrained to the
  serve/elicit/teardown commands only.
- This orchestrates the **run**; it does not change the science. The six bf16 panel models and the
  canonical GT are unchanged — Claude is just a faster hand on the provisioning/serving/debug loop.
