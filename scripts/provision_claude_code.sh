#!/usr/bin/env bash
# Provision a fresh vast.ai GPU box to run Claude Code as the Study A orchestrator.
#
# BOOTSTRAP (the box has no repo yet, so scp this script over first), from your Mac:
#   scp judex-calibration/scripts/provision_claude_code.sh root@<host>:/tmp/
#   ssh root@<host> -p <ssh_port> \
#     "ANTHROPIC_API_KEY='$(security find-generic-password -s anthropic-api-key -w)' \
#      HF_TOKEN='$(security find-generic-password -s hf-token -w)' \
#      bash /tmp/provision_claude_code.sh"
#
# It is INTERACTIVE: it generates an SSH key on the box and pauses for you to add it to GitHub
# (as an ACCOUNT key — a per-repo deploy key can't cover judex's 4 private submodules). Remove the
# key from GitHub and destroy the box when the run is done.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_SSH="${REPO_SSH:-git@github.com:foabodo/judex.git}"
# calibration work lives on this umbrella branch until it merges to main:
REPO_BRANCH="${REPO_BRANCH:-calibration-integration}"

say() { printf '\n== %s ==\n' "$*"; }

say "1/6 Auth check"
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
  echo "!! Set ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN) before running (see the bootstrap header)." >&2
  exit 1
fi
[ -n "${HF_TOKEN:-}" ] || echo "   (warning: HF_TOKEN not set — model weight pulls may rate-limit or 401)"

say "2/6 Base tools"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git tmux curl openssh-client python3-venv >/dev/null

say "3/6 Install Claude Code (native binary, no Node.js)"
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
claude --version

say "4/6 SSH key for the private repos (ADD TO GITHUB WHEN PROMPTED)"
git config --global user.name  "${GIT_NAME:-vast-box}"
git config --global user.email "${GIT_EMAIL:-vast@localhost}"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
KEY="$HOME/.ssh/id_ed25519"
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -f "$KEY" -C "vast-judex-$(date +%s 2>/dev/null || echo run)"
ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
cat <<EOF

==================== ACTION REQUIRED ====================
Add this PUBLIC key to GitHub as an ACCOUNT key:
  github.com -> Settings -> SSH and GPG keys -> New SSH key
(An account key is required: judex has 4 private submodules and one deploy key
 cannot cover them. It grants this box access to your private repos — REMOVE it
 after the run.)
--------------------------------------------------------
$(cat "$KEY.pub")
--------------------------------------------------------
EOF
read -r -p "Press Enter once the key is added to GitHub... " _
if ssh -o BatchMode=yes -T git@github.com 2>&1 | grep -qi "successfully authenticated"; then
  echo "   GitHub SSH: OK"
else
  echo "!! GitHub SSH auth not confirmed — check the key was added, then re-run." >&2
  ssh -o BatchMode=yes -T git@github.com || true
  exit 1
fi

say "5/6 Clone the umbrella (recursive submodules), branch $REPO_BRANCH"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
if [ ! -d judex ]; then
  git clone --recurse-submodules -b "$REPO_BRANCH" "$REPO_SSH" judex
fi
cd judex && git submodule update --init --recursive

say "6/6 Python env + pipeline sanity"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e judex-evaluator -e judex-calibration
cd judex-calibration
PYTHONPATH=src python - <<'PY'
from judex_calibration.aireg import load_cells
from judex_calibration import fewshot
cells = load_cells()
n = len(fewshot.build_fewshot_by_criterion(cells))
print(f"   OK: {len(cells)} AIReg cells (canonical GT), few-shot k={fewshot.default_k()} for {n} Articles")
PY

cat <<EOF

======================= READY =======================
Repo:   $WORKDIR/judex   (CLAUDE.md is auto-loaded — the on-box Claude has the project context)
Env:    . $WORKDIR/judex/.venv/bin/activate
Docs:   judex-calibration/docs/vast_claude_code_orchestration.md  (§5 = the orchestration brief)

Start orchestrating (supervised, survives SSH drops):
  tmux new -s exp
  . $WORKDIR/judex/.venv/bin/activate
  cd $WORKDIR/judex/judex-calibration
  claude          # then paste the §5 brief, or:
  #  claude -p "\$(sed -n '/## 5\./,/## 6\./p' docs/vast_claude_code_orchestration.md)" \\
  #    --dangerously-skip-permissions --max-turns 60 --output-format stream-json --verbose

Reminder: remove the SSH account key from GitHub and destroy this box when done.
=====================================================
EOF
