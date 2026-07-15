#!/bin/bash
# fetch_convert_f16.sh <hf_repo> <outfile.gguf>
# Download a model's bf16 safetensors from HF -> convert to an F16 gguf -> delete the
# download. Serialized so the transient disk peak stays ~(one download + one gguf).
# Used by the LOCAL fp16 science pilot (docs/local_smoke_quickstart.md) — the panel's
# vast legs never touch ggufs (bf16 vLLM from HF weights).
#
# Env: LLAMA_CPP_DIR = a llama.cpp checkout (for convert_hf_to_gguf.py) — REQUIRED.
#      JUDEX_PY      = python with torch+sentencepiece+huggingface_hub (default: judex-arm).
#      HF_TOKEN      = read token (default: Keychain hf-token).
set -euo pipefail
REPO="$1"; OUT="$2"
PY="${JUDEX_PY:-/Users/fabodo/anaconda3/envs/judex-arm/bin/python}"
[ -n "${LLAMA_CPP_DIR:-}" ] || { echo "set LLAMA_CPP_DIR to a llama.cpp checkout" >&2; exit 1; }
DL="${FETCH_TMP_DIR:-${TMPDIR:-/tmp}}/hf-dl/$(echo "$REPO" | tr '/' '_')"
export HF_TOKEN="${HF_TOKEN:-$(security find-generic-password -s hf-token -w 2>/dev/null || true)}"

echo "== [$REPO] free space before: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
mkdir -p "$(dirname "$OUT")" "$DL"
echo "== [$REPO] downloading weights -> $DL"
"$PY" - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("$REPO", local_dir="$DL",
                  allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json",
                                  "*.txt", "*.model", "*.jinja"])
print("download complete")
PYEOF
echo "== [$REPO] converting -> $OUT (f16)"
"$PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$DL" --outfile "$OUT" --outtype f16 2>&1 | tail -3
rm -rf "$DL"
echo "== [$REPO] done: $(ls -lh "$OUT" | awk '{print $5, $9}') | free after: $(df -h "$HOME" | tail -1 | awk '{print $4}')"
