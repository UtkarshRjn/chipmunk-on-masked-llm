#!/usr/bin/env bash
# Download LLaDA-8B checkpoints from HuggingFace.
# Requires `huggingface-cli` (install via `pip install huggingface_hub[cli]`).
# Set HF_TOKEN in env if the repo requires authentication.

set -euo pipefail

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS=(
    "GSAI-ML/LLaDA-8B-Instruct"
    "GSAI-ML/LLaDA-8B-Base"
)

echo "HF cache: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli not found. Install: pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    huggingface-cli download "$model" \
        --cache-dir "$CACHE_DIR" \
        --include "*.safetensors" "*.json" "*.txt" "tokenizer*" \
        --exclude "*.bin" "*.pt"
done

echo "Done. Cached under $CACHE_DIR."
