#!/usr/bin/env bash
# Clone the upstream baseline implementations into the project for reference.
# All clones are gitignored — re-run on each host.

set -euo pipefail

clone() {
    local repo="$1" dest="$2"
    if [[ -d "$dest/.git" ]]; then
        echo "$dest already cloned. Pulling latest."
        git -C "$dest" pull --ff-only
    else
        git clone --depth 1 "$repo" "$dest"
    fi
}

clone https://github.com/sandyresearch/chipmunk.git    chipmunk
clone https://github.com/NVlabs/Fast-dLLM.git          fastdllm
clone https://github.com/horseee/dKV-Cache.git         dkv_cache

echo "Done. References under: chipmunk/, fastdllm/, dkv_cache/."
echo "Note: DualDiffusion (Goyal et al. 2026) has no public repo yet —"
echo "      methods/dualdiffusion.py is a clean-room implementation."
