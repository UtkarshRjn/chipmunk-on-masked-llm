#!/usr/bin/env bash
# Clone the upstream Chipmunk repo into ./chipmunk for reference.
# Gitignored — re-run on each host.
set -euo pipefail

if [[ -d chipmunk/.git ]]; then
  echo "chipmunk/ already cloned. Pulling latest."
  git -C chipmunk pull --ff-only
else
  git clone https://github.com/sandyresearch/chipmunk.git
fi
