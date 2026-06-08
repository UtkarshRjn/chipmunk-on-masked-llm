"""GPU smoke test for the mdm-chipmunk harness.

Validates the full pipeline (model load → generation → eval) on a real MDM
with CUDA. Expected to run in ~30 s on an RTX A5000 with MDLM-170M.

Usage:
    python scripts/smoke_test_gpu.py                    # MDLM-170M, 5 samples
    MODEL=llada_8b_instruct python scripts/smoke_test_gpu.py   # LLaDA-8B

Switch METHOD to test other inference methods once the baselines are confirmed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch

from mdm_chipmunk.eval.harness import run
from mdm_chipmunk.eval.tasks.registry import build_task
from mdm_chipmunk.methods.registry import build_method
from mdm_chipmunk.models.registry import build_model
from mdm_chipmunk.utils.device import device_info
from mdm_chipmunk.utils.logging import get_logger

LOG = get_logger("smoke_gpu")

MODEL = os.environ.get("MODEL", "mdlm_170m")
METHOD = os.environ.get("METHOD", "dense")
TASK = os.environ.get("TASK", "gsm8k")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "5"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "16"))


def main() -> None:
    if not torch.cuda.is_available():
        LOG.error("No CUDA device found — set MDM_DEVICE=cuda and check your driver")
        sys.exit(1)

    info = device_info()
    LOG.info("Device  : %s (%s, %s)", info.name, info.device, info.dtype)
    LOG.info("Config  : model=%s  method=%s  task=%s  samples=%d  steps=%d",
             MODEL, METHOD, TASK, NUM_SAMPLES, NUM_STEPS)

    vram_before = torch.cuda.memory_allocated() / 1e9
    model = build_model(MODEL)
    vram_after = torch.cuda.memory_allocated() / 1e9
    LOG.info("VRAM after model load : %.2f GB  (delta %.2f GB)", vram_after, vram_after - vram_before)

    method = build_method(METHOD)
    task = build_task(TASK, num_samples=NUM_SAMPLES, gen_length=128)

    out = REPO / "results" / f"smoke_gpu_{MODEL}_{METHOD}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = run(
        method=method,
        model=model,
        task=task,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        seed=42,
        out_path=out,
    )

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    LOG.info("Peak VRAM            : %.2f GB", vram_peak)
    LOG.info("Results written to   : %s", out)
    LOG.info("Summary              : %s", summary.to_dict())


if __name__ == "__main__":
    main()
