"""CPU smoke test for the mdm-chipmunk harness.

Runs the dense MDM method on the tiny synthetic model against a handful of
GSM8K samples. The point is to verify the pipeline end-to-end without needing
HuggingFace downloads or a GPU. Accuracy is expected to be 0 — the synthetic
model is random-initialized.

Switch ``MODEL`` to ``mdlm_170m`` to run the real (slow on CPU) baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mdm_chipmunk.eval.harness import run
from mdm_chipmunk.eval.tasks.registry import build_task
from mdm_chipmunk.methods.registry import build_method
from mdm_chipmunk.models.registry import build_model
from mdm_chipmunk.utils.device import device_info
from mdm_chipmunk.utils.logging import get_logger

LOG = get_logger("smoke")
MODEL = "synthetic"
TASK = "synthetic"
METHOD = "dense"
NUM_SAMPLES = 3
NUM_STEPS = 8


def main() -> None:
    info = device_info()
    LOG.info("Smoke test on %s (%s, %s)", info.name, info.device, info.dtype)

    model = build_model(MODEL)
    method = build_method(METHOD)
    task = build_task(TASK, num_samples=NUM_SAMPLES, gen_length=8)

    out = REPO / "results" / "smoke.jsonl"
    summary = run(
        method=method,
        model=model,
        task=task,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        seed=0,
        out_path=out,
    )
    LOG.info("Wrote %s", out)
    LOG.info("Summary: %s", summary.to_dict())


if __name__ == "__main__":
    main()
