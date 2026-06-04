from __future__ import annotations

import dataclasses
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..methods.base import InferenceMethod, Telemetry
from ..models.base import MDMModel
from ..utils.logging import JsonlWriter, get_logger
from .metrics import accuracy, mean_latency, mean_tokens_per_second
from .tasks.base import Task

_LOG = get_logger(__name__)


def _git_short_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _maybe_wandb_init(method: str, model: str, task: str, num_samples: int, num_steps: int, seed: int):
    """Initialise a W&B run if WANDB_PROJECT is set in the env. Returns the run or None."""
    project = os.environ.get("WANDB_PROJECT")
    if not project:
        return None
    try:
        import wandb  # type: ignore[import-not-found]
    except ImportError:
        _LOG.warning("WANDB_PROJECT set but wandb not installed; skipping. pip install mdm-chipmunk[wandb].")
        return None
    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        name=f"{method}-{model}-{task}",
        tags=[method, model, task],
        config={
            "method": method,
            "model": model,
            "task": task,
            "num_samples": num_samples,
            "num_steps": num_steps,
            "seed": seed,
            "git_sha": _git_short_sha(),
        },
        reinit=True,
    )


@dataclass
class RunSummary:
    method: str
    model: str
    task: str
    num_samples: int
    accuracy: float
    mean_latency_s: float
    mean_tokens_per_second: float
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run(
    method: InferenceMethod,
    model: MDMModel,
    task: Task,
    num_samples: int = 10,
    num_steps: int = 128,
    seed: int = 0,
    out_path: str | Path = "results/run.jsonl",
) -> RunSummary:
    """Run `method` on `task` for `num_samples` samples; log per-sample JSONL.

    If the ``WANDB_PROJECT`` env var is set, per-sample telemetry and the final
    summary are also streamed to Weights & Biases (entity from ``WANDB_ENTITY``).
    """
    out_path = Path(out_path)
    writer = JsonlWriter(out_path)
    correct_flags: list[bool] = []
    telemetries: list[Telemetry] = []

    wandb_run = _maybe_wandb_init(
        method=method.name,
        model=model.config.name,
        task=task.name,
        num_samples=num_samples,
        num_steps=num_steps,
        seed=seed,
    )

    with writer:
        for sample in task.iter_samples(num_samples=num_samples):
            prompt_ids = model.tokenize(sample.prompt)
            result = method.generate(
                model=model,
                prompt_ids=prompt_ids,
                gen_length=task.gen_length,
                num_steps=num_steps,
                seed=seed + sample.sample_idx,
            )
            correct = task.score(result.output_text, sample.gold)
            correct_flags.append(correct)
            telemetries.append(result.telemetry)

            row = {
                "method": method.name,
                "model": model.config.name,
                "task": task.name,
                "sample_idx": sample.sample_idx,
                "gold": sample.gold,
                "prediction_text": result.output_text,
                "correct": correct,
                "latency_s": result.telemetry.total_time_s,
                "tokens_per_second": result.telemetry.tokens_per_second,
                "num_steps": result.telemetry.num_steps,
                "gen_length": result.telemetry.gen_length,
                "peak_memory_bytes": result.telemetry.peak_memory_bytes,
                "nfe": result.telemetry.extras.get("nfe"),
            }
            writer.write(row)
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in row.items() if k != "prediction_text"})
            _LOG.info(
                "[%s/%s] %s | correct=%s | %.2fs",
                sample.sample_idx + 1,
                num_samples,
                task.name,
                correct,
                result.telemetry.total_time_s,
            )

    summary = RunSummary(
        method=method.name,
        model=model.config.name,
        task=task.name,
        num_samples=len(correct_flags),
        accuracy=accuracy(correct_flags),
        mean_latency_s=mean_latency(telemetries),
        mean_tokens_per_second=mean_tokens_per_second(telemetries),
        output_path=str(out_path),
    )
    _LOG.info("Summary: %s", summary.to_dict())
    if wandb_run is not None:
        wandb_run.summary.update(summary.to_dict())
        wandb_run.finish()
    return summary
