from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..methods.base import InferenceMethod, Telemetry
from ..models.base import MDMModel
from ..utils.logging import JsonlWriter, get_logger
from .metrics import accuracy, mean_latency, mean_tokens_per_second
from .tasks.base import Task

_LOG = get_logger(__name__)


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
    """Run `method` on `task` for `num_samples` samples; log per-sample JSONL."""
    out_path = Path(out_path)
    writer = JsonlWriter(out_path)
    correct_flags: list[bool] = []
    telemetries: list[Telemetry] = []

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

            writer.write(
                {
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
                }
            )
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
    return summary
