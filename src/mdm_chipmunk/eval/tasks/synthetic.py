from __future__ import annotations

import random
from typing import Any, Iterator

from .base import Task, TaskSample


class SyntheticTask(Task):
    """Tiny in-memory task. No HF / network — for CPU smoke tests."""

    name = "synthetic"
    gen_length = 8

    def __init__(self, num_samples: int = 5, seed: int = 0, gen_length: int = 8, **cfg: Any):
        super().__init__(**cfg)
        self.gen_length = gen_length
        self._rng = random.Random(seed)
        self._samples = [
            TaskSample(
                sample_idx=i,
                prompt=" ".join(f"tok{j}" for j in range(self._rng.randint(2, 5))),
                gold=str(self._rng.randint(0, 9)),
                metadata={},
            )
            for i in range(num_samples)
        ]

    def iter_samples(self, num_samples: int | None = None) -> Iterator[TaskSample]:
        n = len(self._samples) if num_samples is None else min(num_samples, len(self._samples))
        yield from self._samples[:n]

    def score(self, prediction_text: str, gold: Any) -> bool:
        return prediction_text.strip().endswith(str(gold))
