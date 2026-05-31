from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class TaskSample:
    sample_idx: int
    prompt: str
    gold: Any
    metadata: dict[str, Any]


class Task(ABC):
    """Evaluation task. Yields TaskSamples and scores predictions."""

    name: str
    gen_length: int = 256

    @abstractmethod
    def iter_samples(self, num_samples: int | None = None) -> Iterator[TaskSample]: ...

    @abstractmethod
    def score(self, prediction_text: str, gold: Any) -> bool: ...

    def __init__(self, **cfg: Any):
        self.cfg = cfg
