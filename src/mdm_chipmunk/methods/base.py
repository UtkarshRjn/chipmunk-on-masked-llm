from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from ..models.base import MDMModel


@dataclass
class Telemetry:
    method: str
    model: str
    num_steps: int
    gen_length: int
    total_time_s: float = 0.0
    step_times_s: list[float] = field(default_factory=list)
    peak_memory_bytes: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.gen_length / self.total_time_s if self.total_time_s > 0 else 0.0


@dataclass
class GenerationResult:
    output_ids: torch.LongTensor
    output_text: str
    telemetry: Telemetry


class InferenceMethod(ABC):
    """Abstract MDM inference method. Subclasses implement a denoising loop."""

    name: str

    def __init__(self, **cfg: Any):
        self.cfg = cfg

    @abstractmethod
    def generate(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_steps: int,
        seed: int | None = None,
    ) -> GenerationResult: ...

    def _reset_peak_memory(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def _peak_memory(self, device: torch.device) -> int:
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
        return 0
