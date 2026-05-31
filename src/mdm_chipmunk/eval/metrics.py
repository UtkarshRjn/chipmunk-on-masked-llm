from __future__ import annotations

from statistics import mean

from ..methods.base import Telemetry


def accuracy(correct: list[bool]) -> float:
    return mean(int(c) for c in correct) if correct else 0.0


def mean_latency(telemetries: list[Telemetry]) -> float:
    return mean(t.total_time_s for t in telemetries) if telemetries else 0.0


def mean_tokens_per_second(telemetries: list[Telemetry]) -> float:
    rates = [t.tokens_per_second for t in telemetries if t.total_time_s > 0]
    return mean(rates) if rates else 0.0


def peak_memory_bytes(telemetries: list[Telemetry]) -> int:
    return max((t.peak_memory_bytes for t in telemetries), default=0)
