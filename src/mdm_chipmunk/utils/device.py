from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    dtype: torch.dtype
    name: str
    is_cuda: bool
    is_mps: bool
    is_cpu: bool


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    override = os.environ.get("MDM_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device | str | None = None) -> torch.dtype:
    dev = torch.device(device) if isinstance(device, str) else (device or get_device())
    override = os.environ.get("MDM_DTYPE")
    if override:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[override]
    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        return torch.bfloat16 if major >= 8 else torch.float16
    if dev.type == "mps":
        return torch.float16
    return torch.float32


def device_info(prefer: str | None = None) -> DeviceInfo:
    dev = get_device(prefer)
    dtype = get_dtype(dev)
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev)
    elif dev.type == "mps":
        name = "Apple MPS"
    else:
        name = "CPU"
    return DeviceInfo(
        device=dev,
        dtype=dtype,
        name=name,
        is_cuda=dev.type == "cuda",
        is_mps=dev.type == "mps",
        is_cpu=dev.type == "cpu",
    )
