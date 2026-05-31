import os

import torch

from mdm_chipmunk.utils.device import device_info, get_device, get_dtype


def test_get_device_falls_back_to_cpu_when_forced(monkeypatch):
    monkeypatch.setenv("MDM_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_get_dtype_fp32_on_cpu():
    assert get_dtype(torch.device("cpu")) == torch.float32


def test_device_info_consistent():
    info = device_info(prefer="cpu")
    assert info.is_cpu and not info.is_cuda and not info.is_mps
    assert info.device.type == "cpu"
    assert info.dtype == torch.float32
    assert info.name == "CPU"


def test_dtype_env_override(monkeypatch):
    monkeypatch.setenv("MDM_DTYPE", "fp16")
    assert get_dtype(torch.device("cpu")) == torch.float16
    monkeypatch.delenv("MDM_DTYPE")
    assert get_dtype(torch.device("cpu")) == torch.float32
