"""Runtime compatibility shims for torch / transformers version differences.

Each function is a no-op on the version where the feature is native.
"""

from __future__ import annotations

import inspect
from typing import Any


def patch_flash_attn_wrap_triton() -> None:
    """Stub torch.library.wrap_triton for torch < 2.6.0.

    flash_attn 2.8+ calls torch.library.wrap_triton() to register triton kernels
    with the torch dispatcher (needed for torch.compile / export). In eager
    inference mode the wrapper is a no-op, so a passthrough lambda is safe.
    On torch >= 2.6.0 the attribute already exists and this function does nothing.
    """
    import torch.library as _tl

    if not hasattr(_tl, "wrap_triton"):
        _tl.wrap_triton = lambda fn: fn  # type: ignore[attr-defined]


def hf_from_pretrained_dtype_kwarg() -> str:
    """Return the correct dtype keyword for AutoModel.from_pretrained().

    transformers renamed torch_dtype -> dtype somewhere around 4.45.
    Inspect the live signature so this works on any installed version.
    """
    try:
        from transformers import AutoModel

        sig = inspect.signature(AutoModel.from_pretrained)
        return "dtype" if "dtype" in sig.parameters else "torch_dtype"
    except Exception:
        return "torch_dtype"


def hf_load_kwargs(dtype: Any) -> dict[str, Any]:
    """Build the dtype keyword dict for from_pretrained(), version-aware."""
    return {hf_from_pretrained_dtype_kwarg(): dtype}
