"""Runtime compatibility shims for torch / transformers version differences.

Each function is a no-op on the version where the feature is native.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from types import ModuleType
from typing import Any


def install_flash_attn_fallback() -> bool:
    """Provide a pure-PyTorch ``flash_attn`` when the real one can't be built.

    flash-attn has no Pascal (sm_61) support and ships no wheels for it, yet the
    upstream MDLM modeling file hard-imports ``flash_attn`` for two things only:
    non-interleaved RoPE (``layers.rotary.apply_rotary_emb_qkv_``) and full
    bidirectional attention (``flash_attn_interface.flash_attn_varlen_qkvpacked_func``
    with ``causal=False``). Both have exact eager equivalents.

    Registers a fake ``flash_attn`` package in ``sys.modules`` so that both
    ``transformers``' static ``check_imports`` and the runtime forward pass see a
    working module. No-op (returns False) when the real flash_attn is importable.
    """
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    if "flash_attn" in sys.modules and getattr(sys.modules["flash_attn"], "_mdm_fallback", False):
        return True

    import torch
    import torch.nn.functional as F

    def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb_qkv_(qkv, cos, sin, *args, **kwargs):
        # qkv: (b, s, 3, h, d). cos/sin: (s, d/2). Rotates q and k in place, leaves v.
        cos_f = torch.cat((cos, cos), dim=-1)[None, :, None, :]  # (1, s, 1, d)
        sin_f = torch.cat((sin, sin), dim=-1)[None, :, None, :]
        for i in (0, 1):  # q, k — v (index 2) is left untouched
            x = qkv[:, :, i]
            qkv[:, :, i] = x * cos_f + _rotate_half(x) * sin_f
        return qkv

    def flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, dropout_p=0.0, *args, causal=False, softmax_scale=None, **kwargs
    ):
        # qkv: ((b s), 3, h, d) with uniform, unpadded sequences. Plain SDPA.
        seq_len = int(max_seqlen)
        total, _, h, d = qkv.shape
        batch = total // seq_len
        qkv = qkv.view(batch, seq_len, 3, h, d)
        q, k, v = (qkv[:, :, j].transpose(1, 2) for j in range(3))  # (b, h, s, d)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
        )
        return out.transpose(1, 2).reshape(total, h, d)

    flash_attn = ModuleType("flash_attn")
    flash_attn._mdm_fallback = True  # type: ignore[attr-defined]
    layers = ModuleType("flash_attn.layers")
    rotary = ModuleType("flash_attn.layers.rotary")
    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_  # type: ignore[attr-defined]
    interface = ModuleType("flash_attn.flash_attn_interface")
    interface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func  # type: ignore[attr-defined]
    layers.rotary = rotary  # type: ignore[attr-defined]
    flash_attn.layers = layers  # type: ignore[attr-defined]
    flash_attn.flash_attn_interface = interface  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "flash_attn": flash_attn,
            "flash_attn.layers": layers,
            "flash_attn.layers.rotary": rotary,
            "flash_attn.flash_attn_interface": interface,
        }
    )
    return True


def patch_autocast_bf16_fallback() -> bool:
    """Make ``autocast(dtype=bfloat16)`` a no-op on GPUs without bf16 support.

    MDLM's backbone hardcodes ``torch.cuda.amp.autocast(dtype=torch.bfloat16)``,
    whose constructor raises on pre-Ampere cards (e.g. GTX 10-series) even when
    the region is effectively disabled. Since we load MDLM in fp32, the faithful
    fallback is to run that region in fp32 (enabled=False). No-op on bf16-capable
    devices. Returns True if the patch was installed.
    """
    import torch

    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return False
    _amp = torch.cuda.amp
    if getattr(_amp.autocast, "_mdm_patched", False):
        return True
    _orig = _amp.autocast

    def _autocast(*args, **kwargs):
        if kwargs.get("dtype") is torch.bfloat16:
            kwargs["dtype"] = torch.float16
            kwargs["enabled"] = False
        return _orig(*args, **kwargs)

    _autocast._mdm_patched = True  # type: ignore[attr-defined]
    _amp.autocast = _autocast  # type: ignore[assignment]
    return True


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
