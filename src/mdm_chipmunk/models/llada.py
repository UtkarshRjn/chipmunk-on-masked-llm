from __future__ import annotations

from typing import Any

import torch

from ..utils.device import get_device, get_dtype
from ..utils.logging import get_logger
from .base import MDMModel, ModelConfig

_LOG = get_logger(__name__)


class LLaDA(MDMModel):
    """HuggingFace loader for Nie et al. 2025 LLaDA-8B.

    Defaults to ``GSAI-ML/LLaDA-8B-Instruct``. Set ``lazy=True`` to instantiate
    tokenizer and config only — useful for laptop CI where the 16 GB weight
    download is impractical.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        lazy: bool = False,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self._device = torch.device(device) if device else get_device()
        self._dtype = dtype or get_dtype(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.hf_id, trust_remote_code=config.trust_remote_code
        )
        if lazy:
            _LOG.info("LLaDA(lazy=True): skipping weight load — tokenizer + config only.")
            self._model = None
            return
        _LOG.info("Loading LLaDA weights from %s on %s (%s)", config.hf_id, self._device, self._dtype)
        self._model = AutoModel.from_pretrained(
            config.hf_id,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @torch.no_grad()
    def forward_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("LLaDA was loaded with lazy=True; cannot run forward_logits.")
        out = self._model(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device) if attention_mask is not None else None,
        )
        # LLaDA's HF wrapper returns logits directly or under `.logits` depending on version.
        return out.logits if hasattr(out, "logits") else out


DEFAULT_LLADA_INSTRUCT_CONFIG = ModelConfig(
    name="llada-8b-instruct",
    hf_id="GSAI-ML/LLaDA-8B-Instruct",
    mask_token_id=126336,  # per LLaDA model card / inference example
    seq_len=4096,
    num_layers=32,
    architecture="llada",
    trust_remote_code=True,
)

DEFAULT_LLADA_BASE_CONFIG = ModelConfig(
    name="llada-8b-base",
    hf_id="GSAI-ML/LLaDA-8B-Base",
    mask_token_id=126336,
    seq_len=4096,
    num_layers=32,
    architecture="llada",
    trust_remote_code=True,
)
