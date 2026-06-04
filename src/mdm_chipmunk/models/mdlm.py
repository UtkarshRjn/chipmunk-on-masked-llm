from __future__ import annotations

from typing import Any

import torch

from ..utils.device import get_device, get_dtype
from ..utils.logging import get_logger
from .base import MDMModel, ModelConfig

_LOG = get_logger(__name__)


class MDLM(MDMModel):
    """HuggingFace loader for Sahoo et al. 2024 MDLM (small open MDM).

    Default checkpoint: ``kuleshov-group/mdlm-owt`` (~170M params, OWT-trained).
    Used as the CPU/laptop smoke-test workhorse since it fits in a few GB of RAM.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        lazy: bool = False,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.config = config
        self._device = torch.device(device) if device else get_device()
        self._dtype = dtype or get_dtype(self._device)
        # The MDLM HF repo ships a custom MDLMConfig but no AutoTokenizer mapping,
        # so AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True) raises.
        # MDLM uses GPT-2's BPE plus one extra mask token at id 50257 (see paper §3).
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self._tokenizer.mask_token is None:
            self._tokenizer.add_special_tokens({"mask_token": "<mask>"})
        if lazy:
            _LOG.info("MDLM(lazy=True): skipping weight load — tokenizer + config only.")
            self._model = None
            return
        _LOG.info("Loading MDLM weights from %s on %s (%s)", config.hf_id, self._device, self._dtype)
        self._model = AutoModelForMaskedLM.from_pretrained(
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
            raise RuntimeError("MDLM was loaded with lazy=True; cannot run forward_logits.")
        out = self._model(
            input_ids=input_ids.to(self._device),
            attention_mask=attention_mask.to(self._device) if attention_mask is not None else None,
        )
        return out.logits


DEFAULT_MDLM_CONFIG = ModelConfig(
    name="mdlm-170m",
    hf_id="kuleshov-group/mdlm-owt",
    mask_token_id=50257,  # GPT-2 BPE + 1 extra mask slot per MDLM convention; override per checkpoint
    seq_len=1024,
    architecture="mdlm",
    trust_remote_code=True,
)
