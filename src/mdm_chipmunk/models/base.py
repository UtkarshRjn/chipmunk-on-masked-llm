from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ModelConfig:
    """Static description of an MDM checkpoint. Loaded from YAML."""

    name: str
    hf_id: str
    mask_token_id: int
    seq_len: int = 1024
    num_layers: int = 0
    vocab_size: int = 0
    architecture: str = ""
    trust_remote_code: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


class MDMModel(ABC):
    """Abstract masked-diffusion language model.

    The contract is intentionally narrow so that LLaDA-style HF wrappers, MDLM,
    and the tiny synthetic model in tests can all implement it without leaking
    implementation details into inference methods. Caching state lives in the
    inference method, never here.
    """

    config: ModelConfig

    @abstractmethod
    def forward_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B, L, V) for a possibly masked input."""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def tokenizer(self) -> Any: ...

    @property
    def mask_token_id(self) -> int:
        return self.config.mask_token_id

    @property
    def seq_len(self) -> int:
        return self.config.seq_len

    def tokenize(self, text: str, max_length: int | None = None) -> torch.LongTensor:
        max_length = max_length or self.seq_len
        out = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        return out["input_ids"].to(self.device)

    def detokenize(self, ids: torch.LongTensor) -> str:
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
