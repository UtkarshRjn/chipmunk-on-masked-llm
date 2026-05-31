from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import MDMModel, ModelConfig


class _ToyTokenizer:
    """Whitespace tokenizer over a tiny synthetic vocabulary. Test-only."""

    def __init__(self, vocab_size: int, mask_token_id: int):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = 0

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        max_length = max_length or 64
        ids_batch = []
        for t in text:
            toks = [hash(w) % (self.vocab_size - 2) + 2 for w in t.split()][:max_length]
            toks += [self.pad_token_id] * (max_length - len(toks))
            ids_batch.append(toks)
        return {"input_ids": torch.tensor(ids_batch, dtype=torch.long)}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.mask_token_id, self.pad_token_id)]
        return " ".join(f"<{i}>" for i in ids)


class SyntheticMDM(MDMModel):
    """Tiny random-init transformer. Used in unit tests to avoid HF downloads."""

    def __init__(
        self,
        seq_len: int = 32,
        vocab_size: int = 64,
        num_layers: int = 2,
        d_model: int = 32,
        n_heads: int = 2,
        mask_token_id: int = 1,
    ):
        self.config = ModelConfig(
            name="synthetic",
            hf_id="",
            mask_token_id=mask_token_id,
            seq_len=seq_len,
            num_layers=num_layers,
            vocab_size=vocab_size,
            architecture="synthetic-transformer",
            trust_remote_code=False,
        )
        self._tokenizer = _ToyTokenizer(vocab_size, mask_token_id)
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def to(self, device: torch.device | str) -> "SyntheticMDM":
        self._device = torch.device(device)
        self.embed = self.embed.to(self._device)
        self.backbone = self.backbone.to(self._device)
        self.head = self.head.to(self._device)
        return self

    def eval(self) -> "SyntheticMDM":
        self.embed.eval()
        self.backbone.eval()
        self.head.eval()
        return self

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
        x = self.embed(input_ids.to(self._device))
        if attention_mask is not None:
            src_key_padding_mask = attention_mask.to(self._device) == 0
        else:
            src_key_padding_mask = None
        h = self.backbone(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(h)
