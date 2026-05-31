"""dKV-Cache (Ma et al. 2025) port.

Two variants mirroring the upstream horseee/dKV-Cache:

- ``decode``: one-step-delayed KV caching after a token transitions from masked
  to decoded; periodic refresh via ``cache_reloading_step``. Their Algorithm 1.
- ``greedy``: aggressive cache lifespan with windowed schedule. Quadratic total
  complexity but higher per-step speed-up at some quality cost.

Both cache modes require the dKV-Cache-modified LLaDA model under
``dkv_cache/models/modeling_llada_dkv_cache_*.py`` (see the ``DKVCache.set_qkv_cache``
contract below). A third ``no_cache`` mode runs the same denoising schedule
without any cache and works with any :class:`MDMModel` — useful for isolating
the algorithmic remasking schedule from the kernel/cache contribution.

Adapted from horseee/dKV-Cache (MIT licensed) generation_utils/llada_dkv_cache_*.py.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from ..models.base import MDMModel
from .base import GenerationResult, InferenceMethod, Telemetry


class _SupportsDKVCache(Protocol):
    """The contract every dKV-Cache-modified LLaDA model fork exposes."""

    def set_qkv_cache(self, query: bool, kv: bool) -> None: ...
    def get_pos_rotary_embedding_cache(self, mask: torch.Tensor) -> None: ...
    def reset_pos_rotary_embedding_cache(self) -> None: ...
    def __call__(
        self,
        x: torch.Tensor,
        *,
        past_query_key_values: Any = None,
        use_cache: bool = False,
        preprocess_cache: bool = False,
    ) -> Any: ...


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _num_transfer_tokens(block_mask: torch.Tensor, steps: int) -> torch.Tensor:
    total = block_mask.sum(dim=1, keepdim=True)
    base = total // steps
    rem = total - base * steps
    out = torch.zeros(total.size(0), steps, device=block_mask.device, dtype=torch.long) + base
    for b in range(total.size(0)):
        out[b, : int(rem[b].item())] += 1
    return out


def _select_topk_transfer(
    confidence: torch.Tensor,
    quotas: torch.Tensor,
    mask_index: torch.Tensor,
) -> torch.Tensor:
    B, L = confidence.shape
    transfer = torch.zeros(B, L, dtype=torch.bool, device=confidence.device)
    for b in range(B):
        k = int(quotas[b].item())
        if k <= 0:
            continue
        _, idx = torch.topk(confidence[b], k=k)
        transfer[b, idx] = True
    return transfer & mask_index


class DKVCache(InferenceMethod):
    name = "dkv_cache"

    VALID_MODES = ("no_cache", "decode", "greedy")

    def __init__(
        self,
        mode: str = "decode",
        block_length: int = 32,
        cache_reloading_step: int = 4,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        **cfg: Any,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        super().__init__(
            mode=mode,
            block_length=block_length,
            cache_reloading_step=cache_reloading_step,
            temperature=temperature,
            remasking=remasking,
            **cfg,
        )
        self.mode = mode
        self.block_length = block_length
        self.cache_reloading_step = cache_reloading_step
        self.temperature = temperature
        self.remasking = remasking

    @torch.no_grad()
    def generate(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_steps: int,
        seed: int | None = None,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)

        device = model.device
        self._reset_peak_memory(device)
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if gen_length % self.block_length != 0:
            raise ValueError(
                f"gen_length={gen_length} must be divisible by block_length={self.block_length}"
            )
        num_blocks = gen_length // self.block_length
        if num_steps % num_blocks != 0:
            raise ValueError(
                f"num_steps={num_steps} must be divisible by num_blocks={num_blocks}"
            )
        steps_per_block = num_steps // num_blocks

        telemetry = Telemetry(
            method=self.name,
            model=model.config.name,
            num_steps=num_steps,
            gen_length=gen_length,
            extras={
                "mode": self.mode,
                "block_length": self.block_length,
                "cache_reloading_step": self.cache_reloading_step,
            },
        )

        wall = time.perf_counter()
        if self.mode == "no_cache":
            x = self._generate_no_cache(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        elif self.mode == "decode":
            x = self._generate_decode(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        else:
            x = self._generate_greedy(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        telemetry.total_time_s = time.perf_counter() - wall
        telemetry.peak_memory_bytes = self._peak_memory(device)

        gen_ids = x[:, prompt_ids.shape[1]:]
        text = model.detokenize(gen_ids[0])
        return GenerationResult(output_ids=gen_ids, output_text=text, telemetry=telemetry)

    def _init_x(self, prompt_ids: torch.LongTensor, gen_length: int, mask_id: int) -> torch.LongTensor:
        B, Lp = prompt_ids.shape
        x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=prompt_ids.device)
        x[:, :Lp] = prompt_ids
        return x

    def _proposal(self, logits: torch.Tensor, x: torch.Tensor, mask_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits_with_noise = _add_gumbel_noise(logits, self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        if self.remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif self.remasking == "random":
            x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
        else:
            raise NotImplementedError(self.remasking)
        x0 = torch.where(mask_index, x0, x)
        return x0, x0_p

    def _generate_no_cache(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> torch.LongTensor:
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)

            for i in range(steps_per_block):
                step_start = time.perf_counter()
                mask_index = (x == mask_id)
                logits = model.forward_logits(x)
                x0, x0_p = self._proposal(logits, x, mask_index)
                x0_p = x0_p.clone()
                x0_p[:, e:] = -np.inf
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
                transfer = _select_topk_transfer(confidence, num_tx[:, i], mask_index)
                x = torch.where(transfer, x0, x)
                telemetry.step_times_s.append(time.perf_counter() - step_start)

        return x

    def _require_cached_model(self, model: MDMModel) -> Any:
        """Return the underlying HF module if it supports the dKV-Cache contract."""
        inner = getattr(model, "_model", None)
        if inner is None:
            raise RuntimeError(
                "DKVCache decode/greedy modes require a dKV-Cache-modified LLaDA model. "
                "Load via the upstream `models/modeling_llada_dkv_cache_decode.py` or "
                "`modeling_llada_dkv_cache_greedy.py`, then wrap with the LLaDA loader."
            )
        for attr in ("set_qkv_cache", "get_pos_rotary_embedding_cache", "reset_pos_rotary_embedding_cache"):
            if not hasattr(inner, attr):
                raise RuntimeError(
                    f"Loaded model is missing dKV-Cache hook `{attr}`. Use the modified "
                    "LLaDA fork from horseee/dKV-Cache."
                )
        return inner

    def _generate_decode(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> torch.LongTensor:
        inner = self._require_cached_model(model)
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)

            past_qkv: Any = None
            prv_transfer_idx: torch.Tensor | None = None
            cur_transfer_idx: torch.Tensor | None = None

            for i in range(steps_per_block):
                step_start = time.perf_counter()
                mask_index = (x == mask_id)

                if i % self.cache_reloading_step != 0 and i > 1 and prv_transfer_idx is not None:
                    inner.set_qkv_cache(True, True)
                    inner.get_pos_rotary_embedding_cache(~prv_transfer_idx)
                    next_x = x[~prv_transfer_idx].view(x.shape[0], -1)
                    outputs = inner(
                        next_x,
                        past_query_key_values=past_qkv,
                        use_cache=True,
                        preprocess_cache=True,
                    )
                    inner.set_qkv_cache(False, False)
                    inner.reset_pos_rotary_embedding_cache()
                elif i > 0:
                    outputs = inner(
                        x, past_query_key_values=past_qkv,
                        use_cache=False, preprocess_cache=True,
                    )
                else:
                    outputs = inner(
                        x, past_query_key_values=None,
                        use_cache=False, preprocess_cache=False,
                    )

                logits = outputs.logits
                past_qkv = outputs.past_key_values

                x0, x0_p = self._proposal(logits, x, mask_index)
                x0_p = x0_p.clone()
                x0_p[:, e:] = -np.inf

                if x0_p.shape[1] < x.shape[1] and prv_transfer_idx is not None:
                    refill_p = torch.full(
                        (x.shape[0], x.shape[1]), -np.inf, device=x0_p.device, dtype=x0_p.dtype
                    )
                    reorder = torch.nonzero(~prv_transfer_idx, as_tuple=True)[1].view(x0_p.shape[0], -1)
                    refill_p.scatter_(1, reorder, x0_p)
                    x0_p = refill_p
                    x0_p[:, e:] = -np.inf
                    refill_x0 = torch.full(
                        (x.shape[0], x.shape[1]), mask_id, device=x0.device, dtype=x0.dtype
                    )
                    refill_x0.scatter_(1, reorder, x0)
                    x0 = refill_x0
                    x0 = torch.where(mask_index, x0, x)

                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
                transfer = _select_topk_transfer(confidence, num_tx[:, i], mask_index)
                x = torch.where(transfer, x0, x)

                prv_transfer_idx, cur_transfer_idx = cur_transfer_idx, (x != mask_id)
                past_qkv = [past_qkv, (prv_transfer_idx, cur_transfer_idx)]
                telemetry.step_times_s.append(time.perf_counter() - step_start)

        return x

    def _generate_greedy(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> torch.LongTensor:
        inner = self._require_cached_model(model)
        if self.remasking != "random":
            raise NotImplementedError(
                "dKV-Cache greedy variant only supports remasking='random' upstream."
            )
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)
            cumtx = torch.cumsum(
                torch.cat(
                    [torch.zeros(num_tx.size(0), 1, dtype=num_tx.dtype, device=num_tx.device), num_tx],
                    dim=-1,
                ),
                dim=-1,
            )

            x0_p = torch.rand(x.shape, device=x.device)
            x0_p[:, e:] = -np.inf
            x0_p[:, :s] = -np.inf
            denoising_order = torch.sort(x0_p, descending=True).indices

            past_qkv: Any = None
            prv_transfer_idx: torch.Tensor | None = None
            cur_transfer_idx: torch.Tensor | None = None

            for i in range(steps_per_block):
                step_start = time.perf_counter()
                mask_index = (x == mask_id)

                if i % self.cache_reloading_step != 0 and cur_transfer_idx is not None:
                    inner.set_qkv_cache(True, True)
                    inner.get_pos_rotary_embedding_cache(cur_transfer_idx)
                    next_x = x[cur_transfer_idx].view(x.shape[0], -1)
                    outputs = inner(
                        next_x,
                        past_query_key_values=past_qkv,
                        use_cache=True,
                        preprocess_cache=True,
                    )
                    inner.set_qkv_cache(False, False)
                    inner.reset_pos_rotary_embedding_cache()
                else:
                    outputs = inner(
                        x, past_query_key_values=past_qkv,
                        use_cache=False, preprocess_cache=(i > 0),
                    )

                logits = outputs.logits
                past_qkv = outputs.past_key_values
                x0 = torch.argmax(_add_gumbel_noise(logits, self.temperature), dim=-1)
                x0 = torch.where(mask_index, x0, x)

                lo = int(cumtx[0, i].item())
                hi = int(cumtx[0, i + 1].item())
                step_indices = denoising_order[:, lo:hi]
                transfer = torch.zeros_like(x, dtype=torch.bool)
                transfer.scatter_(1, step_indices, True)
                transfer &= mask_index
                x = torch.where(transfer, x0, x)

                prv_transfer_idx, cur_transfer_idx = cur_transfer_idx, (x != mask_id)
                past_qkv = [past_qkv, (prv_transfer_idx, cur_transfer_idx)]
                telemetry.step_times_s.append(time.perf_counter() - step_start)

        return x
