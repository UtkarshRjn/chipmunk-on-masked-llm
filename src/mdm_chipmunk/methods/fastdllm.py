"""Fast-dLLM (Wu et al. 2025) port.

Three modes mirroring the upstream NVlabs/Fast-dLLM `v1/llada/generate.py`:

- ``vanilla``: block-wise semi-AR denoising with confidence-threshold parallel
  decoding. No KV cache. Runs on any ``MDMModel``.
- ``prefix_cache``: caches K/V for the prompt once per block. Requires the host
  model to accept ``use_cache=True`` and return ``past_key_values`` (i.e. the
  Fast-dLLM-modified LLaDA model under ``fastdllm/v1/llada/model/``).
- ``dual_cache``: prefix + current-block KV cache with ``replace_position``.
  Same requirement as ``prefix_cache``.

Algorithm and ``get_transfer_index`` are adapted from Fast-dLLM, which is
Apache-2.0 licensed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..models.base import MDMModel
from .base import GenerationResult, InferenceMethod, Telemetry


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _num_transfer_tokens(block_mask: torch.Tensor, steps: int) -> torch.Tensor:
    total = block_mask.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    out = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    cols = torch.arange(steps, device=block_mask.device).unsqueeze(0)
    return out + (cols < rem.unsqueeze(1)).to(torch.long)


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: torch.Tensor | None,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = _add_gumbel_noise(logits, temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(f"remasking={remasking!r}")

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer = mask_index & (confidence >= threshold)
        max_conf = torch.argmax(confidence, dim=1, keepdim=True)
        force = torch.zeros_like(transfer).scatter_(1, max_conf, True)
        transfer = (transfer | force) & mask_index
        return x0, transfer

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    k = torch.clamp(num_transfer_tokens.to(dtype=torch.long, device=confidence.device), min=0)

    _, sort_idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    select_sorted = cols < k.unsqueeze(1).expand(B, L)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, sort_idx, select_sorted.to(torch.int8))
    return x0, transfer_int.bool() & mask_index


class FastDLLM(InferenceMethod):
    name = "fastdllm"

    VALID_MODES = ("vanilla", "prefix_cache", "dual_cache")

    def __init__(
        self,
        mode: str = "dual_cache",
        block_length: int = 32,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        threshold: float | None = 0.9,
        **cfg: Any,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        super().__init__(
            mode=mode,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            **cfg,
        )
        self.mode = mode
        self.block_length = block_length
        self.temperature = temperature
        self.remasking = remasking
        self.threshold = threshold

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

        telemetry = Telemetry(
            method=self.name,
            model=model.config.name,
            num_steps=num_steps,
            gen_length=gen_length,
            extras={"mode": self.mode, "block_length": self.block_length},
        )

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

        wall = time.perf_counter()
        if self.mode == "vanilla":
            x, nfe = self._generate_vanilla(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        elif self.mode == "prefix_cache":
            x, nfe = self._generate_prefix_cache(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        else:
            x, nfe = self._generate_dual_cache(
                model, prompt_ids, gen_length, num_blocks, steps_per_block, telemetry
            )
        telemetry.total_time_s = time.perf_counter() - wall
        telemetry.peak_memory_bytes = self._peak_memory(device)
        telemetry.extras["nfe"] = nfe

        gen_ids = x[:, prompt_ids.shape[1]:]
        text = model.detokenize(gen_ids[0])
        return GenerationResult(output_ids=gen_ids, output_text=text, telemetry=telemetry)

    def _init_x(self, prompt_ids: torch.LongTensor, gen_length: int, mask_id: int) -> torch.LongTensor:
        B = prompt_ids.shape[0]
        Lp = prompt_ids.shape[1]
        x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=prompt_ids.device)
        x[:, :Lp] = prompt_ids
        return x

    def _generate_vanilla(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> tuple[torch.LongTensor, int]:
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]
        nfe = 0

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)

            i = 0
            while True:
                step_start = time.perf_counter()
                mask_index = (x == mask_id)
                logits = model.forward_logits(x)
                mask_index[:, e:] = False
                quota = None if self.threshold is not None else num_tx[:, i]
                x0, transfer = _get_transfer_index(
                    logits, self.temperature, self.remasking, mask_index, x, quota, self.threshold
                )
                x = torch.where(transfer, x0, x)
                nfe += 1
                i += 1
                telemetry.step_times_s.append(time.perf_counter() - step_start)
                if (x[:, s:e] == mask_id).sum() == 0:
                    break

        return x, nfe

    def _model_forward_with_cache(
        self,
        model: MDMModel,
        x: torch.Tensor,
        past_key_values: Any = None,
        replace_position: torch.Tensor | None = None,
    ) -> Any:
        """Call into a Fast-dLLM-modified LLaDA model. Returns the raw HF output."""
        if not hasattr(model, "_model") or model._model is None:
            raise RuntimeError(
                "FastDLLM prefix/dual cache modes require a fully loaded HF model "
                "(MDLM/LLaDA wrappers expose ._model). Got lazy or non-HF model."
            )
        kwargs: dict[str, Any] = {"use_cache": True}
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if replace_position is not None:
            kwargs["replace_position"] = replace_position
        return model._model(x, **kwargs)

    def _generate_prefix_cache(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> tuple[torch.LongTensor, int]:
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]
        nfe = 0

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)

            step_start = time.perf_counter()
            out_full = self._model_forward_with_cache(model, x)
            past = out_full.past_key_values
            mask_index = (x == mask_id)
            mask_index[:, e:] = False
            quota0 = None if self.threshold is not None else num_tx[:, 0]
            x0, transfer = _get_transfer_index(
                out_full.logits, self.temperature, self.remasking, mask_index, x, quota0, self.threshold
            )
            x = torch.where(transfer, x0, x)
            nfe += 1
            telemetry.step_times_s.append(time.perf_counter() - step_start)

            past = tuple(tuple(t[:, :, :s] for t in layer) for layer in past)

            i = 1
            while (x[:, s:e] == mask_id).sum() > 0:
                step_start = time.perf_counter()
                tail = x[:, s:]
                tail_mask = (tail == mask_id)
                tail_mask[:, self.block_length:] = False
                out_blk = self._model_forward_with_cache(model, tail, past_key_values=past)
                quota_i = None if self.threshold is not None else num_tx[:, i]
                x0_tail, transfer_tail = _get_transfer_index(
                    out_blk.logits, self.temperature, self.remasking, tail_mask, tail, quota_i, self.threshold
                )
                tail_new = torch.where(transfer_tail, x0_tail, tail)
                x = torch.cat([x[:, :s], tail_new], dim=1)
                nfe += 1
                i += 1
                telemetry.step_times_s.append(time.perf_counter() - step_start)

        return x, nfe

    def _generate_dual_cache(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_blocks: int,
        steps_per_block: int,
        telemetry: Telemetry,
    ) -> tuple[torch.LongTensor, int]:
        mask_id = model.mask_token_id
        x = self._init_x(prompt_ids, gen_length, mask_id)
        Lp = prompt_ids.shape[1]
        nfe = 0

        for nb in range(num_blocks):
            s = Lp + nb * self.block_length
            e = s + self.block_length
            block_mask = (x[:, s:e] == mask_id)
            num_tx = _num_transfer_tokens(block_mask, steps_per_block)

            step_start = time.perf_counter()
            out_full = self._model_forward_with_cache(model, x)
            past = out_full.past_key_values
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, s:e] = True

            global_mask = (x == mask_id)
            global_mask[:, e:] = False
            quota0 = None if self.threshold is not None else num_tx[:, 0]
            x0, transfer = _get_transfer_index(
                out_full.logits, self.temperature, self.remasking, global_mask, x, quota0, self.threshold
            )
            x = torch.where(transfer, x0, x)
            nfe += 1
            telemetry.step_times_s.append(time.perf_counter() - step_start)

            for i in range(1, steps_per_block):
                if (x[:, s:e] == mask_id).sum() == 0:
                    break
                step_start = time.perf_counter()
                logits_blk = self._model_forward_with_cache(
                    model, x[:, s:e], past_key_values=past, replace_position=replace_position
                ).logits
                mask_blk = (x[:, s:e] == mask_id)
                quota_i = None if self.threshold is not None else num_tx[:, i]
                x0_blk, transfer_blk = _get_transfer_index(
                    logits_blk, self.temperature, self.remasking, mask_blk, x[:, s:e], quota_i, self.threshold
                )
                blk_new = torch.where(transfer_blk, x0_blk, x[:, s:e])
                x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                nfe += 1
                telemetry.step_times_s.append(time.perf_counter() - step_start)

        return x, nfe
