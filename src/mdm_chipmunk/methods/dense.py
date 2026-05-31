from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from ..models.base import MDMModel
from .base import GenerationResult, InferenceMethod, Telemetry


class DenseMDM(InferenceMethod):
    """Vanilla MDM denoising with low-confidence remasking (LLaDA-style).

    No caching. This is the quality ceiling and speed floor against which every
    other method in this repo is compared. Loosely follows Algorithm 1 of
    Nie et al. 2025 (LLaDA).
    """

    name = "dense"

    def __init__(
        self,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        eos_token_id: int | None = None,
        **cfg: Any,
    ):
        super().__init__(temperature=temperature, remasking=remasking, **cfg)
        self.temperature = temperature
        self.remasking = remasking
        self.eos_token_id = eos_token_id

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
        B, P = prompt_ids.shape
        L = P + gen_length

        mask_id = model.mask_token_id
        x = torch.full((B, L), mask_id, device=device, dtype=torch.long)
        x[:, :P] = prompt_ids
        prompt_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        prompt_mask[:, :P] = True

        telemetry = Telemetry(
            method=self.name,
            model=model.config.name,
            num_steps=num_steps,
            gen_length=gen_length,
        )

        steps_remaining_mask = torch.ones((B, L), device=device, dtype=torch.bool)
        steps_remaining_mask[prompt_mask] = False
        num_to_unmask_total = int(steps_remaining_mask.sum().item() // B)

        unmasked_per_step = _linear_unmask_schedule(num_to_unmask_total, num_steps)

        wall_start = time.perf_counter()
        for step_idx in range(num_steps):
            step_start = time.perf_counter()

            logits = model.forward_logits(x)
            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)
                confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=-1)
                confidence, sampled = probs.max(dim=-1)

            currently_masked = x == mask_id
            currently_masked &= ~prompt_mask

            k = unmasked_per_step[step_idx]
            if k == 0:
                telemetry.step_times_s.append(time.perf_counter() - step_start)
                continue

            masked_confidence = confidence.masked_fill(~currently_masked, float("-inf"))
            for b in range(B):
                available = int(currently_masked[b].sum().item())
                if available == 0:
                    continue
                pick = min(k, available)
                topk = torch.topk(masked_confidence[b], pick)
                idx = topk.indices
                x[b, idx] = sampled[b, idx]

            telemetry.step_times_s.append(time.perf_counter() - step_start)

            if (x[:, P:] != mask_id).all():
                break

        telemetry.total_time_s = time.perf_counter() - wall_start
        telemetry.peak_memory_bytes = self._peak_memory(device)

        gen_ids = x[:, P:]
        text = model.detokenize(gen_ids[0])
        return GenerationResult(output_ids=gen_ids, output_text=text, telemetry=telemetry)


def _linear_unmask_schedule(total: int, steps: int) -> list[int]:
    """Distribute `total` unmasks across `steps` steps as evenly as possible."""
    if steps <= 0:
        return []
    per = [total // steps] * steps
    remainder = total - sum(per)
    for i in range(remainder):
        per[i] += 1
    return per
