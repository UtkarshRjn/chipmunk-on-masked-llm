"""DualDiffusion (Goyal et al. 2026) — speculative decoding for MDMs.

No upstream code release as of June 2026 (arXiv:2604.05250, posted April 2026).
This is a clean-room implementation following the paper's described algorithm
as summarized in the research proposal §2.2 / §3.2-C5:

    Drafter (cheap MDM method) runs `draft_steps` denoising steps.
    Verifier (full MDM, typically LLaDA-8B) evaluates the drafted sequence in
    one forward pass. Tokens are accepted if the verifier agrees (KL-divergence
    or confidence threshold); rejected tokens are remasked and the loop repeats.

The verification rules below are heuristic — the proposal §6 flags DualDiffusion's
verification as "no formal error bound", which is one of the gaps Contribution 3
of the proposal targets.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from ..models.base import MDMModel
from .base import GenerationResult, InferenceMethod, Telemetry


class DualDiffusion(InferenceMethod):
    name = "dualdiffusion"

    VALID_RULES = ("kl", "confidence", "both")

    def __init__(
        self,
        drafter_method_name: str = "fastdllm",
        drafter_method_cfg: dict[str, Any] | None = None,
        draft_steps: int = 4,
        verify_rule: str = "confidence",
        confidence_threshold: float = 0.7,
        kl_threshold: float = 0.5,
        temperature: float = 0.0,
        max_outer_loops: int = 16,
        **cfg: Any,
    ):
        if verify_rule not in self.VALID_RULES:
            raise ValueError(f"verify_rule must be one of {self.VALID_RULES}, got {verify_rule!r}")
        super().__init__(
            drafter_method_name=drafter_method_name,
            drafter_method_cfg=drafter_method_cfg or {},
            draft_steps=draft_steps,
            verify_rule=verify_rule,
            confidence_threshold=confidence_threshold,
            kl_threshold=kl_threshold,
            temperature=temperature,
            max_outer_loops=max_outer_loops,
            **cfg,
        )
        self.drafter_method_name = drafter_method_name
        self.drafter_method_cfg = drafter_method_cfg or {}
        self.draft_steps = draft_steps
        self.verify_rule = verify_rule
        self.confidence_threshold = confidence_threshold
        self.kl_threshold = kl_threshold
        self.temperature = temperature
        self.max_outer_loops = max_outer_loops

    @torch.no_grad()
    def generate(
        self,
        model: MDMModel,
        prompt_ids: torch.LongTensor,
        gen_length: int,
        num_steps: int,
        seed: int | None = None,
        verifier_model: MDMModel | None = None,
        drafter_model: MDMModel | None = None,
    ) -> GenerationResult:
        """Run drafter→verifier→accept/reject loop.

        ``model`` is the verifier (slow, full LLaDA). ``drafter_model`` defaults
        to the same model if not provided — in that case the cheapness of the
        draft comes purely from drafter_method using a cheaper inference path.
        """
        if seed is not None:
            torch.manual_seed(seed)

        verifier = verifier_model or model
        drafter = drafter_model or model

        device = verifier.device
        self._reset_peak_memory(device)
        prompt_ids = prompt_ids.to(device)
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        from .registry import build_method  # local to avoid circular import
        drafter_method = build_method(self.drafter_method_name, **self.drafter_method_cfg)

        mask_id = verifier.mask_token_id
        B, Lp = prompt_ids.shape
        x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
        x[:, :Lp] = prompt_ids

        telemetry = Telemetry(
            method=self.name,
            model=verifier.config.name,
            num_steps=num_steps,
            gen_length=gen_length,
            extras={
                "drafter": drafter.config.name,
                "drafter_method": self.drafter_method_name,
                "verify_rule": self.verify_rule,
                "accepted_per_loop": [],
                "rejected_per_loop": [],
                "outer_loops": 0,
            },
        )

        wall = time.perf_counter()
        for loop in range(self.max_outer_loops):
            outer_start = time.perf_counter()
            if (x[:, Lp:] != mask_id).all():
                break

            draft_result = drafter_method.generate(
                model=drafter,
                prompt_ids=x[0, : (x[0] != mask_id).sum().item() or Lp].unsqueeze(0)
                if False else prompt_ids,
                gen_length=gen_length,
                num_steps=self.draft_steps,
                seed=(seed or 0) + loop,
            )
            drafted_full = torch.cat([prompt_ids, draft_result.output_ids], dim=1)

            verify_logits = verifier.forward_logits(drafted_full)
            verify_probs = F.softmax(verify_logits.to(torch.float64), dim=-1)
            drafted_token_p = torch.gather(
                verify_probs, dim=-1, index=drafted_full.unsqueeze(-1)
            ).squeeze(-1)

            mask_region = torch.zeros_like(drafted_full, dtype=torch.bool)
            mask_region[:, Lp:] = (x[:, Lp:] == mask_id)

            accept = torch.zeros_like(drafted_full, dtype=torch.bool)
            if self.verify_rule in ("confidence", "both"):
                accept |= drafted_token_p >= self.confidence_threshold
            if self.verify_rule in ("kl", "both"):
                draft_logits = drafter.forward_logits(drafted_full)
                draft_probs = F.softmax(draft_logits.to(torch.float64), dim=-1)
                kl = (draft_probs * (draft_probs.clamp_min(1e-12).log() - verify_probs.clamp_min(1e-12).log())).sum(dim=-1)
                accept |= kl <= self.kl_threshold

            accept &= mask_region
            x = torch.where(accept, drafted_full, x)

            telemetry.extras["accepted_per_loop"].append(int(accept.sum().item()))
            telemetry.extras["rejected_per_loop"].append(
                int(mask_region.sum().item()) - int(accept.sum().item())
            )
            telemetry.step_times_s.append(time.perf_counter() - outer_start)

            if int(accept.sum().item()) == 0:
                break

        if (x[:, Lp:] == mask_id).any():
            still_masked = x == mask_id
            final_logits = verifier.forward_logits(x)
            final_argmax = torch.argmax(final_logits, dim=-1)
            x = torch.where(still_masked, final_argmax, x)

        telemetry.total_time_s = time.perf_counter() - wall
        telemetry.peak_memory_bytes = self._peak_memory(device)
        telemetry.extras["outer_loops"] = len(telemetry.extras["accepted_per_loop"])

        gen_ids = x[:, Lp:]
        text = verifier.detokenize(gen_ids[0])
        return GenerationResult(output_ids=gen_ids, output_text=text, telemetry=telemetry)
