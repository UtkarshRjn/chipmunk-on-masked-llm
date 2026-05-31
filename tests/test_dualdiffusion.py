import pytest
import torch

from mdm_chipmunk.methods.dualdiffusion import DualDiffusion
from mdm_chipmunk.models.synthetic import SyntheticMDM


def test_dualdiffusion_runs_with_same_drafter_verifier():
    """Smoke: when drafter == verifier (synthetic model both sides), the loop
    must run, terminate, and fill all masked positions."""
    model = SyntheticMDM(seq_len=24, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    method = DualDiffusion(
        drafter_method_name="fastdllm_vanilla",
        drafter_method_cfg={"mode": "vanilla", "block_length": 4, "threshold": None},
        draft_steps=2,
        verify_rule="confidence",
        confidence_threshold=0.0,  # accept everything — guarantees forward progress
        max_outer_loops=4,
    )
    result = method.generate(
        model=model, prompt_ids=prompt, gen_length=8, num_steps=8, seed=0
    )
    assert result.output_ids.shape == (1, 8)
    assert (result.output_ids != model.mask_token_id).all()
    assert result.telemetry.extras["outer_loops"] >= 1


def test_dualdiffusion_invalid_verify_rule():
    with pytest.raises(ValueError, match="verify_rule"):
        DualDiffusion(verify_rule="bogus")


def test_dualdiffusion_kl_rule_is_callable():
    """Ensure KL-rule path doesn't blow up on shape mismatch."""
    model = SyntheticMDM(seq_len=24, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    method = DualDiffusion(
        drafter_method_name="fastdllm_vanilla",
        drafter_method_cfg={"mode": "vanilla", "block_length": 4, "threshold": None},
        draft_steps=2,
        verify_rule="kl",
        kl_threshold=10.0,  # very permissive
        max_outer_loops=3,
    )
    result = method.generate(
        model=model, prompt_ids=prompt, gen_length=8, num_steps=8, seed=0
    )
    assert result.output_ids.shape == (1, 8)
