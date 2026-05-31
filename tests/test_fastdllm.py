import pytest
import torch

from mdm_chipmunk.methods.fastdllm import FastDLLM, _get_transfer_index, _num_transfer_tokens
from mdm_chipmunk.models.synthetic import SyntheticMDM


def test_num_transfer_tokens_sums_match_total_masks():
    block_mask = torch.tensor([[True, True, True, False, True], [False, True, True, True, True]])
    out = _num_transfer_tokens(block_mask, steps=3)
    assert out.shape == (2, 3)
    # Per-row sums must equal the row's total mask count.
    assert torch.equal(out.sum(dim=1), block_mask.sum(dim=1))


def test_get_transfer_index_threshold_mode_forces_at_least_one():
    B, L, V = 1, 6, 8
    logits = torch.randn(B, L, V)
    x = torch.zeros(B, L, dtype=torch.long)
    mask = torch.tensor([[True, True, True, False, False, True]])
    x0, transfer = _get_transfer_index(
        logits, temperature=0.0, remasking="low_confidence",
        mask_index=mask, x=x, num_transfer_tokens=None, threshold=10.0,  # impossibly high
    )
    # threshold=10 admits no positions on its own; the force-max rule must still pick at least one.
    assert transfer.any()
    # Forced position must have been masked.
    assert (transfer & mask).sum() == transfer.sum()


def test_get_transfer_index_topk_mode_respects_quota():
    B, L, V = 1, 8, 16
    logits = torch.randn(B, L, V)
    x = torch.zeros(B, L, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    quota = torch.tensor([3], dtype=torch.long)
    _, transfer = _get_transfer_index(
        logits, temperature=0.0, remasking="low_confidence",
        mask_index=mask, x=x, num_transfer_tokens=quota, threshold=None,
    )
    assert int(transfer.sum().item()) == 3


def test_fastdllm_vanilla_runs_on_synthetic_model():
    model = SyntheticMDM(seq_len=24, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    method = FastDLLM(mode="vanilla", block_length=4, threshold=None)
    result = method.generate(
        model=model, prompt_ids=prompt, gen_length=8, num_steps=4, seed=0
    )
    assert result.output_ids.shape == (1, 8)
    assert (result.output_ids != model.mask_token_id).all()
    assert result.telemetry.extras["mode"] == "vanilla"
    assert result.telemetry.extras["nfe"] >= 2  # at least 2 forward passes


def test_fastdllm_cache_modes_reject_synthetic_model():
    """Prefix/dual cache require an HF model. Synthetic should raise cleanly."""
    model = SyntheticMDM(seq_len=16, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    method = FastDLLM(mode="prefix_cache", block_length=4, threshold=None)
    prompt = torch.tensor([[5, 6]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="prefix/dual cache"):
        method.generate(model=model, prompt_ids=prompt, gen_length=8, num_steps=4)


def test_fastdllm_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        FastDLLM(mode="bogus")
