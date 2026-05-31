import pytest
import torch

from mdm_chipmunk.methods.dkv_cache import DKVCache, _num_transfer_tokens, _select_topk_transfer
from mdm_chipmunk.models.synthetic import SyntheticMDM


def test_num_transfer_tokens_distributes_remainder():
    mask = torch.tensor([[True, True, True, True, True, True, True]])  # 7 masks
    out = _num_transfer_tokens(mask, steps=3)
    # 7 = 3 + 2 + 2 → remainder of 1 goes to the first step
    assert out.tolist() == [[3, 2, 2]]


def test_select_topk_transfer_respects_quota_and_mask():
    conf = torch.tensor([[0.9, 0.1, 0.5, 0.7]])
    quotas = torch.tensor([2], dtype=torch.long)
    mask = torch.tensor([[True, False, True, True]])  # position 1 not maskable
    transfer = _select_topk_transfer(conf, quotas, mask)
    # Top-2 by confidence: positions 0 (0.9) and 3 (0.7). Position 1 excluded by mask.
    assert transfer.tolist() == [[True, False, False, True]]


def test_dkv_cache_no_cache_runs_on_synthetic_model():
    model = SyntheticMDM(seq_len=24, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    method = DKVCache(mode="no_cache", block_length=4)
    result = method.generate(
        model=model, prompt_ids=prompt, gen_length=8, num_steps=4, seed=0
    )
    assert result.output_ids.shape == (1, 8)
    assert (result.output_ids != model.mask_token_id).all()
    assert result.telemetry.extras["mode"] == "no_cache"


def test_dkv_cache_decode_rejects_synthetic_model():
    model = SyntheticMDM(seq_len=16, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    method = DKVCache(mode="decode", block_length=4)
    prompt = torch.tensor([[5, 6]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="dKV-Cache-modified LLaDA"):
        method.generate(model=model, prompt_ids=prompt, gen_length=8, num_steps=4)


def test_dkv_cache_greedy_requires_random_remasking():
    with pytest.raises(NotImplementedError, match="random"):
        method = DKVCache(mode="greedy", remasking="low_confidence")
        model = SyntheticMDM(seq_len=16, vocab_size=32, num_layers=1, d_model=16, n_heads=2)

        class _Fake:
            def set_qkv_cache(self, *a, **k): pass
            def get_pos_rotary_embedding_cache(self, *a, **k): pass
            def reset_pos_rotary_embedding_cache(self, *a, **k): pass

        model._model = _Fake()
        method.generate(
            model=model, prompt_ids=torch.tensor([[1]], dtype=torch.long),
            gen_length=4, num_steps=2,
        )


def test_dkv_cache_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        DKVCache(mode="bogus")
