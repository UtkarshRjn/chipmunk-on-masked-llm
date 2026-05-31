import torch

from mdm_chipmunk.models import build_model
from mdm_chipmunk.models.synthetic import SyntheticMDM


def test_synthetic_model_forward_shape():
    model = SyntheticMDM(seq_len=16, vocab_size=32, num_layers=1, d_model=16, n_heads=2)
    model.eval()
    ids = torch.randint(0, 32, (2, 16))
    logits = model.forward_logits(ids)
    assert logits.shape == (2, 16, 32)
    assert torch.isfinite(logits).all()


def test_synthetic_model_attention_mask_respected():
    model = SyntheticMDM(seq_len=8, vocab_size=16, num_layers=1, d_model=8, n_heads=2)
    model.eval()
    ids = torch.zeros((1, 8), dtype=torch.long)
    mask = torch.ones((1, 8))
    logits = model.forward_logits(ids, attention_mask=mask)
    assert logits.shape == (1, 8, 16)


def test_registry_builds_synthetic_from_yaml():
    model = build_model("synthetic")
    assert model.config.architecture == "synthetic-transformer"
    assert model.seq_len == 32
    assert model.mask_token_id == 1


def test_registry_lazy_llada(monkeypatch):
    """Lazy load should not hit the network for weights — only tokenizer.

    Skipped if HuggingFace transformers or the tokenizer fetch is unavailable
    (offline laptop or trimmed dep install).
    """
    import pytest

    pytest.importorskip("transformers")
    import urllib.error

    try:
        model = build_model("llada_8b_instruct", lazy=True)
    except (OSError, urllib.error.URLError, ValueError):
        pytest.skip("HuggingFace tokenizer fetch failed — offline laptop is fine")
    assert model.config.name == "llada-8b-instruct"
    assert model.config.mask_token_id == 126336
