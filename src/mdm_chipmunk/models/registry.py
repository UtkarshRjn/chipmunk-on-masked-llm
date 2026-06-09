from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..utils.logging import get_logger
from .base import MDMModel, ModelConfig

_LOG = get_logger(__name__)
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "models"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config at {path}. Available: {list_models()}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_models() -> list[str]:
    if not _CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


def build_model(name: str, lazy: bool = False, **kwargs: Any) -> MDMModel:
    """Build an MDMModel from a config name (looked up in configs/models/<name>.yaml)."""
    cfg_dict = _load_yaml(name)
    arch = cfg_dict.get("architecture", "")
    config = ModelConfig(
        name=cfg_dict.get("name", name),
        hf_id=cfg_dict["hf_id"],
        mask_token_id=int(cfg_dict["mask_token_id"]),
        seq_len=int(cfg_dict.get("seq_len", 1024)),
        num_layers=int(cfg_dict.get("num_layers", 0)),
        vocab_size=int(cfg_dict.get("vocab_size", 0)),
        architecture=arch,
        trust_remote_code=bool(cfg_dict.get("trust_remote_code", True)),
        extras=cfg_dict.get("extras", {}),
    )

    if arch == "llada":
        from .llada import LLaDA

        return LLaDA(config, lazy=lazy, **kwargs)
    if arch == "mdlm":
        from .mdlm import MDLM

        return MDLM(config, lazy=lazy, **kwargs)
    if arch == "synthetic-transformer":
        from .synthetic import SyntheticMDM

        return SyntheticMDM(
            seq_len=config.seq_len,
            vocab_size=config.vocab_size or 64,
            mask_token_id=config.mask_token_id,
        )

    raise ValueError(f"Unknown architecture {arch!r} in model config {name!r}")
