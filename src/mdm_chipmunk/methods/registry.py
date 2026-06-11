from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import InferenceMethod

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "methods"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No method config at {path}. Available: {list_methods()}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_methods() -> list[str]:
    if not _CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


def build_method(name: str, **overrides: Any) -> InferenceMethod:
    cfg = _load_yaml(name)
    kind = cfg.pop("method")
    cfg.pop("name", None)
    cfg.update(overrides)

    if kind == "dense":
        from .dense import DenseMDM

        return DenseMDM(**cfg)
    if kind == "fastdllm":
        from .fastdllm import FastDLLM

        return FastDLLM(**cfg)
    if kind == "dkv_cache":
        from .dkv_cache import DKVCache

        return DKVCache(**cfg)
    if kind == "dualdiffusion":
        from .dualdiffusion import DualDiffusion

        if "drafter_method_cfg" in cfg and cfg["drafter_method_cfg"] is None:
            cfg["drafter_method_cfg"] = {}
        return DualDiffusion(**cfg)

    raise ValueError(f"Unknown method {kind!r} in config {name!r}")
