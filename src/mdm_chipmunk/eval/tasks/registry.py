from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import Task

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "tasks"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No task config at {path}. Available: {list_tasks()}")
    with open(path) as f:
        return yaml.safe_load(f)


def list_tasks() -> list[str]:
    if not _CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


def build_task(name: str, **overrides: Any) -> Task:
    cfg = _load_yaml(name)
    kind = cfg.pop("task")
    cfg.pop("name", None)
    cfg.update(overrides)

    if kind == "gsm8k":
        from .gsm8k import GSM8K

        return GSM8K(**cfg)
    if kind == "synthetic":
        from .synthetic import SyntheticTask

        return SyntheticTask(**cfg)

    raise ValueError(f"Unknown task {kind!r} in config {name!r}")
