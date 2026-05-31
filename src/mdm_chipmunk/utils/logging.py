from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich.logging import RichHandler


def get_logger(name: str = "mdm_chipmunk", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RichHandler(rich_tracebacks=True, show_path=False, show_time=True)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class JsonlWriter:
    """Append-only JSONL writer. One row per `write` call. Safe across crashes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, default=_default_json) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _default_json(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


@contextmanager
def timer():
    """Context manager that yields a callable returning elapsed seconds."""
    start = time.perf_counter()
    elapsed: dict[str, float] = {"value": 0.0}

    def get() -> float:
        return elapsed["value"] or (time.perf_counter() - start)

    try:
        yield get
    finally:
        elapsed["value"] = time.perf_counter() - start
