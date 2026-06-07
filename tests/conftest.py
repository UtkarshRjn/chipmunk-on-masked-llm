import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_addoption(parser):
    parser.addoption("--network", action="store_true", default=False,
                     help="run tests that require HuggingFace network access")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access (skipped without --network)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--network"):
        skip = pytest.mark.skip(reason="network test — pass --network to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)
