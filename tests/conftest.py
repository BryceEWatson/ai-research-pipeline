"""Shared pytest setup.

Every run uses a throwaway vector store, so tests never touch data/chroma_db.
By default a placeholder OpenAI key satisfies settings validation and the
optional keys are blanked, so nothing calls an API. Tests marked
``integration`` call real APIs and only run with ``--run-integration``.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that call OpenAI/Perplexity and need real API keys",
    )


def pytest_configure(config):
    # core.config reads .env relative to the working directory.
    os.chdir(PROJECT_ROOT)
    os.environ["CHROMA_DB_PATH"] = tempfile.mkdtemp(prefix="chroma-test-")
    if not config.getoption("--run-integration"):
        os.environ["OPENAI_API_KEY"] = "sk-offline-placeholder"
        os.environ["VECTARA_API_KEY"] = ""
        os.environ["PERPLEXITY_API_KEY"] = ""


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs real API keys; pass --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
