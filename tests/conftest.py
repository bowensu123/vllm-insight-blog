"""Pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from vllm_insights.db import init_db


@pytest.fixture
def db(tmp_path) -> Path:
    p = tmp_path / "insights.sqlite"
    init_db(p)
    return p
