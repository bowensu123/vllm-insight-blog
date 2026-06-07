"""Schema/migration smoke tests."""
from __future__ import annotations

import sqlite3

from vllm_insights.db import init_db


def test_init_db_is_idempotent(tmp_path):
    p = tmp_path / "x.sqlite"
    init_db(p)
    init_db(p)  # must not raise on a second run
    with sqlite3.connect(p) as c:
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"pull_requests", "releases", "issues", "digest_cache"} <= tables


def test_digest_cache_has_fingerprint_column(tmp_path):
    p = tmp_path / "x.sqlite"
    init_db(p)
    with sqlite3.connect(p) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(digest_cache)")}
    assert {"key", "content", "generated_at", "fingerprint"} <= cols


def test_migration_adds_fingerprint_to_old_digest_cache(tmp_path):
    # Simulate an older DB whose digest_cache predates the fingerprint column.
    p = tmp_path / "x.sqlite"
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE digest_cache (key TEXT PRIMARY KEY, "
                  "content TEXT NOT NULL, generated_at TEXT NOT NULL)")
    init_db(p)  # migration should add the column
    with sqlite3.connect(p) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(digest_cache)")}
    assert "fingerprint" in cols
