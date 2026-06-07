"""Tests for hero signal computation: per-release watch scoping, whole-token
label matching, and the first-release perf window."""
from __future__ import annotations

from helpers import add_issue, add_perf_claim, add_pr, add_release
from vllm_insights import hero


def test_watch_count_scoped_to_release_window(db):
    add_release(db, "v0.9.0", days_ago=30)
    add_release(db, "v1.0.0", days_ago=0)
    add_issue(db, 1, labels="regression", created_days_ago=2)     # in window
    add_issue(db, 2, labels="regression", created_days_ago=400)   # before prev release
    sig = hero._release_signals(db, "v1.0.0", _published_at(db, "v1.0.0"))
    assert sig["watch_n"] == 1


def test_watch_count_matches_whole_label_token(db):
    add_release(db, "v0.9.0", days_ago=30)
    add_release(db, "v1.0.0", days_ago=0)
    add_issue(db, 1, labels="regression", created_days_ago=2)
    add_issue(db, 2, labels="test-regression", created_days_ago=2)   # must NOT count
    add_issue(db, 3, labels="bug, regression", created_days_ago=2)   # comma-space form counts
    sig = hero._release_signals(db, "v1.0.0", _published_at(db, "v1.0.0"))
    assert sig["watch_n"] == 2


def test_open_watch_count_excludes_substring_false_positive(db):
    add_issue(db, 1, labels="regression")
    add_issue(db, 2, labels="non-regression")  # must NOT count
    assert hero._open_watch_count(db) == 1


def test_first_release_perf_window_not_zero(db):
    # Only one release ever -> previous-release window would be zero-width.
    add_release(db, "v1.0.0", days_ago=0)
    add_pr(db, 1, merged_days_ago=5)
    add_perf_claim(db, 1)
    sig = hero._release_signals(db, "v1.0.0", _published_at(db, "v1.0.0"))
    assert sig["perf_n"] == 1


def _published_at(db, tag: str) -> str:
    import sqlite3
    with sqlite3.connect(db) as c:
        return c.execute("SELECT published_at FROM releases WHERE tag=?", (tag,)).fetchone()[0]
