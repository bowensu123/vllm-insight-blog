"""Synthetic-row builders shared by the test modules."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def add_release(db: Path, tag: str, *, days_ago: float = 1.0, prerelease: int = 0) -> None:
    ts = (now() - dt.timedelta(days=days_ago)).isoformat()
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO releases(tag,name,published_at,is_prerelease,url) VALUES(?,?,?,?,?)",
            (tag, tag, ts, prerelease, f"http://x/rel/{tag}"),
        )


def add_pr(
    db: Path,
    number: int,
    *,
    merged_days_ago: float = 1.0,
    release_tag: str | None = None,
    title: str = "A PR",
    author: str = "alice",
    body: str | None = None,
) -> None:
    n = now()
    created = (n - dt.timedelta(days=merged_days_ago + 1)).isoformat()
    merged = (n - dt.timedelta(days=merged_days_ago)).isoformat()
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO pull_requests(number,title,state,author,created_at,merged_at,"
            "url,release_tag,body) VALUES(?,?,?,?,?,?,?,?,?)",
            (number, title, "MERGED", author, created, merged,
             f"http://x/pr/{number}", release_tag, body),
        )


def add_issue(
    db: Path,
    number: int,
    *,
    labels: str,
    state: str = "OPEN",
    created_days_ago: float = 1.0,
) -> None:
    n = now()
    ts = (n - dt.timedelta(days=created_days_ago)).isoformat()
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO issues(number,title,state,author,created_at,updated_at,labels,url) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (number, f"issue {number}", state, "bob", ts, n.isoformat(),
             labels, f"http://x/iss/{number}"),
        )


def add_perf_claim(db: Path, pr_number: int) -> None:
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO perf_claims(pr_number,snippet,kind,value) VALUES(?,?,?,?)",
            (pr_number, "1.5x faster", "multiplier", 1.5),
        )
