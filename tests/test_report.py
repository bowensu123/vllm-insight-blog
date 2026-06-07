"""Tests for the weekly digest assembly: raw list rendering, nan handling,
fingerprint reuse, and the LLM-failure fallback."""
from __future__ import annotations

import vllm_insights.summarize as sm
from helpers import add_pr, add_release
from vllm_insights import report


def test_raw_list_renders_html_not_collapsed_markdown(db):
    add_pr(db, 101, release_tag=None)
    md = report.generate_weekly_digest(db, include_llm=False)
    # Explicit HTML list so renderers don't collapse it into one blob.
    assert "<ul>" in md and "<li>" in md
    assert 'href="http://x/pr/101"' in md


def test_no_nan_suffix_for_unreleased_pr(db):
    add_pr(db, 101, release_tag=None)
    md = report.generate_weekly_digest(db, include_llm=False)
    assert "nan" not in md.lower()


def test_release_tag_shown_as_code(db):
    add_release(db, "v1.0.0")
    add_pr(db, 1, release_tag="v1.0.0")
    md = report.generate_weekly_digest(db, include_llm=False)
    assert "<code>v1.0.0</code>" in md


def test_title_is_html_escaped(db):
    add_pr(db, 1, title="x <b> & y")
    md = report.generate_weekly_digest(db, include_llm=False)
    assert "&lt;b&gt; &amp; y" in md
    assert "<b>" not in md


def test_fingerprint_reuse_skips_second_llm_call(db, monkeypatch):
    add_pr(db, 1)
    calls = {"n": 0}

    def fake_summarize(*a, **k):
        calls["n"] += 1
        return "## TL;DR\nfresh insight"

    monkeypatch.setattr(sm, "summarize_window", fake_summarize)
    report.generate_weekly_digest(db, include_llm=True)
    md2 = report.generate_weekly_digest(db, include_llm=True)
    assert calls["n"] == 1, "unchanged data should reuse cache, not re-call the LLM"
    assert "fresh insight" in md2


def test_changed_data_triggers_new_llm_call(db, monkeypatch):
    add_pr(db, 1)
    calls = {"n": 0}
    monkeypatch.setattr(sm, "summarize_window",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), "## TL;DR\nx")[1])
    report.generate_weekly_digest(db, include_llm=True)
    add_pr(db, 2)  # window changed -> fingerprint differs
    report.generate_weekly_digest(db, include_llm=True)
    assert calls["n"] == 2


def test_llm_failure_falls_back_to_last_good_digest(db, monkeypatch):
    add_pr(db, 1)
    monkeypatch.setattr(sm, "summarize_window", lambda *a, **k: "## TL;DR\ngood digest")
    report.generate_weekly_digest(db, include_llm=True)  # seeds cache

    add_pr(db, 2)  # force a fresh (failing) generation

    def boom(*a, **k):
        raise RuntimeError("timeout")

    monkeypatch.setattr(sm, "summarize_window", boom)
    md = report.generate_weekly_digest(db, include_llm=True)
    assert "good digest" in md
    assert "last successful digest" in md


def test_llm_failure_without_cache_shows_skip_line(db, monkeypatch):
    add_pr(db, 1)

    def boom(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr(sm, "summarize_window", boom)
    md = report.generate_weekly_digest(db, include_llm=True)
    assert "LLM digest skipped" in md


def test_raw_list_does_not_at_mention_pr_authors(db):
    # Regression: posting the digest to a GitHub issue must not ping PR authors.
    add_pr(db, 1, author="hmellor")
    md = report.generate_weekly_digest(db, include_llm=False)
    assert "@hmellor" not in md                       # no bare mention
    assert 'href="https://github.com/hmellor"' in md  # rendered as a profile link


def test_neutralize_mentions_defangs_only_bare_mentions():
    zwsp = "\u200b"
    out = report.neutralize_mentions("ty @hmellor and @Sunt-ing")
    assert "@hmellor" not in out and "@Sunt-ing" not in out
    assert f"@{zwsp}hmellor" in out and f"@{zwsp}Sunt-ing" in out
    # emails and code spans must be left intact
    assert report.neutralize_mentions("foo@bar.com") == "foo@bar.com"
    assert report.neutralize_mentions("`@x`") == "`@x`"
