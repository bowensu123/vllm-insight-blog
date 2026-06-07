"""Tests for backend detection, model defaults, and LLM-input rendering."""
from __future__ import annotations

import vllm_insights.summarize as sm
from helpers import add_pr


def test_detect_backend_priority(monkeypatch):
    for v in ("LLM_BACKEND", "DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert sm._detect_backend(None) == "anthropic"          # default

    monkeypatch.setenv("GITHUB_TOKEN", "x")
    assert sm._detect_backend(None) == "github"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert sm._detect_backend(None) == "anthropic"          # anthropic beats github

    monkeypatch.setenv("DASHSCOPE_API_KEY", "x")
    assert sm._detect_backend(None) == "bailian"            # bailian wins

    monkeypatch.setenv("LLM_BACKEND", "github")
    assert sm._detect_backend(None) == "github"             # explicit env override

    assert sm._detect_backend("anthropic") == "anthropic"  # explicit arg wins


def test_bailian_is_registered_with_default_model():
    assert sm.DEFAULT_MODELS["bailian"] == "qwen3.7-max"


def test_render_input_includes_pr_body_excerpt(db):
    add_pr(db, 1, body="This PR adds a fused FP8 MoE kernel for H100.")
    payload = sm.collect_window(db, 7)
    rendered = sm.render_input(payload)
    assert "fused FP8 MoE kernel" in rendered


def test_render_input_truncates_long_body(db):
    add_pr(db, 1, body="x" * 5000)
    payload = sm.collect_window(db, 7)
    rendered = sm.render_input(payload, body_chars=100)
    assert "…" in rendered
    assert "x" * 5000 not in rendered


def test_collect_window_excludes_old_prs(db):
    add_pr(db, 1, merged_days_ago=2)
    add_pr(db, 2, merged_days_ago=40)
    payload = sm.collect_window(db, 7)
    numbers = {p["number"] for p in payload["prs"]}
    assert numbers == {1}
