"""Tests for the quantization expander and capability-matrix exclusion."""
from __future__ import annotations

import datetime as dt
import sqlite3

from vllm_insights.capability import (
    render_attention_expander,
    render_capability_matrix,
    render_quantization_expander,
    render_spec_decode_expander,
)


def _add_inventory(db, kind, name, path, sha=None):
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO source_inventory(kind,name,source_path,source_sha,last_seen_at) "
            "VALUES(?,?,?,?,?)",
            (kind, name, path, sha, dt.datetime.now(dt.timezone.utc).isoformat()),
        )


def test_quant_expander_explains_algorithms(db):
    _add_inventory(db, "quantization", "fp8",
                   "vllm/model_executor/layers/quantization/fp8.py", "abc1234deadbeef")
    _add_inventory(db, "quantization", "some_new_thing",
                   "vllm/model_executor/layers/quantization/some_new_thing.py")
    html = render_quantization_expander(db)
    assert html.startswith('<details class="info-expander">')
    assert "(2)" in html                              # count
    assert "<code>fp8</code>" in html
    # tagline (summary) + principle (expanded body) for a known method
    assert "E4M3" in html                             # tagline
    assert "tensor cores run FP8 matmuls natively" in html  # principle text
    # each algorithm is its own nested expander
    assert '<details class="q-item">' in html
    # unknown method -> generic fallback principle, still rendered
    assert "No write-up yet" in html
    # source link pinned to the discovered SHA
    assert "blob/abc1234deadbeef/" in html


def test_attention_expander_explains_backends(db):
    _add_inventory(db, "attention", "flash_attn",
                   "vllm/v1/attention/backends/flash_attn.py")
    _add_inventory(db, "attention", "flashinfer",
                   "vllm/v1/attention/backends/flashinfer.py")
    html = render_attention_expander(db)
    assert "vLLM attention backends" in html
    assert "<code>flash_attn</code>" in html
    assert "online-softmax" in html             # FlashAttention principle
    assert "paged KV cache" in html             # FlashInfer principle


def test_spec_decode_expander_explains_methods(db):
    _add_inventory(db, "spec_decode", "ngram", "vllm/v1/spec_decode/ngram.py")
    _add_inventory(db, "spec_decode", "eagle", "vllm/v1/spec_decode/eagle.py")
    html = render_spec_decode_expander(db)
    assert "vLLM speculative decoding" in html
    assert "prompt-lookup" in html              # ngram principle
    assert "feature level" in html              # EAGLE principle


def test_expanders_empty_when_not_loaded(db):
    assert render_attention_expander(db) == ""
    assert render_spec_decode_expander(db) == ""


def test_quant_expander_empty_when_not_loaded(db):
    assert render_quantization_expander(db) == ""


def test_capability_matrix_excludes_quantization(db):
    _add_inventory(db, "quantization", "fp8",
                   "vllm/model_executor/layers/quantization/fp8.py")
    _add_inventory(db, "attention", "flash_attn",
                   "vllm/v1/attention/backends/flash_attn.py")
    matrix = render_capability_matrix(db, exclude_kinds={"quantization"})
    assert "Attention backends" in matrix          # other kinds still rendered
    assert "Quantization" not in matrix            # excluded (lives in the expander)
    assert "fp8" not in matrix


def test_capability_matrix_excluded_only_returns_empty(db):
    # quantization is the only data, and it's excluded -> matrix stays silent so
    # the section's own empty_state shows instead of a misleading table.
    _add_inventory(db, "quantization", "fp8",
                   "vllm/model_executor/layers/quantization/fp8.py")
    assert render_capability_matrix(db, exclude_kinds={"quantization"}) == ""
