"""Tests for the quantization expander and capability-matrix exclusion."""
from __future__ import annotations

import datetime as dt
import sqlite3

from vllm_insights.capability import (
    render_capability_matrix,
    render_quantization_expander,
)


def _add_inventory(db, kind, name, path, sha=None):
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO source_inventory(kind,name,source_path,source_sha,last_seen_at) "
            "VALUES(?,?,?,?,?)",
            (kind, name, path, sha, dt.datetime.now(dt.timezone.utc).isoformat()),
        )


def test_quant_expander_lists_algorithms_with_descriptions(db):
    _add_inventory(db, "quantization", "fp8",
                   "vllm/model_executor/layers/quantization/fp8.py", "abc1234deadbeef")
    _add_inventory(db, "quantization", "some_new_thing",
                   "vllm/model_executor/layers/quantization/some_new_thing.py")
    html = render_quantization_expander(db)
    assert html.startswith('<details class="quant-expander">')
    assert "(2)" in html                              # count
    assert "<code>fp8</code>" in html
    assert "8-bit float (E4M3)" in html               # known description
    assert "open the source for details" in html      # unknown -> generic fallback
    # source link pinned to the discovered SHA
    assert "blob/abc1234deadbeef/" in html


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
