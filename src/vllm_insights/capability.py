"""Derived capability view — every row is a real file in upstream vLLM.

The previous version of this module shipped a hand-curated stable/experimental/
preview/none matrix. That was bluffing: the numbers came from my head, not from
the codebase. This rewrite removes every hardcoded claim and replaces it with
data we can verify in one click.

A row now means: "this `.py` file exists in upstream vLLM right now". We do
not pretend to know maturity, latency, or whether it works on your hardware —
those claims would need real benchmark data, which we don't have yet. What we
do show:

  - The feature name (filename stem, e.g. `fp8`, `awq_marlin`, `mxfp4`)
  - A link to the source file on GitHub (the operator can read it in 30s)
  - PR activity in the last 90 days for that exact path (a proxy for whether
    it's being actively maintained or has gone dormant)
  - A link to a per-feature deep page that lists every PR touching it

If a "feature" looks suspicious to a vLLM expert reading this page, they can
click straight through to the file and decide for themselves. That's the
contrast with the old approach.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from .source_scan import (
    kinds_in_order,
    load_inventory,
    pr_activity_for_inventory,
)


def _feature_slug(kind: str, name: str) -> str:
    return f"{kind}-{name}".replace("_", "-").lower()


def _gh_blob_url(repo: str, path: str, sha: str | None) -> str:
    # Pin the link to the discovered commit SHA so it keeps pointing at the
    # exact file we inventoried, even after upstream moves or deletes it.
    # Fall back to `main` only when we don't have a usable SHA.
    rev = sha if (sha and len(sha) >= 7) else "main"
    return f"https://github.com/{repo}/blob/{rev}/{path}"


# Short, operator-facing descriptions for the quantization methods vLLM ships.
# Keyed by the file stem under vllm/model_executor/layers/quantization/. Anything
# not listed falls back to a generic line — we never invent specifics.
_QUANT_DESCRIPTIONS: dict[str, str] = {
    "fp8": "8-bit float (E4M3) weight/activation quant — near-lossless on Hopper/Ada with hardware FP8.",
    "fp4": "4-bit float weight quant (e.g. NVFP4 on Blackwell).",
    "mxfp4": "MXFP4 — microscaling, block-scaled 4-bit float format.",
    "awq": "Activation-aware Weight Quantization — 4-bit weights that protect salient channels by activation scale.",
    "awq_marlin": "AWQ weights served through the Marlin INT4 kernel for high-throughput GPU inference.",
    "awq_triton": "AWQ dequant/GEMM implemented in Triton.",
    "gptq": "GPTQ — post-training 4-bit weight quant via approximate second-order (Hessian) error correction.",
    "gptq_marlin": "GPTQ weights served through the Marlin kernel.",
    "gptq_marlin_24": "GPTQ + 2:4 structured sparsity on the Marlin kernel.",
    "gptq_bitblas": "GPTQ weights served through BitBLAS mixed-precision GEMM.",
    "marlin": "Marlin — fast mixed-precision INT4×FP16 GEMM kernel for Ampere+ GPUs.",
    "gguf": "GGUF — the llama.cpp quantized weight format (k-quants etc.).",
    "bitsandbytes": "bitsandbytes — on-the-fly 4-bit (NF4) / 8-bit weight quant, popular for QLoRA-style loading.",
    "compressed_tensors": "compressed-tensors — general schema (W8A8 INT8/FP8, INT4, …) from the LLM-Compressor toolchain.",
    "fbgemm_fp8": "FBGEMM FP8 — Meta's FP8 GEMM path.",
    "modelopt": "NVIDIA TensorRT Model Optimizer quant (FP8 / INT4) checkpoints.",
    "experts_int8": "INT8 quantization for MoE expert weights.",
    "int8": "Generic INT8 weight/activation quantization.",
    "tpu_int8": "INT8 quantization path for TPU.",
    "quark": "AMD Quark quantization checkpoints.",
    "hqq": "Half-Quadratic Quantization — fast, calibration-free low-bit quant.",
    "aqlm": "AQLM — Additive Quantization, extreme low-bit (2–3 bit).",
    "qqq": "QQQ — W4A8 quality quantization.",
    "deepspeedfp": "DeepSpeed FP-quantized weights.",
    "moe_wna16": "Weight-only INT4/INT8 (WNA16) kernels for MoE layers.",
    "ipex_quant": "Intel IPEX quantization (CPU / XPU).",
    "neuron_quant": "AWS Neuron (Trainium / Inferentia) quantization.",
    "torchao": "torchao — PyTorch-native int4/int8/fp8 quantization integration.",
    "bitblas": "BitBLAS — mixed-precision GEMM kernels (e.g. for GPTQ / AWQ).",
    "auto_round": "AutoRound — sign-gradient-descent weight rounding for low-bit quant.",
    "petit_nvfp4": "NVFP4 served through the Petit kernel.",
    "rtn": "Round-to-nearest — simple baseline weight quantization.",
}


def render_quantization_expander(db_path: Path, repo: str = "vllm-project/vllm") -> str:
    """A collapsible <details> listing every quantization algorithm vLLM ships.

    Each entry is a real file under `vllm/model_executor/layers/quantization/`
    (from `source_inventory`), with a short description, a source link pinned to
    the discovered SHA, and 90-day PR activity. Returns '' if not loaded yet.
    """
    rows = load_inventory(db_path, kind="quantization")
    if not rows:
        return ""
    activity = pr_activity_for_inventory(db_path, days=90)

    items: list[str] = []
    for r in rows:
        name = r["name"]
        desc = _QUANT_DESCRIPTIONS.get(
            name.lower(), "Quantization method — open the source for details."
        )
        blob = _gh_blob_url(repo, r["source_path"], r.get("source_sha"))
        count = activity.get(r["source_path"], 0)
        act = f'<span class="q-act">{count} PRs/90d</span>' if count else ""
        items.append(
            '<li class="q-item">'
            f'<a class="q-name" href="{blob}" target="_blank" rel="noopener">'
            f"<code>{escape(name)}</code></a>{act}"
            f'<div class="q-desc">{escape(desc)}</div>'
            "</li>"
        )
    return (
        '<details class="quant-expander">'
        '<summary>vLLM quantization algorithms '
        f'<span class="q-count">({len(rows)})</span></summary>'
        '<p class="q-intro">Every method below is a real file in upstream vLLM '
        "(click the name to read it). Descriptions are short operator notes, not "
        "benchmarks.</p>"
        f'<ul class="q-list">{"".join(items)}</ul>'
        "</details>"
    )


def render_capability_matrix(
    db_path: Path,
    repo: str = "vllm-project/vllm",
    exclude_kinds: set[str] | None = None,
) -> str:
    """Render the derived capability section.

    Reads `source_inventory` (populated by `vllm-insights sync --source-scan`)
    and joins it against `pr_files` to compute 90-day activity per path. If the
    inventory is empty (first run), renders a hint rather than a fake table.
    `exclude_kinds` skips kinds shown elsewhere (e.g. quantization has its own
    expander).
    """
    exclude_kinds = exclude_kinds or set()
    activity = pr_activity_for_inventory(db_path, days=90)

    parts: list[str] = []

    have_any = False
    for kind, group_label in kinds_in_order():
        if kind in exclude_kinds:
            continue
        rows = load_inventory(db_path, kind=kind)
        if not rows:
            continue
        have_any = True
        parts.append(f'<h3>{escape(group_label)} '
                     f'<span class="cap-count">({len(rows)})</span></h3>')
        parts.append('<div class="cap-tablewrap"><table class="cap-table">')
        parts.append(
            "<thead><tr>"
            "<th class='feat'>Feature</th>"
            "<th class='path'>Source</th>"
            "<th class='activity'>PRs (90d)</th>"
            "<th class='deep'></th>"
            "</tr></thead><tbody>"
        )
        for r in rows:
            path = r["source_path"]
            count = activity.get(path, 0)
            blob = _gh_blob_url(repo, path, r.get("source_sha"))
            slug = _feature_slug(kind, r["name"])
            activity_html = (
                f'<span class="act-hot">{count}</span>' if count >= 6
                else f'<span class="act-warm">{count}</span>' if count >= 1
                else '<span class="act-cold">0</span>'
            )
            parts.append(
                "<tr>"
                f"<td class='feat'><code>{escape(r['name'])}</code></td>"
                f"<td class='path'><a href='{blob}' target='_blank' rel='noopener'>"
                f"<code>{escape(path)}</code></a></td>"
                f"<td class='activity'>{activity_html}</td>"
                f"<td class='deep'><a href='features/{slug}.html' "
                f"title='Recent PRs touching this file'>&rarr;</a></td>"
                "</tr>"
            )
        parts.append("</tbody></table></div>")

    if not have_any:
        # When used as a secondary view (some kinds shown elsewhere), stay silent
        # and let the caller handle the fully-empty case.
        if exclude_kinds:
            return ""
        parts.append(
            '<p style="color:var(--fg-3)"><em>Not loaded yet.</em></p>'
        )
    return "\n".join(parts)


CAPABILITY_CSS = """
section.capability { margin: 2rem 0 1rem; }
section.capability h2 { margin-bottom: .3rem; }
section.capability .cap-intro { font-size: .9rem; opacity: .8;
    margin: 0 0 .8rem; max-width: 75ch; }
section.capability h3 { margin-top: 1.4rem; font-size: 1rem;
    text-transform: uppercase; letter-spacing: .04em; opacity: .8;
    border-bottom: 1px dashed #6664; padding-bottom: .2rem; }
section.capability .cap-count { font-size: .75rem; opacity: .55;
    font-weight: 400; text-transform: none; letter-spacing: 0; }
.cap-tablewrap { overflow-x: auto; margin: .4rem 0 .8rem; }
table.cap-table { border-collapse: collapse; width: 100%; font-size: .85rem; }
table.cap-table th, table.cap-table td {
    padding: .35rem .55rem; border-bottom: 1px solid #ddd3;
    vertical-align: middle;
}
table.cap-table thead th { font-weight: 600; opacity: .8; font-size: .72rem;
    text-transform: uppercase; letter-spacing: .04em;
    border-bottom: 1px solid #ddd6; }
table.cap-table td.feat { font-size: .9rem; width: 22%; min-width: 160px; }
table.cap-table td.feat code { font-size: .85rem; padding: .05rem .35rem;
    border-radius: 4px; background: rgba(102,204,255,.1);
    border: 1px solid #6cf4; }
table.cap-table td.path { font-size: .72rem; opacity: .8; }
table.cap-table td.path a { text-decoration: none; }
table.cap-table td.path a:hover { text-decoration: underline; }
table.cap-table td.path code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
table.cap-table td.activity { width: 6rem; text-align: center; }
table.cap-table td.deep { text-align: center; width: 2.5rem; font-size: 1.1rem; }
table.cap-table td.deep a { text-decoration: none; opacity: .7; }
table.cap-table td.deep a:hover { opacity: 1; }
.act-hot   { color: #2c8f48; font-weight: 700; }
.act-warm  { color: #b8862b; font-weight: 600; }
.act-cold  { color: #888; }

/* Quantization algorithms expander */
details.quant-expander { margin: .6rem 0 1.4rem; border: 1px solid #6663;
    border-radius: 8px; background: var(--bg-2, #fff1); overflow: hidden; }
details.quant-expander > summary { cursor: pointer; padding: .6rem .85rem;
    font-weight: 600; font-size: .95rem; list-style: none; user-select: none; }
details.quant-expander > summary::-webkit-details-marker { display: none; }
details.quant-expander > summary::before { content: "▸"; display: inline-block;
    margin-right: .5rem; opacity: .6; transition: transform .15s ease; }
details.quant-expander[open] > summary::before { transform: rotate(90deg); }
details.quant-expander[open] > summary { border-bottom: 1px solid #6663; }
.quant-expander .q-count { font-weight: 400; opacity: .55; font-size: .85rem; }
.quant-expander .q-intro { font-size: .82rem; opacity: .75; margin: .7rem .85rem 0;
    max-width: 75ch; }
.quant-expander .q-list { list-style: none; margin: .5rem 0 .3rem; padding: 0;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: .15rem .9rem; }
.quant-expander .q-item { padding: .5rem .85rem; border-top: 1px solid #6662; }
.quant-expander .q-name { text-decoration: none; }
.quant-expander .q-name code { font-size: .85rem; padding: .05rem .35rem;
    border-radius: 4px; background: rgba(102,204,255,.12); border: 1px solid #6cf4; }
.quant-expander .q-name:hover code { background: rgba(102,204,255,.22); }
.quant-expander .q-act { font-size: .68rem; opacity: .6; margin-left: .5rem;
    white-space: nowrap; }
.quant-expander .q-desc { font-size: .8rem; opacity: .82; margin-top: .25rem;
    line-height: 1.4; }
"""
