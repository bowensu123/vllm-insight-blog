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


# For each quantization method vLLM ships: a short tagline plus a "how it works"
# principle (2-4 sentences). Keyed by the file stem under
# vllm/model_executor/layers/quantization/. Anything not listed falls back to a
# generic note — we never invent specifics.
_QUANT_INFO: dict[str, tuple[str, str]] = {
    "fp8": (
        "8-bit float (E4M3) weights/activations — near-lossless, hardware-accelerated.",
        "Stores values in 8-bit floating point, almost always the E4M3 layout (1 sign, "
        "4 exponent, 3 mantissa bits). Keeping an exponent gives a wide dynamic range from "
        "just a per-tensor or per-channel scale, so accuracy loss is small. Hopper/Ada/"
        "Blackwell tensor cores run FP8 matmuls natively, so W8A8 FP8 is both smaller and "
        "faster, not merely smaller.",
    ),
    "fp4": (
        "4-bit float weights (e.g. NVFP4) for Blackwell.",
        "Like FP8 but 4 bits — typically E2M1 (1 sign, 2 exponent, 1 mantissa) with a shared "
        "micro-scale per small block, and sometimes a second global scale. Blackwell tensor "
        "cores execute FP4 directly for large memory/throughput wins; the block scales keep "
        "the 4-bit elements accurate.",
    ),
    "mxfp4": (
        "Microscaling block-scaled 4-bit float (OCP MX).",
        "Groups weights into small blocks (commonly 32 elements). Each block shares one 8-bit "
        "power-of-two scale (E8M0) and each element is a 4-bit float (E2M1). The shared block "
        "scale captures local magnitude, so the 4-bit elements stay accurate without per-value "
        "metadata.",
    ),
    "awq": (
        "Activation-aware 4-bit weight quant that protects salient channels.",
        "Observes that a small fraction of weight channels — those multiplied by large-"
        "magnitude activations — dominate the output error. Using a calibration set, AWQ "
        "searches a per-channel scaling that shrinks those salient channels before rounding "
        "weights to 4-bit, pushing error onto unimportant channels. Weight-only (W4A16), no "
        "backprop.",
    ),
    "awq_marlin": (
        "AWQ 4-bit weights run through the fast Marlin kernel.",
        "AWQ decides the quantized values; Marlin provides a high-throughput INT4×FP16 GEMM on "
        "Ampere+ GPUs that dequantizes on the fly. The result is small (4-bit) and fast at "
        "serving time.",
    ),
    "awq_triton": (
        "AWQ dequant/GEMM written in Triton for portability.",
        "Same AWQ weights as awq_marlin, but the dequantization and matmul are implemented in "
        "Triton so they run on GPUs/configs where the hand-tuned Marlin kernel isn't available.",
    ),
    "gptq": (
        "Post-training low-bit weight quant with second-order error correction.",
        "Quantizes weights one column at a time and, after each step, updates the remaining "
        "un-quantized weights to compensate for the error just introduced, guided by a layer-"
        "wise second-order (Hessian) objective estimated from calibration activations. This "
        "greedy error correction keeps 3-4 bit weights accurate. Weight-only.",
    ),
    "gptq_marlin": (
        "GPTQ weights run through the fast Marlin kernel.",
        "GPTQ decides the quantized values; the Marlin INT4×FP16 kernel serves them at high "
        "throughput on Ampere+ GPUs with on-the-fly dequant.",
    ),
    "gptq_marlin_24": (
        "GPTQ + 2:4 structured sparsity on a sparse Marlin kernel.",
        "Adds 2:4 structured sparsity (two of every four weights forced to zero) on top of "
        "GPTQ's 4-bit weights, and serves them through a sparse Marlin kernel — stacking "
        "sparsity on quantization for extra speedup on GPUs that support 2:4.",
    ),
    "gptq_bitblas": (
        "GPTQ weights served through BitBLAS GEMM.",
        "GPTQ-quantized weights executed by BitBLAS-generated mixed-precision GEMM kernels, an "
        "alternative kernel backend to Marlin.",
    ),
    "marlin": (
        "A fast mixed-precision INT4×FP16 GEMM kernel (not a scheme).",
        "Marlin isn't a quantization method but a kernel: a highly optimized GEMM that "
        "multiplies INT4 weights by FP16 activations at near-roofline speed on Ampere+ GPUs, "
        "dequantizing inline. AWQ/GPTQ weights are run through it.",
    ),
    "gguf": (
        "Loads llama.cpp GGUF quantized weights (k-quants).",
        "Consumes weights already quantized in llama.cpp's GGUF format — most often the "
        "'k-quant' block schemes (Q4_K, Q5_K, …) that store small blocks with one or two shared "
        "scales/minimums. vLLM dequantizes them at runtime; handy for reusing community GGUF "
        "checkpoints.",
    ),
    "bitsandbytes": (
        "On-the-fly 4-bit (NF4) / 8-bit weight quant, QLoRA-style.",
        "Quantizes weights at load time. The 8-bit path uses vector-wise INT8 with separate "
        "handling of activation outliers (LLM.int8()); the 4-bit path uses NF4, a normalized-"
        "float code optimized for normally-distributed weights, plus double quantization of the "
        "scales to save more memory. Popular for low-memory loading and QLoRA.",
    ),
    "compressed_tensors": (
        "A container format describing per-tensor quant recipes.",
        "Not a single algorithm but a schema (from the LLM-Compressor / neuralmagic toolchain) "
        "that records how each tensor is quantized — W8A8 INT8 or FP8, INT4 weight-only, "
        "structured sparsity, and so on. vLLM reads the recipe and dispatches the matching "
        "kernel per layer.",
    ),
    "fbgemm_fp8": (
        "FP8 weights via Meta's FBGEMM GEMM path.",
        "Serves FP8-quantized weights through Meta's FBGEMM FP8 kernels, typically with per-"
        "channel weight scales and dynamic per-token activation scales.",
    ),
    "modelopt": (
        "Loads NVIDIA TensorRT Model Optimizer checkpoints.",
        "Consumes models quantized by NVIDIA TensorRT Model Optimizer (FP8, or INT4 AWQ), "
        "carrying TRT-MO's scales and recipe into vLLM's kernels.",
    ),
    "experts_int8": (
        "INT8 weights for Mixture-of-Experts expert matrices.",
        "Applies INT8 weight quantization specifically to the MoE expert matrices — where most "
        "MoE parameters live — to cut memory while leaving the rest of the model unquantized.",
    ),
    "int8": (
        "Generic INT8 weight (and optionally activation) quant.",
        "Maps weights to 8-bit integers with a scale per tensor or channel; with W8A8 the "
        "activations are quantized too. Simple and very hardware-friendly — larger than 4-bit "
        "but typically near-lossless.",
    ),
    "tpu_int8": (
        "INT8 quantization path specialized for TPU.",
        "An INT8 weight-quantization path tuned for TPU hardware and the XLA compiler.",
    ),
    "quark": (
        "Loads AMD Quark quantized checkpoints.",
        "Consumes models produced by AMD's Quark quantization toolkit (FP8 / INT4 / …), "
        "targeting ROCm and the MI-series GPUs.",
    ),
    "hqq": (
        "Calibration-free low-bit quant via a half-quadratic solver.",
        "Half-Quadratic Quantization fixes a per-group scale (from the weight range) and then "
        "optimizes the zero-point by minimizing a robust, outlier-tolerant error with a half-"
        "quadratic solver — no calibration data needed. It's fast to apply and holds up at low "
        "bit-widths.",
    ),
    "aqlm": (
        "Extreme 2-3 bit weights via additive vector quantization.",
        "Additive Quantization of Language Models represents each group of weights as a sum of "
        "vectors picked from small learned codebooks (vector quantization). This reaches 2-3 "
        "bits per weight with surprisingly little accuracy loss, at the cost of a heavier decode "
        "step.",
    ),
    "qqq": (
        "W4A8: 4-bit weights, 8-bit activations, with accuracy-preserving smoothing.",
        "Pairs 4-bit weights with 8-bit activations (W4A8). To recover the accuracy normally "
        "lost when activations are quantized, QQQ uses adaptive smoothing of the activation "
        "channels that carry large outliers plus Hessian-based weight compensation, without "
        "retraining. It then ships specialized per-channel and per-group W4A8 GEMM kernels that "
        "deliver the speedup over FP16.",
    ),
    "deepspeedfp": (
        "Loads DeepSpeed FP-quantized weights.",
        "Consumes DeepSpeed's floating-point-quantized weight format (e.g. FP6/FP8) and its "
        "matching kernels.",
    ),
    "moe_wna16": (
        "Weight-only N-bit × FP16 kernels for MoE layers.",
        "WNA16 = weight N-bit, activation 16-bit. Specialized INT4/INT8 weight-only kernels for "
        "Mixture-of-Experts layers, multiplying low-bit expert weights against FP16 activations.",
    ),
    "ipex_quant": (
        "Intel IPEX quantization for CPU / XPU.",
        "Routes quantization through the Intel Extension for PyTorch back-end for CPU and Intel "
        "GPU (XPU) execution.",
    ),
    "neuron_quant": (
        "Quantization for AWS Neuron (Trainium / Inferentia).",
        "Provides the quantization path for AWS Neuron accelerators (Trainium / Inferentia).",
    ),
    "torchao": (
        "Bridges PyTorch-native torchao quantization.",
        "Lets vLLM consume models quantized with PyTorch's `torchao` library — int4/int8 weight-"
        "only, fp8, and related schemes — reusing torchao's kernels.",
    ),
    "bitblas": (
        "A mixed-precision GEMM kernel backend.",
        "BitBLAS (from Microsoft) generates fast mixed-precision GEMMs; vLLM uses it as a kernel "
        "backend to execute GPTQ/AWQ-style low-bit weights.",
    ),
    "auto_round": (
        "Learns better low-bit rounding via sign-gradient descent.",
        "AutoRound (Intel) treats each weight's round-up-or-down choice as a learnable parameter "
        "and tunes it with a few hundred steps of sign-gradient descent against calibration "
        "data, beating naive nearest rounding at low bit-widths (e.g. 4-bit).",
    ),
    "petit_nvfp4": (
        "NVFP4 weights served on AMD GPUs via the Petit kernel.",
        "Runs NVFP4 (4-bit float) weights on AMD Instinct GPUs (CDNA2/CDNA3, e.g. MI250/MI300) "
        "that lack native FP4 hardware. The Petit kernel does a mixed-precision GEMM, "
        "dequantizing the 4-bit weights to BF16/FP16 on the fly and multiplying against "
        "BF16/FP16 activations — letting existing AMD GPUs serve FP4 checkpoints without native "
        "FP4 tensor cores.",
    ),
    "rtn": (
        "Round-to-nearest — the simple baseline.",
        "The simplest scheme: divide by a scale and round each weight to the nearest "
        "quantization level, with no calibration or error correction. Fast but the least "
        "accurate — mostly a reference point for the smarter methods.",
    ),
}

_GENERIC_FALLBACK = (
    "",
    "No write-up yet — open the source to see how it works.",
)


# Attention backends: files under vllm/v1/attention/backends/ (or legacy
# vllm/attention/backends/). Keyed by file stem.
_ATTENTION_INFO: dict[str, tuple[str, str]] = {
    "flash_attn": (
        "FlashAttention — exact attention without the full score matrix.",
        "Computes attention exactly but never materializes the N×N score matrix. It tiles "
        "Q/K/V into blocks that fit in on-chip SRAM and uses the online-softmax trick to "
        "accumulate the result block by block, so memory is linear in sequence length and the "
        "kernel is IO-optimal. vLLM uses FlashAttention-2/3 kernels on supported NVIDIA GPUs.",
    ),
    "flashinfer": (
        "FlashInfer — attention kernels tuned for LLM serving.",
        "A library of attention kernels built for inference: paged KV cache, ragged/variable-"
        "length batches, and both prefill and decode. Often the fastest path for an FP8 KV cache "
        "and for grouped-query / MLA attention on recent NVIDIA GPUs.",
    ),
    "triton_attn": (
        "FlashAttention-style kernel written in Triton.",
        "A FlashAttention-style fused kernel implemented in Triton, so it runs across GPU "
        "architectures/vendors where a hand-tuned CUDA kernel isn't available — trading some "
        "peak speed for portability.",
    ),
    "xformers": (
        "Memory-efficient attention via Meta's xFormers.",
        "Uses xFormers' memory-efficient (FlashAttention-style) fused attention. A broadly "
        "compatible backend for GPUs and cases the native FlashAttention kernels don't cover.",
    ),
    "torch_sdpa": (
        "PyTorch scaled_dot_product_attention.",
        "Routes attention through PyTorch's built-in scaled_dot_product_attention, which itself "
        "dispatches to the best available implementation (FlashAttention / memory-efficient / "
        "math). Used on CPU and as a portable fallback.",
    ),
    "flex_attention": (
        "PyTorch FlexAttention — compiled custom masks/bias.",
        "Built on PyTorch FlexAttention, which compiles a user-defined score-modification "
        "function (masking, bias, ALiBi, sliding window, …) into a single fused attention "
        "kernel — flexible attention patterns without hand-writing CUDA.",
    ),
    "rocm_flash_attn": (
        "FlashAttention for AMD GPUs (ROCm).",
        "The FlashAttention path for AMD Instinct GPUs via ROCm kernels.",
    ),
    "rocm_aiter_fa": (
        "AMD AITER FlashAttention (ROCm).",
        "FlashAttention on AMD GPUs through the AITER kernel library — the tuned MI-series path.",
    ),
    "pallas": (
        "TPU attention written in Pallas.",
        "Attention implemented in Pallas, the JAX/XLA kernel language, for TPU execution.",
    ),
    "cpu_attn": (
        "CPU attention backend.",
        "The attention implementation used on CPU back-ends.",
    ),
    "tree_attn": (
        "Tree attention for speculative verification.",
        "Verifies multiple speculative token branches at once by laying them out as a tree and "
        "applying a custom attention mask, so each candidate only attends to its own ancestors — "
        "used to check several drafted continuations in a single pass.",
    ),
    "flashmla": (
        "FlashAttention kernels for DeepSeek MLA.",
        "Kernels specialized for Multi-head Latent Attention (MLA), which compresses the KV "
        "cache into a low-rank latent; serves DeepSeek-style low-rank-KV attention efficiently "
        "on supported NVIDIA GPUs.",
    ),
    "mla": (
        "Multi-head Latent Attention (DeepSeek).",
        "MLA stores the per-token KV as a small low-rank latent vector instead of full K and V, "
        "shrinking the KV cache; the attention math is refactored to operate on that latent. "
        "These backends implement the MLA-specific kernels (e.g. for DeepSeek-V2/V3).",
    ),
}


# Speculative-decoding methods: files under vllm/v1/spec_decode/ (or legacy
# vllm/spec_decode/). Keyed by file stem.
_SPEC_DECODE_INFO: dict[str, tuple[str, str]] = {
    "ngram": (
        "N-gram / prompt-lookup speculation — no draft model.",
        "Proposes the next few tokens by matching the recent context against earlier text in the "
        "same prompt/output (a prompt-lookup table) and copying what followed last time. The "
        "target model verifies the guesses in one pass. Free to run and very effective when "
        "output echoes the input (RAG, code, summarization).",
    ),
    "eagle": (
        "EAGLE — feature-level draft head.",
        "A small draft head that autoregresses at the hidden-feature level rather than over "
        "tokens, predicting the target model's next features to propose several tokens cheaply; "
        "the target then verifies them in a single forward pass. (EAGLE-2/3 add dynamic draft "
        "trees and feature fusion.)",
    ),
    "eagle3": (
        "EAGLE-3 — feature-fusion input, direct token drafting.",
        "An EAGLE variant that fuses low-, mid-, and high-level features from the target model "
        "as the drafter's input and (unlike EAGLE/EAGLE-2) drafts tokens directly rather than "
        "predicting features, which raises the acceptance rate of the proposed tokens.",
    ),
    "medusa": (
        "Medusa — extra decoding heads + tree verification.",
        "Adds several lightweight decoding heads on top of the frozen base model, each predicting "
        "a token a few positions ahead. Their candidate combinations are assembled into a tree "
        "and verified together in one forward pass via tree attention.",
    ),
    "mlp_speculator": (
        "MLP speculator — predicts several future tokens.",
        "A small MLP (IBM's speculator) that, from the current hidden state, predicts a handful "
        "of future tokens to draft candidates for the target model to verify.",
    ),
}


_EXPANDER_INTRO = (
    "Every item below is a real file in upstream vLLM. Click one to read how it works; "
    "click &ldquo;read the source&rdquo; to open the file. Notes are short explanations, "
    "not benchmarks."
)


def _render_info_expander(
    db_path: Path,
    *,
    kind: str,
    title: str,
    info_map: dict[str, tuple[str, str]],
    repo: str,
) -> str:
    """Shared renderer: a collapsible <details> whose items each expand to a
    'how it works' explanation. Returns '' if the inventory for `kind` is empty.
    """
    rows = load_inventory(db_path, kind=kind)
    if not rows:
        return ""
    activity = pr_activity_for_inventory(db_path, days=90)

    items: list[str] = []
    for r in rows:
        name = r["name"]
        tagline, principle = info_map.get(name.lower(), _GENERIC_FALLBACK)
        blob = _gh_blob_url(repo, r["source_path"], r.get("source_sha"))
        count = activity.get(r["source_path"], 0)
        act = f'<span class="q-act">{count} PRs/90d</span>' if count else ""
        tag_html = f'<span class="q-tag">{escape(tagline)}</span>' if tagline else ""
        items.append(
            '<details class="q-item">'
            f"<summary><code>{escape(name)}</code> {tag_html}{act}</summary>"
            f'<p class="q-desc">{escape(principle)} '
            f'<a class="q-src" href="{blob}" target="_blank" rel="noopener">'
            "read the source &rarr;</a></p>"
            "</details>"
        )
    return (
        '<details class="info-expander">'
        f"<summary>{escape(title)} "
        f'<span class="q-count">({len(rows)})</span></summary>'
        f'<p class="q-intro">{_EXPANDER_INTRO}</p>'
        f'<div class="q-list">{"".join(items)}</div>'
        "</details>"
    )


def render_quantization_expander(db_path: Path, repo: str = "vllm-project/vllm") -> str:
    """Collapsible explainer for the quantization algorithms vLLM ships."""
    return _render_info_expander(
        db_path, kind="quantization", title="vLLM quantization algorithms",
        info_map=_QUANT_INFO, repo=repo,
    )


def render_attention_expander(db_path: Path, repo: str = "vllm-project/vllm") -> str:
    """Collapsible explainer for the attention backends vLLM ships."""
    return _render_info_expander(
        db_path, kind="attention", title="vLLM attention backends",
        info_map=_ATTENTION_INFO, repo=repo,
    )


def render_spec_decode_expander(db_path: Path, repo: str = "vllm-project/vllm") -> str:
    """Collapsible explainer for the speculative-decoding methods vLLM ships."""
    return _render_info_expander(
        db_path, kind="spec_decode", title="vLLM speculative decoding",
        info_map=_SPEC_DECODE_INFO, repo=repo,
    )


_PARAM_DOCS_URL = "https://docs.vllm.ai/en/latest/serving/engine_args.html"

# The engine/serving flags operators most often tune, grouped. Static curated
# reference (not derived from source_inventory) — (flag, tagline, explanation).
_PARAM_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Model & precision", [
        ("--model",
         "Which model to serve.",
         "A Hugging Face repo id (e.g. meta-llama/Llama-3.1-8B-Instruct) or a local path. "
         "Almost everything else — architecture, tokenizer, default context length — is loaded "
         "or derived from it."),
        ("--dtype",
         "Compute/weight precision.",
         "auto (default) picks the model's native precision — usually bfloat16 for modern "
         "models — or you can force float16/bfloat16/float32. Prefer bfloat16 on Ampere+ for "
         "numerical stability; float16 can overflow on some models."),
        ("--max-model-len",
         "Max context length (prompt + output).",
         "The longest total sequence the server accepts. Defaults to the model's trained "
         "maximum. Lowering it shrinks the KV-cache footprint so you fit more concurrent "
         "requests; raising it beyond the training length needs RoPE scaling and can hurt "
         "quality."),
        ("--trust-remote-code",
         "Allow the repo's custom model code.",
         "Executes the model repo's own modeling_*.py. Required for architectures not yet in "
         "transformers/vLLM, but it runs third-party Python — only enable for repos you trust."),
        ("--tokenizer-mode",
         "Which tokenizer implementation.",
         "auto (default) picks the fast HF tokenizer; other values include slow (force the "
         "pure-Python HF tokenizer), mistral (the mistral_common tokenizer, needed for some "
         "Mistral models), custom, and model-specific modes — the set isn't exhaustive."),
        ("--seed",
         "Fix the RNG seed.",
         "Makes sampling reproducible across runs for a given request."),
    ]),
    ("Parallelism & distributed", [
        ("--tensor-parallel-size",
         "TP — shard each layer across N GPUs.",
         "Splits every layer's weights and attention heads across N GPUs that all work on each "
         "token together. The main way to fit a model too large for one GPU and to speed up "
         "each request; it wants fast intra-node links (NVLink). Set it to how many GPUs one "
         "replica should span."),
        ("--pipeline-parallel-size",
         "PP — split layers into sequential stages.",
         "Divides the model's layers into N stages on different GPUs/nodes; tokens flow stage "
         "to stage. Tolerates slower inter-node links than TP, so it's how you scale across "
         "nodes or fit very large models — at the cost of some pipeline latency."),
        ("--data-parallel-size",
         "DP — replicate the model for throughput.",
         "Runs N independent replicas to multiply throughput; for MoE models it also underpins "
         "expert parallelism across replicas. Total GPUs used = DP × TP × PP."),
        ("--enable-expert-parallel",
         "EP — distribute MoE experts.",
         "For Mixture-of-Experts models, spread the experts across GPUs (expert parallelism) "
         "instead of replicating them, cutting per-GPU memory for the large expert weights."),
        ("--distributed-executor-backend",
         "How workers are launched.",
         "mp (multiprocessing, single node) or ray (multi-node clusters). Auto-selected; force "
         "ray when a replica spans multiple nodes."),
    ]),
    ("GPU memory & KV cache", [
        ("--gpu-memory-utilization",
         "Fraction of GPU memory vLLM may use (default 0.92).",
         "After the weights load, the remaining share of each GPU's memory becomes KV cache, so "
         "higher means more concurrent tokens/requests — but too high risks OOM from activation "
         "spikes. Lower it if you share the GPU with other processes."),
        ("--kv-cache-dtype",
         "Quantize the KV cache.",
         "auto (= model dtype) or fp8. fp8 roughly halves KV-cache memory, so you fit about 2× "
         "the tokens / concurrency for a small accuracy cost, given hardware/kernel support."),
        ("--cpu-offload-gb",
         "Offload weights to CPU RAM.",
         "Keeps this many GiB of model weights in CPU memory and streams them in as needed — "
         "lets a model that doesn't fit in VRAM run, at a large speed penalty."),
        ("--block-size",
         "KV-cache page size in tokens.",
         "The KV cache is paged into blocks of this many tokens. Rarely changed; it interacts "
         "with prefix caching and the attention kernel."),
    ]),
    ("Batching & scheduling", [
        ("--max-num-seqs",
         "Max concurrent sequences per batch.",
         "The concurrency cap — how many requests can be in a decode batch at once. Higher "
         "raises throughput until you run out of KV cache or compute."),
        ("--max-num-batched-tokens",
         "Token budget per engine step.",
         "How many tokens one iteration may process. Larger values push prefill throughput "
         "(more prompt tokens per step) but can raise inter-token latency for decodes; tuned "
         "together with chunked prefill."),
        ("--enable-chunked-prefill",
         "Interleave long prefills with decodes.",
         "Splits a long prompt's prefill into token-budget-sized chunks and interleaves them "
         "with ongoing decodes, so one huge prompt doesn't stall every other request. Improves "
         "latency fairness under mixed load (on by default for many V1 configs)."),
        ("--enable-prefix-caching",
         "Reuse KV for shared prompt prefixes.",
         "Caches the KV of common prompt prefixes (system prompts, few-shot examples, shared "
         "document context) and reuses it across requests, skipping recomputation. A big win "
         "for RAG and agent workloads that repeat prefixes (on by default in vLLM V1; disable "
         "with --no-enable-prefix-caching)."),
        ("--scheduling-policy",
         "Order of waiting requests.",
         "fcfs (first-come-first-served, default) or priority, which honors a per-request "
         "priority when choosing what to run next."),
    ]),
    ("Performance & features", [
        ("--quantization",
         "Weight-quantization method.",
         "Load the model with a quantization scheme (awq, gptq, fp8, compressed-tensors, …); "
         "often auto-detected from the checkpoint. Shrinks the weights and can speed up serving "
         "— see the quantization expander above for how each works."),
        ("--enforce-eager",
         "Disable CUDA graphs (eager mode).",
         "Turns off CUDA graph capture. Saves some memory and speeds startup, but steady-state "
         "decode is slower — mostly for debugging or memory-tight setups. Off by default."),
        ("--speculative-config",
         "Enable speculative decoding.",
         "Configures a draft method (n-gram, EAGLE, a small draft model, …) that proposes "
         "several tokens per step which the target model verifies in one pass, cutting latency "
         "when the acceptance rate is high — see the speculative-decoding expander above."),
        ("--compilation-config",
         "torch.compile / fusion level.",
         "Controls how aggressively vLLM compiles and fuses the model (e.g. via torch.compile). "
         "Higher levels can speed up steady-state throughput at the cost of longer startup."),
    ]),
    ("Serving & API", [
        ("--served-model-name",
         "The name clients pass as `model`.",
         "The alias clients send in the OpenAI-style API's `model` field (defaults to the "
         "--model path). Set a short, friendly name."),
        ("--api-key",
         "Require an API key.",
         "Demand this key as a Bearer token on every request to the server."),
        ("--enable-auto-tool-choice / --tool-call-parser",
         "OpenAI-style tool calling.",
         "Turn on function/tool calling and select the parser matching the model's tool-call "
         "format (e.g. hermes, llama3_json, mistral)."),
        ("--chat-template",
         "Override the chat template.",
         "Supply a Jinja chat template to format messages when the model's built-in one is "
         "missing or wrong."),
        ("--enable-lora / --max-lora-rank / --max-loras",
         "Serve LoRA adapters.",
         "Serve LoRA adapters on top of the base model: enable LoRA, cap the adapter rank, and "
         "set how many adapters can be active in a batch at once."),
    ]),
]


def render_params_expander(repo: str = "vllm-project/vllm") -> str:
    """A collapsible <details> explaining the vLLM engine/serving flags operators
    most often tune. Static curated reference (grouped), always rendered."""
    total = sum(len(params) for _, params in _PARAM_GROUPS)
    parts = [
        '<details class="info-expander">',
        f"<summary>vLLM key parameters "
        f'<span class="q-count">({total})</span></summary>',
        '<p class="q-intro">The engine / serving flags you most often tune when running '
        "<code>vllm serve</code>. Click one to read what it does and when to change it. "
        f'Full reference in the <a href="{_PARAM_DOCS_URL}" target="_blank" rel="noopener">'
        "vLLM docs</a>.</p>",
        '<div class="q-list">',
    ]
    for group_label, params in _PARAM_GROUPS:
        parts.append(f'<h4 class="q-group">{escape(group_label)}</h4>')
        for name, tagline, principle in params:
            parts.append(
                '<details class="q-item">'
                f"<summary><code>{escape(name)}</code> "
                f'<span class="q-tag">{escape(tagline)}</span></summary>'
                f'<p class="q-desc">{escape(principle)}</p>'
                "</details>"
            )
    parts.append("</div></details>")
    return "".join(parts)


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
details.info-expander { margin: .6rem 0 1.4rem; border: 1px solid #6663;
    border-radius: 8px; background: var(--bg-2, #fff1); overflow: hidden; }
details.info-expander > summary { cursor: pointer; padding: .6rem .85rem;
    font-weight: 600; font-size: .95rem; list-style: none; user-select: none; }
details.info-expander > summary::-webkit-details-marker { display: none; }
details.info-expander > summary::before { content: "▸"; display: inline-block;
    margin-right: .5rem; opacity: .6; transition: transform .15s ease; }
details.info-expander[open] > summary::before { transform: rotate(90deg); }
details.info-expander[open] > summary { border-bottom: 1px solid #6663; }
.info-expander .q-count { font-weight: 400; opacity: .55; font-size: .85rem; }
.info-expander .q-intro { font-size: .82rem; opacity: .75; margin: .7rem .85rem .2rem;
    max-width: 80ch; }
.info-expander .q-list { margin: 0 0 .3rem; padding: 0; }
/* Each algorithm is its own nested expander. */
details.q-item { border-top: 1px solid #6662; }
details.q-item > summary { cursor: pointer; padding: .5rem .85rem; list-style: none;
    user-select: none; display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap; }
details.q-item > summary::-webkit-details-marker { display: none; }
details.q-item > summary::before { content: "+"; opacity: .5; font-weight: 700;
    width: 1ch; display: inline-block; }
details.q-item[open] > summary::before { content: "\2212"; }  /* minus */
details.q-item > summary code { font-size: .85rem; padding: .05rem .35rem;
    border-radius: 4px; background: rgba(102,204,255,.12); border: 1px solid #6cf4; }
.info-expander .q-tag { font-size: .82rem; opacity: .82; }
.info-expander .q-act { font-size: .68rem; opacity: .55; margin-left: auto;
    white-space: nowrap; }
.info-expander .q-desc { font-size: .82rem; opacity: .9; line-height: 1.5;
    margin: 0 .85rem .7rem 2.35rem; max-width: 80ch; }
.info-expander .q-src { white-space: nowrap; }
.info-expander .q-group { font-size: .7rem; text-transform: uppercase;
    letter-spacing: .06em; opacity: .6; font-weight: 700; margin: .9rem .85rem .1rem; }
.info-expander .q-group:first-child { margin-top: .4rem; }
"""
