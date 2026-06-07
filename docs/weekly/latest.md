# vLLM weekly digest — 2026-06-07 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on stabilizing disaggregated serving, optimizing speculative decoding, and expanding hardware support for AMD and Intel platforms. Key performance wins include splitting mixed prefill-decode batches for Qwen3.5's GDN attention and reusing sparse MLA indices for DeepSeek-V4 multi-token prediction. Additionally, the release resolves critical async KV load deadlocks and introduces zentorch-accelerated quantized inference for AMD Zen CPUs.

## Deep dives

### DeepSeek-V4 MTP Index Sharing for Sparse MLA
DeepSeek's Sparse Multi-head Latent Attention (MLA) uses a sparse indexer to select top-k KV blocks, which is computationally expensive to run on every layer. In Multi-Token Prediction (MTP) speculative decoding, running this indexer for every speculative step adds significant overhead. PR [#44420](https://github.com/vllm-project/vllm/pull/44420) implements "IndexCache" to reuse the top-k token selections across MTP steps, skipping the sparse attention indexer when `skip_topk` is enabled. This avoids redundant sparse indexing, reducing the compute cost of speculative decoding. It directly benefits users running DeepSeek-V3 or V4 with MTP enabled, who should see improved speculative acceptance rates and lower latency without changing their configuration.

### Splitting Mixed Batches for Qwen3.5 GDN Attention
Qwen3.5 uses Gated Delta Net (GDN) attention, which relies on a chunked recurrent kernel for prefills and a fused kernel for decodes. Previously, when prefill and decode tokens were mixed in the same batch, the engine routed all tokens through the chunked kernel, padding each decode to a full chunk (e.g., 64 tokens) and wasting massive amounts of compute. PR [#44700](https://github.com/vllm-project/vllm/pull/44700) splits these mixed batches, routing decode tokens to the efficient recurrent decode kernel while keeping prefills on the chunked kernel. This eliminates the padding overhead, drastically improving decode latency in high-concurrency environments. Anyone serving Qwen3.5 or similar GDN-based models will see immediate throughput and latency improvements under mixed workloads.

### Resolving Async KV Load Deadlocks
In disaggregated serving or tiered offloading, vLLM asynchronously loads KV blocks from external connectors. If the engine aggressively loads blocks for incoming requests, it can consume all available GPU memory, starving in-flight chunked prefills and causing a system-wide deadlock. PR [#44560](https://github.com/vllm-project/vllm/pull/44560) introduces a throttling mechanism that prevents async KV loads from occupying blocks required by in-flight chunked prefills or other active requests. This ensures stable memory management and prevents hangs under heavy load. Users running vLLM V1 with KV connectors like Mooncake, Nixl, or LMCache for prefill-decode disaggregation should upgrade to avoid these deadlocks.

## Kernels & attention
- Fused `concat_mla_q` in ROCm sparse-MLA backend ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — eliminates 61 tensor allocations and kernel launches per decode step for DeepSeek-V3.2 on MI355X.
- Added native HIP W4A16 MoE kernel for AMD RDNA3 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — replaces the Triton path with a fused expert routing and GEMM kernel using `v_dot2` primitives.
- Integrated TRTLLM gen attention kernel for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) — provides a highly optimized attention backend for NVIDIA GPUs.
- Fixed a TileLang software pipeliner miscompile in the mHC fused-RMSNorm kernel ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — resolves silent NaNs for DeepSeek-V4 hidden sizes other than 4096.

## Quantization
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates quantized inference on EPYC/Ryzen with transparent fallback for other hardware.
- Added XPU support for compressed-tensors W4A4 MXFP4 MoE ([#44540](https://github.com/vllm-project/vllm/pull/44540)) — enables 4-bit inference for models like Qwen3-235B on Intel GPUs.
- Supported compressed-tensors WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported asymmetric and mixed-precision quantization schemes.
- Guarded `fused_add_rms_norm` dtype mismatches in RMSNorm + quant fusion ([#44694](https://github.com/vllm-project/vllm/pull/44694)) — fixes a crash in Qwen3.5-FP8 block quantization caused by bf16 inputs and fp32 weights.

## Parallelism & scheduling
- Fixed a deterministic hang in multi-node Ray data-parallel serving ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — excludes the Ray DP backend from deferred port allocation to prevent startup deadlocks.
- Fixed per-group block size and hash indexing in MooncakeStoreConnector ([#44103](https://github.com/vllm-project/vllm/pull/44103)) — ensures correct KV event tracking in hybrid multi-group configurations.
- Allowed DP Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — gives operators finer control over multi-node resource allocation.
- Added objectstore as a secondary tier to multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — enables cheaper, higher-capacity storage for evicted KV blocks.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — an open-weights Mixture-of-Experts code-generation model.
- Enabled Cohere North-Mini-Code ([#44707](https://github.com/vllm-project/vllm/pull/44707)) — adds native support with tool-calling and reasoning parsers for Cohere's latest mini model.
- Added Gemma4 Unified (encoder-free) support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — integrates the new multimodal architecture directly into vLLM V1.
- Fixed HyperCLOVAX loading after upstream HuggingFace repo changes ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — registers the model type natively to bypass stale `auto_map` remote code.

## Hardware
- Enabled transparent sleep mode for Intel XPU ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — introduces an XPU memory allocator with sleep/wake interfaces for CUDA parity.
- Supported CPU KV offloading and tiering on XPU ([#36423](https://github.com/vllm-project/vllm/pull/36423)) — uses cross-layer KV layout kernels to enable efficient memory tiering on Intel GPUs.
- Added XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — expands FP8 quantization support to Intel XPU platforms.
- Fixed `ApplyRotaryEmb` HIP grid overflow for long sequences ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — falls back to native implementation when sequence length exceeds the 65,535 per-dim limit on ROCm.

## API & serving
- Added a Rust frontend `phi4_mini_json` tool parser ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — handles streaming deltas, parallel calls, and nested arguments for Phi-4 mini tool outputs.
- Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns with OpenAI's latest API specifications.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — prevents the engine from generating tool calls when explicitly disabled by the client.
- Supported system role messages inside the messages array for Anthropic compatibility ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — improves drop-in replacement for Anthropic API users.

## Watch list
- Initiated deprecation cycle for the `kv_both` role in NixlConnector ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on bidirectional KV transfer should migrate to explicit prefill/decode roles.
- Introduced Pluggable KVCacheSpec ([#37505](https://github.com/vllm-project/vllm/pull/37505)) — lays the groundwork for custom KV cache layouts and speculative decoding integrations, worth tracking for advanced serving setups.
- Normalized NIXL KV-connector wheel installs to match CUDA major versions ([#44266](https://github.com/vllm-project/vllm/pull/44266)) — prevents `libcudart.so.12` import errors on CUDA 13 images, but highlights ongoing friction with external KV connector dependencies.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (231)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44674">#44674</a> [ROCm][Kernel] Enable permute_cols for ROCm — by <a href="https://github.com/charlifu">charlifu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43087">#43087</a> Modify torch dependency in xpu.txt — by <a href="https://github.com/BramVanroy">BramVanroy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42599">#42599</a> [Dependency] Remove stale cuDNN frontend upper bound — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44051">#44051</a> [CI] Stabilize the multi-audio OpenAI server path — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44378">#44378</a> [Doc] Fix multimodal torch.compile troubleshooting to not use removed VLLM_TORCH_COMPILE_LEVEL — by <a href="https://github.com/DaoyuanLi2816">DaoyuanLi2816</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44103">#44103</a> [Bugfix][Mooncake] Fix per-group block_size/block_hash and group_idx in MooncakeStoreConnector KV events — by <a href="https://github.com/ivanium">ivanium</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44417">#44417</a> [videoloader] implement glm46v video loader — by <a href="https://github.com/JaredforReal">JaredforReal</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44707">#44707</a> [Cohere] Enable Cohere Mini Code model and update Command A-plus test registry — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44420">#44420</a> [feature] add index share feature for DSA MTP — by <a href="https://github.com/JaredforReal">JaredforReal</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44041">#44041</a> [Bugfix] Fix benchmark_moe.py after inplace mechanism removal — by <a href="https://github.com/qyYue1389">qyYue1389</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44540">#44540</a> [XPU] add xpu branch in compressed_tensors_moe_w4a4_mxfp4 — by <a href="https://github.com/zufangzhu">zufangzhu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/37149">#37149</a> [XPU][Feature] transparent sleep mode support for XPU platform — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/36423">#36423</a> [XPU] Support  cpu kv offloading and tiering offloading on XPU platform — by <a href="https://github.com/chaojun-zhang">chaojun-zhang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44699">#44699</a> [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 — by <a href="https://github.com/WoosukKwon">WoosukKwon</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42838">#42838</a> [ROCm][MLA] Replace torch.cat in sparse-MLA forward_mqa with fused concat_mla_q — by <a href="https://github.com/maeehart">maeehart</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44560">#44560</a> [BugFix] Resolve multiple async kv load deadlock — by <a href="https://github.com/njhill">njhill</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44075">#44075</a> [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) — by <a href="https://github.com/JartX">JartX</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44700">#44700</a> [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44694">#44694</a> [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43684">#43684</a> [Bugfix][ROCm] `ApplyRotaryEmb`: fall back to native when flash_attn rotary grid would exceed the HIP per-dim limit — by <a href="https://github.com/amd-fuweiy">amd-fuweiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44559">#44559</a> [Bugfix][Voxtral] Add fetch_audio to MistralCommonFeatureExtractor (transformers&gt;=5.10 compat) — by <a href="https://github.com/Yadan-Wei">Yadan-Wei</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44613">#44613</a> [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig — by <a href="https://github.com/aoshen02">aoshen02</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44593">#44593</a> [Misc] Replaced asserts with proper exceptions to improve UX for pooling — by <a href="https://github.com/taneem-ibrahim">taneem-ibrahim</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44692">#44692</a> [Bugfix][Kernel] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096 — by <a href="https://github.com/zyongye">zyongye</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44213">#44213</a> [Rust Frontend] Add Phi-4 mini JSON tool parser — by <a href="https://github.com/devin-lai">devin-lai</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44574">#44574</a> Preserve layout-changing clones — by <a href="https://github.com/mikekg">mikekg</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44130">#44130</a> [Bugfix] Fix `sequence_parallel_chunk_impl` custom op aliasing its input — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44021">#44021</a> [Cohere] fix RoutingMethodType — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44435">#44435</a> [Doc] Add Llama-3.2-3B-Instruct to batch-invariance tested models — by <a href="https://github.com/DaoyuanLi2816">DaoyuanLi2816</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42832">#42832</a> [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path — by <a href="https://github.com/akii96">akii96</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44669">#44669</a> [Core][Engine] allow DP ray placement groups to be set on specific nodes — by <a href="https://github.com/walterbm">walterbm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44666">#44666</a> Male Mergify comment less spammy — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44330">#44330</a> [Bugfix] GPT-OSS instruction rendering — by <a href="https://github.com/yzong-rh">yzong-rh</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44621">#44621</a> Upgrade tpu-inference to v0.21.0 — by <a href="https://github.com/CienetStingLin">CienetStingLin</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/38804">#38804</a> Fix sarvam forward compatibility with transformers v5 — by <a href="https://github.com/Vikrantpalle">Vikrantpalle</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44648">#44648</a> [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm — by <a href="https://github.com/tjtanaa">tjtanaa</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41968">#41968</a> Add objectstore as a secondary tier to multi-tier kv cache offloading — by <a href="https://github.com/effi-ofer">effi-ofer</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44609">#44609</a> Support MiniCPMV batched preprocessing — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44647">#44647</a> [CI] Bump mypy version `1.19.1` -&gt; `1.20.2` — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44635">#44635</a> Speed up docs build — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44649">#44649</a> [CI] Bump mistral-common — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44588">#44588</a> [Reasoning][Structured Outputs] Add Command A plus tags for structural tags — by <a href="https://github.com/rishitdholakia13">rishitdholakia13</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44561">#44561</a> [DSV4] Move more ops out of eager breakpoint — by <a href="https://github.com/WoosukKwon">WoosukKwon</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44615">#44615</a> [Bugfix] Fix gemma4 crash on CPU: guard mem_get_info call — by <a href="https://github.com/adhithyamulticoreware">adhithyamulticoreware</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43167">#43167</a> Remove KV cache scale boilerplate from model weight loading methods — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43150">#43150</a> [BUG] Fix FP64 Gumbel precision coverage — by <a href="https://github.com/tianyu-z">tianyu-z</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44591">#44591</a> [Rust Frontend] Batch auto-abort requests by engine — by <a href="https://github.com/HueCodes">HueCodes</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44066">#44066</a> docs: fix tokenizer optimization typo — by <a href="https://github.com/chunyang-wen">chunyang-wen</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43874">#43874</a> [NixlConnector] Initiate deprecation cycle for `kv_both` role  — by <a href="https://github.com/NickLucche">NickLucche</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44391">#44391</a> [Rust Frontend] Support include_reasoning=false — by <a href="https://github.com/ricky-chaoju">ricky-chaoju</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44622">#44622</a> [Bugfix] Update mistral tokenizer test for continue_final_message fix — by <a href="https://github.com/XuZhou26">XuZhou26</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44603">#44603</a> fix: pad dummy run query_start_loc — by <a href="https://github.com/UranusSeven">UranusSeven</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44618">#44618</a> [Bugfix] Fix test_invocations flaky failure with newer openai SDK — by <a href="https://github.com/XuZhou26">XuZhou26</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44620">#44620</a> [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer — by <a href="https://github.com/Sunt-ing">Sunt-ing</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44617">#44617</a> Fix `LLM.wait_for_completion` output type docstring — by <a href="https://github.com/viiccwen">viiccwen</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41002">#41002</a> [ROCm][perf] Use workspace manager for sparse indexer allocations — by <a href="https://github.com/tuukkjs">tuukkjs</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/40426">#40426</a> [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning — by <a href="https://github.com/hanlin12-AMD">hanlin12-AMD</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44605">#44605</a> [CI/Build] Disable CPU-Compatibility Tests — by <a href="https://github.com/bigPYJ1151">bigPYJ1151</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43720">#43720</a> [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing — by <a href="https://github.com/zixi-qi">zixi-qi</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44571">#44571</a> [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified — by <a href="https://github.com/lucianommartins">lucianommartins</a> → <code>v0.22.1</code></li>
<li><em>…and 171 more</em></li>
</ul>
</details>
