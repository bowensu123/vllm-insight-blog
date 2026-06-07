# vLLM weekly digest — 2026-06-07 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing large-scale disaggregated serving, expanding hardware support for AMD and Intel, and unblocking DeepSeek-V4 and Qwen3.5 workloads. Key performance wins include splitting mixed prefill/decode batches for GDN attention and optimizing Sparse MLA speculative decoding. Model Runner V2 also reaches a major milestone by becoming the default execution path for Llama and Mistral dense models.

## Deep dives

### Splitting Mixed Prefill and Decode Batches for GDN Attention
Gated Delta Net (GDN) attention, used in models like Qwen3.5, relies on a chunked recurrent kernel for efficient sequence modeling. When prefill and decode requests are mixed in a single batch, the engine historically routed all tokens through the chunked kernel, padding short decode sequences to the full chunk size (e.g., 64 tokens) and wasting compute. PR [#44700](https://github.com/vllm-project/vllm/pull/44700) splits mixed batches at the scheduler level, routing pure decode tokens to a specialized, lightweight recurrent kernel (`aiter_kernel`) while keeping prefills on the chunked kernel. This eliminates the massive overhead of processing near-empty chunks for decodes, drastically reducing latency for mixed workloads. Anyone serving Qwen3.5 or other GDN-based architectures with concurrent prefill and decode traffic will see immediate throughput and latency improvements without needing to change flags.

### Index Sharing for DeepSeek Sparse MLA Multi-Token Prediction
DeepSeek’s Sparse Multi-head Latent Attention (MLA) uses a sparse indexer to select the top-k KV blocks for attention. In Multi-Token Prediction (MTP, a form of speculative decoding), running this indexer for every speculative step and every layer introduces severe computational overhead. PR [#44420](https://github.com/vllm-project/vllm/pull/44420) implements an "IndexCache" mechanism that carries `topk_indices` through the forward path, allowing the engine to reuse the top-k block selections across MTP steps and layers when `skip_topk` is enabled. By avoiding redundant sparse attention indexing, this significantly reduces the per-step overhead of speculative decoding, making MTP much more viable for large Sparse MLA models. Users running DeepSeek-V3 or V4 with speculative decoding enabled should test this to see if it improves their acceptance rates and overall generation speed.

### Resolving Async KV Load Deadlocks in Disaggregated Serving
In prefill-decode (PD) disaggregated serving, the decode node asynchronously loads KV cache blocks from the prefill node over the network. If the scheduler aggressively allocates memory for these incoming async loads, it can starve in-flight chunked prefills or other active requests of the physical blocks they need to proceed. PR [#44560](https://github.com/vllm-project/vllm/pull/44560) introduces a throttling mechanism for async KV loads, ensuring they are paused if they would occupy blocks required for other in-flight chunked prefills or async-loading requests. This prevents a deterministic deadlock where the engine hangs because active requests cannot finish and free memory, while async loads wait for memory that will never be freed. Teams operating high-throughput PD disaggregated clusters (using connectors like Mooncake or Nixl) should upgrade to eliminate mysterious multi-node hangs.

## Kernels & attention
- Replaced `torch.cat` with a fused `concat_mla_q` kernel in the ROCm sparse-MLA backend ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — eliminates 61 tensor allocations per decode step on DeepSeek-V3.2.
- Fixed a TileLang software pipeliner miscompile in the mHC fused-RMSNorm kernel ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — prevents silent NaN outputs for DeepSeek-V4 when hidden size isn't 4096.
- Added a fallback to the native rotary embedding on ROCm when sequence length exceeds the HIP grid limit ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — prevents kernel launch aborts on very long contexts.
- Defaulted to the Triton MoE backend on Hopper GPUs ([#44220](https://github.com/vllm-project/vllm/pull/44220)) — provides better out-of-the-box performance for Mixture-of-Experts models on H100/H200.

## Quantization
- Added XPU support for `compressed_tensors_moe_w4a4_mxfp4` ([#44540](https://github.com/vllm-project/vllm/pull/44540)) — enables MXFP4 quantized MoE inference on Intel GPUs.
- Supported compressed-tensors WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported INT8/INT4 weight-only formats.
- Guarded against dtype mismatches in the RMSNorm + quant fusion pass ([#44694](https://github.com/vllm-project/vllm/pull/44694)) — fixes a crash when running Qwen3.5-FP8 with mixed bf16/fp32 weights.
- Added asymmetric support for MoE WNA16 Marlin quantization via compressed-tensors ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows more flexible weight-only quantization schemes for MoE layers.

## Parallelism & scheduling
- Enabled Model Runner V2 (MRV2) by default for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) — brings V1 architecture benefits like chunked prefill optimizations to these popular models.
- Avoided pipeline parallel (PP) bubbles in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — improves GPU utilization and throughput for multi-node PP deployments.
- Fixed an aliasing bug in `sequence_parallel_chunk_impl` ([#44130](https://github.com/vllm-project/vllm/pull/44130)) — prevents `torch.compile` AOT-autograd from rejecting the custom op during memory profiling.
- Allowed data-parallel Ray placement groups to be pinned to specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — improves resource management and stability in multi-node Ray clusters.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — brings native vLLM support to this open-weights MoE code-generation model.
- Introduced Gemma4 Unified (encoder-free) model support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables serving of Google's latest unified architecture with speculative decoding.
- Enabled Cohere's North-Mini-Code model ([#44707](https://github.com/vllm-project/vllm/pull/44707)) — adds native tool-calling and reasoning parser support for Cohere's new coding model.
- Fixed HyperCLOVAX loading by registering the model type natively ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — resolves breakages caused by upstream HuggingFace removing remote code.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates CPU inference with transparent fallback for non-Zen architectures.
- Implemented a native HIP kernel for W4A16 MoE on AMD RDNA3 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — replaces the slower Triton path with fused expert routing and GEMM for consumer GPUs.
- Enabled transparent sleep mode and memory allocator interfaces for Intel XPU ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — matches CUDA parity for memory management and power saving.
- Added CPU KV offloading and tiering offloading support for XPU ([#36423](https://github.com/vllm-project/vllm/pull/36423)) — allows Intel GPU users to spill KV cache to host memory to serve larger batches.

## API & serving
- Added a Rust frontend `phi4_mini_json` tool parser ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — correctly handles Phi-4 mini's specific `functools` tool-call output formatting.
- Stabilized the multi-audio OpenAI server path ([#44051](https://github.com/vllm-project/vllm/pull/44051)) — properly returns a 400 error when audio inputs exceed limits without crashing the server.
- Migrated the `ResponsesParser` to the unified `Parser` interface ([#42977](https://github.com/vllm-project/vllm/pull/42977)) — consolidates reasoning and tool-call parsing logic for cleaner frontend code.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixes a bug where the model would still attempt to invoke tools when explicitly told not to.

## Watch list
- The KV-Cache Layout Refactor is underway (starting with [#44454](https://github.com/vllm-project/vllm/pull/44454)) — aims to standardize KV cache layouts across attention backends, which will impact custom KV connectors.
- NixlConnector has initiated a deprecation cycle for the `kv_both` role ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this specific KV transfer configuration should plan to migrate.
- Model Runner V2 is rapidly expanding to dense models and fixing PP bubbles — expect it to become the default execution path for all models soon.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (232)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44454">#44454</a> [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config construction — by <a href="https://github.com/LucasWilkinson">LucasWilkinson</a></li>
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
<li><em>…and 172 more</em></li>
</ul>
</details>
