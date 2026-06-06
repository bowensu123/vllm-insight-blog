# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on scaling DeepSeek-V4 through targeted Multi-head Latent Attention and Mega MoE optimizations, alongside expanding the Model Runner V2 execution engine to Llama and Mistral dense models. Disaggregated serving saw major multi-tier KV cache offloading and NIXL zero-copy transfer improvements, while AMD Zen CPU users gained significant inference speedups via zentorch kernel routing. Overall, it is a highly infrastructure- and hardware-focused week that unlocks better multi-node throughput and broader hardware acceleration.

## Deep dives

### DeepSeek-V4 MLA and Mega MoE optimizations
DeepSeek-V4 relies on Multi-head Latent Attention (MLA) and a massive Mixture-of-Experts (MoE) architecture, which historically strain inference engines due to complex routing and memory-bound attention. This week, vLLM introduced Expert Parallelism Load Balancing (EPLB) for the DeepSeek-V4 Mega MoE ([#43339](https://github.com/vllm-project/vllm/pull/43339)) to distribute expert weights evenly across GPUs, alongside a new TRT-LLM generation attention kernel tailored for its MLA ([#43827](https://github.com/vllm-project/vllm/pull/43827)). Additionally, sparse FP8 compressor kernels were optimized ([#44161](https://github.com/vllm-project/vllm/pull/44161)) and selective prefix-cache retention for sliding-window KV cache was added ([#43447](https://github.com/vllm-project/vllm/pull/43447)). These changes drastically improve throughput and memory utilization for teams deploying DeepSeek-V4 on multi-node NVIDIA clusters, resolving previous initialization crashes caused by CUTLASS `fmin` incompatibilities ([#44236](https://github.com/vllm-project/vllm/pull/44236)).

### Multi-tier KV cache offloading and NIXL disaggregation
Disaggregated serving separates prefill and decode phases, requiring fast KV cache transfers between nodes or offloading to slower memory tiers to free up expensive GPU HBM. vLLM now supports an object store as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) and introduces zero-copy transfers for the NIXL communicator ([#41633](https://github.com/vllm-project/vllm/pull/41633)). The NIXL connector also gained pipeline-parallel (PP) aware handshake aggregation ([#43720](https://github.com/vllm-project/vllm/pull/43720)) and Mamba prefix caching support ([#42554](https://github.com/vllm-project/vllm/pull/42554)). This matters for large-scale deployments pushing the limits of GPU memory, allowing seamless spillover to CPU RAM or distributed object stores without blocking the critical path, specifically benefiting users of the V1 engine and NIXL-based prefill-decode disaggregation.

### Model Runner V2 expansion and pipeline parallelism
Model Runner V2 (MRV2) is vLLM's next-generation execution engine designed to eliminate scheduling overhead and better support advanced features like speculative decoding and chunked prefill. This week, MRV2 was officially enabled for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) and integrated the FlashInfer sampler ([#42472](https://github.com/vllm-project/vllm/pull/42472)) for faster token generation. Crucially, a fix was introduced to avoid pipeline parallel (PP) bubbles in MRV2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)), ensuring that multi-GPU pipeline stages remain fully utilized. Engineers running Llama or Mistral on multi-GPU setups with pipeline parallelism should test MRV2 to realize lower latency and higher throughput, while speculative decoding also saw Gemma4 Multi-Token Prediction (MTP) support added ([#43241](https://github.com/vllm-project/vllm/pull/43241)).

## Kernels & attention
* Fused RoPE + static Q FP8 quant on the fused RoPE+KV path for ROCm ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — reduces memory bandwidth pressure on AMD GPUs.
* Fixed a fused RMSNorm big-fuse miscompile for hidden sizes other than 4096 ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — ensures correctness for diverse model architectures.
* Refactored the Mamba attention module to LINEAR and unified KDA conv states ([#43556](https://github.com/vllm-project/vllm/pull/43556), [#44539](https://github.com/vllm-project/vllm/pull/44539)) — simplifies state space model execution.
* Used the workspace manager for sparse indexer allocations on ROCm ([#41002](https://github.com/vllm-project/vllm/pull/41002)) — improves memory management for attention.

## Quantization
* Compressed-tensors now supports WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands flexible mixed-precision quantization schemes.
* Refactored CT NVFP4 linear to use a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — cleans up the FP4 quantization codebase for easier maintenance.
* Added asymmetric support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — enables more accurate weight-only quantization for MoE models.
* Guarded `fused_add_rms_norm` input/weight dtype mismatch in RMSNorm + quant fusion to fix Qwen3.5-FP8 nightly failures ([#44694](https://github.com/vllm-project/vllm/pull/44694)).

## Parallelism & scheduling
* Addressed pipeline parallel bubbles in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — keeps multi-GPU stages fully utilized during execution.
* Allowed data-parallel Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — gives finer control over Ray cluster topology.
* Fixed a deterministic hang in multi-node Ray data-parallel serving by excluding the DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)).

## Model support
* Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — an open-weights MoE code-generation model.
* Added Gemma4 Unified (encoder-free) support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) and fixed Gemma4 MTP block table mismatches ([#43982](https://github.com/vllm-project/vllm/pull/43982)).
* Fixed HyperCLOVAX loading by registering the model type natively after upstream HuggingFace removed remote code ([#43860](https://github.com/vllm-project/vllm/pull/43860)).
* Added Granite Speech Plus model support ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — expanding audio and multimodal capabilities.

## Hardware
* Routed W8A8 and W4A16 linear inference through `zentorch` kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly boosts CPU inference performance with transparent fallback.
* Integrated Aiter hipBLASLt GEMM online tuning for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — automatically finds optimal GEMM configurations for AMD GPUs.
* Enabled SHM communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — brings shared-memory optimizations to IBM Power architectures.
* Added an XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and a Triton-based selective scan forward op for Mamba on XPU ([#43421](https://github.com/vllm-project/vllm/pull/43421)).

## API & serving
* Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixes a bug where tools were still invoked when explicitly disabled.
* Unified reasoning and tool-call parsing behind a single `Parser.parse()` interface ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — simplifies the frontend parsing logic.
* Added `/server_info` to the Rust frontend ([#43942](https://github.com/vllm-project/vllm/pull/43942)) and supported streaming for the `generate` endpoint ([#43779](https://github.com/vllm-project/vllm/pull/43779)) — brings the Rust server closer to feature parity.
* Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns with standard system prompt handling.

## Watch list
* The NIXL connector initiated a deprecation cycle for the `kv_both` role ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this for bidirectional KV transfers should migrate to explicit prefill/decode roles.
* FlashInfer JIT cache installation via `--extra-index-url` was removed from Docker builds due to PyPI quarantine ([#44366](https://github.com/vllm-project/vllm/pull/44366)) — custom Docker builds relying on this may need to adjust their FlashInfer installation steps.
* Migrated several custom ops and DeepSeek-V4 kernels to the libtorch stable ABI ([#44365](https://github.com/vllm-project/vllm/pull/44365), [#44334](https://github.com/vllm-project/vllm/pull/44334)) — a foundational shift to prevent C++ ABI breakages across PyTorch versions.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (218)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44694">#44694</a> [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion — @vadiklyutiy</li>
<li><a href="https://github.com/vllm-project/vllm/pull/43684">#43684</a> [Bugfix][ROCm] `ApplyRotaryEmb`: fall back to native when flash_attn rotary grid would exceed the HIP per-dim limit — @amd-fuweiy</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44559">#44559</a> [Bugfix][Voxtral] Add fetch_audio to MistralCommonFeatureExtractor (transformers&gt;=5.10 compat) — @Yadan-Wei</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44613">#44613</a> [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig — @aoshen02</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44593">#44593</a> [Misc] Replaced asserts with proper exceptions to improve UX for pooling — @taneem-ibrahim</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44692">#44692</a> [Bugfix][Kernel] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096 — @zyongye</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44213">#44213</a> [Rust Frontend] Add Phi-4 mini JSON tool parser — @devin-lai</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44574">#44574</a> Preserve layout-changing clones — @mikekg</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44130">#44130</a> [Bugfix] Fix `sequence_parallel_chunk_impl` custom op aliasing its input — @vadiklyutiy</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44021">#44021</a> [Cohere] fix RoutingMethodType — @Terrencezzj</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44435">#44435</a> [Doc] Add Llama-3.2-3B-Instruct to batch-invariance tested models — @DaoyuanLi2816</li>
<li><a href="https://github.com/vllm-project/vllm/pull/42832">#42832</a> [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path — @akii96</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44669">#44669</a> [Core][Engine] allow DP ray placement groups to be set on specific nodes — @walterbm</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44666">#44666</a> Male Mergify comment less spammy — @hmellor</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44330">#44330</a> [Bugfix] GPT-OSS instruction rendering — @yzong-rh</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44621">#44621</a> Upgrade tpu-inference to v0.21.0 — @CienetStingLin</li>
<li><a href="https://github.com/vllm-project/vllm/pull/38804">#38804</a> Fix sarvam forward compatibility with transformers v5 — @Vikrantpalle</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44648">#44648</a> [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm — @tjtanaa</li>
<li><a href="https://github.com/vllm-project/vllm/pull/41968">#41968</a> Add objectstore as a secondary tier to multi-tier kv cache offloading — @effi-ofer</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44609">#44609</a> Support MiniCPMV batched preprocessing — @yma11</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44647">#44647</a> [CI] Bump mypy version `1.19.1` -&gt; `1.20.2` — @hmellor</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44635">#44635</a> Speed up docs build — @hmellor</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44649">#44649</a> [CI] Bump mistral-common — @hmellor</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44588">#44588</a> [Reasoning][Structured Outputs] Add Command A plus tags for structural tags — @rishitdholakia13</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44561">#44561</a> [DSV4] Move more ops out of eager breakpoint — @WoosukKwon</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44615">#44615</a> [Bugfix] Fix gemma4 crash on CPU: guard mem_get_info call — @adhithyamulticoreware</li>
<li><a href="https://github.com/vllm-project/vllm/pull/43167">#43167</a> Remove KV cache scale boilerplate from model weight loading methods — @hmellor</li>
<li><a href="https://github.com/vllm-project/vllm/pull/43150">#43150</a> [BUG] Fix FP64 Gumbel precision coverage — @tianyu-z</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44591">#44591</a> [Rust Frontend] Batch auto-abort requests by engine — @HueCodes → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44066">#44066</a> docs: fix tokenizer optimization typo — @chunyang-wen → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43874">#43874</a> [NixlConnector] Initiate deprecation cycle for `kv_both` role  — @NickLucche → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44391">#44391</a> [Rust Frontend] Support include_reasoning=false — @ricky-chaoju → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44622">#44622</a> [Bugfix] Update mistral tokenizer test for continue_final_message fix — @XuZhou26 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44603">#44603</a> fix: pad dummy run query_start_loc — @UranusSeven → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44618">#44618</a> [Bugfix] Fix test_invocations flaky failure with newer openai SDK — @XuZhou26 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44620">#44620</a> [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer — @Sunt-ing → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44617">#44617</a> Fix `LLM.wait_for_completion` output type docstring — @viiccwen → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41002">#41002</a> [ROCm][perf] Use workspace manager for sparse indexer allocations — @tuukkjs → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/40426">#40426</a> [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning — @hanlin12-AMD → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44605">#44605</a> [CI/Build] Disable CPU-Compatibility Tests — @bigPYJ1151 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43720">#43720</a> [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing — @zixi-qi → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44571">#44571</a> [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified — @lucianommartins → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44569">#44569</a> [DSV4] Refactor DeepseekV4Attention — @WoosukKwon → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44334">#44334</a> [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. — @cleonard530 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42139">#42139</a> [XPU][MoE] support block_fp8_moe on xpu — @zufangzhu → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44500">#44500</a> [Rust Frontend] Skip loading multimodal processor if `--language-model-only` is specified — @BugenZhao → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43926">#43926</a> fix: keep DeepSeek V4 RoPE cache on inv_freq device — @galletas1712 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44539">#44539</a> [mamba] unify KDA conv states into one cache to match 2-state SSM layout — @ZJY0516 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43707">#43707</a> [Logs Refactor] Optimize shutdown logs, easier to follow and consistent — @yewentao256 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41980">#41980</a> use split_group for pytorch process group creation — @tushar00jain → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43307">#43307</a> [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU — @adobrzyn → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44380">#44380</a> [Bugfix] Fix test_cutlass_moe.py — @bnellnm → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44471">#44471</a> [Misc] Add unit tests for pooler head classes — @taneem-ibrahim → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/34894">#34894</a> [DOC] Add INT8 W4A8 docs and Arm&#x27;s supported quantization schemes — @fadara01 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44436">#44436</a> [ROCm][CI] Add test for Aiter unified attn kernel — @divakar-amd → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44057">#44057</a> [Bugfix] Reject non-positive values for ParallelConfig int knobs — @jwzheng96 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44363">#44363</a> [Core] Freeze garbage collector in workers after model initialization — @tlrmchlsmth → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44509">#44509</a> [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count — @tc-mb → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43519">#43519</a> Add model support for granite speech plus — @zvik → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44340">#44340</a> [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings — @mgoin → <code>v0.22.1</code></li>
<li><em>…and 158 more</em></li>
</ul>
</details>
