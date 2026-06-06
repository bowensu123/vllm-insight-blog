# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on maturing the Model Runner V2 (MRV2) execution engine, expanding DeepSeek-V4 optimizations, and hardening disaggregated prefill/decode (PD) infrastructure. We also saw significant hardware-specific wins, including zentorch kernel routing for AMD Zen CPUs and substantial ROCm/Intel XPU backend stabilization. If you are deploying DeepSeek-V4, testing the V2 engine, or building multi-tier KV cache offloading for PD, this update is highly relevant.

## Deep dives

### DeepSeek-V4 Architecture Optimizations
DeepSeek-V4 relies on Multi-head Latent Attention (MLA) and massive Mixture-of-Experts (MoE) layers, which historically strain standard inference engines due to unique memory and routing patterns. This week, vLLM integrated TRT-LLM generation attention kernels specifically for DSV4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)), added Expert Parallelism Load Balancing (EPLB) for its MegaMoE layers ([#43339](https://github.com/vllm-project/vllm/pull/43339)), and implemented selective prefix-cache retention for its sliding window attention ([#43447](https://github.com/vllm-project/vllm/pull/43447)). These changes drastically reduce KV cache memory overhead and improve token generation throughput by optimizing the exact bottlenecks DSV4 exhibits at scale. Teams deploying DSV4 on multi-GPU clusters should upgrade to leverage these targeted performance and memory wins.

### Model Runner V2 Rolls Out to Dense Models
Model Runner V2 (MRV2) is vLLM’s next-generation execution engine, designed to decouple scheduling from execution, eliminate Python-side bottlenecks, and natively support advanced features like speculative decoding. This week, MRV2 was officially enabled for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)), integrated with the highly optimized FlashInfer sampler ([#42472](https://github.com/vllm-project/vllm/pull/42472)), and tuned to avoid pipeline parallel (PP) bubbles ([#42187](https://github.com/vllm-project/vllm/pull/42187)). This matters because it brings V2’s architectural latency and throughput improvements to the most widely deployed model families. Engineers testing the V1/V2 transition should monitor their Llama/Mistral workloads for improved PP scaling and sampling performance.

### Maturing Disaggregated Serving with Multi-Tier KV Offloading
Disaggregated prefill/decode (PD) separates compute-heavy prefill from memory-bound decode, requiring fast, reliable KV cache transfers between nodes without triggering out-of-memory (OOM) errors. vLLM significantly hardened this pipeline by introducing an object store as a secondary tier for KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)), adding zero-copy transfers to the NIXL communicator ([#41633](https://github.com/vllm-project/vllm/pull/41633)), and implementing a Triton fast-path for small CPU-to-GPU block swaps ([#42212](https://github.com/vllm-project/vllm/pull/42212)). This creates a robust fallback hierarchy (GPU VRAM -> CPU RAM -> Object Store) that prevents request drops during bursty traffic. Infrastructure teams building large-scale PD clusters will benefit from this increased resilience and transfer efficiency.

## Kernels & attention
- Optimized CUTLASS FP8 scaled matrix multiplication by bypassing padding ([#43706](https://github.com/vllm-project/vllm/pull/43706)) — yields a 20% kernel performance improvement for FP8 workloads.
- Refactored the DeepSeek-V4 attention module ([#44569](https://github.com/vllm-project/vllm/pull/44569)) and optimized sparse FP8 compressor kernels ([#44161](https://github.com/vllm-project/vllm/pull/44161)) — cleans up the MLA implementation and speeds up sparse routing.
- Added GELU_TANH activation support to CPU, CUTLASS, and WNA16 MoE backends ([#42027](https://github.com/vllm-project/vllm/pull/42027)) — expands compatibility for models using this specific gating function.
- Applied a single-pass min_larger finding and binary search in the Triton Top-p sampling path ([#42191](https://github.com/vllm-project/vllm/pull/42191)) — reduces sampling overhead for large vocabularies.

## Quantization
- Enabled block-scaled W8A8 FP8 execution paths on Intel XPU ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and added W4A16 support to FlashInfer B12x MoE experts ([#43332](https://github.com/vllm-project/vllm/pull/43332)) — broadens quantized inference across hardware.
- Added asymmetric support for MoE WNA16 Marlin kernels via compressed-tensors ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows more flexible weight quantization schemes for MoE models.
- Supported ModelOpt MXFP8 non-gated MoE ([#42958](https://github.com/vllm-project/vllm/pull/42958)) and refactored CT NVFP4 linear classes ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — streamlines deployment of NVIDIA's latest microscaling formats.
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — provides transparent, hardware-accelerated quantized inference on x86.

## Parallelism & scheduling
- Added PP-aware handshake aggregation and intermediate-PP output plumbing to the KV connector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — ensures correct KV cache routing in multi-node pipeline parallel setups.
- Threaded `scheduler_block_size` into the KVCacheManager and KVCacheCoordinator ([#44165](https://github.com/vllm-project/vllm/pull/44165)) — aligns block allocation logic across the scheduling stack.
- Supported Pluggable `KVCacheSpec` ([#37505](https://github.com/vllm-project/vllm/pull/37505)) — allows custom KV cache layouts required for speculative decoding and hybrid SSM/Transformer models.
- Snapshot `max_cudagraph_capture_size` into `FusedMoEConfig` ([#44613](https://github.com/vllm-project/vllm/pull/44613)) — prevents CUDA graph capture failures for large MoE models.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) and Gemma4 Unified encoder-free models ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expands the roster of natively supported open-weights architectures.
- Added Gemma4 Multi-Token Prediction (MTP) support in Model Runner V2 ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — enables speculative decoding for the latest Gemma variants.
- Fixed HyperCLOVAX loading by registering the model type natively ([#43860](https://github.com/vllm-project/vllm/pull/43860)) and vendored MiniCPMV/MiniCPMO processors ([#44282](https://github.com/vllm-project/vllm/pull/44282)) — resolves breakages from upstream HuggingFace `transformers` v5 changes.
- Unified KDA conv states into one cache for Mamba models ([#44539](https://github.com/vllm-project/vllm/pull/44539)) and refactored the Mamba attention module ([#43556](https://github.com/vllm-project/vllm/pull/43556)) — aligns SSM state management with vLLM's 2-state layout.

## Hardware
- Integrated AITER hipBLASLt GEMM online tuning for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) and fused RoPE with static Q FP8 quantization ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — significantly boosts AMD GPU inference performance.
- Added a Triton-based selective scan forward op for Intel XPU ([#43421](https://github.com/vllm-project/vllm/pull/43421)) and enabled RMSNorm/activation quant fusions ([#43968](https://github.com/vllm-project/vllm/pull/43968)) — closes the performance gap for SSMs and quantized models on Intel silicon.
- Enabled shared memory (SHM) communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — facilitates multi-process weight sharing on IBM Power architectures.
- Standardized KV cache layout to "blocks first" on CPU ([#44393](https://github.com/vllm-project/vllm/pull/44393)) and enabled fused kernels for Gated Delta Networks ([#43534](https://github.com/vllm-project/vllm/pull/43534)) — optimizes memory access patterns and CPU execution for hybrid models.

## API & serving
- Migrated the `ResponsesParser` to a unified `Parser` interface ([#42977](https://github.com/vllm-project/vllm/pull/42977)) and consolidated reasoning/tool-call parsing ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — simplifies the frontend codebase for structured outputs.
- Added a Phi-4 mini JSON tool parser to the Rust frontend ([#44213](https://github.com/vllm-project/vllm/pull/44213)) and supported recursive tool parameter conversion ([#44299](https://github.com/vllm-project/vllm/pull/44299)) — improves tool-calling reliability for smaller models.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) and folded developer-role messages into system instructions ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — ensures stricter compliance with OpenAI API semantics.
- Added dynamic LoRA endpoints ([#43778](https://github.com/vllm-project/vllm/pull/43778)) and a server router extension hook ([#43774](https://github.com/vllm-project/vllm/pull/43774)) to the Rust frontend — increases flexibility for custom routing and adapter management.

## Watch list
- **NIXL `kv_both` Deprecation**: The `kv_both` role in `NixlConnector` is entering its deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)); teams using NIXL for KV transfers should update their configurations to use explicit prefill/decode roles.
- **Transformers v5 Compatibility**: Multiple PRs ([#43860](https://github.com/vllm-project/vllm/pull/43860), [#44282](https://github.com/vllm-project/vllm/pull/44282), [#38804](https://github.com/vllm-project/vllm/pull/38804)) are patching model loading and processor vendoring to survive upstream HuggingFace `transformers` v5 breaking changes; expect more churn as the ecosystem migrates.
- **FlashInfer JIT Cache Quarantine**: Docker builds had to stop using `--extra-index-url` for `flashinfer-jit-cache` due to PyPI quarantine issues ([#44366](https://github.com/vllm-project/vllm/pull/44366)); keep an eye on FlashInfer packaging if you build custom vLLM images.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (216)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/43827">#43827</a> [DSv4] Adding TRTLLM gen attention kernel — @zyongye → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44255">#44255</a> [ROCm][CI] Specifying time outs for the lm eval models — @AndreasKaratzas → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44046">#44046</a> [ROCm][CI] Stabilize memory-release in the Hybrid model generation tests — @AndreasKaratzas → <code>v0.22.1</code></li>
<li><em>…and 156 more</em></li>
</ul>
</details>
