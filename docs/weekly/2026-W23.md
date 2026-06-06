# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on productionizing complex architectures and scaling out infrastructure. DeepSeek-V4 received a massive wave of performance and stability optimizations, while the Rust frontend achieved near feature-parity with the Python stack for advanced API routing. On the infrastructure side, multi-tier KV cache offloading and NIXL disaggregation enhancements significantly expand long-context and prefill-decode capabilities, alongside targeted hardware accelerations for AMD Zen CPUs and Intel XPUs.

## Deep dives

### DeepSeek-V4 Production Readiness
DeepSeek-V4 relies on Multi-head Latent Attention (MLA) and massive Mixture-of-Experts (MoE) layers, which demand highly specialized memory and compute routing to avoid bottlenecks. This week, vLLM merged a wave of DSV4 optimizations, including integrating the TensorRT-LLM generation attention kernel ([#43827](https://github.com/vllm-project/vllm/pull/43827)), adding Expert Parallelism Load Balancing (EPLB) for its Mega MoE layers ([#43339](https://github.com/vllm-project/vllm/pull/43339)), and implementing selective prefix-cache retention for its sliding-window KV cache ([#43447](https://github.com/vllm-project/vllm/pull/43447)). These changes, alongside a fix for a critical CUTLASS `fmin` initialization crash ([#44236](https://github.com/vllm-project/vllm/pull/44236)), shift DSV4 from merely loading to running efficiently at scale. Teams deploying DeepSeek-V4 on NVIDIA GPUs will see significantly improved throughput and multi-node stability, particularly when utilizing Expert Parallelism.

### Multi-Tier KV Offloading and NIXL Disaggregation
Disaggregated prefill-decode (PD) serving and long-context workloads require moving massive KV caches between GPUs, CPUs, and external storage without blocking inference. vLLM now supports using an object store as a secondary tier for KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)), while the NIXL KV-connector gained PP-aware handshake aggregation ([#43720](https://github.com/vllm-project/vllm/pull/43720)) and zero-copy transfer optimizations ([#41633](https://github.com/vllm-project/vllm/pull/41633)). This matters because it allows clusters to serve vastly longer contexts by spilling to cheap storage, while simultaneously accelerating the KV handoff between prefill and decode nodes in disaggregated setups. Infrastructure engineers running PD disaggregation or extreme long-context workloads should evaluate the updated NIXL connector and tiered offloading configurations to maximize memory utilization.

### Rust Frontend Feature Parity
vLLM's Rust frontend is designed to handle HTTP routing, request scheduling, and tokenization with lower overhead and better concurrency than the traditional Python Uvicorn stack. This week, the Rust frontend gained substantial feature parity, adding dynamic LoRA endpoints ([#43778](https://github.com/vllm-project/vllm/pull/43778)), streaming `generate` endpoints ([#43779](https://github.com/vllm-project/vllm/pull/43779)), a JSON tool parser for Phi-4 mini ([#44213](https://github.com/vllm-project/vllm/pull/44213)), and batch auto-abort capabilities ([#44591](https://github.com/vllm-project/vllm/pull/44591)). This transforms the Rust frontend from an experimental fast-path into a production-ready alternative capable of handling complex API features like tool calling and dynamic adapter loading natively. Serving engineers aiming to minimize Time-To-First-Token (TTFT) and maximize routing throughput should benchmark the Rust frontend, as it now supports the advanced features previously exclusive to the Python stack.

## Kernels & attention
- Qwen3.5 split mixed prefill+decode batches to route decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — improves decode throughput by avoiding prefill-decode interference in mixed batches.
- ROCm `ApplyRotaryEmb` now falls back to native when the flash_attn rotary grid exceeds HIP per-dim limits ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — prevents crashes on AMD GPUs for models with large head dimensions.
- Fused RoPE and static Q FP8 quantization on the fused RoPE+KV path for ROCm ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — reduces memory bandwidth pressure during attention on AMD hardware.
- Mamba attention module refactored to unify KDA conv states into one cache matching the 2-state SSM layout ([#44539](https://github.com/vllm-project/vllm/pull/44539)) — simplifies state management and improves Mamba execution speed.

## Quantization
- Added support for compressed-tensors WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported INT8/INT4 weight-activation quantization schemes.
- Refactored Compressed Tensors NVFP4 linear to use a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — cleans up the FP4 codebase and standardizes 4-bit floating point inference paths.
- Enabled ModelOpt MXFP8 non-gated MoE support ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — allows efficient FP8 inference for Mixture-of-Experts models using NVIDIA's ModelOpt toolkit.
- Added asymmetric support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — improves accuracy and compatibility for 16-bit weight-only quantized MoE models.

## Parallelism & scheduling
- Fixed a deterministic hang in multi-node Ray data-parallel serving by excluding the Ray DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — unblocks multi-node Ray deployments with multiple API servers.
- Allowed Data Parallel Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — gives operators finer-grained control over hardware allocation in large Ray clusters.
- Model Runner V2 now avoids pipeline parallel bubbles ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — increases GPU utilization and throughput in Pipeline Parallel configurations.
- Added a Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector ([#42212](https://github.com/vllm-project/vllm/pull/42212)) — speeds up KV cache swapping for offloaded blocks.

## Model support
- Added support for JetBrains' Mellum v2, an open-weights MoE code-generation model ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expands vLLM's coverage of specialized coding architectures.
- Added Gemma4 Unified (encoder-free) support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) and Gemma4 MTP (Multi-Token Prediction) support ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — brings Google's latest multimodal and speculative decoding architectures to vLLM.
- Added model support for Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — enables native audio/speech processing capabilities.
- Fixed `OlmoHybridForCausalLM` initialization after upstream checkpoint changes ([#43846](https://github.com/vllm-project/vllm/pull/43846)) and vendored HyperCLOVAX config ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — resolves loading regressions for these specific models.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly accelerates quantized CPU inference on AMD EPYC/Ryzen processors.
- Integrated Aiter hipBLASLt GEMM online tuning for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — automatically finds the best GEMM algorithms for AMD GPUs, boosting matmul performance.
- Added XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and block FP8 MoE support ([#42139](https://github.com/vllm-project/vllm/pull/42139)) — brings advanced FP8 quantization to Intel GPUs.
- Enabled SHM communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — allows shared-memory tensor transfers on IBM Power architectures.

## API & serving
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixes a bug where tools were still being called despite explicit user instructions.
- Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns vLLM's API behavior with standard developer message handling.
- Supported system role messages inside the messages array for the Anthropic API compatibility layer ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — improves drop-in compatibility for Anthropic SDK users.
- Unified reasoning and tool-call parsing behind a single `Parser.parse()` interface ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — simplifies the frontend parsing logic and reduces edge cases in structured outputs.

## Watch list
- The NIXL KV-connector is initiating a deprecation cycle for the `kv_both` role ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — operators using NIXL for PD disaggregation should update their role configurations to avoid future breakage.
- FlashInfer JIT cache installation via `--extra-index-url` has been removed from Docker builds due to PyPI quarantine ([#44366](https://github.com/vllm-project/vllm/pull/44366)) — custom Docker builds relying on this cache may need to adjust their build scripts.
- Migrated several custom CUDA kernels (all-reduce, DeepSeek V4 MLA, MXFP8 MoE) to the libtorch stable ABI ([#44365](https://github.com/vllm-project/vllm/pull/44365)) — improves binary compatibility across PyTorch versions but warrants close monitoring for edge-case kernel regressions.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (215)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44700">#44700</a> [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel — @vadiklyutiy</li>
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
<li><em>…and 155 more</em></li>
</ul>
</details>
