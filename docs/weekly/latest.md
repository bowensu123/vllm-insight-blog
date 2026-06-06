# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on maturing the Model Runner V2 (MRV2) engine, expanding DeepSeek-V4 support with critical Expert Parallelism Load Balancing (EPLB) and TensorRT-LLM attention kernels, and pushing multi-tier KV cache offloading to object stores. Hardware coverage also broadened with AMD Zen CPU zentorch acceleration, Intel XPU Mamba support, and PowerPC shared memory communicators. If you are serving large MoE models, running long-context workloads, or using Pipeline Parallelism, this update contains significant throughput and stability improvements worth investigating.

## Deep dives

### DeepSeek-V4 Mega MoE and Attention Optimizations
DeepSeek-V4 is a massive Mixture-of-Experts (MoE) model that traditionally suffers from expert load imbalance, where some GPU ranks sit idle while others process heavy token traffic. Expert Parallelism Load Balancing (EPLB) solves this by dynamically replicating or routing tokens to balance the compute load across ranks. This week, vLLM added EPLB support specifically for the DeepSeek-V4 Mega MoE architecture ([#43339](https://github.com/vllm-project/vllm/pull/43339)) and integrated the highly optimized TensorRT-LLM generation attention kernel ([#43827](https://github.com/vllm-project/vllm/pull/43827)) to accelerate the decode phase. These changes, alongside fixes for CUTLASS initialization bugs ([#44236](https://github.com/vllm-project/vllm/pull/44236)), unblock high-throughput, multi-node serving of DeepSeek-V4. Teams deploying this model on NVIDIA clusters should test the new EPLB flags to maximize GPU utilization and reduce decode latency.

### Multi-Tier KV Cache Offloading to Object Store
When GPU memory fills up during long-context generation, vLLM offloads evicted KV cache blocks to CPU RAM; however, local CPU memory can also become a bottleneck for massive concurrency or extreme context lengths. To solve this, vLLM now supports an object store (such as S3 or MinIO) as a secondary tier for KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)), working in tandem with updates to the LMCache connector ([#42865](https://github.com/vllm-project/vllm/pull/42865)). This allows the engine to spill KV blocks to distributed storage, effectively decoupling prefix caching and long-context limits from the physical RAM of a single node. Engineers running long-context workloads or heavy prefix-caching who frequently hit CPU OOM limits should evaluate this new object store backend to scale their serving capacity.

### Model Runner V2 Maturation for Dense Models
Model Runner V2 (MRV2) is vLLM’s next-generation execution engine designed to minimize Python overhead and better overlap compute with communication, but it has historically been limited to specific model architectures. This week, MRV2 was enabled by default for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)), bringing its performance benefits to the most widely used model families. Additionally, a major scheduling update eliminates pipeline parallel (PP) "bubbles" ([#42187](https://github.com/vllm-project/vllm/pull/42187))—periods where GPUs sit idle waiting for microbatches—by improving microbatch overlap. Users serving Llama or Mistral models, especially those relying on Pipeline Parallelism across multiple nodes, should upgrade to see immediate throughput and GPU utilization improvements without needing to change their configuration.

## Kernels & attention
- Fused RoPE and static Q FP8 quantization on the ROCm fused RoPE+KV path ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — reduces memory bandwidth pressure during attention on AMD GPUs.
- Fixed a miscompile in the mHC fused-RMSNorm big-fuse kernel when `hidden_size != 4096` ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — resolves silent correctness issues for non-standard model dimensions.
- ROCm `ApplyRotaryEmb` now falls back to native implementations when the flash_attn rotary grid exceeds HIP per-dim limits ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — prevents crashes on large-context AMD workloads.
- Added GELU_TANH activation support to CPU, CUTLASS, and WNA16 MoE backends ([#42027](https://github.com/vllm-project/vllm/pull/42027)) — expands kernel coverage for models using this specific activation function.

## Quantization
- Added support for compressed-tensors WNA8O8Int linear layers and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — broadens the range of supported neural magic quantization formats.
- Enabled ModelOpt MXFP8 non-gated MoE support ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — allows serving NVIDIA ModelOpt quantized Mixture-of-Experts models without gating overhead.
- Added asymmetric quantization support for MoE WNA16 Marlin kernels ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — improves accuracy for heavily compressed MoE models using the Marlin backend.
- Implemented a block-scaled W8A8 FP8 execution path for Intel XPU ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings hardware-accelerated FP8 inference to Intel GPUs.

## Parallelism & scheduling
- Added PP-aware handshake aggregation and intermediate-PP output plumbing for the NixlConnector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — improves KV cache transfer efficiency in disaggregated prefill-decode setups.
- Implemented selective prefix-cache retention for sliding-window KV caches in DeepSeek-V4 ([#43447](https://github.com/vllm-project/vllm/pull/43447)) — prevents unnecessary cache evictions and boosts hit rates for sliding-window models.
- Added an `on_schedule_end()` hook to separate the step lifecycle from event draining in KV offloading ([#44206](https://github.com/vllm-project/vllm/pull/44206)) — prevents scheduling stalls during heavy offload operations.
- Fixed a deterministic hang in multi-node Ray data-parallel serving by excluding the Ray DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — unblocks stable multi-node Ray deployments.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — brings native inference for this open-weights Mixture-of-Experts code-generation model.
- Integrated Gemma4 Unified (encoder-free) architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)) and added MTP support ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — expands coverage for Google's latest multimodal and speculative decoding capabilities.
- Added model support for Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — enables native audio/speech processing for the Granite family.
- Fixed a crash in MiniCPM-V-4.6 video inference caused by placeholder count mismatches ([#44509](https://github.com/vllm-project/vllm/pull/44509)) and added batched preprocessing ([#44609](https://github.com/vllm-project/vllm/pull/44609)) — stabilizes and speeds up video understanding workloads.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly accelerates quantized CPU inference with transparent fallback for non-Zen architectures.
- Integrated Aiter hipBLASLt GEMM online tuning for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — dynamically selects the best GEMM algorithms for AMD GPUs, improving matmul performance.
- Added a Triton-based selective scan forward operation for Mamba models on Intel XPU ([#43421](https://github.com/vllm-project/vllm/pull/43421)) — unblocks efficient state-space model inference on Intel hardware.
- Enabled shared memory (SHM) communicator support for PowerPC architectures ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — allows efficient intra-node tensor parallelism on IBM Power systems.

## API & serving
- Added a Phi-4 mini JSON tool parser ([#44213](https://github.com/vllm-project/vllm/pull/44213)) and a streaming `generate` endpoint ([#43779](https://github.com/vllm-project/vllm/pull/43779)) to the Rust frontend — continues the maturation of the high-performance Rust API server.
- Fixed the Responses API to correctly fold developer-role input messages into system instructions ([#43590](https://github.com/vllm-project/vllm/pull/43590)) and prevented unstreamed tool call args from being dropped ([#44348](https://github.com/vllm-project/vllm/pull/44348)) — ensures OpenAI-compatible API parity.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixes a bug where models would still attempt to generate tool calls when explicitly forbidden.
- Unified reasoning and tool-call parsing behind a single `Parser.parse()` interface ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — simplifies the frontend parsing logic and reduces edge-case bugs.

## Watch list
- The `kv_both` role in `NixlConnector` has entered its deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this role for disaggregated serving should migrate to explicit prefill/decode roles.
- FlashInfer was bumped to [v0.6.12](https://github.com/vllm-project/vllm/releases/tag/v0.6.12) ([#44036](https://github.com/vllm-project/vllm/pull/44036)) and the version check in `topk_topp_sampler` was removed ([#44442](https://github.com/vllm-project/vllm/pull/44442)) — ensures compatibility with the latest FlashInfer sampling optimizations.
- Multiple models (Sarvam, HyperCLOVAX, MiniCPMV) received patches for `transformers` v5 compatibility ([#38804](https://github.com/vllm-project/vllm/pull/38804), [#43860](https://github.com/vllm-project/vllm/pull/43860), [#44282](https://github.com/vllm-project/vllm/pull/44282)) — watch for upstream HuggingFace repo changes as v5 adoption increases.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (217)

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
<li><em>…and 157 more</em></li>
</ul>
</details>
