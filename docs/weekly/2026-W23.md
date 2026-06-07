# vLLM weekly digest — 2026-06-07 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) release focuses heavily on maturing DeepSeek-V4 performance, expanding Intel XPU and AMD ROCm kernel coverage, and hardening disaggregated prefill/decode infrastructure. We also saw significant feature parity updates for the Rust frontend and new multimodal model support, including Gemma4 Unified and Granite Speech. If you are running multi-node clusters or deploying on non-NVIDIA hardware, this week's changes warrant a close look.

## Deep dives

### DeepSeek-V4 performance and routing optimizations
DeepSeek-V4 relies on Sparse Multi-head Latent Attention (MLA) and massive Mixture-of-Experts (MoE) layers, making attention computation and expert routing the primary inference bottlenecks. This week, vLLM decoupled DS-V4 Sparse MLA metadata from V3.2 ([#44699](https://github.com/vllm-project/vllm/pull/44699)), integrated the TRT-LLM generation attention kernel ([#43827](https://github.com/vllm-project/vllm/pull/43827)), and enabled Expert Parallelism Load Balancing (EPLB) for the Mega MoE layers ([#43339](https://github.com/vllm-project/vllm/pull/43339)). It also added selective prefix-cache retention for its sliding-window KV cache ([#43447](https://github.com/vllm-project/vllm/pull/43447)). These changes shift DS-V4 from merely loading to running efficiently at scale by preventing expert starvation and reducing attention overhead. Teams deploying DeepSeek-V4 on multi-GPU clusters should enable EPLB and ensure they are using the latest TRT-LLM attention backends to maximize throughput.

### Intel XPU quantization and memory offloading
Intel XPUs (discrete GPUs like Arc and Max) require specific kernel implementations and memory management strategies to compete with CUDA in LLM inference. A massive wave of XPU PRs landed this week, adding block-scaled W8A8 FP8 linear paths ([#39968](https://github.com/vllm-project/vllm/pull/39968)), W4A4 MXFP4 MoE support ([#44540](https://github.com/vllm-project/vllm/pull/44540)), and CPU KV cache tiering/offloading ([#36423](https://github.com/vllm-project/vllm/pull/36423)). It also introduced transparent sleep mode for power management ([#37149](https://github.com/vllm-project/vllm/pull/37149)). This brings Intel GPUs much closer to CUDA feature parity for modern, heavily quantized MoE models and long-context workloads that exceed VRAM. Intel XPU users running quantized MoE models or long-context workloads can now leverage these native kernels and offloading tiers instead of falling back to slower generic paths.

### Disaggregated serving and tiered KV offloading
Disaggregated serving separates prefill and decode phases across different GPU pools, requiring fast KV cache transfer via connectors like Nixl and smart memory tiering when VRAM is full. The KV connector subsystem saw major upgrades: the Nixl communicator now supports zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)), handshakes are pipeline-parallel aware ([#43720](https://github.com/vllm-project/vllm/pull/43720)), and object stores can act as a secondary tier for multi-tier KV offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)). It also fixed a critical async KV load deadlock ([#44560](https://github.com/vllm-project/vllm/pull/44560)). Zero-copy transfers and PP-aware handshakes drastically reduce the latency of moving KV caches between nodes, while object store tiering prevents OOMs during long-context decode. Engineers building multi-node, disaggregated clusters should test the new Nixl zero-copy paths and configure object store tiering if pushing context lengths to the limit.

## Kernels & attention
- Split mixed prefill+decode batches for Qwen3.5 to route decodes to a faster recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — optimizes throughput for hybrid architectures.
- Replaced `torch.cat` in ROCm sparse-MLA with a fused `concat_mla_q` kernel ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — eliminates a memory bandwidth bottleneck in AMD's MLA implementation.
- Added a fused MoE W4A16 HIP kernel for AMD RDNA3 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings native quantized MoE execution to consumer and prosumer AMD GPUs.
- Applied single-pass min_larger finding and binary search in the Triton Top-p sampling path ([#42191](https://github.com/vllm-project/vllm/pull/42191)) — reduces CPU/GPU synchronization overhead during generation.

## Quantization
- Added support for compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands vLLM's compressed-tensors plugin to new INT8/INT4 weight and activation schemes.
- Enabled NVIDIA ModelOpt MXFP8 for non-gated MoE models ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — allows microscaling FP8 formats to be used on a wider variety of MoE architectures.
- Added asymmetric scale support for MoE WNA16 Marlin kernels ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — improves quantization accuracy for W4A16 models without sacrificing speed.
- Refactored the compressed-tensors NVFP4 linear implementation into a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — cleans up the codebase ahead of Blackwell hardware availability.

## Parallelism & scheduling
- Made KV connector handshakes pipeline-parallel aware with intermediate-PP output plumbing ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — allows PP nodes to correctly participate in disaggregated prefill/decode transfers.
- Optimized the Nixl communicator with zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)) — drastically reduces CPU overhead and latency when moving KV caches between nodes.
- Added object stores as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — prevents OOMs by spilling KV blocks to system RAM or external storage when GPU/CPU memory is full.
- Avoided pipeline parallel bubbles in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — improves GPU utilization in PP setups by overlapping communication and computation more effectively.

## Model support
- Added native support for Gemma4 Unified, Google's latest encoder-free multimodal architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables text and image generation without separate vision encoders.
- Integrated IBM's Granite Speech Plus model ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — expands vLLM's audio and speech multimodal capabilities.
- Implemented a dedicated video loader for the GLM-4V vision-language model ([#44417](https://github.com/vllm-project/vllm/pull/44417)) — ensures correct temporal processing for video inputs.
- Enabled Multi-Token Prediction (speculative decoding) for Gemma4 in Model Runner V2 ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — accelerates Gemma4 inference using draft models.

## Hardware
- Added a block-scaled W8A8 FP8 linear path for Intel XPUs ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings native FP8 inference to Intel discrete GPUs.
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly accelerates CPU inference on EPYC and Ryzen processors.
- Integrated Aiter hipBLASLt GEMM online tuning for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — automatically finds the fastest GEMM algorithms for AMD GPUs at runtime.
- Enabled shared memory (SHM) communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — adds multi-GPU communication capabilities for IBM Power architectures.

## API & serving
- Unified reasoning and tool-call parsing behind a single `Parser.parse()` interface ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — standardizes how vLLM extracts structured outputs and chain-of-thought from diverse models.
- Fixed a bug to correctly honor `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — prevents models from emitting tool calls when explicitly instructed not to.
- Added dynamic LoRA endpoints to the Rust frontend ([#43778](https://github.com/vllm-project/vllm/pull/43778)) — allows the high-performance Rust server to load and route to LoRA adapters on the fly.
- Implemented batch auto-abort for requests in the Rust frontend ([#44591](https://github.com/vllm-project/vllm/pull/44591)) — improves server stability under heavy load by efficiently cancelling dropped client connections.

## Watch list
- The Nixl connector is initiating a deprecation cycle for the `kv_both` role ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — signals a shift toward explicit `prefill` and `decode` roles in disaggregated setups; plan your migration.
- Ongoing migration of custom all-reduce, DeepSeek V4 fused MLA, and MXFP8 MoE to the libtorch stable ABI ([#44365](https://github.com/vllm-project/vllm/pull/44365)) — a major effort to decouple C++ kernels from PyTorch's unstable ABI, which will reduce wheel fragmentation.
- FlashInfer's B12x experts check now accepts W4A16 ([#43332](https://github.com/vllm-project/vllm/pull/43332)) — hints at upcoming Blackwell-specific MoE kernel optimizations landing in the near future.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (225)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44417">#44417</a> [videoloader] implement glm46v video loader — @JaredforReal</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44707">#44707</a> [Cohere] Enable Cohere Mini Code model and update Command A-plus test registry — @Terrencezzj</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44420">#44420</a> [feature] add index share feature for DSA MTP — @JaredforReal</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44041">#44041</a> [Bugfix] Fix benchmark_moe.py after inplace mechanism removal — @qyYue1389</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44540">#44540</a> [XPU] add xpu branch in compressed_tensors_moe_w4a4_mxfp4 — @zufangzhu</li>
<li><a href="https://github.com/vllm-project/vllm/pull/37149">#37149</a> [XPU][Feature] transparent sleep mode support for XPU platform — @yma11</li>
<li><a href="https://github.com/vllm-project/vllm/pull/36423">#36423</a> [XPU] Support  cpu kv offloading and tiering offloading on XPU platform — @chaojun-zhang</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44699">#44699</a> [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 — @WoosukKwon</li>
<li><a href="https://github.com/vllm-project/vllm/pull/42838">#42838</a> [ROCm][MLA] Replace torch.cat in sparse-MLA forward_mqa with fused concat_mla_q — @maeehart</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44560">#44560</a> [BugFix] Resolve multiple async kv load deadlock — @njhill</li>
<li><a href="https://github.com/vllm-project/vllm/pull/44075">#44075</a> [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) — @JartX</li>
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
<li><em>…and 165 more</em></li>
</ul>
</details>
