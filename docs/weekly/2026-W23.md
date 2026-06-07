# vLLM weekly digest — 2026-06-07 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing DeepSeek-V4 execution, expanding Model Runner V2 to dense models, and pushing the boundaries of multi-tier KV cache offloading. We also saw significant hardware-specific optimizations for AMD Zen CPUs and RDNA3 GPUs, alongside new model support for Gemma4 Unified and Granite Speech Plus. If you are running disaggregated serving, deploying DeepSeek-V4, or utilizing AMD hardware, this week's changes offer substantial performance and reliability improvements.

## Deep dives

### DeepSeek-V4 Sparse MLA and TRT-LLM Attention Kernels
DeepSeek-V4 relies on Sparse Multi-head Latent Attention (MLA), which compresses the KV cache into a low-rank latent space to drastically reduce memory usage but requires specialized kernels to decode efficiently. This week, the team decoupled DS-V4 sparse MLA metadata from older versions ([#44699](https://github.com/vllm-project/vllm/pull/44699)), refactored the core attention module ([#44569](https://github.com/vllm-project/vllm/pull/44569)), and integrated the TensorRT-LLM generation attention kernel ([#43827](https://github.com/vllm-project/vllm/pull/43827)). These changes eliminate eager-mode bottlenecks and leverage highly optimized TRT-LLM kernels for the decode phase, significantly boosting token generation speed for this specific architecture. Anyone deploying DeepSeek-V4 on NVIDIA GPUs is affected and should see immediate decode throughput improvements without needing to change any serving flags.

### Multi-Tier KV Cache Offloading with Object Store Support
KV cache offloading moves inactive context states from GPU VRAM to CPU RAM or disk, allowing the engine to handle far more concurrent long-context requests than VRAM alone permits. vLLM now supports using an object store as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)), alongside new Triton fast-paths for small CPU-to-GPU block swaps ([#42212](https://github.com/vllm-project/vllm/pull/42212)). By pushing cold KV cache to distributed object storage like S3, clusters can retain massive context histories across node restarts or scale-downs without exhausting local CPU RAM. Teams serving extremely long-context workloads or running disaggregated prefill/decode setups are affected; you will need to configure the new object store backend in your KV connector settings to utilize it.

### Model Runner V2 Expands to Dense Models and Fixes PP Bubbles
Model Runner V2 (MRV2) is vLLM's redesigned execution loop that unifies the handling of attention, sampling, and scheduling for higher efficiency and better hardware utilization. MRV2 is now enabled by default for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)), adopts the FlashInfer sampler ([#42472](https://github.com/vllm-project/vllm/pull/42472)), and introduces a fix to avoid pipeline parallel (PP) bubbles ([#42187](https://github.com/vllm-project/vllm/pull/42187)). Bringing dense models to MRV2 standardizes the execution path, while eliminating PP bubbles ensures that multi-GPU pipeline stages don't sit idle waiting for micro-batches, maximizing GPU utilization. Users running Llama or Mistral on multi-GPU setups with Pipeline Parallelism are affected and can expect better throughput and lower inter-token latency with no action required.

## Kernels & attention
- Replaced `torch.cat` in sparse-MLA forward MQA with a fused `concat_mla_q` kernel on ROCm ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — avoids memory-bound concatenation to speed up MLA on AMD GPUs.
- Added a fused MoE W4A16 HIP kernel for AMD RDNA3 / gfx1100 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings fast quantized Mixture-of-Experts inference to consumer and prosumer AMD GPUs.
- Split mixed prefill+decode batches for Qwen3.5 to route decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — optimizes hybrid attention by using the faster recurrent kernel for decode steps.
- Fixed a miscompile in the mHC fused-RMSNorm big-fuse kernel for `hidden_size != 4096` ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — resolves silent correctness issues for models with non-standard dimensions.

## Quantization
- Added support for compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands the compressed-tensors plugin to handle weight-only and weight-activation INT8 formats.
- Enabled MXFP8 quantization for non-gated Mixture-of-Experts layers via NVIDIA ModelOpt ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — unlocks FP8 MoE inference for a wider variety of model architectures.
- Added asymmetric quantization support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows 4-bit weight / 16-bit activation MoE models to use asymmetric scales for better accuracy.
- Refactored the compressed-tensors NVFP4 linear implementation to use a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — cleans up the FP4 codebase for easier maintenance and future kernel additions.

## Parallelism & scheduling
- Added PP-aware handshake aggregation and intermediate-PP output plumbing for the KV connector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — enables Pipeline Parallel stages to correctly participate in disaggregated KV cache transfers.
- Added Mamba prefix caching mode support to the Nixl KV connector ([#42554](https://github.com/vllm-project/vllm/pull/42554)) — allows disaggregated prefill/decode setups to transfer prefix caches for SSM-based models.
- Optimized the Nixl communicator with zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)) — improves KV cache transfer speeds in disaggregated serving by eliminating intermediate memory copies.
- Allowed Data Parallel Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — fixes deterministic hangs in multi-node Ray DP serving by controlling port allocation.

## Model support
- Added native support for Gemma4 Unified, Google's latest encoder-free multimodal architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables text and image understanding without a separate vision encoder.
- Added model support for IBM's Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — brings audio understanding and speech-to-text capabilities to vLLM.
- Unified KDA conv states into a single cache to match the 2-state SSM layout for Mamba ([#44539](https://github.com/vllm-project/vllm/pull/44539)) — standardizes state caching, fixing bugs and simplifying hybrid model support.
- Fixed `OlmoHybridForCausalLM` initialization after upstream checkpoint changes to `rope_parameters` ([#43846](https://github.com/vllm-project/vllm/pull/43846)) — unblocks loading for the latest OLMo Hybrid checkpoints.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — drastically speeds up CPU inference on EPYC/Ryzen by bypassing generic oneDNN fallbacks.
- Added transparent sleep mode support for the Intel XPU platform ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — allows Intel GPUs to enter low-power states when idle, saving energy in multi-tenant clusters.
- Enabled shared memory (SHM) communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — brings shared-memory tensor parallelism to IBM Power architectures.
- Upgraded the `tpu-inference` backend to [v0.21.0](https://github.com/vllm-project/vllm/releases/tag/v0.21.0) ([#44621](https://github.com/vllm-project/vllm/pull/44621)) — keeps the TPU execution path aligned with the latest JAX and TPU runtime improvements.

## API & serving
- Added a Phi-4 mini JSON tool parser to the Rust frontend ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — enables reliable structured tool calling for Microsoft's Phi-4 mini via the high-performance Rust API.
- Added Command A plus tags for structural tags in structured outputs ([#44588](https://github.com/vllm-project/vllm/pull/44588)) — improves constrained generation and JSON formatting for Cohere's Command A models.
- Fixed a bug to honor `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — prevents the engine from emitting tool calls when explicitly disabled by the client.
- Folded developer-role input messages into system instructions in the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns vLLM's API semantics with OpenAI's latest message role specifications.

## Watch list
- Initiated the deprecation cycle for the `kv_both` role in the NixlConnector ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — signals a shift in disaggregated serving; users should migrate to explicit `prefill` and `decode` roles.
- Introduced a pluggable `KVCacheSpec` interface ([#37505](https://github.com/vllm-project/vllm/pull/37505)) — lays the groundwork for custom KV cache layouts, which will be critical for upcoming non-standard attention mechanisms.
- Migrated custom all-reduce, DeepSeek V4 fused MLA, and MXFP8 MoE to the libtorch stable ABI ([#44365](https://github.com/vllm-project/vllm/pull/44365)) — ongoing C++ ABI stabilization that will eventually allow shipping pre-compiled wheels across more PyTorch versions.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (221)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/43707">#43707</a> [Logs Refactor] Optimize shutdown logs, easier to follow and consistent — @yewentao256 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41980">#41980</a> use split_group for pytorch process group creation — @tushar00jain → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43307">#43307</a> [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU — @adobrzyn → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44380">#44380</a> [Bugfix] Fix test_cutlass_moe.py — @bnellnm → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44471">#44471</a> [Misc] Add unit tests for pooler head classes — @taneem-ibrahim → <code>v0.22.1</code></li>
<li><em>…and 161 more</em></li>
</ul>
</details>
