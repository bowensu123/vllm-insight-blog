# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing large-scale distributed serving and unblocking massive model deployments, notably fixing DeepSeek-V4 initialization and multi-node Ray data-parallel hangs. Hardware coverage expanded with AMD Zen CPU acceleration via ZenTorch and new Intel XPU kernels, while KV cache management saw major upgrades for distributed offloading. Overall, it’s a highly practical release for operators running extreme-scale or heterogeneous hardware clusters.

## Deep dives

### AMD Zen CPU Inference via ZenTorch
CPU inference in vLLM typically relies on generic oneDNN kernels, which can leave performance on the table for specific microarchitectures. This week, W8A8 (int8 dynamic-symmetric) and W4A16 (GPTQ) linear inference on AMD Zen CPUs are now routed through optimized zentorch kernels ([#41813](https://github.com/vllm-project/vllm/pull/41813)). This matters because it significantly accelerates quantized LLM inference on AMD EPYC and Ryzen processors without requiring a GPU, with transparent fallback for other hardware. If you are running CPU-only deployments on AMD Zen architecture, you should see immediate throughput improvements for quantized models without changing your configuration.

### Unblocking DeepSeek-V4 Initialization
DeepSeek-V4 is a massive Mixture-of-Experts model utilizing Multi-head Latent Attention (MLA), making its initialization and memory allocation highly complex. A CUTLASS `fmin` compatibility issue was breaking model initialization, which is now resolved ([#44236](https://github.com/vllm-project/vllm/pull/44236)), alongside refactoring RoPE initialization ([#44262](https://github.com/vllm-project/vllm/pull/44262)) and moving more ops out of the eager breakpoint ([#44561](https://github.com/vllm-project/vllm/pull/44561)). These fixes are critical because they unblock the actual loading and execution of DSV4 on NVIDIA GPUs, which was previously failing on startup. Anyone deploying DeepSeek-V4 should upgrade to this patch to successfully initialize the model and benefit from the cleaned-up attention refactoring ([#44569](https://github.com/vllm-project/vllm/pull/44569)).

### Multi-Tier KV Cache Offloading to Object Stores
KV cache offloading moves key-value states from GPU VRAM to CPU RAM or disk to support longer contexts or higher concurrency. vLLM now supports using an object store as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)). This matters because it allows the engine to spill KV cache beyond the limits of local CPU memory into distributed storage, drastically increasing the effective context window and concurrent request capacity for extreme workloads. Users running very long-context applications or high-throughput serving clusters can now configure remote object stores to prevent OOMs during massive KV cache accumulation.

### Fixing Multi-Node Ray Data-Parallel Hangs
Ray data-parallel (DP) serving allows vLLM to scale across multiple nodes by running several API servers that distribute incoming requests. A deterministic hang occurred when using `num_api_servers > 1` due to deferred port allocation, which is now fixed by excluding the Ray DP backend from this mechanism ([#43864](https://github.com/vllm-project/vllm/pull/43864)) and allowing DP placement groups on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)). This restores stability for large-scale distributed deployments that rely on Ray for orchestration. If you run multi-node Ray clusters with multiple API servers per node, this patch is essential to prevent your servers from deadlocking during startup.

## Kernels & attention
- Fallback to native `ApplyRotaryEmb` on ROCm when the flash_attn rotary grid exceeds the HIP per-dim limit ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — prevents crashes on long sequences.
- Fuse RoPE and static Q FP8 quantization on the fused RoPE+KV path for ROCm ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — reduces memory bandwidth pressure during prefill.
- Add TRT-LLM generation attention kernel for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) and refactor `DeepseekV4Attention` ([#44569](https://github.com/vllm-project/vllm/pull/44569)) — optimizes MLA decoding for NVIDIA GPUs.
- Sync FlashAttention with upstream ([#44065](https://github.com/vllm-project/vllm/pull/44065)) and fix mHC fused-RMSNorm big-fuse miscompile for hidden sizes other than 4096 ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — keeps attention kernels stable and correct.

## Quantization
- Support compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands the range of supported quantization formats for dense models.
- Add asymmetric support for MoE WNA16 Marlin kernels ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — enables more accurate quantization for Mixture-of-Experts models.
- Support ModelOpt MXFP8 non-gated MoE ([#42958](https://github.com/vllm-project/vllm/pull/42958)) and accept W4A16 in FlashInfer B12x Experts ([#43332](https://github.com/vllm-project/vllm/pull/43332)) — broadens hardware and format coverage for MoE inference.
- Refactor compressed-tensors NVFP4 linear to use a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — simplifies the codebase for FP4 quantization paths.

## Parallelism & scheduling
- Avoid pipeline parallel bubbles in ModelRunnerV2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — improves GPU utilization when using pipeline parallelism.
- Optimize NIXL communicator with zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)) and add PP-aware handshake aggregation ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — drastically speeds up KV cache transfers in disaggregated setups.
- Add Mamba prefix caching mode support for PD NIXL ([#42554](https://github.com/vllm-project/vllm/pull/42554)) — enables prefix caching for state-space models in disaggregated serving.
- Fix `sequence_parallel_chunk_impl` custom op aliasing its input ([#44130](https://github.com/vllm-project/vllm/pull/44130)) — prevents silent data corruption when using sequence parallelism.

## Model support
- Add support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) and Gemma4 Unified encoder-free models ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expands coverage for code generation and modern multimodal architectures.
- Add Granite Speech Plus model support ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — brings a new audio/speech model to the vLLM ecosystem.
- Unify KDA conv states into one cache to match 2-state SSM layout ([#44539](https://github.com/vllm-project/vllm/pull/44539)) and refactor Mamba attention module to LINEAR ([#43556](https://github.com/vllm-project/vllm/pull/43556)) — stabilizes and optimizes Mamba execution.
- Fix `OlmoHybridForCausalLM` initialization ([#43846](https://github.com/vllm-project/vllm/pull/43846)) and HyperCLOVAX loading ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — resolves regressions caused by upstream HuggingFace config changes.

## Hardware
- Add XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and Triton-based selective scan forward op for XPU Mamba ([#43421](https://github.com/vllm-project/vllm/pull/43421)) — significantly boosts Intel GPU performance for quantized and SSM models.
- Enable SHM communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — allows shared-memory tensor transfers on IBM Power architectures.
- Use workspace manager for sparse indexer allocations on ROCm ([#41002](https://github.com/vllm-project/vllm/pull/41002)) — improves memory management for sparse attention on AMD GPUs.
- Add Gemma RMS AR fusion for Intel GPUs ([#42646](https://github.com/vllm-project/vllm/pull/42646)) — reduces kernel launch overhead for Gemma models on XPU.

## API & serving
- Honor `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) and fix unstreamed tool call args dropped in Responses API ([#44348](https://github.com/vllm-project/vllm/pull/44348)) — ensures strict adherence to tool calling constraints.
- Add Command A plus tags for structural tags in structured outputs ([#44588](https://github.com/vllm-project/vllm/pull/44588)) — improves JSON/schema generation for specific model families.
- Fold developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns with OpenAI's latest API message role semantics.
- Add Phi-4 mini JSON tool parser to the Rust frontend ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — enables native tool calling for Microsoft's latest small model.

## Watch list
- Initiate deprecation cycle for the `kv_both` role in `NixlConnector` ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on bidirectional KV transfer roles should migrate to the new unidirectional prefill/decode roles.
- Support Pluggable `KVCacheSpec` ([#37505](https://github.com/vllm-project/vllm/pull/37505)) — a massive underlying refactor that abstracts KV cache management, paving the way for custom cache layouts and advanced offloading strategies.
- Migrate custom all-reduce, DeepSeek V4 fused MLA, and MXFP8 MoE to libtorch stable ABI ([#44365](https://github.com/vllm-project/vllm/pull/44365)) — part of the ongoing effort to decouple vLLM C++ extensions from PyTorch's unstable C++ ABI, improving build stability.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (217)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/43827">#43827</a> [DSv4] Adding TRTLLM gen attention kernel — @zyongye → <code>v0.22.1</code></li>
<li><em>…and 157 more</em></li>
</ul>
</details>
