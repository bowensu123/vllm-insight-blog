# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on scaling DeepSeek-V4 and optimizing the new Model Runner V2, notably by eliminating pipeline parallelism bubbles and introducing Expert-Parallelism Load Balancing for Mega MoE. Hardware support saw significant boosts for AMD Zen CPUs and RDNA3 GPUs via new zentorch and HIP kernels, while multi-tier KV cache offloading was expanded to include object stores for extreme long-context workloads. Overall, it is a strong week for multi-node efficiency, MoE routing, and expanding the hardware footprint beyond Nvidia.

## Deep dives

### DeepSeek-V4 Mega MoE Load Balancing (EPLB)
In Mixture-of-Experts (MoE) models, token routing is often imbalanced, meaning some GPU ranks process significantly more tokens for their assigned experts than others, creating a "straggler" effect that bottlenecks the whole system. Expert-Parallelism Load Balancing (EPLB) solves this by dynamically duplicating heavily loaded experts across underutilized ranks to distribute the compute evenly. PR [#43339](https://github.com/vllm-project/vllm/pull/43339) introduces EPLB support specifically for the DeepSeek-V4 Mega MoE architecture. This matters because it maximizes GPU utilization during expert-parallel inference, preventing fast ranks from idling while waiting for the busiest rank to finish its expert computations. Teams serving DeepSeek-V4 at scale using Expert Parallelism should test this feature to see if it reduces tail latency and improves overall throughput on their specific traffic patterns.

### Eliminating Pipeline Parallelism Bubbles in Model Runner V2
In Pipeline Parallelism (PP), a model is split across multiple GPUs sequentially, which naturally creates "bubbles" where a GPU sits idle waiting for the previous stage to finish processing a micro-batch. These bubbles severely degrade multi-node efficiency and waste expensive compute. PR [#42187](https://github.com/vllm-project/vllm/pull/42187) refactors Model Runner V2 to avoid these pipeline parallel bubbles by optimizing micro-batch scheduling and overlapping communication. This drastically improves the hardware utilization of PP deployments, making multi-node inference much more viable for massive models without sacrificing latency. Anyone running large models across multiple nodes or using PP > 1 should upgrade to [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) and use the V2 engine to yield immediate efficiency gains.

### Multi-Tier KV Cache Offloading to Object Stores
When GPU memory is exhausted, vLLM can offload KV cache to CPU RAM, but for very long contexts or disaggregated prefill/decode setups, even CPU RAM isn't enough. To prevent out-of-memory crashes, the engine needs a third tier like disk or distributed object storage to spill state. PR [#41968](https://github.com/vllm-project/vllm/pull/41968) adds an object store as a secondary tier to the multi-tier KV cache offloading system, while PR [#42212](https://github.com/vllm-project/vllm/pull/42212) adds a Triton fast-path for small CPU-to-GPU block swaps to speed up the retrieval. This enables serving extremely long-context requests or handling massive concurrent prefix-caching workloads by spilling state to cheap, abundant storage. Users running long-context workloads or heavy prefix-caching can now configure an object store backend to prevent OOMs during extreme memory pressure.

## Kernels & attention
- Added TensorRT-LLM generation attention kernel for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) — accelerates the decode phase for this specific architecture.
- Split mixed prefill+decode batches for Qwen3.5 to route decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — improves Qwen3.5 decode performance.
- Defaulted to the Triton MoE backend on Hopper GPUs ([#44220](https://github.com/vllm-project/vllm/pull/44220)) — simplifies configuration and leverages Hopper-specific optimizations.
- Enabled fused kernels for Gated Delta Networks (GDN) on CPU ([#43534](https://github.com/vllm-project/vllm/pull/43534)) — boosts CPU inference performance for state-space models.

## Quantization
- Supported compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands the range of supported INT8/INT4 weight-activation quantization schemes.
- Added support for ModelOpt MXFP8 non-gated MoE ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — enables FP8 inference for specific Mixture-of-Experts variants.
- Added asymmetric support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows asymmetric quantization for 16-bit activation MoE models using the Marlin kernel.
- Added XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings FP8 quantization support to Intel XPU hardware.

## Parallelism & scheduling
- Added Mamba prefix caching mode support to the Nixl PD connector ([#42554](https://github.com/vllm-project/vllm/pull/42554)) — enables state-space model prefix caching in disaggregated setups.
- Optimized the Nixl communicator with zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)) — reduces overhead in Expert Parallelism Load Balancing and KV transfers.
- Supported selective prefix-cache retention for DeepSeek-V4's sliding-window KV cache ([#43447](https://github.com/vllm-project/vllm/pull/43447)) — prevents cache eviction issues specific to sliding-window attention.
- Allowed Data Parallel Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — fixes multi-node Ray DP scheduling and prevents deterministic hangs.

## Model support
- Added support for Gemma4 Unified (encoder-free) multimodal architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expands vision-language model coverage.
- Added Gemma4 Multi-Token Prediction (MTP) support in Model Runner V2 ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — enables speculative decoding for Gemma4.
- Added model support for Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — brings a new audio/speech model to the engine.
- Added JetBrains' Mellum v2 open-weights MoE code-generation model ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expands coverage for specialized coding tasks.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly accelerates CPU inference on AMD hardware.
- Added a fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings fast quantized MoE inference to consumer/prosumer AMD GPUs.
- Enabled SHM (Shared Memory) communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — expands hardware compatibility to PowerPC architectures.
- Enabled RMSNorm and activation quantization fusions on Intel XPU ([#43963](https://github.com/vllm-project/vllm/pull/43963)) — improves performance on Intel accelerators.

## API & serving
- Fixed a bug to properly honor `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — ensures tools aren't hallucinated when explicitly disabled.
- Added Command A plus tags for structural tags in Reasoning/Structured Outputs ([#44588](https://github.com/vllm-project/vllm/pull/44588)) — improves structured generation for specific model families.
- Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns with OpenAI's latest API message role handling.
- Supported system role messages inside the messages array for the Anthropic API compatibility layer ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — improves drop-in replacement fidelity.

## Watch list
- Initiated the deprecation cycle for the `kv_both` role in NixlConnector ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this for PD disaggregation should migrate to the new role definitions.
- Stopped installing `flashinfer-jit-cache` via `--extra-index-url` in Docker due to PyPI quarantine ([#44366](https://github.com/vllm-project/vllm/pull/44366)) — custom Docker builds relying on this cache may need to adjust their build steps.
- Removed KV cache scale boilerplate from model weight loading methods ([#43167](https://github.com/vllm-project/vllm/pull/43167)) — a large-scale refactor that might affect custom model implementations relying on the old scaling hooks.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (216)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/34894">#34894</a> [DOC] Add INT8 W4A8 docs and Arm&#x27;s supported quantization schemes — @fadara01 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44436">#44436</a> [ROCm][CI] Add test for Aiter unified attn kernel — @divakar-amd → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44057">#44057</a> [Bugfix] Reject non-positive values for ParallelConfig int knobs — @jwzheng96 → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44363">#44363</a> [Core] Freeze garbage collector in workers after model initialization — @tlrmchlsmth → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44509">#44509</a> [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count — @tc-mb → <code>v0.22.1</code></li>
<li><em>…and 156 more</em></li>
</ul>
</details>
