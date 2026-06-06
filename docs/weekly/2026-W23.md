# vLLM weekly digest — 2026-06-06 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on stabilizing and optimizing the newly introduced DeepSeek-V4 architecture, alongside significant maturation of the NIXL KV-connector for disaggregated serving. Hardware support saw a notable boost for AMD Zen CPUs via ZenTorch kernels, while the Rust frontend and Model Runner V2 continue to absorb core serving responsibilities. Overall, it’s a highly practical release for teams pushing the limits of long-context offloading, MoE routing, and multi-node Ray deployments.

## Deep dives

### DeepSeek-V4 Architecture Integration and Optimizations
DeepSeek-V4 introduces a novel attention mechanism and a "Mega MoE" (Mixture of Experts) architecture that requires specialized routing and kernel support to run efficiently. This week saw a massive integration push, adding the TRTLLM generation attention kernel ([#43827](https://github.com/vllm-project/vllm/pull/43827)), optimizing sparse FP8 compressor kernels ([#44161](https://github.com/vllm-project/vllm/pull/44161)), and implementing Expert Parallelism Load Balancing (EPLB) for the Mega MoE ([#43339](https://github.com/vllm-project/vllm/pull/43339)), alongside critical initialization fixes ([#44236](https://github.com/vllm-project/vllm/pull/44236)). Without these specialized kernels and routing logic, DSV4 either fails to initialize or suffers from severe performance bottlenecks during generation. Engineers deploying DeepSeek-V4 on NVIDIA GPUs should upgrade to leverage these performance unlocks and ensure stable model loading.

### Multi-Tier KV Cache Offloading and NIXL Maturation
KV cache offloading moves inactive context blocks from GPU VRAM to slower memory tiers (CPU, disk, or object storage) to prevent OOM errors during long-context or high-concurrency serving, while NIXL is vLLM's KV transfer library used for Prefill-Decode (PD) disaggregation. The offloading subsystem now supports object stores as a secondary tier ([#41968](https://github.com/vllm-project/vllm/pull/41968)), while the NIXL connector gained PP-aware handshake aggregation ([#43720](https://github.com/vllm-project/vllm/pull/43720)) and zero-copy transfer optimizations ([#41633](https://github.com/vllm-project/vllm/pull/41633)). Expanding the memory hierarchy allows serving vastly longer contexts, and making NIXL aware of Pipeline Parallelism (PP) ensures disaggregated serving works correctly across multi-stage pipelines. Teams running long-context workloads, high-throughput serving, or PD disaggregated setups should test these new offloading tiers and NIXL configurations to maximize VRAM utilization.

### AMD Zen CPU Acceleration via ZenTorch
Running LLMs on CPUs is typically bottlenecked by memory bandwidth and compute density, often relying on generic libraries like oneDNN that aren't fully optimized for specific microarchitectures. vLLM now routes W8A8 (int8 dynamic-symmetric) and W4A16 (GPTQ) linear inference through AMD's highly optimized ZenTorch kernels on Zen CPUs, with transparent fallback to oneDNN for other hardware ([#41813](https://github.com/vllm-project/vllm/pull/41813)). This unlocks significantly better performance for quantized models on AMD CPUs, making them a viable option for cost-effective or edge deployments without requiring GPUs. Users deploying vLLM on AMD EPYC or Ryzen CPUs should ensure they are using supported quantization formats (W8A8/W4A16) to automatically benefit from the ZenTorch speedups.

## Kernels & attention
- Cutlass FP8 scaled MM bypasses padding for a 20% kernel performance improvement ([#43706](https://github.com/vllm-project/vllm/pull/43706)) — speeds up FP8 inference on Hopper/Ada.
- Triton MoE backend is now used by default on Hopper GPUs ([#44220](https://github.com/vllm-project/vllm/pull/44220)) — simplifies deployment and improves MoE routing performance.
- Fused RoPE and static Q FP8 quantization on the fused RoPE+KV path for ROCm ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — reduces memory bandwidth overhead on AMD GPUs.
- FlashAttention synced with upstream ([#44065](https://github.com/vllm-project/vllm/pull/44065)) — keeps vLLM aligned with the latest FA3/FA2 optimizations and bug fixes.

## Quantization
- Compressed-tensors now supports WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported quantization formats.
- ModelOpt MXFP8 non-gated MoE is now supported ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — enables NVIDIA's microscaling FP8 format for Mixture-of-Experts models.
- Asymmetric support added for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows more flexible weight quantization schemes for MoE layers.
- XPU block-scaled W8A8 FP8 path added ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings FP8 quantization support to Intel GPUs.

## Parallelism & scheduling
- Multi-node Ray data-parallel serving hang fixed by excluding the Ray DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — resolves critical deadlocks in multi-node setups.
- Pipeline parallel bubbles avoided in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — improves GPU utilization when using Pipeline Parallelism.
- DP Ray placement groups can now be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — gives finer control over resource allocation in Ray clusters.
- Mamba prefix caching mode support added to the NIXL PD connector ([#42554](https://github.com/vllm-project/vllm/pull/42554)) — enables prefill-decode disaggregation for state-space models.

## Model support
- Gemma4 Unified (encoder-free) support added ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — brings native support for Google's latest multimodal architecture.
- JetBrains' Mellum v2 code generation model added ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expands the roster of supported open-weights MoE coding models.
- Gemma4 MTP (Multi-Token Prediction) support added to Model Runner V2 ([#43241](https://github.com/vllm-project/vllm/pull/43241)) — enables speculative decoding for Gemma4.
- HyperCLOVAX loading fixed by registering the model type natively ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — resolves breakages from upstream HuggingFace repo changes.

## Hardware
- AITER hipBLASLt GEMM online tuning integrated for ROCm ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — automatically finds optimal GEMM configurations for AMD GPUs.
- XPU Triton-based selective scan forward op for Mamba added ([#43421](https://github.com/vllm-project/vllm/pull/43421)) — accelerates SSM inference on Intel GPUs.
- SHM communicator support enabled for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — expands CPU-based distributed inference to Power architectures.

## API & serving
- Rust frontend now supports batch auto-aborting requests by engine ([#44591](https://github.com/vllm-project/vllm/pull/44591)) — improves resource reclamation when clients disconnect.
- Tool calling unified behind a single `Parser.parse()` interface ([#44267](https://github.com/vllm-project/vllm/pull/44267)) — cleans up the codebase and standardizes tool-call extraction across models.
- Responses API now folds developer-role input messages into system instructions ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns with OpenAI's latest API conventions for system prompts.
- `tool_choice="none"` is now correctly honored in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixes a bug where tools were still being called when explicitly disabled.

## Watch list
- NIXL KV-connector `kv_both` role is entering a deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this role for PD disaggregation should migrate to the new explicit prefill/decode roles.
- FlashInfer JIT cache installation via `--extra-index-url` removed from Docker builds ([#44366](https://github.com/vllm-project/vllm/pull/44366)) — PyPI quarantine means users must rely on the bundled wheels or build from source.
- Model Runner V2 is being enabled for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) — expect the V2 engine core to become the default for more architectures soon.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (218)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/43625">#43625</a> [ROCm] Bump fastsafetensors to v0.3.2 from PyPI, remove git source build — @wjabbour → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42554">#42554</a> [PD][Nixl] Mamba prefix caching mode support  — @NickLucche → <code>v0.22.1</code></li>
<li><em>…and 158 more</em></li>
</ul>
</details>
