# vLLM weekly digest — 2026-06-08 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing the V2 engine, expanding hardware support for Intel XPU and AMD ROCm, and optimizing DeepSeek-V4 inference. Key advancements include multi-tier KV cache offloading to object storage, pipeline parallelism bubble avoidance, and speculative decoding speedups for sparse attention. Overall, it's a strong week for operators scaling long-context workloads and deploying large MoE models across heterogeneous hardware.

## Deep dives

### DeepSeek-V4 Sparse MLA Index Sharing for MTP
DeepSeek's Sparse Multi-Head Latent Attention (MLA) uses a computationally expensive indexer to select top-k KV blocks, which adds overhead during Multi-Token Prediction (MTP) speculative decoding. PR [#44420](https://github.com/vllm-project/vllm/pull/44420) implements "IndexCache" to reuse these top-k indices across MTP steps and layers, safely skipping the indexer when the context hasn't shifted significantly. This matters because it drastically reduces the sparse indexing overhead during speculative generation, unlocking faster decode times for DeepSeek models. Engineers running DeepSeek-V3 or V4 with MTP enabled should test this update to observe throughput improvements in speculative decoding workloads.

### Multi-Tier KV Cache Offloading to Object Storage
KV cache offloading traditionally moves inactive context from GPU VRAM to CPU RAM to serve more concurrent requests, but RAM is still a finite bottleneck. PR [#41968](https://github.com/vllm-project/vllm/pull/41968) introduces an object store as a secondary tier for KV cache offloading, while [#44287](https://github.com/vllm-project/vllm/pull/44287) extends this tiered offloading support to Hybrid Mamba-Attention (HMA) models. This matters because it allows the engine to treat distributed object storage as a massive extension of the KV cache, preventing OOMs and request rejections during extreme long-context or high-concurrency bursts. Operators serving massive batch sizes or ultra-long contexts who have fast network backends should evaluate this to expand their effective serving capacity.

### Pipeline Parallelism Bubble Avoidance in V2 Engine
In Pipeline Parallelism (PP), "bubbles" occur when some devices sit idle waiting for others to finish processing their microbatches, wasting valuable compute. PR [#42187](https://github.com/vllm-project/vllm/pull/42187) updates the V2 Model Runner's scheduling logic to route microbatches in a way that keeps all pipeline stages continuously busy. This matters because it directly increases overall GPU utilization and throughput for large models that must be split across multiple devices. Anyone deploying 70B+ parameter models across multiple GPUs or nodes using `--pipeline-parallel-size` will see improved hardware efficiency and higher token generation rates.

## Kernels & attention
- Fused `concat_mla_q` for ROCm sparse-MLA ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — eliminates 61 tensor allocations and kernel launches per decode step for DeepSeek-V3.2 on MI355X.
- DeepSeek-V4 XPU attention decode path ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — adds Triton kernels for FP8 KV cache and sparse MLA decode on Intel XPUs.
- Fused MoE W4A16 HIP kernel for AMD RDNA3 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings optimized W4A16 MoE execution to gfx1100 consumer/pro GPUs.
- XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — enables FP8 block-scaled linear layers on Intel GPUs.

## Quantization
- zentorch-accelerated W8A8 and W4A16 on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — routes quantized linear inference through optimized CPU kernels with transparent fallback.
- Compressed-tensors WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands support for mixed-precision weight-only and weight-activation quantization schemes.
- Asymmetric support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allows asymmetric quantization in compressed-tensors for MoE models using the Marlin backend.
- ModelOpt MXFP8 non-gated MoE support ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — enables NVIDIA's ModelOpt MXFP8 format for non-gated Mixture-of-Experts layers.

## Parallelism & scheduling
- Multi-node Ray data-parallel hang fix ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — resolves deterministic hangs by excluding the Ray DP backend from deferred port allocation.
- Async KV load deadlock fix ([#44560](https://github.com/vllm-project/vllm/pull/44560)) — prevents deadlocks by throttling async KV loads that would occupy blocks needed for in-flight chunked prefills.
- PP-aware handshake for KV Connectors ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — improves disaggregated prefill/decode routing in pipeline-parallel setups.
- Split mixed prefill+decode batches for Qwen3.5 ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — routes decodes to the recurrent kernel to optimize mixed-batch execution.

## Model support
- JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — adds support for the open-weights MoE code-generation model.
- Gemma4 Unified (encoder-free) support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — integrates the new unified architecture variant of Gemma 4.
- Cohere North-Mini-Code ([#44707](https://github.com/vllm-project/vllm/pull/44707)) — enables the Cohere Mini Code model with tool-calling and reasoning parsers.
- FunASR-Nano initialization crash fix ([#44215](https://github.com/vllm-project/vllm/pull/44215)) — resolves a crash caused by unwrapping non-MoE multimodal models for EPLB.

## Hardware
- XPU transparent sleep mode ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — introduces memory allocator sleep/wake interfaces for Intel GPUs to match CUDA parity.
- CPU KV offloading and tiering on XPU ([#36423](https://github.com/vllm-project/vllm/pull/36423)) — enables swapping KV cache blocks to CPU memory on Intel XPU platforms.
- ROCm Aiter hipBLASLt GEMM online tuning ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — integrates dynamic GEMM tuning for AMD GPUs to optimize matrix multiplications.
- Fused RoPE + static Q FP8 quant on ROCm ([#42832](https://github.com/vllm-project/vllm/pull/42832)) — optimizes the fused RoPE and KV cache path for GPT-OSS models on AMD hardware.

## API & serving
- Rust frontend lifecycle APIs ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — adds `/pause`, `/resume`, and `/is_paused` endpoints for better RL and admin control.
- Rust frontend dynamic LoRA endpoints ([#43778](https://github.com/vllm-project/vllm/pull/43778)) — allows loading and unloading LoRA adapters dynamically via the Rust API.
- Anthropic system role messages inside messages array ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — improves compatibility with Anthropic's API formatting for system prompts.
- Batch auto-abort requests by engine in Rust frontend ([#44591](https://github.com/vllm-project/vllm/pull/44591)) — improves efficiency when canceling multiple in-flight requests.

## Watch list
- NixlConnector `kv_both` role deprecation ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — initiates the deprecation cycle for the `kv_both` role; plan to migrate to explicit prefill/decode roles.
- KV-Cache Layout Refactor ([#44454](https://github.com/vllm-project/vllm/pull/44454)) — begins a multi-part refactor to standardize KV cache layouts and pack K/V into the content dim across attention backends.
- Pluggable KVCacheSpec ([#37505](https://github.com/vllm-project/vllm/pull/37505)) — introduces a pluggable specification for KV cache, paving the way for more flexible memory management and custom cache layouts.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (235)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44470">#44470</a> [XPU] Cap topk/topp Triton BLOCK_SIZE to 4096 to fix Top-p mask difference failures — by <a href="https://github.com/chaojun-zhang">chaojun-zhang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44499">#44499</a> [Rust Frontend] Add /pause, /resume, /is_paused endpoints — by <a href="https://github.com/sahilsGit">sahilsGit</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44828">#44828</a> [BugFix] Use served model name in gemma4 audio-tower error message — by <a href="https://github.com/llsj14">llsj14</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43663">#43663</a> [XPU][CI] Add more test cases in Intel GPU CI — by <a href="https://github.com/zxd1997066">zxd1997066</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44761">#44761</a> [ROCm][CI] Stabilizing teardown and timeout of flaky tests to prevent rare OOMs — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42793">#42793</a> [ROCm][CI] Stage C mirrors — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44771">#44771</a> [XPU][Minor] format moe kernel name and add in kernel list — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44484">#44484</a> [MM][CG] Simplify ViT CUDA graph interfaces — by <a href="https://github.com/shen-shanshan">shen-shanshan</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42953">#42953</a> feat: add DeepSeek-V4 XPU attention decode path — by <a href="https://github.com/majian4work">majian4work</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/39562">#39562</a> [Bugfix]: Fix assertion in MambaManager.allocate_slots() — by <a href="https://github.com/Holworth">Holworth</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44805">#44805</a> Added extra_repr() to pooler classes to improve debuggability — by <a href="https://github.com/taneem-ibrahim">taneem-ibrahim</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44215">#44215</a> [Bugfix] Fix FunASR-Nano crash during initialization — by <a href="https://github.com/SunskyXH">SunskyXH</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42736">#42736</a> [Kernel][Test] Make kernel tests for mamba dual-HW (CUDA + XPU) — by <a href="https://github.com/adobrzyn">adobrzyn</a></li>
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
<li><em>…and 175 more</em></li>
</ul>
</details>
