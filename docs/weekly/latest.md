# vLLM weekly digest — 2026-06-08 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing complex architectures like DeepSeek-V4 and Gemma4 while pushing hardware-specific optimizations for AMD Zen CPUs and Intel XPUs. The Rust frontend matured significantly with new lifecycle and tool-calling endpoints, making it more viable for reinforcement learning and agentic workflows. Additionally, critical bug fixes in FP8 weight layouts and CPU speculative decoding runtimes prevent silent accuracy corruption and severe throughput drops.

## Deep dives

### FP8 Weight Layout Canonicalization
FP8 inference relies on specific memory layouts for matrix multiplication, but differing expectations between kernels can lead to silent data corruption. PR [#44735](https://github.com/vllm-project/vllm/pull/44735) fixes a bug where square FP8 layers were silently corrupted by the Marlin kernel by canonicalizing the weight layout to `(K, N)` at the source rather than relying on fragile shape heuristics. This ensures mathematical correctness across all FP8 linear layers regardless of their dimensions, while PR [#44132](https://github.com/vllm-project/vllm/pull/44132) simultaneously introduces online FP8 per-token activation and per-channel weight (PTPC) quantization to expand available recipes. Engineers running FP8 models, especially those with square hidden dimensions, should upgrade to prevent silent accuracy degradation, and those seeking higher precision FP8 can test the new `--quantization fp8_per_channel` flag.

### DeepSeek-V4 Sparse MLA and MoE Refactoring
DeepSeek-V4 utilizes Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE), which require highly optimized routing and memory layouts to achieve high throughput. This week, PR [#41184](https://github.com/vllm-project/vllm/pull/41184) refactored the MoE subsystem by inverting the relationship between `MoERunner` and `FusedMoE` (now `RoutedExperts`), centralizing expert mapping and capture state. Concurrently, PR [#44699](https://github.com/vllm-project/vllm/pull/44699) decoupled DeepSeek-V4 Sparse MLA metadata from V3.2, and PR [#43827](https://github.com/vllm-project/vllm/pull/43827) integrated TRT-LLM generation attention kernels specifically for DS-V4. These changes clean up internal abstractions for complex architectures while unlocking faster decode paths; teams deploying DeepSeek-V4 or custom MoE models will benefit from the optimized attention kernels, though custom MoE implementations may need to update their weight-loading paths to match the new `.experts.routed_experts` hierarchy.

### CPU Speculative Decoding and OpenMP Runtimes
Speculative decoding on CPU relies heavily on frequent, short parallel regions to evaluate draft tokens quickly, making it highly sensitive to the underlying OpenMP implementation. PR [#44419](https://github.com/vllm-project/vllm/pull/44419) addresses a mysterious 2x throughput drop in CPU speculative decoding by identifying that the GNU `libgomp` runtime handles these short regions much less efficiently than Intel's `libiomp5`. The PR adds an explicit warning when `libiomp5` is not preloaded, explaining why official Docker images outperform standard conda/venv source builds. Engineers running speculative decoding on Intel CPUs must ensure `libiomp5` is in their `LD_PRELOAD` path to avoid leaving half their throughput on the table.

### Rust Frontend Lifecycle and Tool Calling
The Rust frontend in vLLM is designed to provide a high-performance, low-latency alternative to the Python API server, particularly for reinforcement learning and complex agentic workflows. PR [#44499](https://github.com/vllm-project/vllm/pull/44499) introduces `/pause`, `/resume`, and `/is_paused` endpoints, allowing external controllers to halt the scheduler without tearing down the engine. Furthermore, PR [#44213](https://github.com/vllm-project/vllm/pull/44213) adds a native JSON tool parser for Phi-4 mini, and PR [#43778](https://github.com/vllm-project/vllm/pull/43778) introduces dynamic LoRA endpoints. These additions make the Rust frontend viable for production RL environments and tool-calling agents, so developers building custom orchestration layers should explore these new lifecycle hooks to manage engine state more gracefully.

## Kernels & attention
- Replaced `torch.cat` with a fused `concat_mla_q` in the ROCm sparse-MLA forward pass ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — reduces memory bandwidth overhead for MLA on AMD GPUs.
- Routed Qwen3.5 mixed prefill+decode batches to split decodes into the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — optimizes scheduling for hybrid architectures.
- Capped Triton `BLOCK_SIZE` to 4096 for Top-p sampling on XPU ([#44470](https://github.com/vllm-project/vllm/pull/44470)) — fixes deterministic sampling mask failures on Intel GPUs.

## Quantization
- Added block-scaled W8A8 FP8 paths for XPU ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and unified XPU MoE kernel formats for FP8/MXFP8 ([#44771](https://github.com/vllm-project/vllm/pull/44771)) — expands quantization support on Intel hardware.
- Refactored compressed-tensors NVFP4 linear layers into a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) and added asymmetric support for MoE WNA16 Marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — cleans up quantization backend logic.
- Supported compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — broadens the range of supported integer quantization schemes.

## Parallelism & scheduling
- Added PP-aware handshake aggregation and intermediate-PP output plumbing for KV connectors ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — enables robust KV transfer in pipeline-parallel setups.
- Supported selective prefix-cache retention for sliding-window KV cache in DeepSeek-V4 ([#43447](https://github.com/vllm-project/vllm/pull/43447)) — improves cache hit rates for long-context sliding window models.
- Removed the legacy `P2pNcclConnector` ([#44854](https://github.com/vllm-project/vllm/pull/44854)) — consolidates PD disaggregation efforts around the newer ConnectorAPI and NIXL.
- Allowed data-parallel Ray placement groups to be set on specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — improves resource pinning in multi-node clusters.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) and Gemma4 Unified encoder-free models ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expands coverage for open-weights code generation and multimodal architectures.
- Fixed DeepSeek-V4 initialization by resolving a CUTLASS `fmin` compatibility issue ([#44236](https://github.com/vllm-project/vllm/pull/44236)) and non-mega-moe init errors ([#44356](https://github.com/vllm-project/vllm/pull/44356)) — unblocks DS-V4 deployments on latest CUDA.
- Fixed FunASR-Nano initialization crashes ([#44215](https://github.com/vllm-project/vllm/pull/44215)) and HyperCLOVAX loading after upstream HuggingFace repo changes ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — restores stability for audio and vision models.
- Added MTP support for Gemma4 ([#43241](https://github.com/vllm-project/vllm/pull/43241)) and fixed block table batch size mismatches under concurrent load ([#43982](https://github.com/vllm-project/vllm/pull/43982)) — enables speculative decoding for Gemma4.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — significantly accelerates quantized inference on AMD EPYC/Ryzen processors.
- Added a fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings high-performance MoE inference to consumer/prosumer AMD GPUs.
- Enabled `permute_cols` ([#44674](https://github.com/vllm-project/vllm/pull/44674)) and integrated Aiter hipBLASLt GEMM online tuning ([#40426](https://github.com/vllm-project/vllm/pull/40426)) for ROCm — optimizes low-level matrix operations on AMD Instinct accelerators.
- Added XPU-specific decode implementations and Triton kernels for DeepSeek-V4 MLA sparse attention ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — extends advanced attention mechanisms to Intel GPUs.

## API & serving
- Auto-detected and corrected client/server tokenizer mismatches in the benchmarking suite ([#44708](https://github.com/vllm-project/vllm/pull/44708)) — prevents artificial input token inflation during performance testing.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) and folded developer-role messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — improves OpenAI API compatibility.
- Added Command A plus tags for structural tags in structured outputs ([#44588](https://github.com/vllm-project/vllm/pull/44588)) — expands grammar-based generation support for newer Cohere models.

## Watch list
- **KV Cache Layout Refactor**: A multi-part refactor ([#44454](https://github.com/vllm-project/vllm/pull/44454), [#44455](https://github.com/vllm-project/vllm/pull/44455), [#44456](https://github.com/vllm-project/vllm/pull/44456), [#44458](https://github.com/vllm-project/vllm/pull/44458)) is underway to standardize KV cache layouts and pack K/V into the content dim, which will deeply affect custom attention implementations.
- **NIXL KV-Connector Deprecations**: The `kv_both` role in NixlConnector is entering a deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)), and legacy NCCL connectors are being removed ([#44854](https://github.com/vllm-project/vllm/pull/44854)), signaling a hard pivot to NIXL for PD disaggregation.
- **Usage Stats Expansion**: vLLM now reports more granular engine, spec-decode, and expert-parallel configs in aggregate usage stats ([#44595](https://github.com/vllm-project/vllm/pull/44595)); privacy-conscious deployments should review the new fields to ensure compliance with internal telemetry policies.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (238)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44595">#44595</a> [Misc] usage_stats: report more engine, spec-decode, and EP config — by <a href="https://github.com/zlxi02">zlxi02</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44856">#44856</a> [Rust Frontend] [Refactor] Refine utility call interfaces — by <a href="https://github.com/BugenZhao">BugenZhao</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44735">#44735</a> [Bugfix] Canonicalize FP8 weight layout to (K, N) at the source — by <a href="https://github.com/mgoin">mgoin</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44897">#44897</a> [Bugfix][MoE] Fix fused MoE expert mapping helper call sites — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44450">#44450</a> [Model Runner V2] Fix mrv2 mm lora issue — by <a href="https://github.com/yewentao256">yewentao256</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/40470">#40470</a> [Attention] Extract KV-cache update from CPU attention backend — by <a href="https://github.com/dmaniloff">dmaniloff</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41184">#41184</a> [MoE Refactor] FusedMoE/MoERunner inversion refactor — by <a href="https://github.com/bnellnm">bnellnm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44132">#44132</a> [Quantization] add online fp8 ptpc — by <a href="https://github.com/walterbm">walterbm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44708">#44708</a> [Benchmark] Auto-detect and correct client/server tokenizer mismatch for random dataset — by <a href="https://github.com/akii96">akii96</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44819">#44819</a> [CI] Consolidate multimodal entrypoint tests. — by <a href="https://github.com/noooop">noooop</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44852">#44852</a> [CI/Build][CPU] Fix flaky CI image build failure and unexpected warnings — by <a href="https://github.com/bigPYJ1151">bigPYJ1151</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44854">#44854</a> [Connector] Remove `P2pNcclConnector` — by <a href="https://github.com/NickLucche">NickLucche</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44419">#44419</a> [CPU][Spec Decode] Warn about throughput loss when libiomp5 is not preloaded — by <a href="https://github.com/jmamou">jmamou</a></li>
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
<li><em>…and 178 more</em></li>
</ul>
</details>
