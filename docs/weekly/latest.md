# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing large-scale MoE serving, expanding hardware acceleration, and refining the new Rust frontend. Key highlights include the integration of DeepEP v2 for multi-node Expert Parallelism, native ZenTorch acceleration for AMD CPUs, and robust FP8 weight layout canonicalization. Additionally, the release unblocks DeepSeek-V4 initialization and adds support for new architectures like Mellum v2 and Gemma4 Unified.

## Deep dives

### DeepEP v2 for Wide Expert Parallelism
Expert Parallelism (EP) shards Mixture-of-Experts (MoE) layers across GPUs, and "WideEP" extends this across multiple nodes to support massive models like DeepSeek-V4. DeepEP is a specialized communication library designed to optimize the all-to-all token routing required by MoE. PR [#41183](https://github.com/vllm-project/vllm/pull/41183) integrates DeepEP v2 into vLLM's WideEP implementation, bringing upstream communication and routing improvements to the engine. This reduces the multi-node communication bottleneck for large MoE models, improving overall throughput and scaling efficiency across clusters. Engineers serving large MoE models on multi-node NVIDIA clusters are affected, though they must currently manually install NCCL >= 2.30.4 because PyTorch pins an older version.

### Canonicalizing FP8 Weight Layouts
High-performance FP8 GEMM kernels, such as CUTLASS and Marlin, require weight matrices to be in specific memory layouts like (K, N) to maximize memory bandwidth utilization. PR [#44735](https://github.com/vllm-project/vllm/pull/44735) fixes a bug where square layers were silently corrupted by moving the layout canonicalization to (K, N) at the source during weight processing, rather than relying on fragile shape heuristics inside the kernel. This establishes a strict contract at the kernel boundary, eliminating silent numerical corruption and making it safer to add new FP8 backends. Users running FP8 quantized models, especially on ROCm or with custom kernels, benefit from this transparent bugfix which ensures correct outputs without requiring any configuration changes.

### ZenTorch Acceleration for AMD CPUs
CPU-based LLM inference typically falls back to generic math libraries like oneDNN, which may not fully exploit the specific vector instructions of modern server CPUs. PR [#41813](https://github.com/vllm-project/vllm/pull/41813) routes W8A8 (int8 dynamic) and W4A16 (GPTQ) linear operations through AMD's ZenTorch kernels when running on AMD Zen CPUs, while transparently falling back to oneDNN on other hardware. This unlocks hardware-specific performance gains for CPU inference, reducing latency and increasing throughput for quantized models without requiring manual backend selection. Users deploying quantized models on AMD EPYC or Ryzen CPUs will see automatic acceleration, while those on non-Zen CPUs or GPUs will experience no change in behavior.

## Kernels & attention
- Extracted KV-cache updates from the CPU attention backend into a separate method ([#40470](https://github.com/vllm-project/vllm/pull/40470)) — prepares the CPU backend for more flexible attention execution flows.
- Added an XPU-specific decode path for DeepSeek-V4 MLA sparse attention, including Triton kernels for FP8 KV cache ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — enables efficient DeepSeek-V4 inference on Intel GPUs.
- Integrated the TensorRT-LLM generation attention kernel for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) — provides a highly optimized attention fallback for NVIDIA hardware.
- Adopted the FlashInfer sampler for Model Runner V2 ([#42472](https://github.com/vllm-project/vllm/pull/42472)) — standardizes on a faster, more robust sampling implementation for the new engine core.

## Quantization
- Added online FP8 per-token activation and per-channel weight (PTPC) quantization ([#44132](https://github.com/vllm-project/vllm/pull/44132)) — allows dynamic FP8 quantization without offline calibration via `--quantization fp8_per_channel`.
- Refactored Compressed Tensors NVFP4 linear layers to use a single unified class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — simplifies the codebase and reduces duplication for FP4 weight handling.
- Supported compressed-tensors WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported quantization schemes from the compressed-tensors library.

## Parallelism & scheduling
- Split mixed prefill and decode batches for Qwen3.5, routing decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — prevents prefill chunks from stalling decode throughput in hybrid models.
- Removed the legacy `P2pNcclConnector` for prefill-decode disaggregation ([#44854](https://github.com/vllm-project/vllm/pull/44854)) — cleans up deprecated code now that the more robust ConnectorAPI and NIXL are established.
- Added object store support as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — enables cheaper, higher-capacity KV cache spill-over to distributed storage.
- Avoided pipeline parallel bubbles in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — improves GPU utilization when using pipeline parallelism by overlapping computation and communication.

## Model support
- Added support for JetBrains' Mellum v2, an open-weights MoE code-generation model ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expands vLLM's coverage of specialized coding assistants.
- Introduced Gemma4 Unified (encoder-free) multimodal support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables serving Google's latest unified vision-language architecture.
- Added model support for Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — brings IBM's speech-to-text and audio processing models to the vLLM serving stack.
- Fixed a CUTLASS `fmin` compatibility issue that broke DeepSeek-V4 initialization ([#44236](https://github.com/vllm-project/vllm/pull/44236)) — unblocks deployment of DeepSeek-V4 on newer CUDA toolchains.

## Hardware
- Capped the Triton `BLOCK_SIZE` to 4096 for top-k/top-p sampling on Intel XPU ([#44470](https://github.com/vllm-project/vllm/pull/44470)) — resolves deterministic sampling failures and mask differences on Intel GPUs.
- Added a warning when speculative decoding on CPU lacks `libiomp5` in `LD_PRELOAD` ([#44419](https://github.com/vllm-project/vllm/pull/44419)) — prevents a 2x throughput drop caused by falling back to GNU `libgomp`.
- Enabled transparent sleep mode support for the Intel XPU platform ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — allows XPU workers to release memory when idle, improving multi-tenant resource sharing.
- Normalized NIXL KV-connector wheel installs to match the image's CUDA major version ([#44266](https://github.com/vllm-project/vllm/pull/44266)) — prevents `libcudart.so.12` import errors on CUDA 13 images.

## API & serving
- Added `/pause`, `/resume`, and `/is_paused` endpoints to the Rust frontend ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — provides lifecycle control for RL and admin workflows without restarting the engine.
- Fixed a bug where unstreamed tool call arguments were dropped in the Responses API streaming ([#44348](https://github.com/vllm-project/vllm/pull/44348)) — ensures complete tool payloads are delivered to clients.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — prevents the model from hallucinating tool calls when explicitly instructed not to.
- Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns the API behavior with expected system prompt handling.

## Watch list
- **DeepEP NCCL Pinning**: The DeepEP v2 integration ([#41183](https://github.com/vllm-project/vllm/pull/41183)) requires manually installing NCCL >= 2.30.4 because PyTorch pins an older version; watch for PyTorch updates to resolve this friction.
- **KV Cache Layout Refactor**: A multi-part refactor of the KV cache layout is underway ([#44454](https://github.com/vllm-project/vllm/pull/44454)) to standardize Mamba and attention cache formats, which may affect custom connector developers.
- **NixlConnector `kv_both` Deprecation**: The `kv_both` role in NixlConnector is entering its deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)); users relying on this for prefill-decode disaggregation should migrate to explicit roles.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (235)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44929">#44929</a> [Docs] Remove broken link to deleted disaggregated_prefill.sh — by <a href="https://github.com/liulanze">liulanze</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41183">#41183</a> [WideEP] Integrate DeepEP v2 — by <a href="https://github.com/tlrmchlsmth">tlrmchlsmth</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44809">#44809</a> [ROCm][CI] Re-route NixlConnector jobs — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
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
<li><em>…and 175 more</em></li>
</ul>
</details>
