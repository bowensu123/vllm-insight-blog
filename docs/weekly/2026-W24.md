# vLLM weekly digest — 2026-06-08 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release and surrounding merges focused heavily on expanding hardware coverage, particularly for Intel XPUs and AMD ROCm, while laying the groundwork for DeepSeek-V4 and MoE execution refactors. Significant infrastructure work went into multi-tier KV cache offloading and Model Runner V2 pipeline parallelism, improving memory efficiency and multi-node utilization. Additionally, the Rust frontend gained crucial lifecycle and tool-calling endpoints, maturing its viability for production serving.

## Deep dives

### MoE Execution Refactor: Inverting the Runner Relationship
Mixture-of-Experts (MoE) models route tokens to specific "expert" sub-networks, which vLLM handles via a `FusedMoE` layer and an `MoERunner` that executes the kernels. In [#41184](https://github.com/vllm-project/vllm/pull/41184), the relationship between these two components is inverted: the `MoERunner` now owns the `FusedMoE` class (which is renamed to `RoutedExperts`), and the old `FusedMoE` wrapper is removed. This refactor cleans up the execution graph and standardizes weight loading paths (e.g., `.experts.routed_experts.<foo>`). Engineers writing custom MoE layers or debugging MoE weight loading will need to update their code to match the new hierarchy, while end-users running DeepSeek or Qwen MoE models will benefit from a more robust foundation for future kernel optimizations.

### Multi-Tier KV Cache Offloading with Object Store Support
When GPU VRAM is exhausted, vLLM can offload KV cache blocks to CPU memory or disk to keep serving long-context requests. PR [#41968](https://github.com/vllm-project/vllm/pull/41968) introduces an object store as a secondary tier for this multi-tier KV cache offloading, complemented by [#44287](https://github.com/vllm-project/vllm/pull/44287) which enables Hybrid Memory Attention (HMA) models for tiered offloading. This matters because it allows seamless integration with distributed storage systems, drastically expanding the effective memory pool for KV caches beyond a single node's RAM. Teams serving high-concurrency, long-context workloads on constrained GPU clusters should evaluate this to reduce request dropping and swapping overhead.

### Online FP8 Per-Token Per-Channel (PTPC) Quantization
FP8 quantization reduces memory bandwidth and increases throughput, but typically requires offline calibration to determine scaling factors for weights and activations. PR [#44132](https://github.com/vllm-project/vllm/pull/44132) adds support for online FP8 PTPC (per-token activation, per-channel weight) quantization, accessible via `--quantization fp8_per_channel`. This allows the engine to compute scaling factors dynamically during inference, eliminating the need for a separate calibration dataset while preserving the fine-grained accuracy of PTPC. This is a major win for developers deploying custom or rapidly updated models who want FP8 performance without the friction of offline calibration pipelines.

## Kernels & attention
- Added XPU-specific decode implementation for DeepSeek-V4 MLA sparse attention, including Triton kernels for FP8 KV cache operations ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — enables Intel GPU support for DeepSeek-V4.
- Replaced `torch.cat` in ROCm sparse-MLA `forward_mqa` with a fused `concat_mla_q` kernel ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — reduces memory bandwidth overhead for MLA on AMD GPUs.
- Decoupled DeepSeek-V4 Sparse MLA metadata from V3.2 ([#44699](https://github.com/vllm-project/vllm/pull/44699)) and refactored KV cache config construction ([#44454](https://github.com/vllm-project/vllm/pull/44454)) — cleans up the attention backend for future DSV4 optimizations.
- Capped Triton `BLOCK_SIZE` to 4096 in `topk_topp_triton.py` on XPU ([#44470](https://github.com/vllm-project/vllm/pull/44470)) — fixes deterministic sampling mask failures on Intel GPUs.

## Quantization
- Added XPU branch for `compressed_tensors_moe_w4a4_mxfp4` ([#44540](https://github.com/vllm-project/vllm/pull/44540)) and block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — expands quantization kernel coverage to Intel GPUs.
- Refactored Compressed Tensors NVFP4 linear to use a single class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) and added asymmetric support for MoE WNA16 marlin ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — simplifies the quantization backend and broadens format support.
- Guarded `per_token_group_fp8_quant` lookup on non-CUDA platforms ([#44476](https://github.com/vllm-project/vllm/pull/44476)) — prevents compilation crashes when deploying FP8 models on alternative hardware.

## Parallelism & scheduling
- Avoided pipeline parallel bubbles in Model Runner V2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) and enabled MRV2 for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) — improves multi-node PP utilization and expands V2 rollout.
- Fixed a deterministic hang in multi-node Ray data-parallel serving by excluding the DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — unblocks large-scale Ray DP deployments.
- Removed the legacy `P2pNcclConnector` ([#44854](https://github.com/vllm-project/vllm/pull/44854)) and initiated deprecation for the `kv_both` role in NixlConnector ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — consolidates the KV connector API around newer NIXL implementations.
- Added PP-aware handshake aggregation and intermediate-PP output plumbing for KV connectors ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — improves prefill-decode disaggregation reliability in pipeline-parallel setups.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) and Gemma4 Unified encoder-free models ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expands coverage for open-weights code generation and multimodal architectures.
- Added model support for Granite Speech Plus ([#43519](https://github.com/vllm-project/vllm/pull/43519)) and implemented a video loader for GLM-4.6V ([#44417](https://github.com/vllm-project/vllm/pull/44417)) — broadens audio and video multimodal capabilities.
- Fixed DeepSeek-V4 initialization by resolving a CUTLASS `fmin` compatibility issue ([#44236](https://github.com/vllm-project/vllm/pull/44236)) and non-mega-moe init errors ([#44356](https://github.com/vllm-project/vllm/pull/44356)) — unblocks DSV4 deployments on newer CUDA toolchains.
- Fixed `OlmoHybridForCausalLM` initialization after upstream checkpoint changes ([#43846](https://github.com/vllm-project/vllm/pull/43846)) and HyperCLOVAX loading after remote code removal ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — restores compatibility with recent HuggingFace updates.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates CPU inference with transparent fallback for non-Zen architectures.
- Added a fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings high-performance MoE inference to consumer/prosumer AMD GPUs.
- Warned about a ~2x throughput drop in CPU speculative decoding when `libiomp5` is not preloaded ([#44419](https://github.com/vllm-project/vllm/pull/44419)) — helps users avoid hidden performance cliffs in custom environments.
- Enabled `permute_cols` for ROCm ([#44674](https://github.com/vllm-project/vllm/pull/44674)) and integrated Aiter hipBLASLt GEMM online tuning ([#40426](https://github.com/vllm-project/vllm/pull/40426)) — continues the maturation of the AMD ROCm kernel stack.

## API & serving
- Added `/pause`, `/resume`, and `/is_paused` endpoints to the Rust frontend ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — provides lifecycle control for RL and admin workflows.
- Auto-detected and corrected client/server tokenizer mismatches in the benchmarking tool ([#44708](https://github.com/vllm-project/vllm/pull/44708)) — prevents artificial input token inflation during performance testing.
- Honored `tool_choice="none"` in Chat Completions streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) and added a Phi-4 mini JSON tool parser to the Rust frontend ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — improves structured output and tool-calling reliability.
- Folded developer-role input messages into system instructions for the Responses API ([#43590](https://github.com/vllm-project/vllm/pull/43590)) and supported system role messages inside the messages array for Anthropic compatibility ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — enhances multi-provider API parity.

## Watch list
- **KV Cache Layout Refactor**: A multi-part refactor is underway to standardize KV cache layouts, pack K/V into the content dim, and drop `get_transfer_cache_regions` ([#44454](https://github.com/vllm-project/vllm/pull/44454), [#44455](https://github.com/vllm-project/vllm/pull/44455), [#44456](https://github.com/vllm-project/vllm/pull/44456)) — expect shifting internal APIs for custom attention backends.
- **Model Runner V2 Expansion**: MRV2 is actively being enabled for dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) and receiving pipeline parallelism fixes ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — users should test V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) as it becomes the default.
- **FlashInfer JIT Cache Quarantine**: Docker builds stopped using `--extra-index-url` for `flashinfer-jit-cache` due to PyPI quarantine ([#44366](https://github.com/vllm-project/vllm/pull/44366)) — monitor FlashInfer installation paths if building custom images.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (237)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/44621">#44621</a> Upgrade tpu-inference to v0.21.0 — by <a href="https://github.com/CienetStingLin">CienetStingLin</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/38804">#38804</a> Fix sarvam forward compatibility with transformers v5 — by <a href="https://github.com/Vikrantpalle">Vikrantpalle</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44648">#44648</a> [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm — by <a href="https://github.com/tjtanaa">tjtanaa</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41968">#41968</a> Add objectstore as a secondary tier to multi-tier kv cache offloading — by <a href="https://github.com/effi-ofer">effi-ofer</a></li>
<li><em>…and 177 more</em></li>
</ul>
</details>
