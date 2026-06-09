# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing large-scale MoE serving and expanding hardware support, notably integrating DeepEP v2 for multi-node Expert Parallelism and adding zentorch kernels for AMD Zen CPUs. The engine's memory management takes a major architectural step forward with Pluggable KVCacheSpec, decoupling the scheduler from hardcoded Transformer attention layouts to better support hybrid and state-space models. Meanwhile, the Rust frontend matures with new lifecycle endpoints and structured output fixes, while DeepSeek-V4 initialization and execution paths receive critical bug fixes across NVIDIA, AMD, and Intel hardware.

## Deep dives

### DeepEP v2 integration for Wide Expert Parallelism
Expert Parallelism (EP) distributes Mixture-of-Experts (MoE) layers across multiple GPUs, requiring heavy all-to-all communication to route tokens to the correct experts. DeepEP is a specialized communication library designed to optimize this dispatch and combine phase. PR [#41183](https://github.com/vllm-project/vllm/pull/41183) integrates DeepEP v2 into vLLM's Wide EP implementation, bringing features like `ElasticBuffer` to handle dynamic shapes more efficiently. This matters because it significantly reduces the communication bottleneck when scaling MoE models across multi-node clusters. Engineers serving large MoE models on multi-node setups should test this, but note that it currently requires manually installing NCCL >= 2.30.4 since PyTorch pins an older version.

### Pluggable KVCacheSpec for custom memory layouts
Historically, vLLM's block manager assumed a uniform, monolithic KV cache layout tailored to standard Transformer attention. However, modern architectures like Mamba, hybrid SSMs, and sliding-window models require fundamentally different state management. PR [#37505](https://github.com/vllm-project/vllm/pull/37505) introduces `Pluggable KVCacheSpec`, allowing model implementations to define their own cache allocation specifications and block sizes. This decouples the core scheduling engine from hardcoded attention assumptions, enabling correct memory profiling and allocation for complex architectures. Model contributors adding non-standard state spaces will need to implement this spec, while end-users will see more accurate `max_model_len` calculations for hybrid models.

### Inversion of the MoE execution hierarchy
In vLLM's Mixture-of-Experts implementation, the `FusedMoE` class previously owned the execution logic while `MoERunner` handled specific kernel dispatch, leading to tangled state management during CUDA graph captures. PR [#41184](https://github.com/vllm-project/vllm/pull/41184) inverts this relationship so that `MoERunner` owns the execution (renaming `FusedMoE` to `RoutedExperts`), and the old `FusedMoE` class is removed. This refactor centralizes execution state and capture logic, making it much cleaner to integrate new MoE kernels and manage weight loading without state leakage. Developers working on custom MoE models or kernel backends should update their weight loading paths, as MoE weights now reside under an extra `.experts.routed_experts.` namespace.

## Kernels & attention
- ROCm sparse-MLA replaces `torch.cat` with fused `concat_mla_q` ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — reduces memory and compute overhead in Multi-head Latent Attention on AMD GPUs.
- TRTLLM gen attention kernel added for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) — provides a highly optimized attention path for DSV4 on NVIDIA GPUs.
- CPU attention backend extracts KV-cache updates into a separate method ([#40470](https://github.com/vllm-project/vllm/pull/40470)) — prepares the CPU backend for v1 architecture where KV updates are decoupled from the forward pass.
- DeepSeek-V4 XPU attention decode path added ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — enables DSV4 inference on Intel GPUs.

## Quantization
- Online FP8 per-token activation and per-channel weight (PTPC) quantization added ([#44132](https://github.com/vllm-project/vllm/pull/44132)) — allows dynamic FP8 quantization without requiring offline calibration datasets.
- FP8 weight layout canonicalized to (K, N) at the source ([#44735](https://github.com/vllm-project/vllm/pull/44735)) — fixes silent corruption in square layers by establishing a strict layout contract for FP8 kernels.
- Compressed-tensors adds support for WNA8O8Int linears and WNInt embeddings ([#43440](https://github.com/vllm-project/vllm/pull/43440)) — expands the range of supported quantization schemes via the compressed-tensors format.
- XPU adds a block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings FP8 inference capabilities to Intel GPUs.

## Parallelism & scheduling
- `P2pNcclConnector` removed in favor of the generic ConnectorAPI ([#44854](https://github.com/vllm-project/vllm/pull/44854)) — simplifies the PD disaggregation codebase by deprecating a rigid, early-stage connector.
- PP-aware handshake aggregation and intermediate-PP output plumbing added for KVConnector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — enables Pipeline Parallelism to work seamlessly with Prefill-Decode disaggregation.
- Multi-node Ray data-parallel serving hang fixed by excluding DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — resolves deterministic deadlocks when scaling API servers across nodes.
- Mamba prefix caching mode support added for PD Nixl ([#42554](https://github.com/vllm-project/vllm/pull/42554)) — allows state-space models to leverage prefix caching in disaggregated prefill setups.

## Model support
- JetBrains' Mellum v2 added ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — brings support for the open-weights MoE code-generation model.
- Gemma4 Unified (encoder-free) support added ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables inference for the new unified Gemma4 architecture.
- Granite Speech Plus model support added ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — expands vLLM's audio and speech processing capabilities.
- DeepSeek-V4 CUTLASS `fmin` compatibility issue resolved ([#44236](https://github.com/vllm-project/vllm/pull/44236)) — fixes initialization crashes for DSV4 on newer CUDA toolchains.

## Hardware
- AMD Zen CPUs route W8A8 and W4A16 linear inference through zentorch kernels ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates quantized inference on AMD CPUs with transparent fallback.
- ROCm Fused MoE W4A16 HIP kernel enabled for RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings fast MoE inference to consumer/prosumer AMD GPUs.
- XPU topk/topp Triton `BLOCK_SIZE` capped to 4096 ([#44470](https://github.com/vllm-project/vllm/pull/44470)) — fixes deterministic sampling failures and mask differences on Intel GPUs.
- CPU speculative decoding warns about throughput loss if `libiomp5` is not preloaded ([#44419](https://github.com/vllm-project/vllm/pull/44419)) — prevents a 2x performance drop caused by falling back to GNU libgomp.

## API & serving
- Rust frontend adds `/pause`, `/resume`, and `/is_paused` endpoints ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — provides lifecycle control for RL and admin workflows without restarting the engine.
- Rust frontend sets a structured-output backend to prevent HTTP 500 errors ([#44729](https://github.com/vllm-project/vllm/pull/44729)) — fixes a regression where bypassing the Python processor left the grammar backend unresolved.
- Responses API folds developer-role input messages into system instructions ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — aligns the API with standard system prompt handling for developer messages.
- Benchmark script auto-detects and corrects client/server tokenizer mismatches ([#44708](https://github.com/vllm-project/vllm/pull/44708)) — prevents input token inflation during performance testing when tokenizers drift.

## Watch list
- DeepEP v2 requires manual NCCL upgrade: The integration ([#41183](https://github.com/vllm-project/vllm/pull/41183)) requires NCCL >= 2.30.4, but PyTorch pins 2.28.9, forcing users to manually install the newer NCCL wheel for multi-node MoE.
- NIXL KV-connector wheel normalization: Docker builds now strictly match the NIXL wheel to the image's CUDA major version ([#44266](https://github.com/vllm-project/vllm/pull/44266)), which may affect custom CI pipelines that relied on loose CUDA 12/13 compatibility.
- Pluggable KVCacheSpec adoption: As [#37505](https://github.com/vllm-project/vllm/pull/37505) merges, model contributors must migrate non-standard state spaces (like Mamba or sliding window) to the new spec to ensure correct memory profiling in v1.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (236)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/43844">#43844</a> [Bugfix][MiniCPM-o] Fix cuda/cpu device mismatch in Resampler2_5 pos_embed — by <a href="https://github.com/parthash0804">parthash0804</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44952">#44952</a> [Bugfix][CI] Gemma3 Transformers multimodal encoder profiling and build prompt-embedding fixtures — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43595">#43595</a> fix: prevent MM cache hang from stale LRU order keys — by <a href="https://github.com/jeffye-dev">jeffye-dev</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44729">#44729</a> [Bugfix][Rust Frontend] Set a structured-output backend so requests do not 500 — by <a href="https://github.com/Sunt-ing">Sunt-ing</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42978">#42978</a> [ROCm][MLA][Bugfix] Reserve FP8 prefill workspace before lock for Kimi-K2.5 — by <a href="https://github.com/xaguilar-amd">xaguilar-amd</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44750">#44750</a> [Bugfix] Propagate ImportError from load_audio_pyav when vllm[audio] … — by <a href="https://github.com/littlecircle0730">littlecircle0730</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/40576">#40576</a> [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference  — by <a href="https://github.com/grYe99">grYe99</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44940">#44940</a> [XPU][CI] fix test case path — by <a href="https://github.com/jikunshang">jikunshang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44264">#44264</a> [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention — by <a href="https://github.com/liulanze">liulanze</a></li>
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
<li><em>…and 176 more</em></li>
</ul>
</details>
