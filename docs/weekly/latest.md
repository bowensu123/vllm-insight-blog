# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focused on stabilizing multi-node serving, expanding hardware support for AMD and Intel, and maturing the experimental Rust frontend. Significant performance wins were landed for FP8 MoE models on ROCm and Qwen3-Next on H100, while DeepEP v2 integration pushed Wide Expert Parallelism forward. Engineers running disaggregated prefill or custom KV routing should pay close attention to a breaking schema change in ZMQ cache events.

## Deep dives

### DeepEP v2 for Wide Expert Parallelism
Expert Parallelism (EP) distributes Mixture-of-Experts (MoE) layers across multiple GPUs, and "Wide EP" extends this across multiple nodes to support massive models like DeepSeek-V3. PR [#41183](https://github.com/vllm-project/vllm/pull/41183) integrates DeepEP v2, an optimized communication library for MoE all-to-all token dispatch and combine. This upgrade improves multi-node MoE scaling, though it currently requires users to manually install NCCL >= 2.30.4 because PyTorch pins an older version. It matters because it reduces the communication bottleneck that typically limits multi-node MoE throughput. Engineers running large MoE models across multiple nodes should test this, but be aware of the manual NCCL upgrade and a known limitation on 8xB200 NVLink-only setups.

### ROCm Fused All-Reduce, RMSNorm, and FP8 Quantization
In FP8 blockwise quantized models like DeepSeek-V3.2, every transformer block ends with a sequence of an all-reduce, RMSNorm, and per-group FP8 quantization. Executing these as separate kernels incurs significant memory bandwidth overhead and launch latency. PR [#42864](https://github.com/vllm-project/vllm/pull/42864) fuses this entire chain into a single AITER call on AMD GPUs, eliminating roughly 535 microseconds per decode step on MI355X with TP4. This matters because it drastically reduces decode latency, directly increasing generation throughput for FP8 MoE workloads. AMD users serving DeepSeek-V3.2 or similar architectures will see immediate performance gains without needing to change any flags.

### Rust Frontend Reaches Production Parity
vLLM's experimental Rust frontend is designed to handle HTTP routing and tokenization with much lower overhead than the Python FastAPI server. This week, a wave of PRs brought it closer to feature parity: [#44321](https://github.com/vllm-project/vllm/pull/44321) added API key authentication, [#44222](https://github.com/vllm-project/vllm/pull/44222) introduced `/tokenize` and `/detokenize` endpoints, and [#44901](https://github.com/vllm-project/vllm/pull/44901) added support for Kimi K2 tool-call IDs. Additionally, [#44729](https://github.com/vllm-project/vllm/pull/44729) fixed a critical bug where structured output requests returned HTTP 500 errors. This matters because it makes the Rust frontend viable for secure, production-grade deployments that rely on tool calling and structured generation. Engineers evaluating the Rust frontend can now rely on it for standard API security and complex agentic workflows.

### KV Cache Event Schema Switches to Map Encoding
When using disaggregated prefill or external KV cache managers, vLLM publishes KV cache events over ZMQ to coordinate block allocation and transfers. Historically, these events used a positional array format, meaning any schema addition risked breaking downstream subscribers that hadn't been updated. PR [#42892](https://github.com/vllm-project/vllm/pull/42892) migrates this encoding to a map (dictionary) format, making the schema extensible and backward-compatible. This matters because it stabilizes the integration contract for external routing daemons and cache managers, preventing silent failures during vLLM upgrades. If you maintain custom KV routing or disaggregation infrastructure, you must update your ZMQ subscribers to parse map-encoded msgpack payloads.

## Kernels & attention
- Fused QK RMSNorm, RoPE, and gate copy into a single kernel for Qwen3.5 ([#44176](https://github.com/vllm-project/vllm/pull/44176)) — reduces kernel launch overhead for a modest throughput bump.
- Fixed a crash in `minimax_qk_norm_fusion` when FlashInfer's AllReduce is enabled ([#44983](https://github.com/vllm-project/vllm/pull/44983)) — resolves FP32 variance incompatibility that broke TP serving.
- Added TRT-LLM generation attention kernel support for DeepSeek-V4 ([#43827](https://github.com/vllm-project/vllm/pull/43827)) — expands high-performance backend options for this architecture.
- Routed Qwen3.5 decode steps to the recurrent kernel by splitting mixed prefill+decode batches ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — optimizes decode latency in chunked prefill.

## Quantization
- Tuned the Triton `fused_moe` FP8 configuration for Qwen3-Next-80B (TP4) on H100 ([#44830](https://github.com/vllm-project/vllm/pull/44830)) — delivers a 25% speedup at production batch sizes.
- Canonicalized FP8 weight layouts to (K, N) at the source ([#44735](https://github.com/vllm-project/vllm/pull/44735)) — prevents downstream shape mismatches and loading crashes.
- Refactored the compressed-tensors NVFP4 linear layer to use a single unified class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — simplifies the FP4 codebase for future kernel additions.
- Added support for compressed-tensors WNA8O8Int linear layers and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands INT8/INT4 weight-only quantization options.

## Parallelism & scheduling
- Added an object store as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — enables more flexible memory hierarchies for long-context serving.
- Fixed a deterministic hang in multi-node Ray data-parallel serving ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — excludes the DP backend from deferred port allocation to unblock startup.
- Resolved a deadlock in asynchronous KV loads ([#44560](https://github.com/vllm-project/vllm/pull/44560)) — ensures stable prefix caching and disaggregated prefill under high concurrency.
- Enabled selective prefix-cache retention for DeepSeek-V4's sliding-window KV cache ([#43447](https://github.com/vllm-project/vllm/pull/43447)) — improves cache hit rates for long-context workloads.

## Model support
- Added native support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — brings an open-weights MoE code-generation model to the engine.
- Supported Gemma4 Unified (encoder-free) architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables speculative decoding and multimodal extensions for Google's latest models.
- Fixed Cohere2 MoE weight loading and tool-call parsing for Transformers >= 5.10 ([#44747](https://github.com/vllm-project/vllm/pull/44747), [#44907](https://github.com/vllm-project/vllm/pull/44907)) — restores compatibility after upstream config changes.
- Resolved a device mismatch crash in Qwen3-Omni by moving `cu_seqlens` to the GPU ([#44264](https://github.com/vllm-project/vllm/pull/44264)) — fixes engine initialization for the 30B thinking variants.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates CPU inference with transparent fallback for other architectures.
- Stabilized sleep-mode memory release on ROCm ([#43022](https://github.com/vllm-project/vllm/pull/43022)) — properly cycles the virtual address reservation to prevent intermittent HIP OOMs.
- Added XPU support for DeepSeek-V4's fused MHC post+pre operations and attention decode paths ([#41444](https://github.com/vllm-project/vllm/pull/41444), [#42953](https://github.com/vllm-project/vllm/pull/42953)) — brings Intel GPU performance closer to CUDA/AMD parity.
- Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — unlocks fast weight-only 4-bit inference on consumer/prosumer AMD GPUs.

## API & serving
- Advanced the Rust frontend with API key auth, tokenization endpoints, and structured output fixes ([#44321](https://github.com/vllm-project/vllm/pull/44321), [#44222](https://github.com/vllm-project/vllm/pull/44222), [#44729](https://github.com/vllm-project/vllm/pull/44729)) — pushes the experimental server closer to production parity.
- Fixed a bug in the Responses API where unstreamed tool call arguments were dropped ([#44348](https://github.com/vllm-project/vllm/pull/44348)) — ensures agentic clients receive complete payloads.
- Allowed Data Parallel Ray placement groups to be pinned to specific nodes ([#44669](https://github.com/vllm-project/vllm/pull/44669)) — improves multi-node cluster scheduling and resource isolation.
- Added `/pause`, `/resume`, and `/is_paused` endpoints to the Rust frontend ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — enables external load balancers to gracefully drain traffic.

## Watch list
- **DeepEP v2 NCCL Requirement**: The new Wide EP integration ([#41183](https://github.com/vllm-project/vllm/pull/41183)) requires NCCL >= 2.30.4, but PyTorch pins 2.28.9; users must manually upgrade NCCL until the pin is bumped.
- **KV Event Schema Breaking Change**: The shift to map-encoded KV events ([#42892](https://github.com/vllm-project/vllm/pull/42892)) will break existing custom ZMQ subscribers that expect positional arrays; update routing daemons before upgrading.
- **NIXL Wheel Normalization**: Docker images now strictly keep only the NIXL wheel matching the CUDA major version ([#44266](https://github.com/vllm-project/vllm/pull/44266)), fixing `libcudart` errors but potentially affecting custom builds relying on the old multi-wheel layout.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (229)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44176">#44176</a> [Perf] fuse qk rmsnorm rope gate for qwen3.5 — by <a href="https://github.com/ZJY0516">ZJY0516</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44983">#44983</a> [Bugfix] Fix minimax_qk_norm_fusion — by <a href="https://github.com/jeejeelee">jeejeelee</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44907">#44907</a> [Cohere] Cohere2 moe parser fix — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44747">#44747</a> [Cohere] Fix Cohere2MoE weight loading when using Transformers ≥5.10 — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44629">#44629</a> [PD][Bugfix] Fix KV Cache sharing with HMA — by <a href="https://github.com/NickLucche">NickLucche</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44901">#44901</a> [Rust Frontend] Support Kimi K2 tool call IDs — by <a href="https://github.com/cinnamonica02">cinnamonica02</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44481">#44481</a> [XPU][CI] Refine docker image build and pull/create lock mechanism in Intel GPU CI — by <a href="https://github.com/zxd1997066">zxd1997066</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44222">#44222</a> [Rust Frontend] Add /tokenize and /detokenize endpoints — by <a href="https://github.com/TanNgocDo">TanNgocDo</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42864">#42864</a> [ROCm][Compile] Fuse AR + RMSNorm + per-group FP8 quant (+ DSv3.2 indexer fan-out) — by <a href="https://github.com/maeehart">maeehart</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42892">#42892</a> [KV Events] Switch event structs from array to map encoding — by <a href="https://github.com/sagearc">sagearc</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44830">#44830</a> [Kernel][Perf] Tune fused_moe FP8 config for Qwen3-Next-80B tp=4 on H100 (+25% at batch 96-512) — by <a href="https://github.com/qyYue1389">qyYue1389</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44321">#44321</a> [Rust Frontend] Support API key authentication — by <a href="https://github.com/ricky-chaoju">ricky-chaoju</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44918">#44918</a> [CI/Docs] Remove stale disagg prefill links — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44144">#44144</a> [DSV4][XPU] Add MHC fused_post_pre support — by <a href="https://github.com/majian4work">majian4work</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43022">#43022</a> [ROCm][CI] Stabilize sleep-mode memory release — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44903">#44903</a> [Bugfix][CI] Fix `test_offloading_connector.py::test_fs_tiering_offloading` — by <a href="https://github.com/NickLucche">NickLucche</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44947">#44947</a> [CI] Reorganize entrypoints CI — by <a href="https://github.com/noooop">noooop</a></li>
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
<li><em>…and 169 more</em></li>
</ul>
</details>
