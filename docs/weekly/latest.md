# vLLM weekly digest — 2026-06-10 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focused heavily on hardening multimodal serving, closing critical security gaps in audio and image preprocessing, and unblocking next-generation architectures like Gemma4 and DeepSeek-V4 on modern hardware. Alongside these stability fixes, the Rust frontend continues to rapidly close the feature parity gap with the Python serving layer, and disaggregated KV transfer saw major bandwidth optimizations for multi-NIC nodes. It is a highly recommended upgrade for anyone running public-facing multimodal endpoints or deploying large Mixture-of-Experts models on Hopper/Blackwell and AMD MI300X clusters.

## Deep dives

### FlashAttention 4 and Multimodal Prefixes for Gemma4
Gemma4 uses heterogeneous head dimensions across its layers (256 for sliding-window attention and 512 for global attention), which previously forced vLLM to fall back to the slower Triton attention backend because older FlashAttention versions capped head dimensions at 256. PR `[#42175](https://github.com/vllm-project/vllm/pull/42175)` upgrades the default backend to FlashAttention 4 (FA4) on Hopper and Blackwell GPUs, which natively supports head dimensions up to 512, and simultaneously adds `mm_prefix` (bidirectional attention) support to the FlashAttention backend. This matters because it eliminates the Triton bottleneck for Gemma4, yielding substantial throughput gains, while the new prefix support enables proper multimodal prefix caching for vision-language tasks. Engineers running Gemma4 on SM90/SM100+ GPUs will see this apply automatically, but should verify their `VLLM_ATTENTION_BACKEND` environment variables aren't forcefully overriding the new default.

### Plugging Multimodal Denial-of-Service and Preprocessing Gaps
When accepting raw files via API, the server must carefully manage memory allocation and pixel formatting to prevent abuse and ensure model accuracy. PR `[#44970](https://github.com/vllm-project/vllm/pull/44970)` fixes a critical DoS vulnerability where a small 25MB compressed OPUS audio file could expand into a 5.7GB PCM float32 tensor in memory before the duration guard could reject it, while PR `[#44974](https://github.com/vllm-project/vllm/pull/44974)` ensures EXIF orientation tags are transposed and PNG tRNS transparency is properly composited against a background before RGB conversion. These changes matter because they prevent trivial OOM crashes on public speech-to-text endpoints and ensure vision models process images exactly as a human sees them, eliminating silent interpretation biases. Anyone exposing `/v1/audio/transcriptions` or vision endpoints to untrusted traffic must upgrade immediately to mitigate the decompression bomb vector.

### Unlocking Full RDMA Bandwidth for Disaggregated KV Transfer
In disaggregated prefill setups, KV cache transfers rely on RDMA over Host Channel Adapters (HCAs/NICs), but modern high-end nodes often have more HCAs than GPUs (e.g., 16 HCAs for 8 GPUs). Previously, vLLM's Mooncake connector pinned each worker to a single HCA based on its GPU index, leaving half the network interfaces completely idle. PR `[#43799](https://github.com/vllm-project/vllm/pull/43799)` removes this strict 1:1 GPU-to-HCA mapping, allowing the Mooncake transfer engine to bind across all configured network devices on the host. This matters because it instantly doubles the available network bandwidth for KV transfers on multi-NIC nodes, drastically reducing Time-To-First-Token (TTFT) in disaggregated architectures. Operators using Mooncake for PD disaggregation on B300 or H200 multi-NIC hosts should see immediate transfer speedups without changing their launch scripts.

## Kernels & attention
* Added TRTLLM generation attention kernel specifically for DeepSeek-V4 (`[#43827](https://github.com/vllm-project/vllm/pull/43827)`) — provides a highly optimized decode path for DeepSeek's sparse MLA architecture on NVIDIA GPUs.
* Re-enabled the fused softplus-sqrt-topk router under the AITER fused-MoE path on ROCm (`[#44945](https://github.com/vllm-project/vllm/pull/44945)`) — recovers lost performance for MoE routing on AMD Instinct accelerators.
* Reverted the warp-shuffle reduction optimization for `silu_and_mul_per_block_quant` (`[#45066](https://github.com/vllm-project/vllm/pull/45066)`) — unblocks CI after the initial PR (`[#44173](https://github.com/vllm-project/vllm/pull/44173)`) caused build failures on ROCm; a proper cross-platform fix is in progress.

## Quantization
* Added online FP8 per-token per-channel (PTPC) quantization support (`[#44132](https://github.com/vllm-project/vllm/pull/44132)`) — enables more granular dynamic quantization for models that benefit from channel-wise scaling.
* Supported compressed-tensors WNA8O8Int linears and WNInt embeddings (`[#44340](https://github.com/vllm-project/vllm/pull/44340)`) — expands the matrix of supported quantization schemes for heavily compressed models.
* Canonicalized FP8 weight layout to (K, N) at the source (`[#44735](https://github.com/vllm-project/vllm/pull/44735)`) — prevents silent shape mismatches and performance cliffs when loading FP8 checkpoints on ROCm.

## Parallelism & scheduling
* Integrated DeepEP v2 for Wide Expert Parallelism (`[#41184](https://github.com/vllm-project/vllm/pull/41184)`) — improves communication overlap and scaling for massive Mixture-of-Experts models across large multi-node clusters.
* Added selective prefix-cache retention for DeepSeek-V4's sliding-window KV cache (`[#43447](https://github.com/vllm-project/vllm/pull/43447)`) — prevents the eviction of critical global attention tokens when the sliding window advances, preserving context quality.
* Split mixed prefill+decode batches for Qwen3.5 to route decodes to the recurrent kernel (`[#44700](https://github.com/vllm-project/vllm/pull/44700)`) — prevents decode tokens from being bottlenecked by prefill-optimized attention paths, improving mixed-batch throughput.

## Model support
* Added support for JetBrains' Mellum v2 (`[#43992](https://github.com/vllm-project/vllm/pull/43992)`) — brings native execution for this new open-weights Mixture-of-Experts code-generation model.
* Added Gemma4 Unified (encoder-free) support (`[#44429](https://github.com/vllm-project/vllm/pull/44429)`) — enables serving the latest Gemma4 vision variants that process images directly through the language model backbone.
* Fixed a DeepSeek-V4 OOM issue on H200 during initialization (`[#44914](https://github.com/vllm-project/vllm/pull/44914)`) — resolves memory allocation failures when loading the massive MoE model with expert parallelism enabled.
* Fixed weight loading regressions for Qwen3.5 and Step3.5 MoE models (`[#45054](https://github.com/vllm-project/vllm/pull/45054)`, `[#45002](https://github.com/vllm-project/vllm/pull/45002)`) — repairs expert routing mappings broken by the recent FusedMoE refactor.

## Hardware
* Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs (`[#41813](https://github.com/vllm-project/vllm/pull/41813)`) — significantly accelerates quantized inference on AMD EPYC processors with transparent fallback for other architectures.
* Added a hybrid CDNA4 swizzle gate for A8W4 MoE on ROCm (`[#44804](https://github.com/vllm-project/vllm/pull/44804)`) — prevents silent scale tensor corruption by falling back to strided layouts when tensor parallelism exceeds 2.
* Introduced transparent sleep mode support and CPU KV offloading/tiering for Intel XPU (`[#37149](https://github.com/vllm-project/vllm/pull/37149)`, `[#36423](https://github.com/vllm-project/vllm/pull/36423)`) — allows Intel GPU deployments to gracefully yield memory and spill KV cache to system RAM during idle periods.

## API & serving
* Added `seed_oss` and `step3p5` reasoning parsers to the Rust frontend (`[#44552](https://github.com/vllm-project/vllm/pull/44552)`) — closes the feature parity gap for streaming complex chain-of-thought models with custom delimiters.
* Extracted Mistral-specific tool calling and parsing logic into a dedicated `MistralParser` (`[#44596](https://github.com/vllm-project/vllm/pull/44596)`) — cleans up the generic serving paths and correctly handles Mistral's strict 9-char alphanumeric tool IDs.
* Added the Berkeley Function Calling Leaderboard (BFCL) dataset to the benchmarking tool (`[#42457](https://github.com/vllm-project/vllm/pull/42457)`) — provides a standardized, realistic workload for measuring tool-calling latency and throughput.

## Watch list
* **FusedMoE Refactor Fallout:** The massive FusedMoE/MoERunner inversion (`[#41184](https://github.com/vllm-project/vllm/pull/41184)`) introduced several weight-loading regressions in Qwen and Step models this week. Watch for further patches as the new MoE abstraction settles and edge cases in custom checkpoint mappings are discovered.
* **KV Connector Deprecations:** The NIXL KV-connector wheel installs were normalized for CUDA 13 (`[#44266](https://github.com/vllm-project/vllm/pull/44266)`), but the broader KV connector API is shifting. Keep an eye on the initiated deprecation cycle for the `kv_both` role (`[#43874](https://github.com/vllm-project/vllm/pull/43874)`) if you are building custom disaggregated prefill proxies.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (222)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44804">#44804</a> [ROCm][gpt-oss] Hybrid CDNA4 swizzle gate for A8W4 MoE — by <a href="https://github.com/xiaohuguo2023">xiaohuguo2023</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42457">#42457</a> [Bench] Add BFCL dataset for vllm bench serve tool-calling workloads — by <a href="https://github.com/laviier">laviier</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44999">#44999</a> Model/colbert autoweightsloader — by <a href="https://github.com/yufufi">yufufi</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45058">#45058</a> Change from owning configs to owning config utils — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44686">#44686</a> Fix Harmony tool descriptions for optional fields — by <a href="https://github.com/shenoyvvarun">shenoyvvarun</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44552">#44552</a> [Rust Frontend] Add seed_oss and step3p5 reasoning parsers — by <a href="https://github.com/yzhan1">yzhan1</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45057">#45057</a> [Bugfix] Handle HWC images in ImageProcessorItems.get_image_size — by <a href="https://github.com/YellowFoxH4XOR">YellowFoxH4XOR</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45081">#45081</a> [Refactor] Remove dead states from chat completion serving — by <a href="https://github.com/sfeng33">sfeng33</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45054">#45054</a> [Bugfix] Fix weight loading issues caused by #41184 — by <a href="https://github.com/bnellnm">bnellnm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45085">#45085</a> [Bugfix][CI/Build] Fix Rust frontend build after chat conversion refactor — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43756">#43756</a> [Bench] benchmark_serving_multi_turn: make non-standard conversation_id payload opt-in — by <a href="https://github.com/Change72">Change72</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42175">#42175</a> [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support — by <a href="https://github.com/lucianommartins">lucianommartins</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44596">#44596</a> [Refactor][Mistral] Extract parsing logic into MistralParser — by <a href="https://github.com/sfeng33">sfeng33</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44884">#44884</a> [Rust Frontend] Extract shared options in route helper params — by <a href="https://github.com/BugenZhao">BugenZhao</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44914">#44914</a> [Bug] Fix deepseek v4 OOM issue — by <a href="https://github.com/yewentao256">yewentao256</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44678">#44678</a> [ROCm][CI] fix test_rope_kvcache_fusion.py — by <a href="https://github.com/charlifu">charlifu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45066">#45066</a> Revert &quot;[Kernel] Speed up silu_and_mul_per_block_quant with warp-shuf… — by <a href="https://github.com/micah-wil">micah-wil</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44516">#44516</a> feat(multi-turn-bench): add api_key and custom headers for multi turn benchmark — by <a href="https://github.com/jimmy-evo">jimmy-evo</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45002">#45002</a> [Bugfix] fix qwen3.5 ep weight loading — by <a href="https://github.com/ZJY0516">ZJY0516</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44936">#44936</a> [ROCm][V2] Fix failed assertion in Llama models when using EAGLE with `ROCM_AITER_FA` — by <a href="https://github.com/micah-wil">micah-wil</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43799">#43799</a> [Mooncake] Use all HCAs on multi-NIC hosts instead of GPU-indexed RNIC selection — by <a href="https://github.com/Dao007forever">Dao007forever</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44945">#44945</a> [ROCm][Perf] Use fused softplus-sqrt-topk router under AITER fused-MoE — by <a href="https://github.com/Fangzhou-Ai">Fangzhou-Ai</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44173">#44173</a> [Kernel] Speed up silu_and_mul_per_block_quant with warp-shuffle reduction + vectorized I/O — by <a href="https://github.com/yangdian96">yangdian96</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44974">#44974</a> [Security] Fix image EXIF orientation and tRNS transparency handling — by <a href="https://github.com/jperezdealgaba">jperezdealgaba</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44408">#44408</a> Fix MiDashengLM TP&gt;1 crash in audio encoder attention — by <a href="https://github.com/mganczarenko">mganczarenko</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44415">#44415</a> [Docs] Add KV offloading usage guide (single- and multi-tier) — by <a href="https://github.com/ronensc">ronensc</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44970">#44970</a> [Security] Fix DoS via audio decompression bomb in speech-to-text endpoint — by <a href="https://github.com/jperezdealgaba">jperezdealgaba</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44663">#44663</a> [Bugfix] Add X-Session-ID from conversation_id in multi-turn benchmark — by <a href="https://github.com/tykow">tykow</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44040">#44040</a> [ROCm][CI] Stabilize ModernBERT token-classification parity against Hugging Face — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42262">#42262</a> [WIP][XPU] upgrade torch-xpu to 2.12 — by <a href="https://github.com/jikunshang">jikunshang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/39425">#39425</a> Remove `raw_inputs` from transformers backend — by <a href="https://github.com/zucchini-nlp">zucchini-nlp</a></li>
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
<li><em>…and 162 more</em></li>
</ul>
</details>
