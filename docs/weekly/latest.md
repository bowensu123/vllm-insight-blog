# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses heavily on hardening the server against multimodal edge cases, maturing the Rust frontend, and squeezing out low-level performance on AMD and Intel hardware. Notable engineering wins include fusing the AllReduce-RMSNorm-FP8 chain for DeepSeek V3.2 on ROCm and closing a critical audio decompression bomb vulnerability. Additionally, the KV cache event schema was migrated to a map-based format, significantly improving the stability of external disaggregated prefill routers.

## Deep dives

### Audio Decompression Bomb DoS Mitigation
Speech-to-text endpoints accept compressed audio formats like OPUS, which must be decoded into raw PCM floats in memory before processing. A maliciously crafted or simply very long 25MB compressed file can expand to over 5GB of float32 PCM, bypassing upload size limits and causing an out-of-memory crash. PR [#44970](https://github.com/vllm-project/vllm/pull/44970) fixes this by enforcing a duration guard during the decode process itself, preventing the massive memory allocation from occurring before the `max_audio_clip_s` limit is checked. This matters because it closes a trivial denial-of-service vector that could take down a server with just a few concurrent requests. Anyone exposing the `/v1/audio/transcriptions` endpoint to untrusted or public traffic should upgrade immediately to prevent OOM kills.

### ROCm AITER AllReduce, RMSNorm, and FP8 Quantization Fusion
In FP8 blockwise models like DeepSeek V3.2, the end of every transformer block requires an all-reduce, followed by RMSNorm, and then per-group FP8 quantization. Executing these as separate kernel launches incurs significant memory bandwidth overhead and launch latency, bottlenecking the decode phase. PR [#42864](https://github.com/vllm-project/vllm/pull/42864) introduces a new pattern in the `RocmAiterAllReduceFusionPass` that fuses this entire chain into a single AITER kernel call. This matters because it eliminates roughly 535 microseconds of overhead per decode step, substantially increasing decode throughput for large models. Engineers serving DeepSeek V3.2 or similar FP8-blockwise architectures on AMD MI355X (and MI300) hardware with tensor parallelism will see immediate performance gains without changing their serving configuration.

### KV Cache Events Schema Migration to Map Encoding
vLLM publishes KV cache events over ZMQ to external subscribers, such as disaggregated prefill routers and distributed cache managers. Historically, these events were encoded as positional msgpack arrays, meaning any addition to the schema would silently break older subscribers that relied on strict index mapping. PR [#42892](https://github.com/vllm-project/vllm/pull/42892) migrates the event structs from array to map (dictionary) encoding, attaching explicit keys to every field. This matters because it decouples the vLLM core schema evolution from external infrastructure, allowing new KV metadata to be added without breaking existing routing daemons. Engineers building custom KV cache managers or disaggregated serving layers need to update their subscribers to parse map-based msgpack payloads instead of positional arrays.

## Kernels & attention
- Fused QK RMSNorm, partial RoPE, and gate copy into a single kernel launch for Qwen3.5 ([#44176](https://github.com/vllm-project/vllm/pull/44176)) — reduces kernel launch overhead and boosts throughput for Qwen3.5 models.
- Added a tuned Triton `fused_moe` FP8 configuration for Qwen3-Next-80B on H100 TP=4 ([#44830](https://github.com/vllm-project/vllm/pull/44830)) — yields a ~25% speedup at production batch sizes (96-512) over the default config.
- Fixed a fallback crash in `minimax_qk_norm_fusion` when FlashInfer AllReduce is enabled ([#44983](https://github.com/vllm-project/vllm/pull/44983)) — resolves an FP32 variance incompatibility that broke MiniMax model serving.

## Quantization
- Canonicalized FP8 weight layouts to (K, N) at the source during weight loading ([#44735](https://github.com/vllm-project/vllm/pull/44735)) — prevents shape mismatch errors on ROCm when loading certain FP8 checkpoints.
- Added support for online FP8 per-token-per-channel (PTPC) quantization ([#44132](https://github.com/vllm-project/vllm/pull/44132)) — enables dynamic FP8 quantization for models without offline calibration.
- Refactored Compressed Tensors NVFP4 linear layers to use a single unified class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — simplifies the codebase and eases future FP4 kernel integrations.

## Parallelism & scheduling
- Fixed a deterministic hang in multi-node Ray data-parallel serving when `num_api_servers > 1` ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — restores stability for large-scale Ray DP deployments by excluding the DP backend from deferred port allocation.
- Fixed KV cache sharing crashes when using Hierarchical Memory Allocator (HMA) with connectors ([#44629](https://github.com/vllm-project/vllm/pull/44629)) — ensures layers that don't need KV cache are excluded from the spec, preventing setup failures.
- Initiated the deprecation cycle for the `kv_both` role in `NixlConnector` ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — signals a shift toward more explicit KV transfer roles in disaggregated prefill setups.

## Model support
- Added support for JetBrains' Mellum v2, an open-weights Mixture-of-Experts code-generation model ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expands vLLM's coverage of specialized coding MoE architectures.
- Added support for Gemma4 Unified (encoder-free) multimodal models ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables serving of Google's latest unified vision-language architecture.
- Fixed `Cohere2MoE` weight loading failures on Transformers >= 5.10 ([#44747](https://github.com/vllm-project/vllm/pull/44747)) — resolves a `KeyError` caused by upstream config changes to `mlp_layer_types`.
- Fixed a tensor parallel crash in the MiDashengLM audio encoder ([#44408](https://github.com/vllm-project/vllm/pull/44408)) — corrects a shape mismatch in attention reshaping when `tensor_parallel_size > 1`.

## Hardware
- Routed W8A8 and W4A16 linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates CPU inference on AMD Zen architectures with transparent fallback for other CPUs.
- Added XPU support for DeepSeek-V4 MHC fused post/pre operations ([#44144](https://github.com/vllm-project/vllm/pull/44144)) — matches the AMD/CUDA fused execution pattern for Intel GPUs, improving DSV4 decode loops.
- Fixed a ROCm sleep-mode memory release issue where virtual addresses weren't cycled ([#43022](https://github.com/vllm-project/vllm/pull/43022)) — prevents intermittent HIP OOMs during sleep/wake transitions on MI300.

## API & serving
- Added `/tokenize` and `/detokenize` endpoints to the Rust frontend ([#44222](https://github.com/vllm-project/vllm/pull/44222)) — brings the Rust server to feature parity with the Python OpenAI server for text processing.
- Added API key authentication support to the Rust frontend via `--api-key` and `VLLM_API_KEY` ([#44321](https://github.com/vllm-project/vllm/pull/44321)) — secures Rust-based deployments without needing an external reverse proxy.
- Fixed a 500 error for structured outputs in the Rust frontend ([#44729](https://github.com/vllm-project/vllm/pull/44729)) — ensures the grammar backend is correctly resolved when bypassing the Python Processor.
- Fixed image EXIF orientation and PNG tRNS transparency handling in multimodal preprocessing ([#44974](https://github.com/vllm-project/vllm/pull/44974)) — ensures the model sees pixel data that matches human visual perception.

## Watch list
- The `NixlConnector` `kv_both` role is entering deprecation ([#43874](https://github.com/vllm-project/vllm/pull/43874)); teams using NIXL for disaggregated prefill should plan to migrate to explicit prefill/decode roles.
- KV cache event subscribers must update their msgpack parsers from positional arrays to maps ([#42892](https://github.com/vllm-project/vllm/pull/42892)) before upgrading, or external cache managers will fail to parse events.
- Audio endpoint users should review the new decode-time duration limits ([#44970](https://github.com/vllm-project/vllm/pull/44970)) to ensure legitimate long-form audio use cases aren't inadvertently truncated by the default 30s guard.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (230)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><em>…and 170 more</em></li>
</ul>
</details>
