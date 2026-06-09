# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release and surrounding merges focus heavily on architectural refactoring for Mixture-of-Experts and non-standard attention, alongside deep hardware optimizations for AMD and Intel. We saw major structural changes to the KV cache manager and MoE execution paths to natively support hybrid models like DeepSeek-V4 and Qwen3.5. Additionally, the Rust frontend continues to mature with new tokenization and authentication endpoints, while critical security and stability fixes were applied to multimodal and disaggregated serving paths.

## Deep dives

### Pluggable KVCacheSpec
vLLM's block manager historically assumed a uniform, dense block layout for standard multi-head attention. As models like Mamba (SSM), DeepSeek (MLA), and hybrid architectures emerge, they require fundamentally different cache structures, such as recurrent states or shared caches. PR [#37505](https://github.com/vllm-project/vllm/pull/37505) introduces a pluggable `KVCacheSpec` interface, allowing model-specific cache allocation and management logic to be injected directly into the core scheduler. This unblocks native, optimized support for non-standard attention mechanisms without requiring invasive hacks to the core block manager, paving the way for better memory utilization. Anyone running hybrid, SSM, or MLA models will benefit from more stable memory management, while developers extending vLLM for new architectures now have a clean API for cache specification.

### FusedMoE and MoERunner Inversion
Mixture-of-Experts (MoE) execution previously tightly coupled the routing logic with the underlying kernel execution inside the `FusedMoE` class, making it difficult to support diverse quantization formats and hardware backends. PR [#41184](https://github.com/vllm-project/vllm/pull/41184) inverts this control flow, making `MoERunner` the primary abstraction that dictates how experts are mapped, loaded, and executed, while `FusedMoE` becomes a specialized implementation detail. This drastically simplifies adding new MoE quantization formats, custom routing strategies, and hardware-specific kernels, while resolving deep-seated weight-loading bugs across different tensor-parallel configurations. Developers extending MoE support will find the codebase much more approachable, and users running large MoE models like Qwen3 or DeepSeek will see improved stability, as evidenced by immediate follow-up fixes for Qwen3.5 EP ([#45002](https://github.com/vllm-project/vllm/pull/45002)) and Cohere2 MoE ([#44747](https://github.com/vllm-project/vllm/pull/44747)).

### DeepSeek-V4 Sparse MLA and Attention Kernels
DeepSeek-V4 introduces Sparse Multi-head Latent Attention (MLA), which differs significantly from V3.2 in how it handles indexing, cache compression, and decoder fan-out. PR [#44699](https://github.com/vllm-project/vllm/pull/44699) decouples V4's Sparse MLA metadata from the V3.2 implementation, and PR [#43827](https://github.com/vllm-project/vllm/pull/43827) integrates the TensorRT-LLM generation attention kernel specifically tailored for DSv4's unique shapes. Decoupling the metadata removes rigid assumptions that bottlenecked V4, while the TRT-LLM kernel unlocks highly optimized, low-latency decode phases on NVIDIA GPUs, avoiding the overhead of generic attention paths. Anyone deploying DeepSeek-V4 on NVIDIA hardware will see substantial latency reductions and higher throughput during the generation phase; no flag changes are required as this is the new default path.

## Kernels & attention
- Fused QK RMSNorm, RoPE, and gate for Qwen3.5 ([#44176](https://github.com/vllm-project/vllm/pull/44176)) — reduces kernel launch overhead and boosts throughput for this specific architecture.
- Fused softplus-sqrt-topk router under AITER fused-MoE for ROCm ([#44945](https://github.com/vllm-project/vllm/pull/44945)) — speeds up MoE routing on AMD GPUs by keeping operations in a single kernel.
- Reverted warp-shuffle reduction for `silu_and_mul_per_block_quant` ([#45066](https://github.com/vllm-project/vllm/pull/45066)) — unblocks ROCm CI while a proper fix is developed for the FP8 MoE speedup.

## Quantization
- Tuned Triton `fused_moe` FP8 config for Qwen3-Next-80B on H100 ([#44830](https://github.com/vllm-project/vllm/pull/44830)) — yields a ~25% speedup at batch sizes 96-512 for TP=4 by optimizing block shapes.
- Added online FP8 per-token-per-channel (PTPC) quantization ([#44132](https://github.com/vllm-project/vllm/pull/44132)) — enables dynamic FP8 quantization for models without offline calibration.
- Canonicalized FP8 weight layout to (K, N) at the source ([#44735](https://github.com/vllm-project/vllm/pull/44735)) — fixes weight loading bugs on ROCm by ensuring consistent memory layouts across backends.

## Parallelism & scheduling
- Mooncake KV connector now uses all HCAs on multi-NIC hosts ([#43799](https://github.com/vllm-project/vllm/pull/43799)) — prevents leaving network interfaces idle on systems like B300 with 8 GPUs and 16 HCAs.
- Nixl communicator optimized with zero-copy transfers ([#41633](https://github.com/vllm-project/vllm/pull/41633)) — reduces CPU overhead and latency during disaggregated prefill KV cache transfers.
- Split mixed prefill+decode batches for Qwen3.5 to route decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — improves scheduling efficiency for hybrid models.

## Model support
- Added support for JetBrains' Mellum v2 MoE code-gen model ([#43992](https://github.com/vllm-project/vllm/pull/43992)) and Gemma4 Unified encoder-free architecture ([#44429](https://github.com/vllm-project/vllm/pull/44429)).
- Fixed HyperCLOVAX loading by natively registering the model type ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — adapts to upstream HuggingFace removing remote code in newer transformers.
- Fixed Cohere2 MoE weight loading for Transformers >= 5.10 ([#44747](https://github.com/vllm-project/vllm/pull/44747)) — handles the new `mlp_layer_types` config structure to prevent dense layer-0 crashes.

## Hardware
- AMD Zen CPUs route W8A8/W4A16 inference through zentorch kernels ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — accelerates CPU inference with transparent fallback for non-Zen chips.
- Fused AllReduce + RMSNorm + per-group FP8 quant for DeepSeek V3.2 on ROCm ([#42864](https://github.com/vllm-project/vllm/pull/42864)) — eliminates ~535us per decode step on MI355X by combining ops.
- Added XPU block-scaled W8A8 FP8 path ([#39968](https://github.com/vllm-project/vllm/pull/39968)) and transparent sleep mode support ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — expands quantization and power-management for Intel GPUs.

## API & serving
- Switched KV cache event structs from array to map encoding ([#42892](https://github.com/vllm-project/vllm/pull/42892)) — prevents breaking external routing subscribers when the msgpack schema is extended.
- Fixed DoS vulnerability in audio transcription via decompression bombs ([#44970](https://github.com/vllm-project/vllm/pull/44970)) — enforces decode limits to prevent OOM kills from compressed files.
- Added `/tokenize` and `/detokenize` endpoints to the Rust frontend ([#44222](https://github.com/vllm-project/vllm/pull/44222)) — matches Python server functionality for fast, in-process encoding.

## Watch list
- The `kv_both` role in NixlConnector is entering deprecation ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this for disaggregated prefill should migrate to explicit roles.
- The FusedMoE refactor ([#41184](https://github.com/vllm-project/vllm/pull/41184)) is shaking out weight-loading bugs ([#45002](https://github.com/vllm-project/vllm/pull/45002)) — monitor for further MoE mapping fixes as the new abstraction settles.
- KV Events schema change to map encoding ([#42892](https://github.com/vllm-project/vllm/pull/42892)) is a breaking change for out-of-tree subscribers that haven't updated their msgpack parsing logic.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (225)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><em>…and 165 more</em></li>
</ul>
</details>
