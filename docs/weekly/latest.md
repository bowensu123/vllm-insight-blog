# vLLM weekly digest — 2026-06-10 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focused on stabilizing large-scale Mixture-of-Experts (MoE) serving, specifically targeting DeepSeek-V4 and Gemma4 architectures. We saw significant performance unlocks for Hopper and Blackwell GPUs via FlashAttention 4, alongside critical network bandwidth fixes for disaggregated serving on multi-NIC hosts. Additionally, the Rust frontend continues to mature with new tokenization and authentication endpoints, while security patches addressed critical audio and image preprocessing vulnerabilities.

## Deep dives

### Gemma4 Unified FlashAttention 4 & Multimodal Prefixes
Gemma4 uses heterogeneous head dimensions across its layers (256 for sliding-window, 512 for global attention), which previously forced vLLM to fall back to the slower Triton attention backend. With FlashAttention 4 (FA4) now supporting head dimensions up to 512, PR [#42175](https://github.com/vllm-project/vllm/pull/42175) enables FA4 as the default backend for Gemma4 on Hopper and Blackwell GPUs. This change also introduces `mm_prefix` support to the FlashAttention backend, enabling bidirectional attention for multimodal inputs. Users running Gemma4 on SM90+ hardware will see immediate throughput improvements without needing to change flags, and multimodal workloads can now leverage prefix caching.

### Mooncake Multi-NIC HCA Utilization for KV Transfer
In disaggregated prefill/decode serving, KV cache transfers rely on RDMA over Host Channel Adapters (HCAs). High-end hosts like the B300 feature more HCAs than GPUs (e.g., 16 HCAs for 8 GPUs), but previous logic pinned each Mooncake worker to a single GPU-indexed HCA, leaving half the network bandwidth idle. PR [#43799](https://github.com/vllm-project/vllm/pull/43799) removes this explicit per-GPU RNIC lookup, allowing the Mooncake transfer engine to bind across all configured network devices. This matters for teams running PD disaggregation on multi-NIC clusters, as it maximizes KV transfer throughput and reduces decode time-to-first-token (TTFT).

### DeepEP v2 Integration for Wide Expert Parallelism
Expert Parallelism (EP) distributes Mixture-of-Experts (MoE) layers across multiple GPUs, requiring heavy all-to-all communication to route tokens to the correct experts. DeepEP is a specialized communication library optimized specifically for these MoE all-to-all patterns. PR [#41183](https://github.com/vllm-project/vllm/pull/41183) integrates DeepEP v2 into vLLM's WideEP path, significantly reducing communication overhead compared to standard NCCL. This is a major win for teams serving massive MoE models like DeepSeek-V3 or V4 across multi-node clusters, as it scales EP efficiency and reduces network bottlenecks during the MoE routing phase.

## Kernels & attention
* Fused `all_reduce -> RMSNorm -> per-group FP8 quant` for DeepSeek V3.2 on ROCm (`[#42864](https://github.com/vllm-project/vllm/pull/42864)`) — eliminates ~535us of standalone kernel launches per decode step.
* Added a fused `softplus-sqrt-topk` router for AITER fused-MoE on ROCm (`[#44945](https://github.com/vllm-project/vllm/pull/44945)`) — optimizes the expert routing path for DeepSeek-V4.
* Reverted a warp-shuffle optimization for `silu_and_mul_per_block_quant` (`[#45066](https://github.com/vllm-project/vllm/pull/45066)`) — unblocks CI after it broke the ROCm build, with a proper fix pending.

## Quantization
* AMD Zen CPUs now route W8A8 and W4A16 (GPTQ) linear inference through zentorch kernels (`[#41813](https://github.com/vllm-project/vllm/pull/41813)`) — provides hardware-accelerated quantized inference on AMD CPUs with transparent fallback.
* Added online FP8 per-token-per-channel (PTPC) quantization support (`[#44132](https://github.com/vllm-project/vllm/pull/44132)`) — enables dynamic FP8 quantization for models without offline calibration.
* Canonicalized FP8 weight layout to `(K, N)` at the source (`[#44735](https://github.com/vllm-project/vllm/pull/44735)`) — fixes weight loading mismatches on ROCm.

## Parallelism & scheduling
* Fixed a deterministic hang in multi-node Ray data-parallel serving (`[#43864](https://github.com/vllm-project/vllm/pull/43864)`) — excludes the Ray DP backend from deferred port allocation to prevent startup deadlocks.
* Qwen3.5 now splits mixed prefill+decode batches (`[#44700](https://github.com/vllm-project/vllm/pull/44700)`) — routes decodes to the recurrent kernel for better throughput.
* Replaced `P2pNcclConnector` with newer KV transfer mechanisms (`[#44854](https://github.com/vllm-project/vllm/pull/44854)`) — cleans up legacy point-to-point NCCL code in favor of modern connectors.

## Model support
* Added support for JetBrains' Mellum v2 (`[#43992](https://github.com/vllm-project/vllm/pull/43992)`) — brings native support for this open-weights Mixture-of-Experts code-generation model.
* Fixed DeepSeek-V4 initialization and OOM issues on H200 (`[#44914](https://github.com/vllm-project/vllm/pull/44914)`, `0decac0d`) — resolves a CUTLASS `fmin` compatibility bug and adjusts memory allocations.
* Fixed HyperCLOVAX loading after upstream HuggingFace removed its remote code (`[#43860](https://github.com/vllm-project/vllm/pull/43860)`) — registers the model natively in vLLM to prevent startup failures.
* Unified KDA conv states into a single cache for Mamba models (`[#44539](https://github.com/vllm-project/vllm/pull/44539)`) — matches the 2-state SSM layout for correct recurrent state management.

## Hardware
* Added a fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) GPUs (`[#44075](https://github.com/vllm-project/vllm/pull/44075)`) — brings optimized quantized MoE inference to consumer/prosumer AMD hardware.
* Implemented the DeepSeek-V4 XPU attention decode path and fused post-pre support (`[#42953](https://github.com/vllm-project/vllm/pull/42953)`, `[#44144](https://github.com/vllm-project/vllm/pull/44144)`) — expands DeepSeek-V4 support to Intel GPUs.
* Upgraded `torch-xpu` to 2.12 (`[#42262](https://github.com/vllm-project/vllm/pull/42262)`) — keeps Intel GPU dependencies current for XPU backend stability.

## API & serving
* Fixed a DoS vulnerability in the `/v1/audio/transcriptions` endpoint (`[#44970](https://github.com/vllm-project/vllm/pull/44970)`) — prevents OOMs from compressed audio decompression bombs by enforcing decode limits.
* Extracted Mistral-specific tool calling logic into a dedicated `MistralParser` (`[#44596](https://github.com/vllm-project/vllm/pull/44596)`) — cleans up the unified parser interface and handles v11+ grammar extraction.
* Added `/tokenize` and `/detokenize` endpoints to the Rust frontend (`[#44222](https://github.com/vllm-project/vllm/pull/44222)`) — enables in-process text processing without involving the inference engine.
* Fixed image preprocessing for EXIF orientation and PNG tRNS transparency (`[#44974](https://github.com/vllm-project/vllm/pull/44974)`) — prevents multimodal interpretation bias from raw pixel mismatches.

## Watch list
* The `kv_both` role in `NixlConnector` has entered a deprecation cycle (`[#43874](https://github.com/vllm-project/vllm/pull/43874)`) — users relying on it for bidirectional KV transfer should migrate to the new PD disaggregation patterns.
* A major refactor of `FusedMoE` and `MoERunner` inversion landed (`[#41184](https://github.com/vllm-project/vllm/pull/41184)`) — changes how MoE experts are mapped and executed, so watch for edge cases in custom MoE implementations.
* The Rust frontend is rapidly maturing, gaining API key authentication (`[#44321](https://github.com/vllm-project/vllm/pull/44321)`), structured output backends (`[#44729](https://github.com/vllm-project/vllm/pull/44729)`), and Kimi K2 tool call ID preservation (`[#44901](https://github.com/vllm-project/vllm/pull/44901)`) — worth evaluating if you are building custom high-performance routing layers.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (223)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><em>…and 163 more</em></li>
</ul>
</details>
