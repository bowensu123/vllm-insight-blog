# vLLM weekly digest — 2026-06-09 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on stabilizing DeepSeek-V4 inference, hardening multimodal security, and advancing the Rust frontend toward production parity. Key performance wins include deep kernel optimizations for DeepSeek's block-FP8 MoE routing and full multi-NIC RDMA utilization for Mooncake KV transfers. If you are serving DeepSeek-V4, running disaggregated prefill/decode with Mooncake, or exposing multimodal endpoints to the public, this update is highly recommended.

## Deep dives

### Optimizing DeepSeek Block-FP8 MoE Quantization
In Mixture-of-Experts (MoE) models using block-wise FP8 quantization, the routing and activation functions must be heavily fused to avoid memory bandwidth bottlenecks. The `silu_and_mul_per_block_quant` kernel handles the fused `SiLU(gate) * up` activation alongside per-group FP8 quantization for DeepSeek's experts. PR [#44173](https://github.com/vllm-project/vllm/pull/44173) rewrites this kernel from a thread-block-per-group approach with shared-memory tree reductions to a highly efficient one-warp-per-group design using warp-shuffle reductions and vectorized I/O. This eliminates the memory-latency bounds seen on Hopper GPUs, significantly speeding up the expert computation path. Engineers running DeepSeek-V3/V4 or similar block-FP8 MoE models on H100s will see immediate throughput improvements without changing any flags.

### Maximizing RDMA Bandwidth in Mooncake KV Transfers
In disaggregated prefill/decode (PD) serving, the KV cache must be transferred over the network from prefill nodes to decode nodes, making RDMA bandwidth critical. Previously, the Mooncake KV connector pinned each GPU worker to a single Host Channel Adapter (HCA) based on its GPU index, which left extra NICs idle on multi-NIC hosts (e.g., 8 GPUs with 16 HCAs). PR [#43799](https://github.com/vllm-project/vllm/pull/43799) removes this strict 1:1 mapping, allowing Mooncake's transfer engine to bind across all configured network devices. This change unlocks the full aggregate network bandwidth for KV transfers. Teams running PD disaggregation on dense-NIC hardware should upgrade to eliminate this network bottleneck.

### Hardening Multimodal Endpoint Security
Serving multimodal models requires parsing diverse file formats, which introduces attack surfaces like decompression bombs and pixel-interpretation biases. PR [#44970](https://github.com/vllm-project/vllm/pull/44970) fixes a critical Denial-of-Service (DoS) vulnerability where a small, highly compressed OPUS audio file could expand into gigabytes of PCM data in memory before the duration guard could reject it, easily OOM-killing the server. Additionally, PR [#44974](https://github.com/vllm-project/vllm/pull/44974) ensures EXIF orientation tags are transposed and PNG tRNS transparency is properly composited before RGB conversion, preventing the model from seeing rotated or improperly blended images. Anyone exposing audio transcription or vision endpoints to untrusted user inputs must apply these fixes immediately.

## Kernels & attention
- Fused the `all_reduce -> RMSNorm -> per-group FP8 quant` chain into a single AITER call for DeepSeek-V3.2 on ROCm ([#42864](https://github.com/vllm-project/vllm/pull/42864)) — eliminates ~535us of standalone kernel launch overhead per decode step.
- Added a tuned Triton `fused_moe` FP8 configuration for Qwen3-Next-80B on H100 TP=4 ([#44830](https://github.com/vllm-project/vllm/pull/44830)) — yields a ~25% speedup for batch sizes between 96 and 512.
- Fused the split, QK-RMSNorm, partial RoPE, and gate copy operations into a single kernel for Qwen3.5 ([#44176](https://github.com/vllm-project/vllm/pull/44176)) — improves overall token throughput by reducing kernel launch overhead.
- Replaced `torch.cat` with a fused `concat_mla_q` kernel in the sparse-MLA `forward_mqa` path for ROCm ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — streamlines Multi-head Latent Attention (MLA) execution on AMD GPUs.

## Quantization
- Routed W8A8 and W4A16 (GPTQ) linear inference through zentorch kernels on AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — provides transparent, hardware-accelerated quantized inference on AMD CPUs with fallback to oneDNN.
- Refactored the compressed-tensors NVFP4 linear implementation to use a single unified class ([#42443](https://github.com/vllm-project/vllm/pull/42443)) — simplifies the codebase and standardizes FP4 weight handling.
- Added support for compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands the range of supported INT8 quantization schemes via the compressed-tensors format.
- Canonicalized FP8 weight layouts to `(K, N)` at the source during weight loading ([#44735](https://github.com/vllm-project/vllm/pull/44735)) — prevents downstream shape mismatches and bugs in ROCm quantization kernels.

## Parallelism & scheduling
- Fixed KV cache sharing crashes when using Hierarchical Memory Architecture (HMA) by filtering out layers that don't need a KV cache ([#44629](https://github.com/vllm-project/vllm/pull/44629)) — ensures stable disaggregated serving for models with shared or omitted KV layers.
- Integrated DeepEP v2 for Wide Expert Parallelism ([#41183](https://github.com/vllm-project/vllm/pull/41183)) — advances the scheduling and communication primitives for massive MoE expert parallelism.
- Fixed a deterministic hang in multi-node Ray data-parallel serving by excluding the Ray DP backend from deferred port allocation ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — unblocks large-scale multi-node data-parallel deployments.
- Added an object store as a secondary tier for multi-tier KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — enables more flexible and scalable KV cache eviction policies to external storage.

## Model support
- Added support for JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — brings native inference for this open-weights Mixture-of-Experts code-generation model.
- Implemented native support for Gemma4 Unified (encoder-free) ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — enables efficient serving of Google's latest unified multimodal architecture.
- Fixed DeepSeek-V4 initialization by resolving a CUTLASS `fmin` compatibility issue (0decac0d) — unblocks DSV4 loading on recent CUDA toolchains.
- Fixed `Cohere2MoE` weight loading failures with Transformers >= 5.10 by normalizing the `mlp_layer_types` config ([#44747](https://github.com/vllm-project/vllm/pull/44747)) — ensures Cohere MoE models load correctly with newer HuggingFace versions.

## Hardware
- Stabilized sleep-mode memory release on ROCm by properly cycling the virtual address reservation ([#43022](https://github.com/vllm-project/vllm/pull/43022)) — prevents intermittent HIP OOM errors when waking up MI300 GPUs.
- Added XPU support for the DeepSeek-V4 fused MHC post+pre path ([#44144](https://github.com/vllm-project/vllm/pull/44144)) — aligns Intel XPU decoder loops with the optimized AMD/CUDA patterns.
- Implemented a fused W4A16 HIP MoE kernel for AMD RDNA3 (gfx1100) ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — brings dedicated MoE acceleration to consumer/prosumer AMD GPUs.
- Upgraded the `torch-xpu` dependency to 2.12 ([#42262](https://github.com/vllm-project/vllm/pull/42262)) — keeps the Intel GPU backend aligned with the latest PyTorch XPU features and fixes.

## API & serving
- Added `/tokenize` and `/detokenize` endpoints to the Rust frontend ([#44222](https://github.com/vllm-project/vllm/pull/44222)) — brings the high-performance Rust server closer to feature parity with the Python OpenAI-compatible server.
- Implemented API key authentication middleware in the Rust frontend ([#44321](https://github.com/vllm-project/vllm/pull/44321)) — secures Rust-based deployments without needing an external reverse proxy.
- Preserved Kimi K2 model-generated tool-call IDs in the Rust frontend output path ([#44901](https://github.com/vllm-project/vllm/pull/44901)) — ensures streamed tool calls match the Python backend's behavior for downstream agents.
- Normalized Cohere Command streaming tool-call deltas to omit placeholder fields ([#44907](https://github.com/vllm-project/vllm/pull/44907)) — ensures OpenAI-compatible tool-calling clients don't choke on empty intermediate function names.

## Watch list
- **KV Event Schema Change**: The KV cache event structs published over ZMQ are switching from positional arrays to map encoding ([#42892](https://github.com/vllm-project/vllm/pull/42892)) — external subscribers (like routing daemons) must update their parsers to handle the new dictionary-based schema.
- **NixlConnector Deprecation**: The `kv_both` role in `NixlConnector` is entering its deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this specific KV transfer role should plan to migrate to the newer HMA defaults.
- **P2pNcclConnector Removal**: The legacy `P2pNcclConnector` has been completely removed ([#44854](https://github.com/vllm-project/vllm/pull/44854)) — any custom disaggregated prefill setups still referencing this connector must migrate to the modern KV connector APIs.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (226)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/44828">#44828</a> [BugFix] Use served model name in gemma4 audio-tower error message — by <a href="https://github.com/llsj14">llsj14</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43663">#43663</a> [XPU][CI] Add more test cases in Intel GPU CI — by <a href="https://github.com/zxd1997066">zxd1997066</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44761">#44761</a> [ROCm][CI] Stabilizing teardown and timeout of flaky tests to prevent rare OOMs — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42793">#42793</a> [ROCm][CI] Stage C mirrors — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><em>…and 166 more</em></li>
</ul>
</details>
