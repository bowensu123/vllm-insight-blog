# vLLM weekly digest — 2026-06-04 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (237)

<details><summary>Click to expand the raw list</summary>

- [#43519](https://github.com/vllm-project/vllm/pull/43519) Add model support for granite speech plus — @zvik → `nan`
- [#44340](https://github.com/vllm-project/vllm/pull/44340) [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings — @mgoin → `nan`
- [#43827](https://github.com/vllm-project/vllm/pull/43827) [DSv4] Adding TRTLLM gen attention kernel — @zyongye → `nan`
- [#44255](https://github.com/vllm-project/vllm/pull/44255) [ROCm][CI] Specifying time outs for the lm eval models — @AndreasKaratzas → `nan`
- [#44046](https://github.com/vllm-project/vllm/pull/44046) [ROCm][CI] Stabilize memory-release in the Hybrid model generation tests — @AndreasKaratzas → `nan`
- [#43625](https://github.com/vllm-project/vllm/pull/43625) [ROCm] Bump fastsafetensors to v0.3.2 from PyPI, remove git source build — @wjabbour → `nan`
- [#42554](https://github.com/vllm-project/vllm/pull/42554) [PD][Nixl] Mamba prefix caching mode support  — @NickLucche → `nan`
- [#44476](https://github.com/vllm-project/vllm/pull/44476) [Bugfix][Compile] Guard per_token_group_fp8_quant lookup on non-CUDA platforms — @QiliangCui2023 → `nan`
- [#44534](https://github.com/vllm-project/vllm/pull/44534) Add GH token to docs build pre run check — @hmellor → `nan`
- [#42443](https://github.com/vllm-project/vllm/pull/42443) Refactor CT NVFP4 linear to use a single class — @dsikka → `nan`
- [#44205](https://github.com/vllm-project/vllm/pull/44205) [Bugfix] fix EVS for qwen3-vl — @garrygale → `nan`
- [#43556](https://github.com/vllm-project/vllm/pull/43556) [Attention] Mamba attention module refactor - LINEAR — @wangxiyuan → `nan`
- [#42646](https://github.com/vllm-project/vllm/pull/42646) [perf] Add gemma RMS AR fusion — @jiahanc → `nan`
- [#44493](https://github.com/vllm-project/vllm/pull/44493) [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata — @Kevin-XiongC → `nan`
- [#43447](https://github.com/vllm-project/vllm/pull/43447) [Prefix Caching] DeepSeekv4 - Support selective prefix-cache retention for sliding-window KV cache — @wzhao18 → `nan`
- [#44497](https://github.com/vllm-project/vllm/pull/44497) [CI] Reverted gitignore changes — @AndreasKaratzas → `nan`
- [#44479](https://github.com/vllm-project/vllm/pull/44479) [Frontend] Consolidate online serving utils. — @noooop → `nan`
- [#42129](https://github.com/vllm-project/vllm/pull/42129) [Inductor] Fast-path Inductor fallback for vllm::*/vllm_aiter::* custom ops — @okorzh-amd → `nan`
- [#44463](https://github.com/vllm-project/vllm/pull/44463) [CI] Resolve release V2 docker build after ROCm CI wheels change — @AndreasKaratzas → `nan`
- [#41633](https://github.com/vllm-project/vllm/pull/41633) [EPLB] Nixl communicator optimization. Zero-copy transfers — @ilmarkov → `nan`
- [#44230](https://github.com/vllm-project/vllm/pull/44230) optimize the compressor 128 split cutedsl kernel  — @Jie-Fang → `nan`
- [#41471](https://github.com/vllm-project/vllm/pull/41471) [Refactor] Remove dead code in tests and parallel_state — @yewentao256 → `nan`
- [#41759](https://github.com/vllm-project/vllm/pull/41759) [MM][Perf][CG] Support ViT full CUDA graph for InternVL — @oguzhankir → `nan`
- [#42865](https://github.com/vllm-project/vllm/pull/42865) [KV Connector] Update lmcache kv_offloading_backend to use LMCacheMPConnector — @maobaolong → `nan`
- [#44410](https://github.com/vllm-project/vllm/pull/44410) [Bugfix] Fix VLLMNotFoundError when using LoRA adapter name in poolin… — @wanghenshui → `nan`
- [#43241](https://github.com/vllm-project/vllm/pull/43241) [Model Runner V2][Spec Decode] Add Gemma4 MTP support — @TheEpicDolphin → `nan`
- [#44289](https://github.com/vllm-project/vllm/pull/44289) [XPU] skip unapplied UT in test_gpu_model_runner.py — @yma11 → `nan`
- [#43982](https://github.com/vllm-project/vllm/pull/43982) [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load — @Dymasik → `nan`
- [#35078](https://github.com/vllm-project/vllm/pull/35078) Bump actions/stale from 10.1.1 to 10.3.0 — @dependabot[bot] → `nan`
- [#44442](https://github.com/vllm-project/vllm/pull/44442) [Minor] Remove FlashInfer version check in topk_topp_sampler — @WoosukKwon → `nan`
- [#44253](https://github.com/vllm-project/vllm/pull/44253) [Bug Fix][Model Runner V2][Spec Decode] Warmup & capture with different attention states for speculator prefill — @TheEpicDolphin → `nan`
- [#42752](https://github.com/vllm-project/vllm/pull/42752) [Bugfix] Honor tool_choice="none" in Chat Completions streaming — @hoobnn → `nan`
- [#42453](https://github.com/vllm-project/vllm/pull/42453) [Feature] Support batch invariant rms norm with residual — @yewentao256 → `nan`
- [#44429](https://github.com/vllm-project/vllm/pull/44429) [Model] Add Gemma4 Unified (encoder-free)  support — @lucianommartins → `nan`
- [#44413](https://github.com/vllm-project/vllm/pull/44413) [LoRA] Fix dedup for post-replacement module aliases — @linitra24 → `nan`
- [#44122](https://github.com/vllm-project/vllm/pull/44122) [Refactor] Remove dead code fp quant — @yewentao256 → `nan`
- [#44370](https://github.com/vllm-project/vllm/pull/44370) [ROCm][CI] Move Model Executor test step from MI250 to MI300 (gfx942) — @JartX → `nan`
- [#44365](https://github.com/vllm-project/vllm/pull/44365) [10b/n] Migrate custom all-reduce, DeepSeek V4 fused MLA, MiniMax reduce-RMS, and MXFP8 MoE to libtorch stable ABI — @cleonard530 → `nan`
- [#43659](https://github.com/vllm-project/vllm/pull/43659) Handle spinloop ext load failure gracefully — @pschlan-amd → `nan`
- [#44207](https://github.com/vllm-project/vllm/pull/44207) fix(config): validate max_num_scheduled_tokens >= 0 on all paths — @Oxygen56 → `nan`
- [#37505](https://github.com/vllm-project/vllm/pull/37505) [KVCache] Support Pluggable KVCacheSpec — @MengqingCao → `nan`
- [#44174](https://github.com/vllm-project/vllm/pull/44174) [CI] Align PD tests to HMA on by default — @NickLucche → `nan`
- [#44425](https://github.com/vllm-project/vllm/pull/44425) [CI/Build] Fix LoRA testing — @jeejeelee → `nan`
- [#42472](https://github.com/vllm-project/vllm/pull/42472) [Model Runner V2] Use FlashInfer sampler — @njhill → `nan`
- [#43590](https://github.com/vllm-project/vllm/pull/43590) [Frontend][Responses API] Fold developer-role input messages into system instructions — @chaunceyjiang → `nan`
- [#44346](https://github.com/vllm-project/vllm/pull/44346) [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers — @sfeng33 → `nan`
- [#39968](https://github.com/vllm-project/vllm/pull/39968) [XPU] Add XPU block-scaled W8A8 fp8 path — @xwu-intel → `nan`
- [#43942](https://github.com/vllm-project/vllm/pull/43942) [Rust Frontend] Add /server_info to Rust frontend — @Xunzhuo → `nan`
- [#43689](https://github.com/vllm-project/vllm/pull/43689) [SharedOffloadRegion] Align blocks to page-size   — @varun-sundar-rabindranath → `nan`
- [#44393](https://github.com/vllm-project/vllm/pull/44393) [Attention][CPU] Standardize kv layout to blocks first — @bigPYJ1151 → `nan`
- [#44212](https://github.com/vllm-project/vllm/pull/44212) [Perf] Improve multimodal item handling from O(n) to O(log n) per step — @andylolu2 → `nan`
- [#42212](https://github.com/vllm-project/vllm/pull/42212) [Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector — @Etelis → `nan`
- [#43759](https://github.com/vllm-project/vllm/pull/43759) [XPU]fallback to TRITON_ATTN for vit attn on xpu when use float32 dtype — @yma11 → `nan`
- [#44348](https://github.com/vllm-project/vllm/pull/44348) [Bugfix] Fix unstreamed tool call args dropped in Responses API streaming — @sfeng33 → `nan`
- [#44347](https://github.com/vllm-project/vllm/pull/44347) [Bugfix] Update TrtLLM MoE routing methods — @wzhao18 → `nan`
- [#44388](https://github.com/vllm-project/vllm/pull/44388) [Doc] Update ViT CUDA graph interfaces — @shen-shanshan → `nan`
- [#44311](https://github.com/vllm-project/vllm/pull/44311) [Rust Frontend] Fix several hf chat template rendering issues — @BugenZhao → `nan`
- [#43778](https://github.com/vllm-project/vllm/pull/43778) [Rust Frontend] Add dynamic LoRA endpoints — @Xunzhuo → `nan`
- [#43774](https://github.com/vllm-project/vllm/pull/43774) [Rust Frontend] Add server router extension hook — @NolanHo → `nan`
- [#44287](https://github.com/vllm-project/vllm/pull/44287) [KV Offloading] Enable HMA models for Tiering Offloading — @varun-sundar-rabindranath → `nan`
- _…and 177 more_

</details>
