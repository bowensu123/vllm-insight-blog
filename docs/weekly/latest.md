# vLLM weekly digest — 2026-06-03 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (244)

<details><summary>Click to expand the raw list</summary>

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
- [#44251](https://github.com/vllm-project/vllm/pull/44251) [Perf] Add tuned selective_state_update configs for H200 and RTX PRO … — @Majid-Taheri → `nan`
- [#36949](https://github.com/vllm-project/vllm/pull/36949) [ROCm][CI] Optimize ROCm Docker build: registry cache, DeepEP, and ci-bake script — @AndreasKaratzas → `nan`
- [#42758](https://github.com/vllm-project/vllm/pull/42758) Enable perf_token_group_quant/_C_stable_libtorch for ROCm — @charlifu → `nan`
- [#44244](https://github.com/vllm-project/vllm/pull/44244) [Benchmark] Enable reasoning-model (thinking) benchmarking via `--chat-template-kwargs` for client-rendered datasets — @qiching → `nan`
- [#43862](https://github.com/vllm-project/vllm/pull/43862) [Bugfix] fix crash in postprocess for null tool args  — @william-rom → `nan`
- [#44236](https://github.com/vllm-project/vllm/pull/44236) fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init — @Oxygen56 → `nan`
- [#44293](https://github.com/vllm-project/vllm/pull/44293) Nit Changes in Tiered KV Offload — @rshavitt → `nan`
- [#44352](https://github.com/vllm-project/vllm/pull/44352) [CI] Add missing vllm/parser/ CI trigger and fix test_parse.py  — @sfeng33 → `nan`
- [#44042](https://github.com/vllm-project/vllm/pull/44042) [CI] Reject out-of-vocabulary  before they reach the GPU logprob path — @AndreasKaratzas → `nan`
- [#44369](https://github.com/vllm-project/vllm/pull/44369) [ROCm][CI] Skip fp8 reload tests on gfx90a (MI250) — @JartX → `nan`
- [#44368](https://github.com/vllm-project/vllm/pull/44368) [ROCm][CI] Fix stale wvSplitK GEMM fallback test for N=5 — @JartX → `nan`
- [#43838](https://github.com/vllm-project/vllm/pull/43838) [Platform] Add is_cumem_allocator_available — @wangxiyuan → `nan`
- [#44366](https://github.com/vllm-project/vllm/pull/44366) [docker] Stop using extra-index-url for flashinfer-jit-cache — @khluu → `nan`
- [#44356](https://github.com/vllm-project/vllm/pull/44356) [Bugfix] Fix Deepseek v4 non-mega-moe model init error — @wzhao18 → `nan`
- [#42191](https://github.com/vllm-project/vllm/pull/42191) [Perf] Apply single-pass min_larger finding and binary search in Triton Top-p path. — @cakeng → `nan`
- [#44367](https://github.com/vllm-project/vllm/pull/44367) [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts — @WoosukKwon → `nan`
- [#44128](https://github.com/vllm-project/vllm/pull/44128) [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it — @DaoyuanLi2816 → `nan`
- [#43332](https://github.com/vllm-project/vllm/pull/43332) [MoE/b12x] Accept W4A16 (kNvfp4Static, None) in FlashInferB12xExperts supports check — @ECMGit → `nan`
- [#44036](https://github.com/vllm-project/vllm/pull/44036) [CI/Build] Bump flashinfer to v0.6.12 — @vadiklyutiy → `nan`
- [#44345](https://github.com/vllm-project/vllm/pull/44345) [BugFix] Fix sparse NCCL weight transfer test construction — @bedeks → `nan`
- [#42027](https://github.com/vllm-project/vllm/pull/42027) [Kernel][MoE] Add GELU_TANH to CPU, CUTLASS, and WNA16 MoE backends — @lesj0610 → `nan`
- [#42187](https://github.com/vllm-project/vllm/pull/42187) [ModelRunnerV2] Avoid pipeline parallel bubbles — @njhill → `nan`
- [#44350](https://github.com/vllm-project/vllm/pull/44350) [Misc] Remove stray empty file — @MatthewBonanni → `nan`
- [#44082](https://github.com/vllm-project/vllm/pull/44082) [Bugfix] Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask — @ivanium → `nan`
- [#44338](https://github.com/vllm-project/vllm/pull/44338) [MRV2] Remove assignment of graph_pool in cudagraph_utils — @WoosukKwon → `nan`
- [#39667](https://github.com/vllm-project/vllm/pull/39667) Bump actions/github-script from 8.0.0 to 9.0.0 — @dependabot[bot] → `nan`
- [#43458](https://github.com/vllm-project/vllm/pull/43458) [MRV2] Also enable MRV2 for Llama and Mistral dense models  — @njhill → `nan`
- [#44283](https://github.com/vllm-project/vllm/pull/44283) [Anthropic] Support system role messages inside messages array — @chaunceyjiang → `nan`
- [#43339](https://github.com/vllm-project/vllm/pull/43339) [Feature] Support EPLB for DeepSeek v4 Mega Moe — @wzhao18 → `nan`
- [#43669](https://github.com/vllm-project/vllm/pull/43669) [Bugfix] flashinfer: fail fast when --kv-cache-dtype nvfp4 used on unsupported arch — @Kartavyasonar → `nan`
- [#43100](https://github.com/vllm-project/vllm/pull/43100) [BugFix] Fix Humming MoE deploy error — @adotdad → `nan`
- [#43963](https://github.com/vllm-project/vllm/pull/43963) [XPU] Enable rms_norm/act quant fusions — @zhenwei-intel → `nan`
- [#44279](https://github.com/vllm-project/vllm/pull/44279) [Refactor] Remove dead code from parser infrastructure — @sfeng33 → `nan`
- [#44274](https://github.com/vllm-project/vllm/pull/44274) [Core] Move `max_concurrent_batches` to `VllmConfig` — @njhill → `nan`
- [#44025](https://github.com/vllm-project/vllm/pull/44025) [compressed-tensors] Asymmetric support for MoE WNA16 marlin — @brian-dellabetta → `nan`
- [#43843](https://github.com/vllm-project/vllm/pull/43843) [Misc] Support local image encoding in benchmarks — @xiaozcy → `nan`
- [#44013](https://github.com/vllm-project/vllm/pull/44013) Migrate header files to torch stable abi — @cleonard530 → `nan`
- [#44320](https://github.com/vllm-project/vllm/pull/44320) [Rust Frontend] Cover different thinking modes in roundtrip tests — @BugenZhao → `nan`
- [#44308](https://github.com/vllm-project/vllm/pull/44308) [ROCm] Fix AITER RMSNormQuantFusion for Kimi-Linear — @pschlan-amd → `nan`
- [#44299](https://github.com/vllm-project/vllm/pull/44299) [Rust Frontend] Support recursive tool parameter conversion — @BugenZhao → `nan`
- [#44168](https://github.com/vllm-project/vllm/pull/44168) [XPU] [Bug] remove xpuw4a16 output size check — @zufangzhu → `nan`
- [#43978](https://github.com/vllm-project/vllm/pull/43978) [BugFix] [GDN] Read linear_key_head_dim from hf_text_config for multimodal models — @IdoAtadTD → `nan`
- [#44065](https://github.com/vllm-project/vllm/pull/44065) [FlashAttention] Sync FA with upstream — @MatthewBonanni → `nan`
- [#44282](https://github.com/vllm-project/vllm/pull/44282) [Bugfix] Vendor MiniCPMV/MiniCPMO processors to unblock Transformers v5  — @wjinxu → `nan`
- [#42958](https://github.com/vllm-project/vllm/pull/42958) Support ModelOpt MXFP8 non-gated MoE — @TomerBN-Nvidia → `nan`
- [#44232](https://github.com/vllm-project/vllm/pull/44232) [Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor — @lucianommartins → `nan`
- _…and 184 more_

</details>
