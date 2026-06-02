# vLLM weekly digest — 2026-06-02 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (229)

<details><summary>Click to expand the raw list</summary>

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
- [#42967](https://github.com/vllm-project/vllm/pull/42967) [Bugfix] Sync block_size from EngineCore to frontend for hybrid Mamba… — @Gruner-atero → `nan`
- [#44170](https://github.com/vllm-project/vllm/pull/44170) [Frontend] Consolidate dev entrypoints. — @noooop → `nan`
- [#42971](https://github.com/vllm-project/vllm/pull/42971) Fix DFlash prefix cache corruption due to missing lookahead block — @shreyas269 → `nan`
- [#43421](https://github.com/vllm-project/vllm/pull/43421) [XPU][Mamba] Triton-based selective scan forward op for XPU — @mfylcek → `nan`
- [#44206](https://github.com/vllm-project/vllm/pull/44206) [KV Offload] Add `on_schedule_end()` hook to separate step lifecycle from event draining — @ronensc → `nan`
- [#43754](https://github.com/vllm-project/vllm/pull/43754) [HARDWARE][POWER] Enable SHM communicator support for PowerPC — @Rukhaiya2004 → `nan`
- [#44126](https://github.com/vllm-project/vllm/pull/44126) [Multimodal] Automatically select registered video loader for VLM — @Isotr0py → `nan`
- [#42977](https://github.com/vllm-project/vllm/pull/42977) [Parser] Migrate `ResponsesParser` to unified `Parser` interface — @albertoperdomo2 → `nan`
- [#41627](https://github.com/vllm-project/vllm/pull/41627) [EC Connector] Non blocking EC Connector lookup — @omerpaz95 → `nan`
- [#44165](https://github.com/vllm-project/vllm/pull/44165) [Core][Refactor]: thread `scheduler_block_size` into KVCacheManager and KVCacheCoordinator — @ivanium → `nan`
- [#43883](https://github.com/vllm-project/vllm/pull/43883) [Rust Frontend] add  --enable-request-id-headers flag support. — @cinnamonica02 → `nan`
- [#44177](https://github.com/vllm-project/vllm/pull/44177) [kv_offload] Add `@override` decorators to subclass method implementations — @ronensc → `nan`
- [#43534](https://github.com/vllm-project/vllm/pull/43534) [CPU][Perf] Enable fused kernels for GDN's gated delta rules — @fadara01 → `nan`
- [#44220](https://github.com/vllm-project/vllm/pull/44220) [Perf] use triton moe backend on hopper by default — @ZJY0516 → `nan`
- [#44267](https://github.com/vllm-project/vllm/pull/44267) [Refactor] Unify reasoning + tool-call parsing behind Parser.parse() — @sfeng33 → `nan`
- [#43991](https://github.com/vllm-project/vllm/pull/43991) [Model Runner V2] Use actual batch max_seq_len for attn metadata — @izhuhaoran → `nan`
- [#43990](https://github.com/vllm-project/vllm/pull/43990) [Model Runner V2] Support zeroing freshly allocated KV blocks for hybrid + fp8 KVCache — @izhuhaoran → `nan`
- [#43798](https://github.com/vllm-project/vllm/pull/43798) [Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl — @Isotr0py → `nan`
- [#41714](https://github.com/vllm-project/vllm/pull/41714) [MM][CG] Profile encoder CUDA graph pool memory — @BWAAEEEK → `nan`
- [#43930](https://github.com/vllm-project/vllm/pull/43930) [XPU][Bugfix] Fix per_token_group_fp8_quant missing dummy args on XPU — @chaojun-zhang → `nan`
- [#42959](https://github.com/vllm-project/vllm/pull/42959) [BugFix][kv_offload]: Prevent offloading stale sliding window blocks — @orozery → `nan`
- [#38053](https://github.com/vllm-project/vllm/pull/38053) [BugFix] Fix TypeError in MiniCPM-O audio feature unpadding — @Krishnachaitanyakc → `nan`
- [#44131](https://github.com/vllm-project/vllm/pull/44131) [CI] Stabilize OpenAI schema fuzzing for malformed structural tags — @AndreasKaratzas → `nan`
- [#44017](https://github.com/vllm-project/vllm/pull/44017) [Refactor] Move unstreamed tool-arg flush from serving layer to parser — @sfeng33 → `nan`
- [#44266](https://github.com/vllm-project/vllm/pull/44266) [Bugfix][CI] Normalize NIXL connector CUDA wheel installs — @alec-flowers → `nan`
- [#44265](https://github.com/vllm-project/vllm/pull/44265) [ROCm] Upgrade AITER to v0.1.13.post1 — @micah-wil → `nan`
- [#43742](https://github.com/vllm-project/vllm/pull/43742) [Bugfix][Mooncake] Release GPU pin on failed store in MooncakeStoreConnector — @Dao007forever → `nan`
- [#44262](https://github.com/vllm-project/vllm/pull/44262) [DSV4] Refactor RoPE initialization — @WoosukKwon → `nan`
- [#44256](https://github.com/vllm-project/vllm/pull/44256) [ROCm][CI] Skip unbacked dynamic shapes tests on PyTorch < 2.11 — @JartX → `nan`
- [#44246](https://github.com/vllm-project/vllm/pull/44246) [DSV4] Remove unncessary classes & functions — @WoosukKwon → `nan`
- [#44234](https://github.com/vllm-project/vllm/pull/44234) [Test][BugFix] Fix double-BOS in PD+specdec acceptance test — @njhill → `nan`
- [#44248](https://github.com/vllm-project/vllm/pull/44248) [BugFix][CI] Fix added `_has_module` tests — @njhill → `nan`
- [#40096](https://github.com/vllm-project/vllm/pull/40096) [Frontend][Core] Add sparse NCCL weight transfer support for in-place updates — @bedeks → `nan`
- [#43779](https://github.com/vllm-project/vllm/pull/43779) [Rust Frontend] Support streaming `generate` endpoint — @Xunzhuo → `nan`
- [#41294](https://github.com/vllm-project/vllm/pull/41294) [ROCm][CI] Fix and stabilize EAGLE3 acceptance tests — @AndreasKaratzas → `nan`
- [#44161](https://github.com/vllm-project/vllm/pull/44161) [Kernel][DSv4] Optimize sparse FP8 compressor kernels — @zyongye → `nan`
- [#43992](https://github.com/vllm-project/vllm/pull/43992) [Feature] Add support for JetBrains' Mellum v2 code generation model — @shadeMe → `nan`
- [#43706](https://github.com/vllm-project/vllm/pull/43706) [Perf] Optimize cutlass fp8 scaled mm bypassing padding, 20% kernel performance improvement — @yewentao256 → `nan`
- [#44146](https://github.com/vllm-project/vllm/pull/44146) [XPU][CI] Fix test_audio_in_video flake by using module-scoped server fixture — @chaojun-zhang → `nan`
- _…and 169 more_

</details>
