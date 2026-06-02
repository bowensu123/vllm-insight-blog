# vLLM weekly digest — 2026-06-02 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (218)

<details><summary>Click to expand the raw list</summary>

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
- [#43770](https://github.com/vllm-project/vllm/pull/43770) [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model. — @zzt93 → `nan`
- [#43481](https://github.com/vllm-project/vllm/pull/43481) [Rust Frontend] Add InternLM2 tool parser — @willamhou → `nan`
- [#44153](https://github.com/vllm-project/vllm/pull/44153) [Frontend] Resettle generative scoring entrypoint. — @noooop → `nan`
- [#42944](https://github.com/vllm-project/vllm/pull/42944) fix: glm5.1 pp model loading — @UranusSeven → `nan`
- [#42730](https://github.com/vllm-project/vllm/pull/42730) [CPU][RISC-V] Add missing RVV cpu_types helpers for WNA16 — @wcynb1023 → `nan`
- [#44159](https://github.com/vllm-project/vllm/pull/44159) [Docs] Replace broken video url in examples — @Isotr0py → `nan`
- [#44035](https://github.com/vllm-project/vllm/pull/44035) [BugFix] Fix `_has_module` to verify native deps via trial import — @jeffreywang-anyscale → `nan`
- [#44078](https://github.com/vllm-project/vllm/pull/44078) [MRV2] Remove Eagle's dedicated CUDA graph pool — @LucasWilkinson → `nan`
- [#36254](https://github.com/vllm-project/vllm/pull/36254) [Misc] Use VLLMValidationError consistently in chat completion and completion protocol validators — @umut-polat → `nan`
- [#44118](https://github.com/vllm-project/vllm/pull/44118) docs: fix MLA attention docstring examples — @nightcityblade → `nan`
- [#43956](https://github.com/vllm-project/vllm/pull/43956) [CI/Build] Enable Step3p7ForConditionalGeneration testing — @jeejeelee → `nan`
- [#41813](https://github.com/vllm-project/vllm/pull/41813) [CPU][Zen] Route W8A8 and W4A16 linear inference through zentorch on AMD Zen CPUs — @aadwived → `nan`
- [#44050](https://github.com/vllm-project/vllm/pull/44050) [MRV2] Support breakable CUDA graph — @WoosukKwon → `nan`
- [#43909](https://github.com/vllm-project/vllm/pull/43909) [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered` — @yewentao256 → `nan`
- [#44047](https://github.com/vllm-project/vllm/pull/44047) [Governance] Add @BugenZhao as Rust frontend code owner — @BugenZhao → `nan`
- [#43817](https://github.com/vllm-project/vllm/pull/43817) [ROCm] Add attention sink support to AITer flash attention backend — @sphinx07 → `nan`
- [#42379](https://github.com/vllm-project/vllm/pull/42379) [Bugfix] Fix RMSNorm kernels to multiply in weight's native dtype — @liulanze → `nan`
- [#43571](https://github.com/vllm-project/vllm/pull/43571) [BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py — @Liangliang-Ma → `nan`
- [#43881](https://github.com/vllm-project/vllm/pull/43881) [ROCm] cmake: support PYTORCH_FOUND_HIP for torch 2.13 native HIP language support — @nemanjaudovic → `nan`
- [#44028](https://github.com/vllm-project/vllm/pull/44028) [ROCm][CI] Fix failure in the Phi3V pooling test — @AndreasKaratzas → `nan`
- [#43997](https://github.com/vllm-project/vllm/pull/43997) [Refactor] Remove dead current_tool_name_sent assignments from tool parsers — @sfeng33 → `nan`
- [#43792](https://github.com/vllm-project/vllm/pull/43792) offload prompt_embeds decode in render_prompts_async to avoid blocking — @gagandhakrey → `nan`
- [#38445](https://github.com/vllm-project/vllm/pull/38445) [PERF]MiniMax-M2 gate kernel — @jeejeelee → `nan`
- [#44033](https://github.com/vllm-project/vllm/pull/44033) Revert "[MoE Refactor] Migrate MoeWNA16Method quantization to MK orac… — @bnellnm → `nan`
- [#43974](https://github.com/vllm-project/vllm/pull/43974) [CI] Fix smoke test step key to bypass block gate — @khluu → `nan`
- [#44023](https://github.com/vllm-project/vllm/pull/44023) [CI] Remove duplicate Harmony test coverage — @sfeng33 → `nan`
- _…and 158 more_

</details>
