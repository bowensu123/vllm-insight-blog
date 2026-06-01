# vLLM weekly digest — 2026-06-01 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (202)

<details><summary>Click to expand the raw list</summary>

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
- [#43108](https://github.com/vllm-project/vllm/pull/43108) [MoE Refactor] Remove supports_expert_map — @bnellnm → `nan`
- [#42647](https://github.com/vllm-project/vllm/pull/42647) [MoE Refactor] Migrate MoeWNA16Method quantization to MK oracle — @bnellnm → `nan`
- [#44009](https://github.com/vllm-project/vllm/pull/44009) [Frontend] Clean up stop_token_ids override for Harmony — @yzong-rh → `nan`
- [#43346](https://github.com/vllm-project/vllm/pull/43346) [Metrics] Exclude KV transfer tokens from iteration_tokens_total — @tlrmchlsmth → `nan`
- [#43688](https://github.com/vllm-project/vllm/pull/43688) [Feature] SSL support for dp supervisor — @yewentao256 → `nan`
- [#44019](https://github.com/vllm-project/vllm/pull/44019) Add @khluu to CODEOWNERS — @khluu → `nan`
- [#44011](https://github.com/vllm-project/vllm/pull/44011) [CI] Remove redundant test_chat_with_tool_reasoning.py — @sfeng33 → `nan`
- [#43971](https://github.com/vllm-project/vllm/pull/43971) [CI] Make Model Executor test hangs fail fast with a traceback — @khluu → `nan`
- [#44005](https://github.com/vllm-project/vllm/pull/44005) [Bug] Fix torch device issue for MOE permute — @yewentao256 → `nan`
- [#43998](https://github.com/vllm-project/vllm/pull/43998) [Bugfix] Fix Ray placement group allocation with grouped nodes — @czhu-cohere → `nan`
- [#43988](https://github.com/vllm-project/vllm/pull/43988) [Bugfix] Use storage_block_size in KV cache reshape for compressed specs (DeepSeek V4) — @zixi-qi → `nan`
- [#43219](https://github.com/vllm-project/vllm/pull/43219) [EPLB] Make async EPLB default — @ilmarkov → `nan`
- [#42553](https://github.com/vllm-project/vllm/pull/42553) [MoE Refactor] WNA16 MoE backend selection into oracle module — @bnellnm → `nan`
- [#43616](https://github.com/vllm-project/vllm/pull/43616) [Bugfix] Disable allreduce_rms_fusion when pipeline_parallel_size > 1 — @zixi-qi → `nan`
- [#43818](https://github.com/vllm-project/vllm/pull/43818) [Misc] added unit tests for the core pooling methods — @taneem-ibrahim → `nan`
- [#43922](https://github.com/vllm-project/vllm/pull/43922) docs: clarify ITL acronym in optimization docs — @chunyang-wen → `nan`
- [#43857](https://github.com/vllm-project/vllm/pull/43857) Add vLLM library info to Hugging Face Hub requests — @Wauplin → `nan`
- [#43977](https://github.com/vllm-project/vllm/pull/43977) [Bugfix][CPU] Remove invalid extra deps — @bigPYJ1151 → `nan`
- [#43972](https://github.com/vllm-project/vllm/pull/43972) Skip docs build if PR doesn't affect docs — @hmellor → `nan`
- [#43961](https://github.com/vllm-project/vllm/pull/43961) [Bugfix] Corrupted MLA + linear attention — @gau-nernst → `nan`
- [#42982](https://github.com/vllm-project/vllm/pull/42982) [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts) — @frida-andersson → `nan`
- [#42595](https://github.com/vllm-project/vllm/pull/42595) [Bugfix] [ROCm] [DSV4] Fix AITER MXFP4 MoE weight loading and shuffle… — @MHYangAMD → `nan`
- [#41394](https://github.com/vllm-project/vllm/pull/41394) [Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16 — @JartX → `nan`
- [#37622](https://github.com/vllm-project/vllm/pull/37622) [Bugfix] Fix Step3 pipeline parallel KeyError for residual tensor — @JMonde → `v0.22.0`
- [#43871](https://github.com/vllm-project/vllm/pull/43871) [CI] Nixl+SimpleCPUOffloadingConnector unit tests — @NickLucche → `v0.22.0`
- [#43565](https://github.com/vllm-project/vllm/pull/43565) [XPU] support MTP of gdn attention — @mayuyuace → `v0.22.0`
- [#43703](https://github.com/vllm-project/vllm/pull/43703) [CI][ROCm] Don't skip MoRI-IO Connector tests — @simondanielsson → `v0.22.0`
- [#43947](https://github.com/vllm-project/vllm/pull/43947) [XPU] fix xpu install document triton-xpu version — @jikunshang → `v0.22.0`
- [#43945](https://github.com/vllm-project/vllm/pull/43945) [ROCm][CI] Fix AITER unified attention for encoder-decoder cross-attention — @AndreasKaratzas → `v0.22.0`
- [#43761](https://github.com/vllm-project/vllm/pull/43761) [Frontend]Responses API supports chat_template_kwargs — @chaunceyjiang → `v0.22.0`
- [#43898](https://github.com/vllm-project/vllm/pull/43898) [ROCm][DSv4] Remove device pipeline stall in sparse attention — @kliuae → `v0.22.0`
- _…and 142 more_

</details>
