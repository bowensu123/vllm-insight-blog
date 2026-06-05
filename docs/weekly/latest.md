# vLLM weekly digest — 2026-06-05 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) — 2026-05-29 10:28 UTC

## PRs merged this window (224)

<details><summary>Click to expand the raw list</summary>

- [#44066](https://github.com/vllm-project/vllm/pull/44066) docs: fix tokenizer optimization typo — @chunyang-wen → `nan`
- [#43874](https://github.com/vllm-project/vllm/pull/43874) [NixlConnector] Initiate deprecation cycle for `kv_both` role  — @NickLucche → `nan`
- [#44391](https://github.com/vllm-project/vllm/pull/44391) [Rust Frontend] Support include_reasoning=false — @ricky-chaoju → `nan`
- [#44622](https://github.com/vllm-project/vllm/pull/44622) [Bugfix] Update mistral tokenizer test for continue_final_message fix — @XuZhou26 → `nan`
- [#44603](https://github.com/vllm-project/vllm/pull/44603) fix: pad dummy run query_start_loc — @UranusSeven → `nan`
- [#44618](https://github.com/vllm-project/vllm/pull/44618) [Bugfix] Fix test_invocations flaky failure with newer openai SDK — @XuZhou26 → `nan`
- [#44620](https://github.com/vllm-project/vllm/pull/44620) [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer — @Sunt-ing → `nan`
- [#44617](https://github.com/vllm-project/vllm/pull/44617) Fix `LLM.wait_for_completion` output type docstring — @viiccwen → `nan`
- [#41002](https://github.com/vllm-project/vllm/pull/41002) [ROCm][perf] Use workspace manager for sparse indexer allocations — @tuukkjs → `nan`
- [#40426](https://github.com/vllm-project/vllm/pull/40426) [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning — @hanlin12-AMD → `nan`
- [#44605](https://github.com/vllm-project/vllm/pull/44605) [CI/Build] Disable CPU-Compatibility Tests — @bigPYJ1151 → `nan`
- [#43720](https://github.com/vllm-project/vllm/pull/43720) [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing — @zixi-qi → `nan`
- [#44571](https://github.com/vllm-project/vllm/pull/44571) [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified — @lucianommartins → `nan`
- [#44569](https://github.com/vllm-project/vllm/pull/44569) [DSV4] Refactor DeepseekV4Attention — @WoosukKwon → `nan`
- [#44334](https://github.com/vllm-project/vllm/pull/44334) [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. — @cleonard530 → `nan`
- [#42139](https://github.com/vllm-project/vllm/pull/42139) [XPU][MoE] support block_fp8_moe on xpu — @zufangzhu → `nan`
- [#44500](https://github.com/vllm-project/vllm/pull/44500) [Rust Frontend] Skip loading multimodal processor if `--language-model-only` is specified — @BugenZhao → `nan`
- [#43926](https://github.com/vllm-project/vllm/pull/43926) fix: keep DeepSeek V4 RoPE cache on inv_freq device — @galletas1712 → `nan`
- [#44539](https://github.com/vllm-project/vllm/pull/44539) [mamba] unify KDA conv states into one cache to match 2-state SSM layout — @ZJY0516 → `nan`
- [#43707](https://github.com/vllm-project/vllm/pull/43707) [Logs Refactor] Optimize shutdown logs, easier to follow and consistent — @yewentao256 → `nan`
- [#41980](https://github.com/vllm-project/vllm/pull/41980) use split_group for pytorch process group creation — @tushar00jain → `nan`
- [#43307](https://github.com/vllm-project/vllm/pull/43307) [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU — @adobrzyn → `nan`
- [#44380](https://github.com/vllm-project/vllm/pull/44380) [Bugfix] Fix test_cutlass_moe.py — @bnellnm → `nan`
- [#44471](https://github.com/vllm-project/vllm/pull/44471) [Misc] Add unit tests for pooler head classes — @taneem-ibrahim → `nan`
- [#34894](https://github.com/vllm-project/vllm/pull/34894) [DOC] Add INT8 W4A8 docs and Arm's supported quantization schemes — @fadara01 → `nan`
- [#44436](https://github.com/vllm-project/vllm/pull/44436) [ROCm][CI] Add test for Aiter unified attn kernel — @divakar-amd → `nan`
- [#44057](https://github.com/vllm-project/vllm/pull/44057) [Bugfix] Reject non-positive values for ParallelConfig int knobs — @jwzheng96 → `nan`
- [#44363](https://github.com/vllm-project/vllm/pull/44363) [Core] Freeze garbage collector in workers after model initialization — @tlrmchlsmth → `nan`
- [#44509](https://github.com/vllm-project/vllm/pull/44509) [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count — @tc-mb → `nan`
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
- _…and 164 more_

</details>
