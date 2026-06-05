# vLLM weekly digest — 2026-06-05 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: ANTHROPIC_API_KEY not set for anthropic backend_

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (221)

<details><summary>Click to expand the raw list</summary>

- [#43167](https://github.com/vllm-project/vllm/pull/43167) Remove KV cache scale boilerplate from model weight loading methods — @hmellor → `nan`
- [#43150](https://github.com/vllm-project/vllm/pull/43150) [BUG] Fix FP64 Gumbel precision coverage — @tianyu-z → `nan`
- [#44591](https://github.com/vllm-project/vllm/pull/44591) [Rust Frontend] Batch auto-abort requests by engine — @HueCodes → `v0.22.1`
- [#44066](https://github.com/vllm-project/vllm/pull/44066) docs: fix tokenizer optimization typo — @chunyang-wen → `v0.22.1`
- [#43874](https://github.com/vllm-project/vllm/pull/43874) [NixlConnector] Initiate deprecation cycle for `kv_both` role  — @NickLucche → `v0.22.1`
- [#44391](https://github.com/vllm-project/vllm/pull/44391) [Rust Frontend] Support include_reasoning=false — @ricky-chaoju → `v0.22.1`
- [#44622](https://github.com/vllm-project/vllm/pull/44622) [Bugfix] Update mistral tokenizer test for continue_final_message fix — @XuZhou26 → `v0.22.1`
- [#44603](https://github.com/vllm-project/vllm/pull/44603) fix: pad dummy run query_start_loc — @UranusSeven → `v0.22.1`
- [#44618](https://github.com/vllm-project/vllm/pull/44618) [Bugfix] Fix test_invocations flaky failure with newer openai SDK — @XuZhou26 → `v0.22.1`
- [#44620](https://github.com/vllm-project/vllm/pull/44620) [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer — @Sunt-ing → `v0.22.1`
- [#44617](https://github.com/vllm-project/vllm/pull/44617) Fix `LLM.wait_for_completion` output type docstring — @viiccwen → `v0.22.1`
- [#41002](https://github.com/vllm-project/vllm/pull/41002) [ROCm][perf] Use workspace manager for sparse indexer allocations — @tuukkjs → `v0.22.1`
- [#40426](https://github.com/vllm-project/vllm/pull/40426) [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning — @hanlin12-AMD → `v0.22.1`
- [#44605](https://github.com/vllm-project/vllm/pull/44605) [CI/Build] Disable CPU-Compatibility Tests — @bigPYJ1151 → `v0.22.1`
- [#43720](https://github.com/vllm-project/vllm/pull/43720) [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing — @zixi-qi → `v0.22.1`
- [#44571](https://github.com/vllm-project/vllm/pull/44571) [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified — @lucianommartins → `v0.22.1`
- [#44569](https://github.com/vllm-project/vllm/pull/44569) [DSV4] Refactor DeepseekV4Attention — @WoosukKwon → `v0.22.1`
- [#44334](https://github.com/vllm-project/vllm/pull/44334) [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. — @cleonard530 → `v0.22.1`
- [#42139](https://github.com/vllm-project/vllm/pull/42139) [XPU][MoE] support block_fp8_moe on xpu — @zufangzhu → `v0.22.1`
- [#44500](https://github.com/vllm-project/vllm/pull/44500) [Rust Frontend] Skip loading multimodal processor if `--language-model-only` is specified — @BugenZhao → `v0.22.1`
- [#43926](https://github.com/vllm-project/vllm/pull/43926) fix: keep DeepSeek V4 RoPE cache on inv_freq device — @galletas1712 → `v0.22.1`
- [#44539](https://github.com/vllm-project/vllm/pull/44539) [mamba] unify KDA conv states into one cache to match 2-state SSM layout — @ZJY0516 → `v0.22.1`
- [#43707](https://github.com/vllm-project/vllm/pull/43707) [Logs Refactor] Optimize shutdown logs, easier to follow and consistent — @yewentao256 → `v0.22.1`
- [#41980](https://github.com/vllm-project/vllm/pull/41980) use split_group for pytorch process group creation — @tushar00jain → `v0.22.1`
- [#43307](https://github.com/vllm-project/vllm/pull/43307) [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU — @adobrzyn → `v0.22.1`
- [#44380](https://github.com/vllm-project/vllm/pull/44380) [Bugfix] Fix test_cutlass_moe.py — @bnellnm → `v0.22.1`
- [#44471](https://github.com/vllm-project/vllm/pull/44471) [Misc] Add unit tests for pooler head classes — @taneem-ibrahim → `v0.22.1`
- [#34894](https://github.com/vllm-project/vllm/pull/34894) [DOC] Add INT8 W4A8 docs and Arm's supported quantization schemes — @fadara01 → `v0.22.1`
- [#44436](https://github.com/vllm-project/vllm/pull/44436) [ROCm][CI] Add test for Aiter unified attn kernel — @divakar-amd → `v0.22.1`
- [#44057](https://github.com/vllm-project/vllm/pull/44057) [Bugfix] Reject non-positive values for ParallelConfig int knobs — @jwzheng96 → `v0.22.1`
- [#44363](https://github.com/vllm-project/vllm/pull/44363) [Core] Freeze garbage collector in workers after model initialization — @tlrmchlsmth → `v0.22.1`
- [#44509](https://github.com/vllm-project/vllm/pull/44509) [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count — @tc-mb → `v0.22.1`
- [#43519](https://github.com/vllm-project/vllm/pull/43519) Add model support for granite speech plus — @zvik → `v0.22.1`
- [#44340](https://github.com/vllm-project/vllm/pull/44340) [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings — @mgoin → `v0.22.1`
- [#43827](https://github.com/vllm-project/vllm/pull/43827) [DSv4] Adding TRTLLM gen attention kernel — @zyongye → `v0.22.1`
- [#44255](https://github.com/vllm-project/vllm/pull/44255) [ROCm][CI] Specifying time outs for the lm eval models — @AndreasKaratzas → `v0.22.1`
- [#44046](https://github.com/vllm-project/vllm/pull/44046) [ROCm][CI] Stabilize memory-release in the Hybrid model generation tests — @AndreasKaratzas → `v0.22.1`
- [#43625](https://github.com/vllm-project/vllm/pull/43625) [ROCm] Bump fastsafetensors to v0.3.2 from PyPI, remove git source build — @wjabbour → `v0.22.1`
- [#42554](https://github.com/vllm-project/vllm/pull/42554) [PD][Nixl] Mamba prefix caching mode support  — @NickLucche → `v0.22.1`
- [#44476](https://github.com/vllm-project/vllm/pull/44476) [Bugfix][Compile] Guard per_token_group_fp8_quant lookup on non-CUDA platforms — @QiliangCui2023 → `v0.22.1`
- [#44534](https://github.com/vllm-project/vllm/pull/44534) Add GH token to docs build pre run check — @hmellor → `v0.22.1`
- [#42443](https://github.com/vllm-project/vllm/pull/42443) Refactor CT NVFP4 linear to use a single class — @dsikka → `v0.22.1`
- [#44205](https://github.com/vllm-project/vllm/pull/44205) [Bugfix] fix EVS for qwen3-vl — @garrygale → `v0.22.1`
- [#43556](https://github.com/vllm-project/vllm/pull/43556) [Attention] Mamba attention module refactor - LINEAR — @wangxiyuan → `v0.22.1`
- [#42646](https://github.com/vllm-project/vllm/pull/42646) [perf] Add gemma RMS AR fusion — @jiahanc → `v0.22.1`
- [#44493](https://github.com/vllm-project/vllm/pull/44493) [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata — @Kevin-XiongC → `v0.22.1`
- [#43447](https://github.com/vllm-project/vllm/pull/43447) [Prefix Caching] DeepSeekv4 - Support selective prefix-cache retention for sliding-window KV cache — @wzhao18 → `v0.22.1`
- [#44497](https://github.com/vllm-project/vllm/pull/44497) [CI] Reverted gitignore changes — @AndreasKaratzas → `v0.22.1`
- [#44479](https://github.com/vllm-project/vllm/pull/44479) [Frontend] Consolidate online serving utils. — @noooop → `v0.22.1`
- [#42129](https://github.com/vllm-project/vllm/pull/42129) [Inductor] Fast-path Inductor fallback for vllm::*/vllm_aiter::* custom ops — @okorzh-amd → `v0.22.1`
- [#44463](https://github.com/vllm-project/vllm/pull/44463) [CI] Resolve release V2 docker build after ROCm CI wheels change — @AndreasKaratzas → `v0.22.1`
- [#41633](https://github.com/vllm-project/vllm/pull/41633) [EPLB] Nixl communicator optimization. Zero-copy transfers — @ilmarkov → `v0.22.1`
- [#44230](https://github.com/vllm-project/vllm/pull/44230) optimize the compressor 128 split cutedsl kernel  — @Jie-Fang → `v0.22.1`
- [#41471](https://github.com/vllm-project/vllm/pull/41471) [Refactor] Remove dead code in tests and parallel_state — @yewentao256 → `v0.22.1`
- [#41759](https://github.com/vllm-project/vllm/pull/41759) [MM][Perf][CG] Support ViT full CUDA graph for InternVL — @oguzhankir → `v0.22.1`
- [#42865](https://github.com/vllm-project/vllm/pull/42865) [KV Connector] Update lmcache kv_offloading_backend to use LMCacheMPConnector — @maobaolong → `v0.22.1`
- [#44410](https://github.com/vllm-project/vllm/pull/44410) [Bugfix] Fix VLLMNotFoundError when using LoRA adapter name in poolin… — @wanghenshui → `v0.22.1`
- [#43241](https://github.com/vllm-project/vllm/pull/43241) [Model Runner V2][Spec Decode] Add Gemma4 MTP support — @TheEpicDolphin → `v0.22.1`
- [#44289](https://github.com/vllm-project/vllm/pull/44289) [XPU] skip unapplied UT in test_gpu_model_runner.py — @yma11 → `v0.22.1`
- [#43982](https://github.com/vllm-project/vllm/pull/43982) [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load — @Dymasik → `v0.22.1`
- _…and 161 more_

</details>
