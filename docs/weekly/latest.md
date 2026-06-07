# vLLM weekly digest — 2026-06-07 (W23)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week’s [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) patch release focuses on targeted performance optimizations for emerging architectures like DeepSeek-V4 and Qwen3.5, alongside critical bug fixes for multi-node Ray serving and model loading regressions. Hardware support expands with zentorch acceleration for AMD Zen CPUs and new Intel XPU memory management features. Disaggregated serving also sees major under-the-hood improvements with zero-copy KV transfers and pipeline-parallel handshakes.

## Deep dives

### DeepSeek-V4 MTP Index Sharing
DeepSeek models use Sparse Multi-head Latent Attention (MLA), which relies on a sparse indexer to select top-k tokens and reduce compute. When using Multi-Token Prediction (MTP) for speculative decoding, running this indexer for every speculative step becomes a significant bottleneck. PR [#44420](https://github.com/vllm-project/vllm/pull/44420) implements an "IndexCache" mechanism that reuses the top-k token selections across MTP layers and steps when `skip_topk` is enabled. This eliminates redundant sparse indexing compute during the verification phase, substantially accelerating MTP for DeepSeek models. Users running DeepSeek-V3 or V4 with MTP speculative decoding will see immediate throughput improvements without needing to change their configuration.

### Qwen3.5 GDN Attention Batch Splitting
Qwen3.5 utilizes Gated Delta Net (GDN) attention, a linear recurrent mechanism that vLLM previously processed using a chunked kernel designed for long prefills. In mixed prefill-decode batches, this forced small decode chunks to be padded to a fixed size (e.g., 64 tokens), wasting significant compute. PR [#44700](https://github.com/vllm-project/vllm/pull/44700) splits these mixed batches, routing decode tokens to a specialized recurrent kernel while prefills continue using the chunked kernel. This prevents massive compute waste on decode steps, dropping decode latency significantly and improving overall throughput. Anyone serving Qwen3.5 or other GDN-based models on the V1 engine will benefit from this automatic optimization.

### Nixl KV-Connector Zero-Copy Transfers
In disaggregated prefill-decode serving, KV caches must be transferred from prefill nodes to decode nodes, which traditionally incurs memory copy overhead. PR [#41633](https://github.com/vllm-project/vllm/pull/41633) optimizes the Nixl communicator by introducing zero-copy transfers for KV cache migration. By avoiding intermediate buffer copies, this change reduces CPU/GPU overhead and memory bandwidth consumption during state transfer. This directly lowers time-to-first-token (TTFT) in disaggregated architectures. Teams deploying disaggregated setups using the Nixl KV connector should upgrade to realize these latency and bandwidth gains.

## Kernels & attention
- Fused `concat_mla_q` in ROCm sparse-MLA `forward_mqa` ([#42838](https://github.com/vllm-project/vllm/pull/42838)) — replaces `torch.cat` to eliminate 61 tensor allocations per decode step on MI355X.
- Native HIP W4A16 MoE kernel for AMD RDNA3 ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — replaces the Triton path with `v_dot2` primitives for faster expert routing and GEMM.
- Fix for mHC fused-RMSNorm big-fuse miscompile ([#44692](https://github.com/vllm-project/vllm/pull/44692)) — resolves silently wrong `layer_input` (NaNs) in DeepSeek-V4 when hidden size != 4096.

## Quantization
- Support for compressed-tensors WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expands mixed-precision integer quantization schemes for diverse model weights.
- ModelOpt MXFP8 non-gated MoE support ([#42958](https://github.com/vllm-project/vllm/pull/42958)) — enables NVIDIA's ModelOpt MXFP8 quantization for non-gated Mixture-of-Experts layers.
- Block-scaled W8A8 FP8 path for Intel XPU ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — brings block-scaled FP8 linear inference to Intel XPU platforms.
- Asymmetric MoE WNA16 marlin support ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — adds asymmetric quantization handling to the Marlin backend for compressed-tensors.

## Parallelism & scheduling
- PP-aware handshake aggregation for NixlConnector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — improves pipeline parallelism coordination during KV cache handshakes in disaggregated serving.
- Multi-node Ray data-parallel hang fix ([#43864](https://github.com/vllm-project/vllm/pull/43864)) — resolves deterministic hangs by excluding the Ray DP backend from deferred port allocation.
- Objectstore as a secondary tier for KV cache offloading ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — allows offloading evicted KV blocks to an object store, expanding memory capacity for long contexts.
- Pipeline parallel bubble reduction in ModelRunnerV2 ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — optimizes scheduling to reduce idle time in pipeline parallel configurations.

## Model support
- JetBrains' Mellum v2 ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — adds support for the open-weights MoE code-generation model.
- Cohere Mini Code model ([#44707](https://github.com/vllm-project/vllm/pull/44707)) — enables the North-Mini-Code model with native tool-calling and reasoning parsers.
- Gemma4 Unified (encoder-free) support ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — adds native support for Google's encoder-free multimodal architecture.
- HyperCLOVAX loading fix ([#43860](https://github.com/vllm-project/vllm/pull/43860)) — registers the model type natively to bypass removed remote code in newer `transformers` versions.

## Hardware
- zentorch kernels for AMD Zen CPUs ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — routes W8A8 and W4A16 linear inference through zentorch for accelerated CPU inference with transparent fallback.
- ROCm `ApplyRotaryEmb` fallback for large sequences ([#43684](https://github.com/vllm-project/vllm/pull/43684)) — prevents HIP grid dimension overflow crashes when sequence lengths exceed 524k tokens.
- Transparent sleep mode for Intel XPU ([#37149](https://github.com/vllm-project/vllm/pull/37149)) — introduces memory allocator sleep/wake interfaces to match CUDA parity.
- SHM communicator support for PowerPC ([#43754](https://github.com/vllm-project/vllm/pull/43754)) — enables shared memory communication for PowerPC platforms.

## API & serving
- Phi-4 mini JSON tool parser in Rust Frontend ([#44213](https://github.com/vllm-project/vllm/pull/44213)) — adds robust tool-call parsing for Phi-4 mini's `functools` output format.
- Responses API developer-role folding ([#43590](https://github.com/vllm-project/vllm/pull/43590)) — automatically folds developer-role inputs into system instructions for OpenAI compatibility.
- Multi-audio OpenAI server path stabilization ([#44051](https://github.com/vllm-project/vllm/pull/44051)) — ensures the server correctly returns 400 for over-limit audio inputs without crashing subsequent requests.
- Anthropic system role messages inside array ([#44283](https://github.com/vllm-project/vllm/pull/44283)) — improves compatibility with Anthropic's API formatting for system prompts.

## Watch list
- KV-Cache Layout Refactor ([#44454](https://github.com/vllm-project/vllm/pull/44454)) — the first in a series of PRs standardizing KV cache layouts across attention backends; watch for potential memory layout changes in custom integrations.
- NixlConnector `kv_both` deprecation ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — initiates the deprecation cycle for the `kv_both` role; users should migrate to explicit prefill/decode roles.
- Model Runner V2 expansion ([#43458](https://github.com/vllm-project/vllm/pull/43458)) — MRV2 is now enabled for Llama and Mistral dense models, signaling the impending default switch for the V1 engine architecture.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (231)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/44454">#44454</a> [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config construction — by <a href="https://github.com/LucasWilkinson">LucasWilkinson</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44674">#44674</a> [ROCm][Kernel] Enable permute_cols for ROCm — by <a href="https://github.com/charlifu">charlifu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43087">#43087</a> Modify torch dependency in xpu.txt — by <a href="https://github.com/BramVanroy">BramVanroy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42599">#42599</a> [Dependency] Remove stale cuDNN frontend upper bound — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44051">#44051</a> [CI] Stabilize the multi-audio OpenAI server path — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44378">#44378</a> [Doc] Fix multimodal torch.compile troubleshooting to not use removed VLLM_TORCH_COMPILE_LEVEL — by <a href="https://github.com/DaoyuanLi2816">DaoyuanLi2816</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44103">#44103</a> [Bugfix][Mooncake] Fix per-group block_size/block_hash and group_idx in MooncakeStoreConnector KV events — by <a href="https://github.com/ivanium">ivanium</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44417">#44417</a> [videoloader] implement glm46v video loader — by <a href="https://github.com/JaredforReal">JaredforReal</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44707">#44707</a> [Cohere] Enable Cohere Mini Code model and update Command A-plus test registry — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44420">#44420</a> [feature] add index share feature for DSA MTP — by <a href="https://github.com/JaredforReal">JaredforReal</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44041">#44041</a> [Bugfix] Fix benchmark_moe.py after inplace mechanism removal — by <a href="https://github.com/qyYue1389">qyYue1389</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44540">#44540</a> [XPU] add xpu branch in compressed_tensors_moe_w4a4_mxfp4 — by <a href="https://github.com/zufangzhu">zufangzhu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/37149">#37149</a> [XPU][Feature] transparent sleep mode support for XPU platform — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/36423">#36423</a> [XPU] Support  cpu kv offloading and tiering offloading on XPU platform — by <a href="https://github.com/chaojun-zhang">chaojun-zhang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44699">#44699</a> [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 — by <a href="https://github.com/WoosukKwon">WoosukKwon</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42838">#42838</a> [ROCm][MLA] Replace torch.cat in sparse-MLA forward_mqa with fused concat_mla_q — by <a href="https://github.com/maeehart">maeehart</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44560">#44560</a> [BugFix] Resolve multiple async kv load deadlock — by <a href="https://github.com/njhill">njhill</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44075">#44075</a> [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) — by <a href="https://github.com/JartX">JartX</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44700">#44700</a> [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44694">#44694</a> [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43684">#43684</a> [Bugfix][ROCm] `ApplyRotaryEmb`: fall back to native when flash_attn rotary grid would exceed the HIP per-dim limit — by <a href="https://github.com/amd-fuweiy">amd-fuweiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44559">#44559</a> [Bugfix][Voxtral] Add fetch_audio to MistralCommonFeatureExtractor (transformers&gt;=5.10 compat) — by <a href="https://github.com/Yadan-Wei">Yadan-Wei</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44613">#44613</a> [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig — by <a href="https://github.com/aoshen02">aoshen02</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44593">#44593</a> [Misc] Replaced asserts with proper exceptions to improve UX for pooling — by <a href="https://github.com/taneem-ibrahim">taneem-ibrahim</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44692">#44692</a> [Bugfix][Kernel] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096 — by <a href="https://github.com/zyongye">zyongye</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44213">#44213</a> [Rust Frontend] Add Phi-4 mini JSON tool parser — by <a href="https://github.com/devin-lai">devin-lai</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44574">#44574</a> Preserve layout-changing clones — by <a href="https://github.com/mikekg">mikekg</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44130">#44130</a> [Bugfix] Fix `sequence_parallel_chunk_impl` custom op aliasing its input — by <a href="https://github.com/vadiklyutiy">vadiklyutiy</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44021">#44021</a> [Cohere] fix RoutingMethodType — by <a href="https://github.com/Terrencezzj">Terrencezzj</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44435">#44435</a> [Doc] Add Llama-3.2-3B-Instruct to batch-invariance tested models — by <a href="https://github.com/DaoyuanLi2816">DaoyuanLi2816</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42832">#42832</a> [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path — by <a href="https://github.com/akii96">akii96</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44669">#44669</a> [Core][Engine] allow DP ray placement groups to be set on specific nodes — by <a href="https://github.com/walterbm">walterbm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44666">#44666</a> Male Mergify comment less spammy — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44330">#44330</a> [Bugfix] GPT-OSS instruction rendering — by <a href="https://github.com/yzong-rh">yzong-rh</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44621">#44621</a> Upgrade tpu-inference to v0.21.0 — by <a href="https://github.com/CienetStingLin">CienetStingLin</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/38804">#38804</a> Fix sarvam forward compatibility with transformers v5 — by <a href="https://github.com/Vikrantpalle">Vikrantpalle</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44648">#44648</a> [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm — by <a href="https://github.com/tjtanaa">tjtanaa</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41968">#41968</a> Add objectstore as a secondary tier to multi-tier kv cache offloading — by <a href="https://github.com/effi-ofer">effi-ofer</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44609">#44609</a> Support MiniCPMV batched preprocessing — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44647">#44647</a> [CI] Bump mypy version `1.19.1` -&gt; `1.20.2` — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44635">#44635</a> Speed up docs build — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44649">#44649</a> [CI] Bump mistral-common — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44588">#44588</a> [Reasoning][Structured Outputs] Add Command A plus tags for structural tags — by <a href="https://github.com/rishitdholakia13">rishitdholakia13</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44561">#44561</a> [DSV4] Move more ops out of eager breakpoint — by <a href="https://github.com/WoosukKwon">WoosukKwon</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44615">#44615</a> [Bugfix] Fix gemma4 crash on CPU: guard mem_get_info call — by <a href="https://github.com/adhithyamulticoreware">adhithyamulticoreware</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43167">#43167</a> Remove KV cache scale boilerplate from model weight loading methods — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43150">#43150</a> [BUG] Fix FP64 Gumbel precision coverage — by <a href="https://github.com/tianyu-z">tianyu-z</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44591">#44591</a> [Rust Frontend] Batch auto-abort requests by engine — by <a href="https://github.com/HueCodes">HueCodes</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44066">#44066</a> docs: fix tokenizer optimization typo — by <a href="https://github.com/chunyang-wen">chunyang-wen</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43874">#43874</a> [NixlConnector] Initiate deprecation cycle for `kv_both` role  — by <a href="https://github.com/NickLucche">NickLucche</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44391">#44391</a> [Rust Frontend] Support include_reasoning=false — by <a href="https://github.com/ricky-chaoju">ricky-chaoju</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44622">#44622</a> [Bugfix] Update mistral tokenizer test for continue_final_message fix — by <a href="https://github.com/XuZhou26">XuZhou26</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44603">#44603</a> fix: pad dummy run query_start_loc — by <a href="https://github.com/UranusSeven">UranusSeven</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44618">#44618</a> [Bugfix] Fix test_invocations flaky failure with newer openai SDK — by <a href="https://github.com/XuZhou26">XuZhou26</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44620">#44620</a> [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer — by <a href="https://github.com/Sunt-ing">Sunt-ing</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44617">#44617</a> Fix `LLM.wait_for_completion` output type docstring — by <a href="https://github.com/viiccwen">viiccwen</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/41002">#41002</a> [ROCm][perf] Use workspace manager for sparse indexer allocations — by <a href="https://github.com/tuukkjs">tuukkjs</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/40426">#40426</a> [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning — by <a href="https://github.com/hanlin12-AMD">hanlin12-AMD</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44605">#44605</a> [CI/Build] Disable CPU-Compatibility Tests — by <a href="https://github.com/bigPYJ1151">bigPYJ1151</a> → <code>v0.22.1</code></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43720">#43720</a> [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing — by <a href="https://github.com/zixi-qi">zixi-qi</a> → <code>v0.22.1</code></li>
<li><em>…and 171 more</em></li>
</ul>
</details>
