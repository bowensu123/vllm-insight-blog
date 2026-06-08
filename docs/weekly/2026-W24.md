# vLLM weekly digest — 2026-06-08 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

## TL;DR
This week's [v0.22.1](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) release focuses on deep architectural refactors for Mixture-of-Experts and DeepSeek-V4 speculative decoding, alongside significant expansions in hardware support for AMD Zen CPUs and Intel XPUs. The prefill-decode disaggregation stack continues to mature with the removal of legacy NCCL connectors and new multi-tier KV offloading capabilities. Overall, it's a heavy infrastructure and performance week that lays the groundwork for Model Runner V2 becoming the default for dense models.

## Deep dives

### MoE Execution Refactor: Inverting the Runner and Expert Relationship
In Mixture-of-Experts (MoE) models, the routing mechanism decides which tokens go to which experts, and the execution layer computes the expert outputs. Previously, the `FusedMoE` class owned the `MoERunner`, which tangled execution state with routing logic. This relationship is now inverted in [#41184](https://github.com/vllm-project/vllm/pull/41184): `MoERunner` owns `FusedMoE` (renamed to `RoutedExperts`), and the old `FusedMoE` class is removed. This cleans up the abstraction boundary, moving CUDA graph capture state out of the expert layer and into the runner, which simplifies custom backend integration. Engineers writing custom MoE layers or integrating new MoE backends will need to update their weight loading paths to the new `.experts.routed_experts.<foo>` hierarchy.

### Accelerating DeepSeek-V4 Speculative Decoding via Index Sharing
DeepSeek-V4 uses Sparse Multi-head Latent Attention (MLA), which relies on a sparse indexer to select top-k tokens for attention. In Multi-Token Prediction (MTP) speculative decoding, recomputing these indices for every draft token and every layer is computationally expensive. Update [#44420](https://github.com/vllm-project/vllm/pull/44420) implements "IndexCache", allowing the top-k token selections to be reused across MTP speculative steps and layers by carrying `topk_indices` through the forward path. This avoids redundant sparse attention indexing, significantly reducing the overhead of the draft model phase and increasing overall speculative decoding throughput. Users running DeepSeek-V4 with MTP enabled will see immediate performance gains without needing to adjust their serving flags.

### Online FP8 Per-Token Per-Channel (PTPC) Quantization
FP8 quantization reduces memory bandwidth and increases compute throughput, but static per-tensor quantization can hurt accuracy on outlier activations. Per-Token Per-Channel (PTPC) quantization dynamically scales activations per token and weights per channel to preserve precision. PR [#44132](https://github.com/vllm-project/vllm/pull/44132) adds support for online FP8 PTPC quantization, activatable via the new `--quantization fp8_per_channel` flag. This provides a highly accurate, dynamic FP8 method that doesn't require offline calibration datasets, making it much easier to deploy FP8 models with minimal accuracy degradation. Engineers deploying FP8 models on Hopper or Ada GPUs who want better accuracy than standard per-tensor FP8 should test this new flag.

## Kernels & attention
- DeepSeek-V4 XPU attention decode path adds Triton kernels for FP8 KV cache and sparse MLA decode on Intel XPUs ([#42953](https://github.com/vllm-project/vllm/pull/42953)) — enabling native high-performance inference for DS-V4 on Intel hardware.
- ROCm fused MoE W4A16 HIP kernel enables optimized execution for AMD RDNA3 (gfx1100) GPUs ([#44075](https://github.com/vllm-project/vllm/pull/44075)) — bringing fast 4-bit MoE inference to consumer and workstation AMD cards.
- Qwen3.5 mixed prefill+decode batches are split to route decodes to the recurrent kernel ([#44700](https://github.com/vllm-project/vllm/pull/44700)) — optimizing execution paths and reducing latency for hybrid workloads.
- FlashAttention is synced with upstream to bring in the latest performance and correctness fixes ([#44065](https://github.com/vllm-project/vllm/pull/44065)) — ensuring vLLM stays aligned with the fastest attention implementations.

## Quantization
- Compressed-tensors adds support for WNA8O8Int linears and WNInt embeddings ([#44340](https://github.com/vllm-project/vllm/pull/44340)) — expanding the flexible quantization formats available via the vLLM compressor.
- Asymmetric support for MoE WNA16 Marlin is added to compressed-tensors ([#44025](https://github.com/vllm-project/vllm/pull/44025)) — allowing more diverse weight distributions and better accuracy in 4-bit MoE models.
- XPU block-scaled W8A8 FP8 path is enabled for Intel GPUs ([#39968](https://github.com/vllm-project/vllm/pull/39968)) — bringing dynamic FP8 linear inference and memory savings to Intel hardware.

## Parallelism & scheduling
- Pipeline parallel bubbles are avoided in Model Runner V2 by optimizing the scheduling of micro-batches ([#42187](https://github.com/vllm-project/vllm/pull/42187)) — maximizing GPU utilization during multi-node pipeline execution.
- Multi-tier KV cache offloading now supports an object store as a secondary tier ([#41968](https://github.com/vllm-project/vllm/pull/41968)) — enabling massive host-side or remote KV cache storage for long-context workloads.
- Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector ([#42212](https://github.com/vllm-project/vllm/pull/42212)) — reducing latency during KV cache tiering and offloading operations.
- PP-aware handshake aggregation is added to the KV Connector ([#43720](https://github.com/vllm-project/vllm/pull/43720)) — improving multi-node prefill-decode handshakes when pipeline parallelism is involved.

## Model support
- JetBrains' Mellum v2, an open-weights MoE code-generation model, is now natively supported ([#43992](https://github.com/vllm-project/vllm/pull/43992)) — expanding the roster of specialized coding models available out-of-the-box.
- Gemma4 Unified (encoder-free) architecture is added ([#44429](https://github.com/vllm-project/vllm/pull/44429)) — expanding multimodal and speculative decoding capabilities for Google's latest vision-language models.
- Cohere North-Mini-Code is enabled with native tool-calling and reasoning parser support ([#44707](https://github.com/vllm-project/vllm/pull/44707)) — allowing seamless deployment of Cohere's specialized coding agents.
- Granite Speech Plus model support is added ([#43519](https://github.com/vllm-project/vllm/pull/43519)) — bringing new audio and multimodal capabilities to the engine.

## Hardware
- AMD Zen CPUs now route W8A8 and W4A16 linear inference through zentorch kernels ([#41813](https://github.com/vllm-project/vllm/pull/41813)) — providing transparent, hardware-accelerated quantized inference on AMD processors.
- CPU speculative decoding explicitly warns if `libiomp5` is not preloaded ([#44419](https://github.com/vllm-project/vllm/pull/44419)) — preventing a 2x throughput drop caused by GNU `libgomp` overhead in short parallel regions.
- Intel XPU MoE kernels are unified and expanded to include block FP8 and MXFP8 formats ([#44771](https://github.com/vllm-project/vllm/pull/44771)) — standardizing MoE execution and enabling newer quantization schemes on Intel GPUs.
- XPU topk/topp Triton `BLOCK_SIZE` is capped at 4096 ([#44470](https://github.com/vllm-project/vllm/pull/44470)) — fixing deterministic sampling mask failures and ensuring correct generation on Intel hardware.

## API & serving
- The Rust frontend gains `/pause`, `/resume`, and `/is_paused` endpoints ([#44499](https://github.com/vllm-project/vllm/pull/44499)) — enabling better lifecycle management and pausing for RL and admin workflows.
- Benchmarking tools auto-detect and correct client/server tokenizer mismatches by probing the `/tokenize` endpoint ([#44708](https://github.com/vllm-project/vllm/pull/44708)) — preventing silent input token inflation during performance testing.
- Tool calling in the Responses API now correctly honors `tool_choice="none"` during streaming ([#42752](https://github.com/vllm-project/vllm/pull/42752)) — fixing a bug where tools were still invoked despite explicit user instructions.
- The Rust frontend adds dynamic LoRA endpoints ([#43778](https://github.com/vllm-project/vllm/pull/43778)) — allowing hot-swapping and management of adapters via the API without restarting the server.

## Watch list
- The `kv_both` role in `NixlConnector` has entered a deprecation cycle ([#43874](https://github.com/vllm-project/vllm/pull/43874)) — users relying on this for bidirectional KV transfer should migrate to explicit prefill/decode roles.
- Model Runner V2 is now enabled by default for Llama and Mistral dense models ([#43458](https://github.com/vllm-project/vllm/pull/43458)) — signaling the imminent retirement of the V1 runner for these core architectures.
- The legacy `P2pNcclConnector` has been fully removed ([#44854](https://github.com/vllm-project/vllm/pull/44854)) — users implementing custom prefill-decode disaggregation must now use the generalized `ConnectorAPI`.

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (239)

<details>
<summary>Click to expand the raw list</summary>

<ul>
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
<li><a href="https://github.com/vllm-project/vllm/pull/44771">#44771</a> [XPU][Minor] format moe kernel name and add in kernel list — by <a href="https://github.com/yma11">yma11</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44484">#44484</a> [MM][CG] Simplify ViT CUDA graph interfaces — by <a href="https://github.com/shen-shanshan">shen-shanshan</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42953">#42953</a> feat: add DeepSeek-V4 XPU attention decode path — by <a href="https://github.com/majian4work">majian4work</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/39562">#39562</a> [Bugfix]: Fix assertion in MambaManager.allocate_slots() — by <a href="https://github.com/Holworth">Holworth</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44805">#44805</a> Added extra_repr() to pooler classes to improve debuggability — by <a href="https://github.com/taneem-ibrahim">taneem-ibrahim</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44215">#44215</a> [Bugfix] Fix FunASR-Nano crash during initialization — by <a href="https://github.com/SunskyXH">SunskyXH</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42736">#42736</a> [Kernel][Test] Make kernel tests for mamba dual-HW (CUDA + XPU) — by <a href="https://github.com/adobrzyn">adobrzyn</a></li>
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
<li><em>…and 179 more</em></li>
</ul>
</details>
