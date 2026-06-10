# vLLM weekly digest — 2026-06-10 (W24)

_Window: last 7 days · upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm)_

_LLM digest skipped: RuntimeError: DASHSCOPE_API_KEY not set for bailian backend_

## Releases this window

- [`v0.22.1`](https://github.com/vllm-project/vllm/releases/tag/v0.22.1) — 2026-06-05 10:10 UTC

## PRs merged this window (220)

<details>
<summary>Click to expand the raw list</summary>

<ul>
<li><a href="https://github.com/vllm-project/vllm/pull/45127">#45127</a> [Model] Remove obsolete ERNIE models — by <a href="https://github.com/xianbaoqian">xianbaoqian</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/35669">#35669</a> Feature/offloading manager stats — by <a href="https://github.com/Srinivasoo7">Srinivasoo7</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44683">#44683</a> [Bugfix][Rust Frontend] Fix missing added tokens in hf/fastokens tokenizer — by <a href="https://github.com/Isotr0py">Isotr0py</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/39498">#39498</a> [Bugfix] Add deepseek_v32 to Quark dynamic MXFP4 model type check — by <a href="https://github.com/shantipriya-amd">shantipriya-amd</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44981">#44981</a> [Rust Frontend] [CI] Unify Rust artifact builds with setuptools-rust — by <a href="https://github.com/BugenZhao">BugenZhao</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44744">#44744</a> [Security] Fix remote DoS via invalid recovered token reinjection — by <a href="https://github.com/jperezdealgaba">jperezdealgaba</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45110">#45110</a> [BUGFIX][XPU] fix xpu `flash_attn_varlen_func` interface — by <a href="https://github.com/jikunshang">jikunshang</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45011">#45011</a> [Refactor] Rename rocm_moe.py to rocm_moe_rdna.py — by <a href="https://github.com/JartX">JartX</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44823">#44823</a> [ROCm][CI] Defer AITER sampler import and isolate server test PYTHONPATH — by <a href="https://github.com/AndreasKaratzas">AndreasKaratzas</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45029">#45029</a> Revert &quot;[Bugfix][CI] Gemma3 Transformers multimodal encoder profiling and build prompt-embedding fixtures&quot; — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/39419">#39419</a> [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding — by <a href="https://github.com/EanWang211123">EanWang211123</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44804">#44804</a> [ROCm][gpt-oss] Hybrid CDNA4 swizzle gate for A8W4 MoE — by <a href="https://github.com/xiaohuguo2023">xiaohuguo2023</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42457">#42457</a> [Bench] Add BFCL dataset for vllm bench serve tool-calling workloads — by <a href="https://github.com/laviier">laviier</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44999">#44999</a> Model/colbert autoweightsloader — by <a href="https://github.com/yufufi">yufufi</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45058">#45058</a> Change from owning configs to owning config utils — by <a href="https://github.com/hmellor">hmellor</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44686">#44686</a> Fix Harmony tool descriptions for optional fields — by <a href="https://github.com/shenoyvvarun">shenoyvvarun</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44552">#44552</a> [Rust Frontend] Add seed_oss and step3p5 reasoning parsers — by <a href="https://github.com/yzhan1">yzhan1</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45057">#45057</a> [Bugfix] Handle HWC images in ImageProcessorItems.get_image_size — by <a href="https://github.com/YellowFoxH4XOR">YellowFoxH4XOR</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45081">#45081</a> [Refactor] Remove dead states from chat completion serving — by <a href="https://github.com/sfeng33">sfeng33</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45054">#45054</a> [Bugfix] Fix weight loading issues caused by #41184 — by <a href="https://github.com/bnellnm">bnellnm</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45085">#45085</a> [Bugfix][CI/Build] Fix Rust frontend build after chat conversion refactor — by <a href="https://github.com/mmangkad">mmangkad</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/43756">#43756</a> [Bench] benchmark_serving_multi_turn: make non-standard conversation_id payload opt-in — by <a href="https://github.com/Change72">Change72</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/42175">#42175</a> [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support — by <a href="https://github.com/lucianommartins">lucianommartins</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44596">#44596</a> [Refactor][Mistral] Extract parsing logic into MistralParser — by <a href="https://github.com/sfeng33">sfeng33</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44884">#44884</a> [Rust Frontend] Extract shared options in route helper params — by <a href="https://github.com/BugenZhao">BugenZhao</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44914">#44914</a> [Bug] Fix deepseek v4 OOM issue — by <a href="https://github.com/yewentao256">yewentao256</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44678">#44678</a> [ROCm][CI] fix test_rope_kvcache_fusion.py — by <a href="https://github.com/charlifu">charlifu</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45066">#45066</a> Revert &quot;[Kernel] Speed up silu_and_mul_per_block_quant with warp-shuf… — by <a href="https://github.com/micah-wil">micah-wil</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/44516">#44516</a> feat(multi-turn-bench): add api_key and custom headers for multi turn benchmark — by <a href="https://github.com/jimmy-evo">jimmy-evo</a></li>
<li><a href="https://github.com/vllm-project/vllm/pull/45002">#45002</a> [Bugfix] fix qwen3.5 ep weight loading — by <a href="https://github.com/ZJY0516">ZJY0516</a></li>
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
<li><em>…and 160 more</em></li>
</ul>
</details>
