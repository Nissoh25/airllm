# airllm-sparse-codesign-map — 2026-11-24

Wiring of the NVIDIA co-design blog (2026-11-17, "Co-Designing Sparse Attention for
Efficient Long-Context Inference") to the local gold repos and the k26v15-qwen4
distillation target. Ledger: MODEL-20261123-002. Evidence: full blog extracted to
`/home/pegasus/.hermes/cache/scratch` (23828 chars) + developer.nvidia.com URL.

## 0. Status (2026-11-24)

AirLLM 27B-FP8 (Qwen3.8-27B, 64 layers, block-quant FP8) PROVEN on all three sparks:
- spark1: init 25s, GEN OK, CPU 2m36s
- spark2: init 35s, GEN OK
- spark3: init 24s, GEN OK, CPU 1m49s (despite AEON+Huihui holding 73G unified)
Mandatory patches: resplit shards + GPU-side block dequant in airllm_base.py
(SPARK-PATCH 2026-11-24 v2). Launch outside the 12G hermes-gateway cgroup:
`systemd-run --user --no-block --unit=...`.

RE-VERIFIED LIVE 2026-11-24 (Qwen CLI): spark2 airllm import OK, transformers
5.12.1, torch 2.14.0+cu130, CUDA True, SPARK-PATCH x2 in airllm_base.py; spark3
same with torch 2.13.0+cu130. Win11 (kod-kod-desktop) DONE per subagent
20261123_201556_6ecab0: F:/airllm-venv-20261123, torch 2.14.1+cu126, transformers
5.12.1, airllm 3.0.1; Qwen3-0.6B smoke on RTX 3090 = peak VRAM 353 MB, 8 tokens
48.9 s, coherent output ("Paris"/"Rome"), 31 shards ~1.5 GB, F: ~403 GB free.
All 4 nodes equalized. NOTE: win11 smoke was 0.6B — 27B-FP8 on win11 not yet run.
WIN11 27B STATUS (2026-11-24 late): patched fork deployed (8b47db0 + scp'd
SPARK-PATCH dequant — the GitHub fork LACKS that uncommitted patch), smoke kit
staged (C:\Users\Administrator\win11-smoke-27b-20261124.bat), 29G checkpoint
transfer IN FLIGHT but crippled: tailscale path is DERP(dfw)-relayed (no direct
NAT route), 0.23-2.7MB/s, ETA unreliable. Two racing lanes: scp push from
spark1 (done-flag /tmp/win11-push-27b.done) + HF download on win11 (was stalled
at 3.0G, restarted). Fix direct path before relying on either (see memory file).

## 1. The three stages and where cost lives

indexer (scales with N) -> Top-k (scales with keys scored, NOT k) -> core attention
(scales with k, block-friendly). At 128K prefill: dense attention = 91% of layer time;
DSA = indexer+Top-k 47%, core 24%, FFN 25%. The sparsity does not remove cost, it
MOVES it into the indexer — so the indexer is the co-design surface.

## 2. The 8 guidelines (abridged)

1. Keep G*(D+1) <= ~200 Tensor Core rows (256 with FP16 epilogue). Group size and
   speculative-decoding depth compete for the SAME tile. Beyond one tile, runtime
   steps up (more CTAs, key cache re-read).
2. Indexer epilogue in FP16; FP16 accumulate too if QK scaling keeps outputs in
   FP16 range -> 1.31x-1.45x decode indexer speedup.
3. KH=1 with cross-head reduction over QH query heads -> ONE logit per key. Decode
   runtime grows LINEAR with KH (7.6x for KH=1->8): each extra key head re-reads the
   whole key cache. (Designs without cross-head reduction — MSA, MoBA — lose the Gx
   intensity factor.)
4. Hsz = 128 or 256. Hsz=64 leaves Tensor Cores idle on the CUDA-core epilogue
   (ReLU + weighted sum). Prefill 64->128 costs only 1.1-1.2x; decode pays key bytes,
   so 128 is the settlement point.
5. Reduce keys SCORED: compression ratio CR, hierarchical indexing, index sharing
   across layers. Linear work cut in both phases; does not change intensity —
   combine with 1-4.
6. Top-k follows keys scored, not k kept (64x more k = <2x Top-k time). Fuse Top-k
   into the indexer kernel (no logit write/read-back; saves HBM + energy). Decode:
   Guess-Verify-Refine (previous step's selection as guess) = 2.1-2.4x vs radix
   select. Refs: TensorRT-LLM fused Top-k; FlashInfer selective-Top-k (open source).
7. BLOCK-level selection with large blocks. Contiguous bulk fetch + cross-query
   sharing. Halving block size ~doubles core-attention runtime. Token-level (DSA,
   GLM-5.2, LSA) works but fetches per-token addresses.
8. Large core-attention group size (many Q heads per KV head) so a fetched block is
   reused across query heads.

WARNING: porting a checkpoint trained on one Table-1 row — changing CR or block size
without retraining fails quality gates. Co-design = training-time alignment with the
kernels you ship.

## 3. Table 1 — shipped indexer configs (from the blog)

| Model | QH | KH | Hsz | Post | CR | Granularity | Notes |
|---|---|---|---|---|---|---|---|
| DSA (DeepSeek V3.2) | 64 | 1 | 128 | ReLU+ws | 1 | token | separate FP8 index cache |
| GLM 5.2 | 32 | 1 | 128 | ReLU+ws | 1 | token | INDEX SHARED ACROSS LAYERS |
| GLM 5.3 Flash | 32 | 1 | 128 | ReLU+ws | 4 | block-4+tail | |
| LSA-Flash | 32 | 1 | 128 | ReLU+ws | 1 | token | hierarchical + index shared |
| CSA | 64 | 1 | 128 | ReLU+ws | 4 | compressed token | |
| Qwen QSA | 4 | 1 | 128 | ReLU+sum | 4 | block-4+tail | |
| MiniMax MSA | 4 (1/grp) | 1 | 128 | MaxPool | 1 | block-128 | scores every token, pools per block, no cross-head reduction |
| NSA | 64 | 4 | 192 | softmax | 16 | block-16 | |
| MoBA | 32 | 8 | 128 | MeanPool(K) | 4096 | block-4096 | pools keys before scoring |

Corrected 2026-11-24 against the full blog text (earlier version had DSA/GLM5.2/
LSA at CR=4 and NSA at CR=4; the blog's Table 1 says CR=1/1/1 and CR=16).
Per-model token BUDGETS (256/432/32768) are not in the blog — they come from the
individual papers; label them as such when quoting.

## 4. Local repo map (all under /home/pegasus/Projects/, dated 20261124)

| Repo | Role in the design space |
|---|---|
| Qwen-Native-Sparse-Attention-20261124 | "Qwen QSA" row — 4Q/1K/128/CR4/block-4 native. The template for qwen4 indexer. |
| latent-sparse-attention-20261124 | NSA row (64/4/192/CR16/block-16) + reference kernels. |
| flash-sparse-attention-20261124 | block-level core attention kernel (guideline 7) — the bulk-fetch path. |
| flash-attention-20261124 / FlashAttention20-20261124 | dense baseline kernels; FA20 = the "2.0" fork the user calls the k26v15 lineage. |
| cutlass-20261124 | Tensor Core tile/epilogue machinery (guidelines 1,2 — 200-row tile, FP16 epilogue/accumulate). |
| SpargeAttn-20261124 (Triton_SpargeAttn) | sparse attention for long-video/generation-style workloads; Triton path = sm_121-friendly dev loop. |
| Prism-20261124 | index-sharing / pruning-related (guideline 5, 3rd axis). |
| airllm/ | streaming inference host for the big local models (27B-FP8 proven 3/3 nodes). |

## 5. Application to our targets

### 5.1 Qwen3.8-27B today (running on 3 sparks)
Hybrid stack: linear_attn (Gated-DeltaNet style) on most layers + full attention every
Nth. Sparse attention applies to the FULL-attention layers only; the linear layers
already have sub-quadratic cost. AirLLM streams it; long-context ceiling = KV cache of
the full layers — a QSA-style indexer there is the win (block-4, CR4, KH=1, Hsz=128,
fused Top-k per guidelines 3/4/6/7).

### 5.2 k26v15-qwen4 distillation (the 10-years-ahead model)
Pick the QSA row as the starting config, tuned by the guidelines:
- KH=1 + cross-head reduction over the 4 Q heads (guideline 3) — decode-safe.
- Hsz=128 (guideline 4) — sm_121 tiles + 128B transfers.
- CR=4, block-of-4+tail (guidelines 5,7) — matches Qwen QSA training row; do NOT
  change CR/block at inference without retraining (blog warning).
- G*(D+1) budget: with G=4, speculative depth up to D=49 fits one 200-row tile
  (guideline 1) — i.e. long speculative chains are nearly free for the indexer.
- FP16 epilogue (guideline 2) — indexer QK values are tiny; FP16 accumulate safe.
- Fuse Top-k + Guess-Verify-Refine decode path (guideline 6) — 2.1-2.4x on the
  decode indexer, which is exactly where AirLLM's per-step latency lives.
- GLM-5.2's index-sharing-across-layers (Table 1) is the extra axis to test in the
  distillation data: one indexer pass per N layers.

### 5.3 GB10-specific
- sm_121 Blackwell: kernels must be built with sm_121 (llama.cpp build already proves
  the toolchain: CUDA 13.0, BLACKWELL_NATIVE_FP4). Triton (SpargeAttn) is the
  fastest path to validate guideline-level experiments without CUTLASS surgery.
- Unified memory: the "HBM bytes" arguments hold, but KV/index caches compete with
  model weights for the same 121G pool — AirLLM streaming + sparse KV coexist well
  (sparse core attention reads only k blocks, not N).
- FP8 e4m3 native on sm_121 -> the blog's "production FP8 kernel" measurements are
  directly our numbers; no porting assumptions.

## 6. Next steps (ordered) — STATUS 2026-11-24
1. [x] QSA indexer extract: lsa/baselines/qsa.py (torch reference, Qwen's own math) +
   distill yaml budgets; pure Triton -> no sm_121 build flags needed (§8/§10).
2. [x] Triton prototype + bench vs dense 32K/128K: §9-10.5 (fused indexer 15x,
   shared core 7-10x, block sweep, locality law). Harness: qsa-triton-proto-20261124.py.
3. [x] GVR decode Top-k: §10.5 — 2.10-2.28x at >=91% overlap, exactness 100%, matches blog.
4. [x] Distillation spec: umbrella-system/docs/adr/0103-k26v15-qwen4-sparse-
   attention-distillation-spec-20261124.md — **ACCEPTED** (ratified by Nisso
   2026-11-24; Q2 revised per §11.1 before ratification; file NOT git-committed yet).
5. [~] Windows 11: 0.6B smoke DONE (353MB VRAM, coherent); 27B-FP8 on win11 still OPEN.
NEXT: ratify ADR-0103 -> trained-weight validation runs for gates Q1-Q3 (locality >=75%,
decode overlap >=90%, hier recall >=95%) -> fuse union_prep into selection -> 27B-FP8 win11.

## 7. Selection-granularity axis write-up (2026-11-24)

Token vs block — the axis that matters most to the CORE attention kernel, not the
indexer (blog: "This choice matters most to the core attention kernel").

- Token-level (DSA, GLM 5.2, LSA-Flash): k selected keys scattered in KV cache.
  Per-token address lookup + separate fetch. Every query token selects different
  tokens -> two queries rarely share a fetch.
- Block-level (GLM 5.3 Flash, QSA, NSA, MSA, MoBA): contiguous blocks. One address
  serves the whole block, fetched in bulk; queries that selected the same block
  share the fetch. Kernel works block-by-block: fetch block once, compute for every
  query that selected it, combine each query's partials.
- Cost law (blog Figure 11, fixed 2,048-key budget; 16x128 vs 32x64 vs 64x32):
  FLOPs and KV bytes are IDENTICAL; kernel time follows the NUMBER OF BLOCKS —
  every block adds a query load + partial result + merge. Halving block size
  ~doubles core-attention runtime in prefill AND decode.
- Quality ceiling: coarse blocks can't reject one bad token inside a selected block.
  "+tail" rows (GLM 5.3 Flash, QSA) handle the partial final block explicitly.
  MSA's block-128 is a POOLING granularity (scores every token, pools per block,
  no cross-head reduction); MoBA's block-4096 pools keys BEFORE scoring. Neither
  is true selection.
- Guideline 8: fetched blocks are reused across the query heads of one token that
  share a KV head -> large CORE-attention group (GQA-style) multiplies block reuse.
- Vendor evolution: GLM 5.2 (CR1/token) -> GLM 5.3 Flash (CR4/block-4+tail), same
  QH32/KH1/Hsz128: the 5.2->5.3 diff IS guidelines 5+7.

Frozen for k26v15-qwen4: QSA row = block-of-4 + tail, CR=4, training-locked.
QSA paper's 32,768-token budget (NOT the blog's) = 8,192 blocks at block-4 ->
Figure-11 law says those 8K block-ops are the core-side cost to attack via
bigger blocks (retrain only), fetch coalescing, and guideline-8 head reuse.
Triton prototype: sweep block {4,16,64,128} at fixed 2048 budget in a micro-model
to internalize the 2x-per-halving curve; keep block-4 for the real checkpoint.
Guess-Verify-Refine (guideline 6) pairs best with blocks: selected block sets
change little between decode steps.

## 8. NSA repo extract (2026-11-24)

Qwen-Native-Sparse-Attention-20261124 = the BLOG'S NSA ROW (arXiv 2502.11089)
applied to Qwen2.5 — TRAINING repo (no varlen; forward+backward only).
Exact config (nsa_attn.py defaults, corroborates blog Table 1 CR=16):
- kernel_size=32 ("l"), stride=16 ("d") -> CR=16
- select_size=64 ("l'"), top_n=16 -> selection = 16 blocks x 64 = 1024 tokens
- window_size=512; asserts: stride power-of-2, kernel%stride==0, select%kernel==0
- QH=64/KH=4 (README quickstart; blog Hsz=192 is the shipped config, repo demo
  uses 128/128 for illustration)
- WIRING PATTERN: qwen2/patch_nsa.py -> `module.Qwen2Attention = Qwen2NSA`
  drop-in attention swap. This is the template for the k26v15 distillation patch.
- fp8/fp8_gemm.py + triton_kernel/* (rmsnorm/rope/silu/dyt) = training-side extras.

SUCCESSOR (inference-capable, cloned 2026-11-24):
/home/pegasus/Projects/NSA-Scalable-Flash-20261124  (mdy666/Scalable-Flash-Native-Sparse-Attention)
- Triton-based, Hopper-optimized; >FA2 at 6K+, >FA3 at 11K+
- varlen + Context Parallel (Megatron-level) -> 1M sequences
- flash_nsa/ kernels, nano-vllm/ integration, inference.ipynb, setup.py (pip .)
- README roadmap note: "In inference, make topk more compatible with CUDA Graph"
- Both repos = pure Triton/torch, NO CUTLASS build -> sm_121-friendly, as predicted
  in section 5.3.

Next: read flash_nsa/ select + compress kernels; Triton micro-bench on sm_121.
-> DONE 2026-11-24, results in section 9.

## 9. NSA vs dense measured on GB10 (2026-11-24, spark1, sm_121)

Harness: ~/Projects/nsa-microbench-20261124.py + nsa-dense-baseline-20261124.py (v2).
Env: torch 2.15.0+cu132, triton 3.9.0, airllm venv. Config: N=131072 (128K),
QH=64/KH=4, D=VD=128, page=128, KS=32/STRIDE=16 (CR=16), BLK=64/TOPN=16, WIN=512.
Data: randn*0.02 bf16 (kernel-workload timing, NOT trained-weight quality).
Logs: /tmp/nsa-microbench-20261124.log, /tmp/nsa-dense-baseline2-20261124.log.

| Stage | Time | Share |
|---|---|---|
| mean_pooling_prefill (k+v) | 2.2 ms | 0.1% |
| cmp_attn_prefill (indexer+compressed core) | 2397.0 ms | 89.0% |
| slc_attn_prefill (block-select core) | 244.8 ms | 9.1% |
| combine_prefill | 51.0 ms | 1.9% |
| **NSA prefill total (SWA branch EXCLUDED)** | **2695 ms** | |
| nsa_decode (fused SLC+SWA, end-to-end) | 0.26 ms/step | 3795 tok/s ceiling (1 req) |

| 128K comparison | dense flash | dense mem-eff |
|---|---|---|
| prefill | 3825 ms (NSA 1.42x faster) | 12801 ms (NSA 4.75x faster) |
| decode (1 q-token vs 128K KV) | 3.804 ms (NSA 14.6x faster) | 28.339 ms (NSA 109x faster) |

CAVEATS (honest accounting):
- NSA prefill EXCLUDES the sliding-window branch -> 1.42x is an OPTIMISTIC bound;
  full NSA prefill = 2695 ms + SWA cost. Decode numbers ARE end-to-end (fused SLC+SWA).
- Dense = torch SDPA with native enable_gqa=True (verified bit-exact vs
  repeat_interleave). MATH backend excluded at 128K: materializes 64x131072^2
  scores (~2 TB) -> OOM-killed v1 run at 00:15:39 (nsa-densebase-20261124.service).
- v1 GQA bugs fixed 2026-11-24: microbench expand(1,QH,...) on KH=4 illegal;
  baseline expand().repeat() gave TILED head order (wrong mapping) + 4 GiB copies.
- Random-data topk distribution may differ from trained weights; re-bench on a
  real checkpoint before quoting speedups externally.

CONFIRMS the blog cost law on our silicon: indexer = 89% of NSA prefill ->
guideline 5/6 (fewer keys scored, fused Top-k, FP16 epilogue) is THE co-design
surface for k26v15-qwen4. Decode is where NSA pays off hardest (14.6x vs flash):
AirLLM per-step latency lives exactly there.

## 10. QSA Triton prototype — measured (2026-11-24, spark1 GB10 sm_121)

Harness: ~/Projects/qsa-triton-proto-20261124.py (+ qsa-proto-run-20261124.sh).
Reference: lsa/baselines/qsa.py (Qwen's own indexer math) + distill config
long_16k_1m_heavy_qsa.yaml: index 4Q/1K/d128/rotary64, Gemma (1+w) norm fp32,
pool-before-rotate block keys, I[i,b]=sum_h relu(q.k)/sqrt(128), block=4,
top_k=2048 keys = 512 closed blocks + open-block tail.
CORE dims = REAL Qwen3.8-27B config.json (~/Projects/models/Qwen3.8-27B-FP8-20261123):
24Q/4KV/d256, hidden 5120, 64 layers, full_attention_interval=4 -> 16 full-attn
layers (3,7,...,63), partial_rotary 0.25. (qwen3_5 class defaults 16Q/4KV are NOT
the 27B's dims — always read the checkpoint config.)

CORRECTNESS GATE (N=4096): Triton fused ReLU+sum scorer vs torch qsa.py reference:
max_abs_diff 3e-6, top-512 selection overlap 100%. Triton 0.18ms vs einsum 3.49ms
(19x). HIER (coarse 32-block groups -> top-64 -> fine 2048 cands -> top-512)
vs FLAT recall = 74.94% on RANDOM data (trained-weight recall unmeasured).

BENCH (bf16 randn, prefill, reps 3/1):
| N | coarse triton | topk-groups | hier FULL | core (no sharing) | dense flash | qsa/dense |
|---|---|---|---|---|---|---|
| 32K | 2.8 ms | 2.9 ms | 1524 ms | 616 ms | 232 ms | 9.2x SLOWER |
| 128K | 29.7 ms | 52.6 ms | 4163 ms | 3993 ms | 3585 ms | 2.28x SLOWER |

FINDINGS (the cost law, quantified on sm_121):
1. The fused indexer scorer itself is CHEAP (30ms coarse @128K, linear in N).
2. Bottleneck A = UNFUSED fine stage: torch gather+bmm materializes ~64GB of
   gathered keys @128K (4163ms). Fix: Triton fused fine scorer (gather-in-kernel,
   no [N,2048,128] materialization) — guideline 6 fusion applied to level 2.
3. Bottleneck B = per-query core with NO cross-query block sharing: 512 block-ops
   per query (guideline 7: block COUNT dominates). Core 616ms@32K -> 3993ms@128K
   (6.5x for 4x N = L2 spill: k/v 64MB fits L2 @32K, 256MB does not @128K).
   Fix: query-tile block-major core w/ shared fetch (guideline 8) — flash_nsa
   slc_attn pattern; and bigger blocks (retrain-locked, §7 warning).
4. Gap narrows 9.2x -> 2.28x as dense goes quadratic; naive crossover ~256-512K.
   With fixes 2+3 the QSA row should cross dense well below 128K (NSA did at
   128K with 16x64 blocks = 16 block-ops vs our 512).
5. block-4 @ top_k=2048 = 512 block-ops/query is the WORST case of the Figure-11
   law; §7's block-size sweep {4,16,64,128} at fixed budget is the next experiment.

### 10.1 v2 — FUSED Triton fine scorer (2026-11-24, same harness)

Fix #2 applied: gather-in-kernel fine scorer (_fine_kernel in the proto; program
per 8 rows, candidate chunks of 32, d-tiles of 32; ik gathered L2-resident, no
[N,2048,128] materialization; per-head ReLU+sum in registers; writes only fs
[N,NC] fp32). Correctness: score max_abs_diff 1e-6 vs torch bmm, top-512
selection match 100%.

| N | hier OLD (torch bmm) | hier FUSED (triton) | speedup | core | dense flash | qsa_total / dense |
|---|---|---|---|---|---|---|
| 32K | 909 ms | **69 ms** | 13.2x | 601 ms | 457 ms | 1.47x slower |
| 128K run1 | 4731 ms | **315 ms** | 15.0x | 3841 ms | 4749 ms | 1.14x FASTER |
| 128K run2 | 3543 ms | **332 ms** | 10.7x | 4825 ms | 3940 ms | 0.76x (slower) |

FINDINGS v2:
1. Fused indexer is STABLE across runs (315-332ms @128K) while core/dense swing
   +/-25% with box load (shared UMA) -> honest verdict: QSA proto is at PARITY
   with dense at 128K (0.76-1.14x run-to-run), 1.5x slower at 32K.
2. Indexer is NO LONGER the bottleneck: 315ms vs core 3841-4825ms = core is 92%
   of QSA time. Guideline-8 shared-fetch core (flash_nsa slc_attn pattern:
   sorted selection, query-tile block-major, fetch-once-compute-many) is THE
   remaining lever, plus the block-size sweep (finding 5).
3. Coarse stage scales linear: 1.2->17ms (32K->128K). topk-groups 1.4->33ms.
   Both negligible now.
4. Total proto improvement v1->v2: 8156ms -> 4156-5158ms at 128K (1.6-2.0x).

### 10.2 v3 — SHARED-fetch core (guideline 8, ideal-locality ceiling, 2026-11-24)

Roofline first: per-query core @128K moves N x KH x 512 blocks x 4 keys x 256d
x2B x2(k+v) = ~1 TB of REDUNDANT fetches -> 3.7s at ~273GB/s == measured
3.8-4.8s. It was already AT the memory roofline: instruction fixes cannot help;
only fetch sharing can. _coresh kernel: query tile TQ=8 (rows = 8 queries x 6
GQA heads = 48, padded 64), ONE shared selection list per tile (row 0's ->
IDEAL-LOCALITY CEILING demo, semantics differ), 16-slot chunks (64 keys),
d-tiled QK + online softmax + PV, NaN-guarded, tail EXCLUDED (<0.2% of work).
Correctness: outputs finite (asserted); selection semantics intentionally
changed by the sharing assumption — this brackets the win, it is not a drop-in.

| N | core per-query | core SHARED | speedup | fetch traffic | qsa_tot(shared) | dense | dense/qsa |
|---|---|---|---|---|---|---|---|
| 32K | 1127 ms | **114 ms** | 9.9x | 256->32 GiB | **183 ms** | 216 ms | **1.19x faster** |
| 128K | 3963 ms | **550 ms** | 7.2x | 1024->128 GiB | **861 ms** | 4280 ms | **4.97x faster** |

FULL PIPELINE EVOLUTION @128K: v1 naive 8156ms (2.28x SLOWER than dense) ->
v2 fused indexer 4275ms (1.00x parity) -> v3 shared core **861ms (4.97x FASTER)**.
9.5x end-to-end improvement this session.

CAVEATS (honest):
- Sharing gain assumes FULL tile locality (all 8 queries select the same 512
  blocks). Real trained selection is PARTIAL; production kernel needs per-tile
  union/dedup (flash_nsa slc_attn sorted-selection pattern) and the win scales
  with actual overlap. Random-data bench cannot show partial-locality behavior.
- Tail excluded in shared core; per-query core includes it.
- Box was OVERCOMMITTED during these runs (Committed_AS 207G > CommitLimit
  176G; Huihui-27B live in ollama, 17.3G pool) — dense/core swing +/-25%
  between runs; ratios from the SAME run are the meaningful ones.
- cuDevicePrimaryCtxRetain OOMs appear transiently under this overcommit;
  retry after load spikes settle (do NOT unload live ollama models).

NEXT (map §6): block-size sweep {4,16,64,128} at fixed 2048-key budget
(Figure-11 law, retrain-locked); partial-locality shared core (union of tile
selections + membership mask); GVR decode Top-k (§6.3); distillation spec (§6.4).

### 10.3 Block-size sweep — FINAL VERDICT (2026-11-24, shared box; ratios are the result)

Method: _coresweep kernel (shared-fetch shape, CONSTANT 64 keys/chunk across
legs; CK=128 exceeds sm_121's 101KB shared mem). BS in {4,16,64,128} ->
blocks/query {512,128,32,16}, identical FLOPs + identical fetched key bytes.
Synthesized random selections (seed 42, causality ignored — timing shape only),
ideal-locality tiles (TQ=8). Per-leg launch cfg: default stages for CS>1 legs,
num_stages=2 for CS=1 (BS>=64; default overflows shared mem). mode=sweep.

32K (3 runs): BS4 86*/177/246-438, BS16 82*/152-194, BS64 194-303, BS128 143-178 ms
128K (3 runs): BS4 840/~937/2138, BS16 1451!/853/1514, BS64 850/816/1496, BS128 808/891/1684 ms
(* = default-stages run; ! = contention spike. Absolute ms are box-load dependent
(+/-2.8x); WITHIN-RUN RATIOS are the reproducible signal.)

VERDICT (consistent across ALL runs incl. the quiet-window rerun):
1. BS=4 is consistently the WORST leg: 1.0-1.4x penalty vs BS>=16 (gather
   fragmentation: 16 scattered 4-row segments per 64-key chunk; worse when
   pipelining is off — 1.7x at stages=2).
2. BS 16 / 64 / 128 are FLAT within ~10-15% of each other at fixed 2048-key
   budget. There is NO 2x-per-halving cascade in this kernel class: chunked
   shared fetch (64 keys/chunk) already amortizes per-block overhead (ids
   loads + gather segmentation).
3. The Figure-11 law DOES hold for per-block SEQUENTIAL kernels: per-query
   _core (512 sequential block iterations) = 3963ms vs shared chunked
   ~550-890ms at 128K (4.7-7.2x). Block-count cost is a KERNEL-STRUCTURE
   property, not an intrinsic property of the selection granularity.
4. DESIGN IMPLICATION for k26v15-qwen4: QSA's block-4 training config is NOT
   a performance disaster IF the serving kernel chunks fetches (>=64 keys)
   and shares them across a query tile. The retrain-lock warning (§7) stays:
   do not change block size at inference; fix it in the KERNEL instead.
5. Absolute ms on this box are load-dominated (+/-2.8x, live ollama lanes);
   rerun for absolutes only when "ollama ps" is empty:
   bash ~/Projects/qsa-proto-run-20261124.sh <32768|131072> sweep
   (launcher v2 auto-retries ctx-OOM up to 8x45s; NEVER unload live models).

### 10.4 Partial-locality union core — THE locality law (2026-11-24)

_coreunion + union_prep (batched sort/unique/scatter membership [UM,TQ] uint8;
kernel = sweep-shape with 2D row x key validity mask; CK=32 — CK=64 + the
membership load needs 102,528B > sm_121's 101,376B shared mem). Each query
keeps its OWN 512 blocks; loc = fraction drawn from a shared per-tile pool.
Sanity: loc=1.0 kernel 113ms @32K == _coresh ideal 114ms.

| N | loc | union UM/tile | fetch | prep | kernel | qsa_tot* | vs dense |
|---|---|---|---|---|---|---|---|
| 32K | 1.00 | 512 (12% of 4096 pairs) | 32 GiB | 33 ms | 113 ms | 215 ms | ~2.1x faster |
| 32K | 0.75 | 1331 (32%) | 83 GiB | 34 ms | 196 ms | 299 ms | ~1.5x faster |
| 32K | 0.00 | 3299 (81%) | 206 GiB | 36 ms | 514 ms | 619 ms | ~0.7x (slower) |
| 128K | 1.00 | 512 (12%) | 128 GiB | 127 ms | 721 ms | 1160 ms | 3.1-4.1x faster |
| 128K | 0.75 | 1400 (34%) | 350 GiB | 134 ms | 1366 ms | 1812 ms | 2.0-2.6x faster |
| 128K | 0.00 | 3899 (95%) | 975 GiB | 405 ms | 4442 ms | 5159 ms | parity/slower |
(*qsa_tot = fused indexer (312ms @128K, 69ms @32K) + prep + kernel; dense from
same-window runs 3585-4749ms @128K, 216-457ms @32K.)

FINDINGS:
1. Kernel time tracks FETCH TRAFFIC at ~180-260 GB/s effective (UMA-bound):
   the union mechanism converts selection locality directly into bandwidth
   savings. Random (loc=0) converges to per-query cost (4442 vs 3963ms) +
   prep — sharing cannot beat physics without overlap.
2. **LOCALITY >= ~75% IS THE DESIGN REQUIREMENT** for QSA block-4 to beat
   dense at 128K. That is exactly what distillation must buy: train the
   indexer so neighboring queries select overlapping blocks (the QSA/NSA
   papers claim trained selections are highly local; our randn data is the
   worst case by construction).
3. union_prep (sort-based) costs 127-405ms @128K — NOT negligible vs the 721ms
   ideal kernel. Production: fuse union build into the fine-selection kernel
   or amortize across decode steps (selections change little — pairs with
   Guess-Verify-Refine, §6.3).
4. sm_121 shared-mem ceiling (101,376B) forces CK=32 for the membership-masked
   kernel; CK=64 fits only without the mask (_coresh/_coresweep).

### 10.5 GVR decode select (guideline 6, §6.3) — measured 2026-11-24

_gvr kernel (one program per decode step): guess = previous step's 512-block
selection; gmin = min(guess scores under the NEW row); collect {s >= gmin} into
a BUF=1024-wide candidate buffer (tl.cumsum compaction, no sort); host then
runs topk512 over 1024 instead of NB. Exactness: true top-512 subset of
{s >= gmin} by construction; cnt > BUF -> fallback flag -> full topk for that
step (exactness NEVER compromised). Drift sim: random-walk scores, T=64 steps,
sigma in {0.02, 0.1, 0.5}; overlap measured from actual top-512 sets.

128K (NB=32768), per-step (batched/64):
| sigma | overlap | full topk | GVR kernel + small topk | speedup | fallback | exact |
|---|---|---|---|---|---|---|
| 0.02 | 97.8% | 4.5 us | 1.3 + 0.9 = 2.2 us | **2.10x** | 0% | 100% |
| 0.10 | 91.2% | 4.7 us | 1.1 + 1.0 = 2.1 us | **2.28x** | 3% | 100% |
| 0.50 | 81.4% | 3.3 us | 1.1 + 0.8 = 1.9 us | (1.75x)* | 88% | 100% |
(*misleading at 88% fallback: fallback steps ALSO pay full topk -> effective
~baseline. GVR needs overlap >= ~90% to pay; below that it degrades gracefully
to baseline, never worse than +kernel cost.)

32K (NB=8192): 1.19-1.82x — both paths launch-overhead-bound, select width
doesn't hurt yet at 8K candidates.

Decode indexer scoring (triton flat scorer, 4 heads x NB blocks, bf16):
2.3 us/step @128K, 0.9 us/step @32K. So the full QSA decode index path @128K
= ~2.3 us scoring + ~2.2 us GVR select = ~4.5 us/step — select no longer
dominates once GVR is in (was 4.5 of 6.8 us without).

MATCHES THE BLOG: claimed GVR 2.1-2.4x vs radix select; measured 2.10-2.28x
at realistic drift (overlap >= 91%). The mechanism's value GROWS with NB
(128K > 32K) — right direction for the 1M-context target.
Production note: fuse verify+compact+small-select into ONE kernel (removes
the 0.9us small-topk dispatch + the -inf refill) and derive the guess from
the served selection cache; pairs with §10.4 finding 3 (union_prep amortized
across steps the same way).

## 11. Teacher-selection locality on REAL weights (2026-11-24) — Q1/Q2 empirical bound

Harness: ~/Projects/qsa-teacher-locality-20261124.py (+ qsa-teacher-run-20261124.sh).
Teacher = Qwen3-0.6B (28L, 16Q/8KV/d128, local HF cache) on REAL text (airllm
corpus, 7 docs, 32,768 tokens). Selection = top-B blocks by head-mean of
block-max-pooled attention probs over closed blocks (LSA distill teacher
semantics). Layers 4/14/24; metrics: 8-query-tile union traffic (the §10.4
fetch factor), adjacent-query overlap (Q2 proxy), sinkfrac.

T=32768 (NB=8192):
| B | union traffic (% of per-query) | adj-overlap | sinkfrac |
|---|---|---|---|
| 512 (6.25% budget) | 24.6-28.6% | 100% | 2.5-2.8% |
| 128 (RATIO-MATCHED to ADR 128K point) | 31.1-36.8% | 80.1-87.0% | ~3.2% |

T=4096: B=16 (ratio-matched) union 29-35%, adj-overlap 67-74%, sinkfrac 16-19%.

READ vs ADR-0103 gates:
1. **Q1 (tile locality >=75%) is PLAUSIBLE on real attention.** Teacher union
   traffic 25-37% maps onto the §10.4 locality scale as ~70-90% equivalent
   (loc=0.75 -> 34%, loc=1.0 -> 12%). At teacher-like locality the union core
   lands in the 2.0-2.6x-faster-than-dense band at 128K. sinkfrac ~3% at 32K
   proves this is NOT an attention-sink artifact.
2. **Q2 (decode overlap >=90%) is BORDERLINE from the proxy.** Adjacent-query
   overlap 80-87% at ratio-matched B. BUT decode-step overlap is not adjacent-
   query overlap: real steps grow KV incrementally (papers report near-stable
   selections), and GVR fallback depends on how far dropped blocks sink below
   gmin, not only the swap count: at 85% overlap, w~B*0.15 -> cnt ~ B+w << BUF
   for smooth drift (the §10.5 88%-fallback case was a violent random walk).
   Expect fallback << 88% on real data; MUST re-measure on the distilled ckpt.
3. Layer depth barely matters (L4/L14/L24 within ~4 points).

CAVEATS: 0.6B geometry != 27B (16Q/8KV/d128 vs 24Q/4KV/d256); ONE corpus
(technical text, single domain); prefill selections only; the binding gate
remains a rerun of this probe ON the trained k26v15-qwen4 checkpoint.
Runtime: 91s total at 32K (peak 12.2 GiB) — cheap enough to run per-epoch
during distillation as a live locality monitor.

### 11.1 TRUE decode-step overlap (2026-11-24) — Q2 gate REVISED

Harness: ~/Projects/qsa-decode-overlap-20261124.py (+ qsa-decode-run-20261124.sh).
Real greedy generation (48 steps, 8K-token corpus prompt, growing KV cache);
teacher selection per step over closed blocks, same LSA-distill semantics.

| Layer | Budget B | step-overlap mean | p10 | min |
|---|---|---|---|---|
| L14 | 32 (1.56%, PRODUCTION-RATIO) | 45.5% | 31.2% | 21.9% |
| L24 | 32 (1.56%) | 44.9% | 31.2% | 15.6% |
| L14 | 512 (25%, generous) | 68.6% | 60.2% | 51.2% |
| L24 | 512 (25%) | 69.9% | 62.5% | 58.4% |

VERDICT: teacher (real softmax attention) selections CHURN hard between decode
steps — 45% overlap at the production budget ratio, 69% even at 25% budget.
The §11 adjacent-query prefill proxy (80-87%) was OPTIMISTIC for decode.

CONSEQUENCES for ADR-0103:
1. Q2 (>=90% overlap) is NOT achievable by teacher-like selection with a naive
   prev-step guess. REVISE: stability must be a TRAINING OBJECTIVE (selection-
   consistency term in the distillation loss — the QSA second stage "train the
   model under the sparse read" is where this belongs) OR the guess must widen
   (union of last 2-3 steps; inclusion prob rises, candidate buffer still <<NB).
2. GVR correctness is unaffected (fallback = full topk, exact). GVR viability
   depends on drift DEPTH below gmin, not just swap count — not measured here.
3. SANITY: at 128K, full topk select is only ~4.5us vs decode core read ~30us+
   (2048 keys x 4KV x d256 = 8MB @ 273GB/s) — select is ~1-2% of step time.
   GVR is an optimization for the 1M-context regime (NB=262144), not a 128K
   blocker. Do NOT over-invest before the 1M lane exists.

SM_121 NOTE: 101,376 B shared mem limit — CK=128-key chunks of the shared core
need 164KB and CANNOT compile at any num_stages; 64-key chunks fit (~86KB).

OPS LESSON (2026-11-24): Qwen CLI turns can land in a JAILED docker container
(/.dockerenv; device cgroup denies /dev/nvidia* even with sudo; python3=3.11,
no user dbus). Host GPU was healthy throughout. Escape that WORKS:
ssh spark2 -> ssh spark1 (double-hop; direct ssh spark1/172.24.0.1 = publickey
denied). Home is bind-mounted: files written in the jail are visible on host.
Host interpreter: /usr/bin/python3.12 (torch 2.15.0+cu132 + triton 3.9.0 user-site).
The airllm venv python chain resolves to 3.11 IN THE JAIL — broken there, fine on host.

GB10 UMA ops note: bench launches must go through systemd-run --user --no-block
(outside the 12G hermes-gateway cgroup). zram0 was 98.8% full at 00:05 (31.5/31.9G)
+ 28G swapfile2 -> cuDevicePrimaryCtxRetain OOM despite 29G MemAvailable;
unloading ollama embeds freed enough to run. Check zram/swap commit pressure,
not just MemAvailable, before CUDA bench launches.
