# MiniCPM5-2B megakernel on SpacemiT K3

## Deliverables

- `mega_minicpm_pf.cpp` -> `mega_minicpm_pf` - whole-model single-graph runner:
  one `ggml_backend_graph_compute` per token over all 42 layers (embedding +
  layers + lm_head + argmax). KV history is host-resident and fed to the graph as
  inputs; the current K/V rows are written in-graph. Batched prefill, greedy
  decode, plain decode (no speculation).
- `mega_minicpm_spec.cpp` -> `mega_minicpm_spec` - same runner plus prompt-lookup
  n-gram speculative decoding (`MEGA_DRAFT=k`): draft k tokens from the prompt,
  verify all k+1 rows in one graph pass, accept the longest matching prefix and
  emit the bonus token.
- `mega_fused.cpp`, `mega_minicpm.cpp` - earlier prototypes (short-context fused
  megakernel, `--ref` runner). Kept for reference; superseded by the two above.

## Build

```sh
L=~/workspace/llama-cpp-31953bc
g++ -O3 -std=c++17 -march=rv64gcv1p0 -mabi=lp64d \
  -I$L/include -I$L/ggml/include -I$L/src -I$L/ggml/src \
  mega_minicpm_pf.cpp -o mega_minicpm_pf \
  -L$L/build-smt/bin -lllama -lggml -lggml-cpu -lggml-base \
  -Wl,-rpath,$L/build-smt/bin
```

Current binaries are `mega_minicpm_pf.zp` / `mega_minicpm_spec.zp` (mask
zero-prefix hint). `.new` is the same source without the hint, kept as the A/B
control; rebuild `ggml-base ggml-cpu` before relinking since the hint lives in
the backend (`ggml_flash_attn_ext_set_mask_zero_prefix`).

## Attention-path work (ggml-cpu, spacemit backend)

- Split-KV tiled flash attention (`FA_SPLITKV_TILED=1`, default on): for long KV
  each of the 8 threads owns a KV slice, writes one `(M, S, VKQ[128])` record per
  query row, and the caller merges with an online-softmax pass. Engaged for both
  decode (nr=16) and verify (nr=64/128) and for the 10K prefill tail chunk.
  A `SPLITKV_TILED` trace line reports nr/nek1/nth when it fires.
- Null mask on the decode FA path: the single query at `pos` attends to exactly
  rows `[0, pos]`, so the additive mask is all zeros. Passing `mask = nullptr`
  removes a full f16 mask pass over the KV range (~9 ms/token at 10K) and is
  bit-exact -- token streams are identical with the mask present and absent.
- Prefill batches use the same split path with the real causal mask; tokens are
  unchanged versus the pre-split build.
- Mask zero prefix (`ggml_flash_attn_ext_set_mask_zero_prefix`): the batched
  causal mask for a batch starting at position `p` is exactly zero for KV
  columns `[0, p]`. The producer declares that prefix in op param slot 5 and the
  tiled kernel skips the mask read/add/max for whole KV tiles inside it. Taking
  `+0.0` off cannot change a rounding, so results are bit-exact: the `.zp` build
  at 10K emits the same token stream as the control and prefill rises from
  68.30 to 69.79 t/s (mask was 2.2% of prefill attention; the skipped prefix is
  most of it). Verify batches (k+1 rows) get the same hint; K=4 decode improves
  58.64 -> 57.29 ms/token.

## Validation (K3, Q4_0)

- Single-token forward: top-5 logits match llama.cpp exactly.
- Multi-token contexts: top-5 candidates match llama.cpp exactly.
- Greedy token ids can diverge from llama.cpp at near-ties because the graphs use
  different summation orders; this is a precision difference, not a model error.
  Output remains fluent.
- Decode with the mask dropped reproduces the masked run token-for-token.
- Speculative decoding K=3 at 10K reproduces the plain-decode token stream
  exactly; larger K changes row tiling and can flip a near-tie at some position.

## Performance (10K context, Q4_0, `-t 8`)

Environment for every run:
`SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1`.

| config | prefill | decode ms/tok | decode t/s | acceptance |
|---|---|---|---|---|
| `mega_minicpm_pf.zp`, split-KV on | **69.8 t/s** | 115.5 | 8.66 | - |
| `mega_minicpm_pf` (pre-prefix), split-KV on | 68.3 t/s | 114.5 | 8.73 | - |
| `mega_minicpm_pf`, split-KV off | 68.3 t/s | 126.8 | 7.89 | - |
| `mega_minicpm_spec.zp`, MEGA_DRAFT=2 | 69.8 t/s | 75.0 | 13.3 | 66.7% (2.29) |
| `mega_minicpm_spec.zp`, MEGA_DRAFT=3 | 69.8 t/s | 62.3 | 16.0 | 62.1% (2.78) |
| `mega_minicpm_spec.zp`, **MEGA_DRAFT=4** | **69.8 t/s** | **57.3** | **17.5** | 63.9% (3.42) |
| `mega_minicpm_spec.zp`, MEGA_DRAFT=5 | 69.8 t/s | 69.5 | 14.4 | 46.2% (3.15) |
| `mega_minicpm_spec.zp`, MEGA_DRAFT=6 | 69.8 t/s | 67.8 | 14.8 | 49.5% (3.76) |
| `mega_minicpm_spec.zp`, MEGA_DRAFT=7 | 69.8 t/s | 74.0 | 13.5 | 39.2% (3.47) |

The `.zp` rows for K=2/3/5/6/7 were measured on the pre-prefix build (68.3 t/s
prefill); only the decode column is compared, and the prefix hint changes only
the verify batch's mask work. K=4 was re-measured on `.zp`: 57.29 ms/token.

Reference points: mtmd-backend baseline 40.00 t/s prefill / 5.86 t/s decode;
llama-bench on the same board 9.11 / 8.34 t/s.

`-t 8` is not a tuning choice but a requirement: `-t 12/16` aborts
(`thread_n exceeds perfer_core_ids size 8`), and `-t 4` collapses to 36.3 t/s
prefill / 907 ms per decode token because the backend leaves the packed path.

Prefill at 10K is attention-compute bound, not bandwidth bound. FA profiling of
the prefill phase: 1184.8 ms total, qk 51.5%, pv 34.3%, kx 2.8%, mask 2.2%.
Int8 flash attention was measured and is slower here (prefill 58.5 t/s, decode
206.7 ms/token), so it is not used.

Plain decode is weight-stream bound: the Q4_0 weights are ~1.32 GiB and one
weight pass at the measured GEMM streaming rate (~20 GB/s) costs ~71 ms, so
115 ms/token leaves ~40% of headroom. Only batched verification (speculation)
breaks the one-token-per-pass limit; the best point, K=4, is 3.42 accepted
tokens per weight pass at 57.3 ms/token (17.5 t/s).
Past K=4 acceptance falls faster than the verify batch grows, so the curve peaks
there: the verify pass reads the weights once regardless of k, but each extra
row costs attention over the whole KV history.

## Commands

```sh
cd ~/workspace/mega
export SPACEMIT_TCM_HOLD=1 SPACEMIT_DECODE_CORES=8-15 GGML_TILED_KPACK=1

# plain 10K prefill + decode
./mega_minicpm_pf   -m /dev/shm/mc-q4o.gguf -f ~/opt10k/ctx10k.txt -n 64 -c 12288 -t 8

# speculative (draft length 4 is the 10K optimum)
MEGA_DRAFT=4 ./mega_minicpm_spec.zp -m /dev/shm/mc-q4o.gguf -f ~/opt10k/ctx10k.txt -n 64 -c 12288 -t 8

# diagnostics
FA_TRACE=1   # print when the split-KV path fires
FA_PROF=1    # per-window FA kernel breakdown
MEGA_NO_FA=1 # fall back to mul_mat + soft_max + mul_mat attention
```
