# The end-to-end chain: prompt text in, tokens out

`mega_decode` used to start from a bare token id (`-p 692`) at position 0 and call
that a decode benchmark. That is not a chain: no tokenizer, no prompt, no context,
no weights provenance. This note records what the chain is now, what it measures,
and the one structural bug that the missing chain had been hiding.

## Shape

```
prompt text  --llama-tokenize-->  token ids  --prefill-->  KV cache  --dspark decode-->  token ids --detok.py--> text
```

* **Tokenizer.** The model is GPT-2 BPE, `tokenizer.ggml.model = gpt2`, 130560
  tokens and 129794 merges, byte-level. Prompt ids are produced by
  `llama-tokenize --ids`, so the chain starts from the *reference tokenizer's*
  ids. `detok.py` reverses the same table straight out of the GGUF, which is what
  turns ids back into text for comparison against `llama-simple`.
* **Weights.** `mc-q4o.gguf` is a Q4_0 quantization of the real checkpoint, not a
  toy: `general.name = MiniCPM5 2.6B`, 381 tensors and a 130560-entry vocabulary,
  identical in all three to `models/MiniCPM5-2B-F16.gguf` and
  `models/MiniCPM5-2B-Q8_0.gguf`. The float and Q8 originals are on the board at
  `/home/bianbu/models/`.
* **Prefill.** `target_forward(m, pos0, rowtok, want_logits)` is the same forward
  the decode round uses, driven with a whole prompt chunk at a time. Attention is
  causal per row (`g_attn_np = pos0 + r + 1`) and the KV is written at the row's
  own position, so a chunk needs no extra masking code. Logits are skipped
  (`want_logits = 0`) because a chunk would need `m * VOCAB` floats when only the
  last prompt position's distribution is ever read.
* **Hand-off.** Prefill consumes prompt tokens `0 .. n-2`. The last prompt token
  becomes the decode round's anchor, so round 0 already emits a generated token.

`-f ids.txt` is the prompt entry point; `MEGA_POLL`-style knobs that matter here are
`MEGA_ROWS` (row capacity of the row buffers, default 8) and `MEGA_PFCHUNK` (prefill
chunk, default `MEGA_ROWS`).

## The bug the chain exposed: RoPE pairing

`rope_fn` rotated interleaved pairs `(x[2i], x[2i+1])`, the GPT-NeoX convention.
This checkpoint is llama-architecture and rotates the halves of each head against
each other, `(x[i], x[i+64])`. Nothing had caught it because the only cross-check
in the tree was decode against *itself* (plain greedy against dspark), which is
invariant to a wrong-but-consistent rotation.

Evidence, prompt `The capital of France is` (ids `608 4894 304 6918 357`):

| | generated ids |
|---|---|
| reference `llama-simple` | `8181 33` |
| interleaved (old)      | `8181 350` |
| half-split (now)       | `8181 33` |

`MEGA_ROPE=inter` restores the old pairing. The fix is worth far more than the
alignment itself: the drafter is fed target features, and those features were
computed under the wrong rotation, so its accuracy had been capped by a
representation bug.

| dspark, 21-token prompt, s=3 | before | after |
|---|---|---|
| first-draft accuracy | 68.5% | **77.8%** |
| mean rank of target token in draft logits | 2644 | **47** |
| drafts accepted per round | 1.26 | **1.83** |

## Measured

Real weights, real prompt ids, greedy, 8 participants.

| stage | result |
|---|---|
| prefill, 4 tokens | 0.071 s |
| prefill, 20 tokens | 0.288 s (69.6 tok/s) |
| prefill, 512 tokens | 9.20 s (55.7 tok/s) |
| decode, 21-token context, plain | 17.57 tok/s (56.9 ms/round) |
| decode, 21-token context, dspark s=3 | 21.59 tok/s (131.3 ms/round) |
| decode, 512-token context, plain | 14.91 tok/s (67.1 ms/round) |

At 512 tokens the decode round splits 50.57 ms GEMM / 12.05 ms attention, and
prefill is GEMM-bound.

## Alignment status

Against `llama-simple` with the same Q4_0 file, greedy, same prompt ids:

* 5-token prompt: the first 9 generated tokens match, then `,` vs `.`.
* 21-token prompt: the first 21 generated tokens match, then `,` vs `.`.

Both divergences are a single near-tie at a comma. That is the expected shape of
remaining f32/f16 accumulation-order differences, not a structural error — but it
is not yet "exactly llama.cpp", and the objective asks for exact.

## What is standing between this and the objective

1. **Prefill GEMM is 230 GMAC/s, not 822.** `mega_decode` streams Q4_0 weights
   through `gemm_kernel_i8i4`, and `MEGA_PFSPEED` shows that rate is flat from 32
   to 512 rows (232.0 / 231.6 / 234.7 GMAC/s on `ffn_gate`). `mega3` loads a
   second, int8 weight copy (`-I`) and its `task_pf_gemm_i8` reaches 822. Prefill
   at 200 tok/s needs that representation; decode must keep Q4_0 because it
   streams every byte.
2. **Prefill attention is scalar.** At 10K the attention work is ~9 TMAC, which a
   scalar loop cannot do at all. `mega3`'s int8 tiled `pfa_*` attention does it in
   31.3 s, and that is the number the 174 tok/s prefill was built on.
3. Against those two, 10K prefill is 27.4 s GEMM + 31.3 s attention = 58.7 s =
   174 tok/s. 200 tok/s needs 51.2 s, so attention has to come down by ~7.5 s on
   top of adopting the int8 GEMM path.
4. Decode at 10K also needs a vectorised attention; 12.05 ms at 512 rows scales
   roughly linearly and would swamp the dspark gain.

## The bug that made every prompt turn to noise: shared chunk counters

The prefill GEMM hands 4-row groups to threads through one atomic counter per
column block, so the faster core can simply take more. The counters were indexed
by column block alone and reset once per stage. QKV is three jobs in one stage,
so job 0 (`q_proj`, 32 column blocks) drained counters 0..31 to `mgroups`, and
jobs 1 and 2 (`k_proj`, `v_proj`, only four blocks each) found their counters
already past the end and returned without writing a single element.

The symptom is not a wrong token, it is *no* token: K and V were exactly zero for
every position of the prompt, so the KV cache the decode round read was zero, and
the first generated id came from an empty context. `probe: K(0,0,0)` printed
`0.00000` after a full prefill; exporting the cache showed 0 for all 1024 f16
halves of layer 0.

The fix is to give every (job, column block) its own counter, `NCHUNK = 512`, and
reset the whole array per stage. With it, layer 0 K at position 0 differs from
the decode path's own K for the same token by ~1e-3 in f16 units, i.e. int8
activation noise rather than a layout error.

The chain still does not reproduce the reference ids: for `The capital of France
is` (ids `608 4894 304 6918 357`) `llama-simple` emits `8181 33` (` Paris`,`),
while the chain emits `2781 2781 720`. The f16 attention path (`MEGA_NOI8=1`)
gives the same ids, so int8 attention is not the cause; the deviation grows in
the layer-0 output projection (decode `o-out` 0.00135, prefill 0.0212 added to the
residual). That is larger than per-row int8 activation quantisation can account
for and is the next thing to chase.

## Two more bugs behind the noise: V head 1, and a missing multiply

With K/V actually written, the chain still produced garbage. Two structural
faults explained why, and both are of the same kind: a value that was computed
and then never used, or used with the wrong stride.

* **V half stride.** The int8 V cache is `[kv][dim/64][pos/16]` blocks of
  1024 B, i.e. each kv head owns two blocks per position block. PV addressed it
  as `(kv * npb16 + half * npb16) * 1024`, so kv 0 read blocks 0 and 1 (right)
  and kv 1 read blocks 1 and 2 (kv 0's second half and its own first half).
  Heads 0..7 were correct, heads 8..15 attended over the wrong V: with a single
  visible position the attention output for those heads should equal V itself,
  and it was off by 0.27.
* **`silu(gate) * up`.** Prefill computed the up projection into `g_pfup` and
  then fed `silu(gate)` alone into `ffn_down`. The up half was dead work, and
  the FFN was wrong by a factor that grows with the residual norm.

After both, the layer-0 attention output matches the cached V to 8e-4 (was
0.27), and the first generated id moves from `2781` to `33` - which is the
*second* id of the reference (`8181 33`): one greedy step away instead of
unrelated.

The remaining gap is numerical, and it is localised. Measured at layer 0's V
projection against a scalar f32 reference over the same normalised row:

| activation quantisation | V[0]   |
|---|---|
| f32                    | -0.00250 |
| per-row int8 (prefill) | -0.03885 |
| per-32-block int8 (decode) | -0.00037 |

Per-row scaling has to cover 2048 values including outliers, so its step is
roughly an order of magnitude coarser than per-32-block; on this row it costs
15x the error. Prefill uses per-row because the vmadot kernel accumulates int32
over the whole K; per-block scales would need a float flush every 32 elements,
which is the open question for alignment.

## Alignment: the attention output was layer 0's, for every layer

The int8 PV path wrote its result into `g_pfa_out` only when `layer == 0`
(`if (layer != 0 || !g_pfa_out) continue;` in both PV variants). Every layer past
the first therefore ran its output projection on the *previous* layer 0 attention
output, and the residual accumulated the same 2048-vector 42 times. With that
guard removed, a four-row chunk tracks the f32 reference layer by layer:

| layer | attention+residual | after FFN |
|---|---|---|
| 0  | 2.8e-2 | 3.0e-2 |
| 8  | 8.5e-2 | 8.7e-2 |
| 32 | 8.2e-2 | 8.1e-2 |
| 41 | 7.8e-2 | 1.4e-1 |

A single-row prefill (no attention mixing) sits at 1e-2 to 3e-3 the whole way,
so the remaining drift is the four-row path.

Two smaller faults found while measuring:

* **Activation record stride.** `g_a8_stride` reserved a fixed 512 B for the
  per-block scales. At `A_BLK=256` the k=6144 down projection needs 384 B, so
  everything looked fine; at 64 it needs 1536 B and each group's scale block
  overwrote the next group's payload. Sized from `k/A_BLK` now.
* **No way to see the head.** `argmax` hides how close the logits are, so
  `MEGA_TOP=1` prints the top eight with values. Against the f32 reference for
  the same 5-token prompt (`8181 10.83, 280 9.65, 285 9.38`), the vmadot path
  gave `285 9.68, 280 9.41, 8181 8.97`: the right candidate set, with `8181`
  short by 1.9.

## Exact Q4_0 block scales: the tokens line up

The vmadot path re-quantises the dequantised 4-bit weights to int8 with one
scale per output row. On layer 0 that is 2.0% on the q/k/v projections; the
activation side at `A_BLK=256` adds 9.9%. Running the same projections through
the vendor m4 `i8i4` kernel instead - which keeps each Q4_0 block's own scale
and quantises activations per 32 k elements - matches the reference:

```
prefill: 4 tokens in 0.838 s
step 0 -> 8181 (10.7097)
  top8: 8181(10.71) 280(9.63) 285(9.34) 8266(8.92) 2297(8.69) 310(8.03) ...
```

`-W` (or `MEGA_VENDOR=1`) selects it: `pf_quant()` picks the quantiser and the
`PfJob` weight pointer picks the Q4_0 copy (`b`) over the requantised one
(`b8`); everything downstream, including the f16 KV cache, is unchanged. The
first two greedy ids agree with the f32 reference (`8181`, then `350` - the
reference has `350 20.086` and `33 20.043`, so llama.cpp picking `33` there is
its own numerics on a 0.04 margin).

### The price, at 10K

| prefill GEMM | activation blocks | weights | 10K prefill | first id |
|---|---|---|---|---|
| vmadot | 256 | per-row int8 | 57.6 s (178 tok/s) | 285 |
| vmadot | 128 | per-row int8 | 115.0 s (89 tok/s) | 285 |
| vmadot | 64, m4 flush | per-row int8 | 218.2 s (47 tok/s) | 8181 (margin 0.11) |
| vmadot | 32, m4 flush | per-row int8 | (not run) | 8181 (margin 0.13) |
| vendor `i8i4` | 32 | Q4_0 exact | 93.8 s (109 tok/s) | 8181 (margin 1.08) |

At `A_BLK=64` the flush runs four times as often and the GEMM goes from 25.6 s
to 181 s, far more than the instruction count explains; rewriting the flush as
m4 (128 lanes per op, 12 ops instead of 48 per block) only bought 315 s to
218 s. The accumulator round-trip per block is the suspect. Fine activation
blocks plus exact weights is the configuration that both aligns and stays fast,
and that needs the block loop inside the kernel with the f32 accumulators held
in registers.

### Moving the accumulators into registers: the flush gets cheap, the layout wins

Attempted and reverted. `gemm_i8d_4x64_blk` kept both accumulator sets in vector
registers across the k loop, running the dot 16 steps then folding the block's
four row scales in with `vfcvt` + `vfmacc` before the next block. That does
remove the round-trip: at `A_BLK=256` the 10K GEMM went 69.0 s -> 37.6 s and the
whole prefill 104.8 s -> 71.8 s (142.6 tok/s).

Two hardware behaviours cost a day and are worth recording:

1. `vrgather.vv` with `vd == vs2` raises an illegal instruction on A100. The
   instruction needs a different destination register even though the source is
   dead afterwards.
2. `vmadot vD` writes 64 bits past its own register: with markers in `v17..v31`
   only `v17` changed after `vmadot v16`. A destination must be even, so every
   odd register directly above an accumulator is unusable for anything that
   lives across the dot - which is where the 8 f32 accumulators had been left.

Those two are fixed, but the port is still not numerically equal to
`gemm_i8d_4x64`: the second 32-column pass (`B + 512` -> `C + 128`) disagrees
with the legacy kernel's registers 4-7, while the first pass matches exactly.
`A_BLK=64` would also have been the wrong trade anyway - with the flush in
registers the 10K GEMM at 64 is 114 s against 37.6 s at 256, i.e. four times the
flushes costs 3x, so fine blocks stay expensive even with no memory traffic.
The aligned configurations therefore remain `A_BLK=64` with the memory flush
(69 tok/s) and the exact-Q4 vendor path (109 tok/s).
