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

**Correction, after the fp32 judge was built (see the section further down):**
matching llama.cpp Q4_0 is not the objective and was never achievable as stated,
because llama.cpp's own Q4_0 is 0.66 cosine away from exact fp32 on a 64-token
prompt. On a 512-token prompt the two stacks are `35 242 38 35` for exact fp32
and this chain, and `35 1415 26 242` for llama.cpp Q4_0. The chain is on the
right side of that comparison.

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
| vendor `i8i4_hp_m4` | 32 (K256) | Q4_0 exact | 75.8 s (135 tok/s) | 8181, then `33` |

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

### The vendor K256 hp kernel is the aligned fast path

`gemm_kernel_i8i4_hp_m4` is the kernel llama.cpp's own spacemit prefill uses, and
it is a different trade from the K32 `gemm_kernel_i8i4` the `-W` path calls. It
works on K256 superblocks and keeps the fp32 accumulators in v28-v31 for the
whole superblock, folding each K32 scale in as it goes. Measured on one A100
core with the same 4-row tile:

| kernel | blk_len | GMAC/s |
|---|---|---|
| `gemm_kernel_i8i4_m4` | 32 | 33.3 |
| `gemm_kernel_i8i4_m4` | 32, count_m 8/16/128 | 33.3 (count_m is ignored above 4) |
| `gemm_kernel_i8i8_m4` | 32 | 33.0 |
| `gemm_kernel_i8i4_hp_m4` | 256 | 47.7 |

Wiring it in is a quantiser swap plus a stride change:
`quantize_a_4row_i8_hp(256, ...)` writes 1160 B per 4-row group per 256 k
(8 subblocks of 4 fp16 row scales + 4x32 int8, then a 72 B trailer of per-subblock
sums), and B needs no repacking at all - the existing 576 B per K32 per 32 columns
is exactly the hp subblock, so eight of them in a row are the 4608 B superblock.

At 10K that is GEMM 59.0 s -> 41.1 s and prefill 94.1 s -> 75.8 s, 135.2 tok/s,
with `-X` now the default. It also aligns better than `-W`, not worse: on
`The capital of France is` the reference (`llama-simple`) emits `8181 33`, `-W`
emits `8181 350` (step 1 wrong), and `-X` emits `8181 33`. Getting the same
kernel on both sides of the comparison matters more than f32-versus-fp16 block
scales; the hp path's own dot accuracy is 0.37% rms against the K32 path, and
its step-0 margin is thinner (0.30 against 1.08) even though the id is right.

count_m in the vendor GEMM is worth recording as a dead end: the dispatcher only
checks `count_m >= 4`, the m4 kernel always computes 4 rows, and a benchmark that
scales GMAC/s by count_m is measuring nothing - the wall time is flat from m=4 to
m=128.
### Converting only the new KV positions keeps the token, then loses it

The chunked prefill reconverted the whole prefix in every chunk, so 20 chunks over
10240 rows ran ~107k position conversions per layer, 6.4 s of the 76 s prefill.
Two changes were tried against that:

1. Pin the int8 map strides to the buffer capacity (`g_k8i_npb_cap`,
   `g_v8i_npb16_cap`) instead of `(npos + 63) / 64`. This one is free and now in:
   with the dynamic stride every already-converted block moved each time the
   prefix grew, which was the real reason a chunk could not just convert its own
   rows. Prefill time and the first token are unchanged (6.41 s conv, token 5).
2. Convert only `[done, npos)` per chunk and rescale the prefix only when a new
   maximum appears, with a final whole-prefix rewrite on the last chunk for good
   measure. Conv 6.41 s -> 2.37 s, but 10K prefill then picks token 11127 (9.98)
   instead of 5 (14.36). Toggling the final rewrite off and on changes nothing -
   both give 11127 - so the difference is not the bytes the cache ends up with.
   The greedy chain is following the scale each position carried *while* prefill
   ran; per-chunk scales are numerically different from whole-prefix scales even
   when the last chunk reconciles them. Reverted.

The way to keep the 4 s would be a per-64-position-block scale (`sk` per block
inside the K map, `sv` per 16-position block in the V map) plus a per-block scale
multiply in the QK epilogue and the PV accumulate. Blocks would then never need
the prefix rescanned, and the conversion would be one pass per chunk by
construction rather than by bookkeeping.
### The prefill GEMM is issue-bound, not bandwidth-bound

The obvious explanation for 47.7 GMAC/s per A100 core was weight streaming: a
576 B B block serves 4 rows x 32 columns x 32 k, i.e. 0.14 B per MAC, so 41 s of
10K GEMM reads ~275 GB per core. `MEGA_BHOT=1` replaces every column block with
block 0, keeping 36 KB of B resident for the whole matrix: GEMM 41.06 s -> 40.60 s.
The numbers are garbage but the timing is not, and it says B traffic costs 1%.

Bigger row tiles point the same way. `-M 8` (32 rows per B pass) is 40.72 s,
`-M 16` (64 rows) is 45.50 s and `-M 32` is 88.69 s, because the A record for the
tile outgrows the 384 KB TCM at 390 KB and falls back to re-reading A from DRAM
once per column block. So the 47.7 GMAC/s is the kernel's own issue rate
(26 MAC/cycle at 1.8 GHz, i.e. one `vmadot` per ~19 cycles), and it is the same
number attention sits at: QK 101.8 core-s over 4.5 TMAC is 44 GMAC/s.

That puts the A-cluster floor for 10K prefill at ~61 s of MAC work
(GEMM 41 + QK 20 + PV 14 measured at 47.7/44/42 GMAC/s), and the current 73.4 s is
that floor plus conv, quant, norm, silu and softmax.

### The X100 GEMM split was computing everything twice

`-x100 P` hands the last P% of each matrix's columns to the X100 cluster, but
`task_pf_gemm` sized its column loop from `g_pfj[j].n` and ignored
`g_pfj[j].nlim`, so the A cluster kept computing every column. Both clusters then
wrote the same C elements with different accumulation orders: GEMM 41.06 -> 44.82 s
and a different token. Reading `nlim` gives GEMM 34.88 s and 10K prefill 69.16 s
(148.1 tok/s) at P=30, but the short-prompt chain moves off `8181 33` and P=45
already moves 10K token 0 off `5`. The X100 kernel applies one scale per row over
the whole K while the hp kernel folds a scale per K32 block into an fp32
accumulator, so the two paths cannot agree bit for bit; making the X100 path
usable means giving it the hp A record and the same per-block fold, which is a
kernel of its own. `-x100` therefore stays off by default.

### The X100 cluster does pay for the KV int8 conversion

Conv is a streaming pass over f16 K/V, so its cost follows thread count rather
than per-thread speed: sixteen even bands give 6.41 s -> 4.38 s, while the
weighted split that suits compute-bound work (4:1 in favour of the A cluster, an
X100 core being roughly a quarter as fast at f32 vector work) gave 4.39 s for conv
and made rmsnorm and silu *slower* than the plain eight-way split (silu
1.52 -> 1.94 s). So conv runs on all sixteen, everything else stays on the A
cluster. 10K prefill 75.70 -> 73.43 s (139.4 tok/s), first token `5` at 14.3568,
short prompt `8181 33` unchanged -- the results are bit-identical because the
bands only change which thread quantises which position.

One hardware trap to keep: the 16-byte store in `conv16` must be a single
sixteen-lane step (`e16m2` -> `e8m1`). Chunking it by `vsetvl_e16mf2`, which is
correct on the A cluster, holds only eight lanes on an X100 core (VLEN=256), and
the second half of every 16-byte block was silently left unwritten -- the 10K
cache still looked plausible and only the generated token moved. The same applies
to the max scan, which now walks the head dim in `vsetvl` chunks instead of
assuming one `e16m4` register holds all 128 dims.

Alignment evidence at 10K is still token 0 only: `llama-simple` has no `-c` and
silently windows a 10K prompt down to its 512-token context, so the same-prompt
reference continuation needs a driver that can set `n_ctx`.

## The 10K cache was one position too small

`MAXCTX` was 10240, and a "10K" prompt is 10241 ids: the prefill fills positions
0..10239, then the first generated token lands at position 10240 - one past the
last slot. Both caches are strided by `MAXCTX`, so that token's V row
(`[layer][pos][kv][dim]`) writes over layer L+1's position 0, its K row over the
next kv head's position block, and the rope table's last entry runs 128 floats
past its allocation. Every 10K number and every 10K token id taken before this
was measured with the top of the context scribbling on itself.

`MAXCTX` is now 12288 and the `-F` chain refuses `pos0 + ngen > MAXCTX` instead
of silently overrunning.

## The reference was the wrong file

The alignment numbers in this file were compared against `llama-simple`, which
cannot be told `n_ctx` and therefore truncated any 10K prompt to its 512-token
default. The ids it produced for the short prompt (`8181 33`) are what the F16
and Q8_0 files give; with the same Q4_0 file it prints `285` (` a great idea`),
i.e. the 4-bit file flips that token in llama.cpp's own arithmetic.

`refgreedy.cpp` is the harness that fixes this: it links libllama, feeds the
prompt as token ids straight from a file, takes `n_ctx` on the command line, and
does nothing but argmax, so both sides run the same ids over the same context
and print ids that can be diffed line by line.

## Decode runs a different kernel than llama.cpp

`llama.cpp`'s spacemit backend maps `GGML_TYPE_Q4_0` to `q4_0_32x256_q8_0` for
every shape, and that entry point picks its **one-row** body for `count_m < 4`.
Our decode was feeding the one-row path a K32 (`gemm_kernel_i8i4`) record
instead of the K256 hp record (`gemm_kernel_i8i4_hp`, 290 B per 256 k for one
row: 256 int8, 8 int16 subblock sums, one fp16 block scale at offset 288).

Both are defensible numerically, and they disagree on a knife-edge prompt: on
the 5-token prompt the K32 kernel gives `8181` (the F16 answer) while the hp
one-row kernel gives `285`, which is what llama.cpp gives for the same file.

Padding one row up to a four-row hp group and calling `_hp_m4` does **not**
reproduce the one-row kernel - the two entry points quantise A with different
record layouts, and the padded version gives `30930`. So:

| decode kernel | 5-token first id | short-context rate |
|---|---|---|
| `gemm_kernel_i8i4` (K32, 1 row) | 8181 | 17.3 tok/s |
| `gemm_kernel_i8i4_hp` (K256, 1 row) | 285 (matches llama.cpp) | 6.3 tok/s |
| `_hp_m4` with rows 1..3 zeroed | 30930 | 5.1 tok/s |

`-H` selects the second. It is 2.7x slower than the K32 kernel, so it is an
alignment probe, not the speed path.

## 2-bit weights

`-q2` repacks the Q4_0 source into the vendor `nrow_block_q2_k<32>` layout
(`gemm_kernel_i8i2k`): 2688 B per 32 rows x 256 k = 2.625 bit per weight, so the
decode blob drops from 1416 MB to 738 MB. Short-context decode goes 17.3 ->
24.9 tok/s. The encoding follows llama.cpp's Q2_K scheme (per-16-value min/max,
`d = max_scale/15`, `nib = round(15*scale/max_scale)`, effective multiplier
`nib*d` with no 1/16 - the kernel's `factor_scale` of 1/16 is undone by its
`refactor_scale` of 16) but skips `make_qkx2_quants`, so it is not the same file
as `llama-quantize --pure`. It is a different quantisation, hence its own token
trajectory; it needs its own reference run before it can be called aligned.

## Layer 0, checked against exact fp32: our stack is faithful

`layer0_ref.py` recomputes layer 0 from the same GGUF in float64 and diffs it
against the megakernel's `MEGA_DUMP` output for the last prefilled row (31
tokens, `-X`):

| op | cosine vs exact | max abs diff |
|---|---|---|
| V projection | 0.999151 | 9.4e-3 |
| Q after rope | 0.999981 | 8.5e-4 |
| attention output (pre `wo`) | 0.999778 | 2.3e-3 |
| post-attention residual | 0.999983 | 1.0e-3 |

So rmsnorm, the Q4_0 K256 GEMM, the half-split rope, GQA attention and the
output projection are all right; what is left is int8 activation/attention
rounding at the 1e-3 level.

Two traps this tool exists because of, both of which produced meaningless
"mismatches" first:

* GGUF tensor offsets are relative to the data section, not the file. Reading
  them as file offsets turns the f16 block scales into header bytes and every
  value becomes NaN.
* The `-F` chain prefills `nt-1` ids and keeps the last as the decode anchor, so
  the dumped row `cm-1` is the *second-to-last* id. Comparing it against the
  last id shifts everything by one position and makes a perfect layer look like
  cosine 0.46.

## Why "match llama.cpp" needs a better target than its Q4_0

Same 32-token prompt, comparing the final normed hidden state that both sides
multiply by the output projection:

| pair | cosine |
|---|---|
| llama.cpp F16 vs llama.cpp Q4_0 | 0.962 |
| ours vs llama.cpp Q4_0 | 0.915 |
| ours layer 0 vs exact fp32 | 0.99998 |

llama.cpp's own Q4_0 runs 0.962 away from its F16 on a 32-token prompt, and the
two disagree on the first greedy id of `"The capital of France is"`
(`8181` = " Paris" for F16 and Q8_0, `285` = " a" for Q4_0). A token-level
alignment contract with Q4_0 therefore pins us to one particular kernel's
rounding, not to the model. The F16/Q8_0 trajectory is the one worth matching.

## The judge is exact fp32, and the megakernel is the side that agrees with itThe section above left one question open: when llama.cpp's Q4_0 and this chain
disagree, which one is wrong? `ref_f32.py` answers it. It is a numpy float32
forward pass over the same GGUF, one layer at a time, with no KV reuse and no
sampling - each greedy step re-runs the whole prompt, so it cannot drift.

512-token prompt, greedy, four tokens:

| stack | ids |
|---|---|
| exact fp32 (numpy, whole prompt recomputed per step) | `35 242 38 35` |
| **this megakernel** (`-X`, Q4_0 exact weights) | `35 242 38 35` |
| llama.cpp Q4_0 (`refgreedy`, same GGUF, same ids) | `35 1415 26 242` |

Full logit rows, 64-token prompt, last prompt position:

| pair | cosine | rms diff | top-1 |
|---|---|---|---|
| megakernel vs exact fp32 | **0.999002** | 0.108 | `65401` vs `22354`, gap 0.13 |
| llama.cpp Q4_0 vs exact fp32 | 0.6636 | 1.93 | `22354` vs `22354` |

So the chain is faithful to the model and llama.cpp's Q4_0 is not: 20x the rms
error, and its top-8 order is scrambled (`3371` at rank 11, `318` at rank 1).
The two tokens that separate them are a genuine quantisation artifact of
llama.cpp's own Q4_0 kernel, not of ours.

Consequences for the rest of this document:

* "Aligned with llama.cpp Q4_0" is no longer the acceptance test and never was
  a meaningful one. `-q2` and `-q3` were rejected in the 2-bit section partly
  because they fork from llama.cpp on the first token; the fork is not evidence,
  since llama.cpp Q4_0 forks from exact fp32 on the same prompt.
* What an alignment run has to compare against now is `ref_f32.py` (slow but
  exact) or llama.cpp F16. Both are several hundred tokens of context per
  minute at best, so the practical protocol is: establish the fp32 trajectory
  once on a given prompt, then check candidate configurations against that.

Two traps this cost, worth keeping:

* **Never run two megakernels, or a megakernel and `refgreedy`, at the same
  time.** They take the same TCM blocks from `/dev/tcm_sync_mem`. Running them
  concurrently produced a 244 s prefill, a first token of `100312` instead of
  `220`, and a "non-deterministic prefill" scare that evaporated once the runs
  were serialised: two clean back-to-back 10K runs hash the int8 KV cache
  identically (`K aa4d3369bd72773e`, `V 4a9ac7dc14d233e0`) and agree on the
  first two tokens.
* **`-F` prefills `n-1` ids.** The dumped last row belongs to the second-to-last
  id. A row-for-row diff has to account for it or a perfect layer looks like
  cosine 0.46.
## The 10K decode ceiling: bytes, not kernels

Measured on a quiet board, one run at a time, 10K prompt, greedy, 12 tokens.
Every row prints `decode path:` first so a flag the parser swallowed cannot pass
for a failed optimisation (that had already happened once, see the `-X` note in
`docs/k3-minicpm-megakernel-notes.md`).

| config | GEMM ms | attn ms | decode tok/s |
|---|---|---|---|
| Q4_0, f16 KV | 52.3 | 40.3 | 10.29 |
| Q4_0, int8 KV | 53.8 | 32.9 | **11.18** |
| `-q3` (homemade), int8 KV | 40.8 | 32.8 | 12.86 |
| `-q3`, int8 KV, 4096 window | 42.8 | 15.1 | 16.50 |
| `-q2` (homemade), int8 KV | 36.0 | 32.8 | 13.90 |

Bytes per token for the Q4_0 + int8 KV row: 1265 MB of weights plus 215 MB of
int8 K and V, i.e. 1480 MB in 89.5 ms = **16.5 GB/s**. A plain contiguous stream
with huge pages measures 17.7 GB/s at 8 threads and 19.8 at 16 (`bwprobe.cpp`,
2 GB buffer, three passes), so the decode loop is already at ~85% of what the
memory system will give.

That is the whole story of the 10K number: it is a byte-count problem.
20 tok/s is 50 ms, which at 17 GB/s allows 850 MB per token. Q4_0 weights alone
are 1265 MB, so **no amount of kernel tuning reaches the target while the weights
stay 4-bit.** The two ways to cut bytes are a coarser weight format or fewer
weight reads per token (speculative decoding).

### A coarser weight format is not available at acceptable quality

The logit row for the last prompt position, 64-token prompt, against exact fp32
(`ref_f32.py`):

| weights | cosine vs fp32 | rms diff | first id |
|---|---|---|---|
| Q4_0 (`-X`, vendor K256 hp) | **0.999002** | 0.108 | `65401` (fp32 `22354`, gap 0.13) |
| `-q3`, 1001.6 MB | 0.176614 | 3.53 | `97761` |
| `-q2`, 738 MB | 0.010577 | 3.60 | `117223` |

The in-tree `-q3` / `-q2` packers requantise the *dequantised Q4_0* weights
per k16 and the result is not 3-bit quality, it is noise: cosine 0.18 and 0.01.
Their 12.9 and 13.9 tok/s are therefore speeds for a different, broken model.
This closes the question the earlier "2-bit" section left open: the cliff is not
at 2 bits specifically, it is that these two packers do not preserve anything.

The vendor `gemm_kernel_i8i3k` / `i8i2k` kernels are fine and worthwhile; what is
missing is real Q3_K weights from `llama-quantize` (`MiniCPM5-2B-Q3KM.gguf` is on
the board, PPL 26.63 against Q8_0's 21.86) instead of a repack of Q4_0. Feeding
that file through the existing Q3_K decode branch is the next quality-correct
step, but note the arithmetic it buys: 1001.6 MB + 215 MB of int8 KV is 1216 MB,
which is 74 ms at 16.5 GB/s, i.e. **13.5 tok/s**. Still short of 20.

### Speculative decoding is the only lever left with the required size

Reusing the weight read across several tokens is what breaks the byte wall: a
round that verifies four rows still streams the weights once. At 1 accepted draft
per round the bytes per emitted token halve (1480 MB / 2 rows = 740 MB = 45 ms =
22 tok/s); the measured drafter accepts 1.26 on short context.

The catch, which the short-context numbers hide: at 10K one four-row round cost
2.3x a one-row round on a 21-token context (56.9 -> 131.3 ms), and that factor
came from the GEMM being bandwidth bound. At 10K attention is 37% of the round
and is *not* bandwidth bound: qk takes 14.0 ms for 215 MB of f16 K, and 14.8 ms
for 108 MB of int8 K, i.e. the same time for half the bytes. So four rows will
cost close to 4x on the attention half, and a round lands near
54 + 131 = 185 ms for 2.26 tokens = 12 tok/s. Speculation alone does not get
there either; the attention kernel has to stop being time-per-position bound
first.

Where attention really sits: qk 14.0 ms and pv 23.9 ms for 215 MB each (f16),
which is 15.4 and 9.0 GB/s, and neither number moves when the width halves. The
loop issues 4 loads plus 32 FMAs per 4 dims per 32-position block, so it is
neither at the memory ceiling nor at the FMA issue ceiling; it is stalling on the
32-byte granular loads. The fix with precedent in this tree is the one prefill
already uses: run QK and PV as int8 `vmadot` GEMMs (scores are a 16x32 by 128
GEMM, the output is a 16x128 by 32 GEMM), which is 512 products per instruction
instead of 32.

### Why PV is the slow half, and it is not the arithmetic

`bwprobe.cpp` with `MEGA_STRIDE=n`, which reads one 128-byte block out of every
n*128 and reports only the bytes it used, 2 GB buffer, 8 threads:

| pattern | useful GB/s |
|---|---|
| contiguous | 17.3 |
| 1 of every 2 blocks | 10.8 |
| 1 of every 4 blocks | 9.0 |
| 1 of every 8 blocks | 7.8 |

PV measures 23.9 ms for 215 MB of f16 V at 10K, which is **9.0 GB/s** — the
stride-4 number to the digit. The reason is structural: PV's inner loop is over
positions with the head dimension fixed, so for a given (kv, half) it reads the
128-byte quarter of every 512-byte position stride, and it makes four such passes
to cover both kv heads and both halves. The accumulators force it: 16 f32m1
registers is the most that fits alongside the V loads and the scale broadcast, so
two dimension chunks is the widest a pass can go, which leaves the four-way
split.

The obvious fix, one pass per kv at m4, was written and measured: **2.4x slower**
(4K PV 9.84 -> 23.29 ms). Wide-LMUL FMA on these cores is far more expensive per
MAC than m1, the same property that had already ruled out m2. So the four-pass m1
loop wins on this core and the stride has to be fixed in the data layout instead:
store V as four contiguous planes `[kv][half][pos][32]` so each pass streams
instead of striding. That is a writer-side change in the conv and in the decode
append, with every existing reader either migrated or left on a second plane.

QK, for contrast, is already at 15.4 GB/s because its K plane is stored blocked
`[kv][pos_blk][dim][32]` and each block is 8 KB of contiguous memory. That is the
layout PV needs, and the difference between the two is most of the attention
split: qk 14.0 ms against pv 23.9 ms for the same number of bytes.
