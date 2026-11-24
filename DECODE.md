# Decode: measured state

> Index of conclusions, including every falsified attempt: `FINDINGS.md`.

Numbers below are from `./bench_decode.sh` on the board (16.4 GB RAM, `/dev/shm`
7.9 GB), Q4_0 weights, greedy, `-O1`.

## What the target needs

    decode >= 20 tok/s  ==  50.0 ms/token at any context up to 10K

## Where it is

| item | i8xi4 (Q4_0) | i8xi3k (Q3) |
|---|---|---|
| weight blob | 1265.2 MB | 1001.6 MB |
| GEMM per token | 50.15 ms | 38.90 ms |
| attention (ctx <= 64) | 2.56 ms | 2.56 ms |
| everything else | 4.7-5.2 ms | 4.6-5.1 ms |
| **tok/s** | **17.26** | **20.52** |

The `i8xi3k` column is a short-context measurement of a repack of Q4_0, and its
20.52 does not survive at 10K (12.36 on the real Q3_K file) nor a quality check
(cosine 0.44). See the correction below and `FINDINGS.md` section 7.

GEMM sustains 24.0 GB/s against a measured raw-stream ceiling of 27.5 GB/s, so the
vendor kernel's serial load->use chain is leaving ~13% of the bandwidth on the
table. Everything else in the token costs ~7.3 ms and is flat across the stages.

## Two things this measurement does NOT cover

1. **No real prompt.** `-p` is a single starting token id and the loop sets
   `g_pos = step`, so a run labelled `prompt=10240` has 32 tokens of context, not
   10240. The attention figures above are for short context.
2. **Attention does not scale.** It costs 2.56 ms at ctx 32 and 4.92 ms at ctx 64,
   i.e. 0.074 ms per position. Extrapolated, 10240 positions would be ~755 ms per
   token. The scalar per-position loop has to be replaced with the vectorised
   kernel before a 10K decode number means anything.

## A bug this measurement found

`llama-quantize` leaves `output.weight` at Q6_K by default. `mega_decode` repacked
it as Q4_0 like the other 294 matrices, so 210-byte Q6_K blocks were parsed as
144-byte Q4_0 blocks: the fp16 scales became random bit patterns and the int32
accumulators overflowed. Every logit was `inf` and argmax picked garbage. Both
mega3 and mega_decode had this. Quantize with

    llama-quantize --pure --output-tensor-type q4_0 --token-embedding-type q4_0 \
        MiniCPM5-2B-F16.gguf mc-q4o.gguf Q4_0 8

which puts all 296 tensors at Q4_0, and delete the repack caches afterwards: the
cache is validated by byte size only, and the Q6_K-as-Q4_0 blob has the same size,
so a stale cache loads silently and reproduces the bug.

## Why 3-bit is not the answer on its own

It clears 20 tok/s (20.52) at short context, but it changes the output: greedy
tokens diverge from the Q4_0 run at step 1 with a margin of 0.67.

**Superseded, with the number this time.** On the real `MiniCPM5-2B-Q3KM.gguf`
the 3-bit decode path measures cosine **0.4403** / rms 2.60 against exact fp32 of
that same file, where the Q4_0 path is 0.999002 / 0.108 - and at 10K it runs
12.36 tok/s, not 20.52. The short-context 20.52 was a repack of Q4_0 measured
through a chain with three bugs. So 3-bit is closed on both axes: it never reaches
the speed target at 10K, and it costs 20x the logit error. See `FINDINGS.md`
sections 5 and 7; the weight-domain breakdown (`check_q3loss.py`) shows the packer
is sound and it is the bit count that does it. Q4_0 + int8 KV stays the decode
reference.

## Storage

The model, the repack caches and the binaries were in `/dev/shm`, i.e. RAM: about
3.6 GB of it, wiped on every recovery, and a plausible contributor to the board
dropping off the network. They now live on UFS (`/home/bianbu/workspace/mega`) and
survive reboots. The weights are still *streamed from RAM* at runtime, because
decode needs 24+ GB/s and UFS is an order of magnitude short; they are read from
UFS once at startup into an aligned huge-page blob.

## The number that decides decode: m-row verification cost

Decode at m=1 is entirely the weight stream: 2.28 GMAC of arithmetic against 1.26 GB
of weights, so compute is ~3 ms of a 50 ms token and 47 ms is waiting on DRAM. That
is why no amount of kernel work moves m=1 decode, and why speculative decoding is
the only lever left: if m rows reuse each B tile while it is hot, the stream is paid
once for m tokens.

`probe_mscale.cpp` measures it with the real FFN-gate shape (K=2048) and a weight
plane large enough to defeat cache (134 MB; at the model's own 12.6 MB the matrix is
L2-resident and benchmarks at 51 GB/s, hiding the whole effect):

| m | ms | vs m=1 |
|---|---|---|
| 1 | 3.307 | 1.00 |
| 2 | 3.156 | 0.95 |
| 3 | 4.396 | 1.33 |
| 4 | 5.781 | 1.75 |
| 5 | 7.184 | 2.17 |
| 8 | 11.444 | 3.46 |

Scaled to the measured 50.15 ms decode GEMM, verifying m rows of the target costs
about 50/66/88/109 ms for m = 1/3/4/5. Three drafts are verified for 1.33x the cost
of generating one token, so the economics of drafting are strongly favourable.

Note m=2 coming in *below* m=1. Two kernel calls over the same tile give twice the
outstanding loads, which hides DRAM latency that a single call cannot. Recorded as
observed, not explained.

## What this means for the three routes

- **m=1 kernel work**: capped. Compute is 3 ms of 50; even a perfect GEMM saves 6%.
- **3-bit weights**: measured and closed. 12.36 tok/s at 10K (not 20) and cosine
  0.44 against exact fp32, so it fails both axes. `FINDINGS.md` section 7.
- **dspark speculative decode**: three drafts verified per round at 1.33x the cost
  of one token. Even at a pessimistic 60% acceptance this lands near 27 tok/s, and
  the target is 20. This is the route.

## Multi-row verification is in, and it is exact

`mega_decode` now runs every round as m rows: row 0 is the pending token, rows
1..K are drafts, all sharing one pass over the weights. The GEMM nests rows
innermost inside the n-block walk so the 36 KB B tile is read once and used m
times. `-s K` sets the draft count; K=0 reproduces the old single-token path.

Exactness was checked without any draft model by replaying true greedy output:

    ./mega_decode -n 40 -p 692 MEGA_TOKOUT=/tmp/plain.txt      # 18.00 tok/s
    MEGA_REPLAY=/tmp/plain.txt ./mega_decode -n 40 -p 692 -s 3 # 25.72 tok/s

Acceptance was 3/3 on every round and the two token streams diffed identical. So
anything the verifier accepts is byte-identical to plain greedy by construction --
it only ever emits a draft it has itself reproduced, or its own argmax.

Measured round cost and rate at full acceptance:

| rows | GEMM ms | vs m=1 | tok/s |
|---|---|---|---|
| 1 | 50.5 | 1.00 | 17.90 |
| 2 | 76.9 | 1.52 | 23.02 |
| 3 | 105.4 | 2.09 | 24.33 |
| 4 | 135.7 | 2.69 | 25.78 |
| 5 | 165.8 | 3.28 | 26.29 |
| 6 | 196.9 | 3.90 | 25.88 |

Every extra row costs a flat ~28 ms. That number is the important finding: 2.28
GMAC per row at 95 GMAC/s aggregate. So past m=1 the binding constraint flips from
DRAM bandwidth to IME2 arithmetic throughput -- at m=1 the GEMM runs at 25 GB/s and
only exercises 45 GMAC/s of the engine, but at m=4 the MACs alone need 98 ms.

That has a direct consequence for how to go further. mega3's prefill GEMM uses the
RVV int8 dot kernel at 809.8 GMAC/s, roughly 8x IME2's 95. Verification is exactly
the regime that is MAC-bound rather than stream-bound, so running the rows through
the RVV kernel instead of IME2 should keep an m=4 round near the 50 ms stream floor
instead of 135 ms, i.e. on the order of 60+ tok/s. IME2 stays right for m=1, where
the stream hides everything.

Non-GEMM overhead also grows with rows and is worth trimming later: at m=4 rmsnorm
4.0 + quantA 2.4 + rope 3.2 + misc 2.9 ms is ~13% of the round.

Is it worth noting that at m=1 the whole encode/quantise path (0.58 ms) costs more
than attention (0.59)? Attention at 32 positions is trivially small; at 10K it will
not be, which is why the vectorised attention and int8 KV remain on the list.

## Native m=4: the round stops being MAC-bound

The section above concluded verification was MAC-bound at ~95 GMAC/s aggregate and
pointed at the RVV kernel. That was the wrong place to look first -- the IME2
dispatcher already routes count_m >= 4 to a native four-row kernel, and four rows
through it cost far less than four single-row calls.

`probe_m4.cpp`, same 134 MB plane, both paths against the same weights:

    4x count_m=1 :  5.755 ms    93.3 GMAC/s
    native m=4   :  3.036 ms   176.9 GMAC/s     1.90x
    agreement    :  maxabs 0.000e+00            bit-identical

Not a near-miss: both packings quantise identically (scale = amax/127, sum = -sum(q))
and the results match exactly, so the m4 path cannot drift.

In the real decode the round at K=3 drafts goes from 155.1 ms to 72.5 ms, and the
GEMM portion from 135.7 ms to 54.1 ms -- essentially the m=1 stream floor of 50.5 ms.
Extra rows are nearly free again.

| round | GEMM ms | tok/s |
|---|---|---|
| m=1 (plain) | 50.5 | 17.90 |
| m=4 via 4x m1 | 135.7 | 25.78 |
| m=4 native | 54.1 | 55.17 |

Output still diffs identical to plain greedy, acceptance 3/3 every round.

One detail worth keeping: the fused four-row activation packer was 6.87 ms as plain
C++ (2.8x the RVV single-row quantiser) because it interleaves rows. Written with
RVV it is 1.25 ms -- the work per row is unchanged from quantize_a_row_i8, only the
store offsets differ, so interleaving costs nothing. It requires vlenb=128 (e32m1
gives vl=32, one block) and falls back to the scalar loop otherwise.

Where the round now goes, at 72.5 ms for up to four tokens: GEMM 54.1, then rmsnorm
4.3 + rope 3.3 + logits 3.4 + misc 2.8 + attn 2.0 + quantA 1.3 + silu 1.1. The tail
is ~19 ms and is the next thing to attack, since GEMM is now pinned to the stream.

Against the goal: 55.2 tok/s at perfect acceptance, and at a pessimistic two of three
drafts accepted it is still 3/0.0725 = 41 tok/s. The draft model now has a lot of
room; it does not need to be fast to clear 20.

## dspark: three bugs between "it runs" and "it drafts"

The drafter ran end to end long before it drafted anything. Accepting 0.03 drafts per
round was not a weak model, it was three separate mistakes, and the giveaway was that
zeroing the injected features changed the output by exactly nothing while four different
capture points all scored 2.6%.

**1. Block input was the anchor repeated.** The reference feeds
`[last_token, mask * 6]` (`batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, ...)`),
with `mask_token_id = 75982` from `tokenizer.ggml.mask_token_id`. There is no
`mask_embd` tensor in this variant -- it is created only in the `n_lora_kv > 0` branch.
Without it every slot looked like the anchor and the block was degenerate.

**2. Block attention was made mutual.** `block_uniform = true` means all queries share the
*causal* row of the block's first query, so the visible keys are the prefix plus slot 0.
Letting the seven slots see each other (my first reading of "non-causal") destroyed the
draft.

**3. The drafter left `g_m = 1` behind.** Its markov head is single row, and the target's
final lm_head GEMM did not re-assert the row geometry, so only logits row 0 was written
and rows 1..3 were verified against stale *draft* logits. The target's own output was
diverging from plain greedy within two tokens, which is how this surfaced. The row
geometry is now restored right after the drafter returns, before anything stages A.

Fixing only #3 took first-draft accuracy from 2.6% to 88.3%; all three together give the
numbers below. The rank metric (where the target's choice sits in the draft's
distribution) is what separated "broken" from "weak": it was 57108 of 130560, which is
noise, and is now 2644.

### Measured, 120 tokens from a 40-position context

| config | rows | rounds | accepted/round | tok/s |
|---|---|---|---|---|
| plain greedy | 1 | 120 | -- | 17.31 |
| drafts=2 | 3 | 59 | 1.07 | 13.71 |
| drafts=3 | 4 | 54 | 1.26 | **21.76** |
| drafts=4 | 5 | 51 | 1.39 | 15.98 |
| drafts=5 | 6 | 51 | 1.39 | 14.94 |
| drafts=6 | 7 | 51 | 1.39 | 14.44 |
| drafts=7 | 8 | 51 | 1.39 | 13.89 |

drafts=3 is the only row count that hits the native m4 kernel exactly. m=3 falls back to
three m=1 calls (151 ms/round) and m=5..8 needs two m4 groups (150-172 ms), so the
single-group m=4 round at 104 ms wins despite accepting fewer tokens. First-draft
accuracy is 68.5% and the accepted output is token-identical to plain greedy.

Two capture points were tested for the fused features -- the residual entering a tracked
layer versus leaving it -- and they land within noise of each other (21.76 vs 21.26
tok/s), which is what the `-Dpre`/`-D` switch still exposes.

## Correction: the RoPE pairing was wrong

The 68.5% / 1.26 numbers above were measured under an interleaved RoPE
(`(x[2i], x[2i+1])`, the GPT-NeoX pairing). This checkpoint is llama-architecture
and rotates head halves against each other, `(x[i], x[i+64])`. It survived because
every check in the tree compared the engine against itself, and a wrong-but-
consistent rotation leaves self-comparison invariant.

Once the chain accepted a real prompt from `llama-tokenize`, a comparison against
`llama-simple` on `mc-q4o.gguf` made it visible immediately: after
`The capital of France is` the reference emits `8181 33` (` Paris` `,`) and the
interleaved version emitted `8181 350` (` Paris` `.`). `MEGA_ROPE=inter` keeps the
old behaviour for comparison; half-split is now the default.

The fix also raises drafter quality, since the features the drafter consumes are
target features computed under that same rotation:

| dspark s=3, 21-token prompt, 48 generated | interleaved | half-split |
|---|---|---|
| first-draft accuracy | 68.5% | 77.8% |
| mean rank of target token in draft logits | 2644 | 47 |
| drafts accepted per round | 1.26 | 1.83 |
| decode tok/s | 21.76 | 21.59 |
| plain greedy tok/s, same run | 17.31 | 17.57 |

So the acceptance rate improved by 45% and the speed stayed where it was: the
round is 131 ms regardless of how many drafts it accepts, and the extra acceptance
only pays off against the extra row cost once attention is not the flat part of
the budget. See `CHAIN.md` for the prompt-to-tokens pipeline and the alignment
status.

## Where the 10K round actually spends its time (re-measured)

`-m mc-q4o.gguf -I -X -K8 -n 16 -F /tmp/p10k.txt`, 10K prompt:

| stage | ms/token | share |
|---|---|---|
| gemm (includes the head) | 54.29 | 62.3% |
| attn | 29.59 | 34.0% |
| rmsnorm / silu / rope / misc | 1.80 | 2.1% |
| embed + head argmax | 0.15 | 0.2% |
| **total** | **87.2** | → 11.47 tok/s |

GEMM is the Q4_0 weight stream in 54.29 ms, i.e. 23.3 GB/s against a measured
ceiling just above that, so it is done. That fixes the single-row ceiling: a
perfect attention at the 220 MB int8 K+V byte floor would be ~13 ms, giving
54.3 + 13 + 2 ≈ 69 ms, or 14.5 tok/s.

(An earlier table here said embed 3.26 ms. That was an accounting leftover in an
older binary; per-step timers put the dequant at 0.01 ms, and the current
breakdown above is the one to use.) **Twenty tok/s is not reachable at one
row per weight pass by any amount of kernel work**, which is why the speculator
is not optional.

Attention is 30.78 ms for 220 MB, i.e. 7.1 GB/s, so it is nowhere near its own
byte floor. Section 3 of `FINDINGS.md` has the measurement: the four-row int8
`vmadot` tile (`MEGA_ATTN4_ROWS=4`) costs 41.54 ms at 10K with the three-
instruction exp and 29.34 ms with softmax moved to the X100 cluster
(`MEGA_ATTN4P=1`), and its stage split is qk 0.1 / sm 0.1 / pv 0.1 core-s, i.e.
softmax-led rather than dot-led. It is bit-identical to the 64-row tile on the
row `bench_prefill_attn` checks (worst |diff| 6.889e-04 against scalar fp32), and
the split is bit-identical to the non-split run. So a four-row speculator round is

    54.3 (weights, unchanged: m4 reuses each B column tile) + 29.3 + ~6 = 90 ms

which clears 20 tok/s at 1.84 accepted tokens per round. Short context measured
1.83 with three drafts, so the chain sits on the line and the next lever is
acceptance.

### The four-row round runs at 10K, and it is correct

`MEGA_M4DEC=n` drives `m4_round` in `mega3.cpp`: four rows, one weight pass, the
int8 KV planes prefill already built, and the X100 softmax split. Measured on
`m3c.a4 -m mc-q4o.gguf -F /tmp/p10k.txt -n 0` with `MEGA_ATTN4P=1 MEGA_M4DEC=1
MEGA_M4AX2=1`:

| round | latency | row 0 answer |
|---|---|---|
| 0 (pos 9999) | 106.73 ms | 220 |
| 1 (pos 10003) | 98.94 ms | 220 |
| 2 (pos 10007) | 139.72 ms | 220 |

The single-row chain at position 9999 emits 220 (`step 0 -> 220 (18.4918)`), so
the round reproduces the reference token. Its logit is 17.21 rather than 18.49:
that is int8 attention against the single-row path's f16 KV, and greedy decoding
does not care. Row 0's attention output agrees with the prefill int8 attention for
the same position at cosine 0.999998, which is the check that says the round is
not merely producing a plausible token.

Three bugs had to go, and each one is a trap worth naming:

1. **q, k and v shared one output buffer.** The m4 kernel writes rows at stride
   `ldc = n`, so pointing all three jobs at one buffer let k and v overwrite q's
   first rows and the attention ran on a k that was really v.
2. **The int8 KV planes are per-layer scratch in prefill, not storage.** Prefill
   converts layer L, attends over it, then converts L+1 over the top of the same
   planes. A decode round reads every layer, so it has to keep 42 planes: 268 MB
   at 12K, allocated only when `-MEGA_ATTN4` means the run continues into decode.
   Without this every layer attended over layer 41's K and V.
3. **`g_k8isc`/`g_v8isc` are per-layer globals too.** `pfa_sm_i8` and `pfa_pv_i8`
   dequantise with them, and prefill only keeps them valid for the layer it just
   converted. The round restores this layer's grid from the recorded scales before
   each `pfa_run`. Using layer 41's scales everywhere scaled the attention output
   by about 60x.

Budget against the measurement: 50.6 of GEMM + 29.3 of attention leaves ~19 ms of
tail in a 99 ms round, against the ~6 ms the budget assumed for four rows of
norm, silu, quantise, rope and the KV append. That tail is the next target.
