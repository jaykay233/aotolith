# FINDINGS: what is proven, what is dead, and why

This is the record of every optimisation attempt on the MiniCPM5-2B megakernel for
the SpacemiT K3, in both directions: what moved the number, and what was tried,
measured, and **falsified**. The point of the second half is that a dead end that
is not written down gets re-walked. Each entry carries the number that killed it.

Targets: **prefill >= 200 tok/s and decode >= 20 tok/s at 10K context**, greedy,
standalone megakernel, lossy numerics allowed.

Where the other notes fit: `CHAIN.md` is the prompt-to-tokens pipeline and its
history, `DECODE.md` is the decode breakdown, `DSPARK.md` is the drafter. This
file is the index of conclusions; it does not repeat their derivations.

---

## 导读（中文）

- **10K decode 目前 12.36 tok/s**（真实 Q3_K 权重 + int8 KV），Q4_0 是 11.18。
  投机解码（Q4_0 + dspark，逐 slot markov 权重 0.8/0.25/0.25/0.25）在同一块板上
  是 **17.89 tok/s**：每轮 119.8 ms、接受 1.143 个草案。剩下的差距在 drafter
  结构上——四个 slot 都只看前缀 + slot 0，slot 1-3 无法条件于 slot 0 的结果。
  20 tok/s 不是内核问题，是**每 token 要搬的字节数**问题：Q4_0 权重 1265 MB
  + int8 KV 215 MB = 1480 MB，在 16.5 GB/s 上就是 89.5 ms。要 20 tok/s 必须
  让一次权重读取产出多个 token（dspark 投机解码），或者把权重压到 2-3 bit。
- **prefill 200 tok/s 的差距不是调参**：对齐的 4-bit 厂商内核（K256 hp m4）
  10K 是 139 tok/s（73.4 s），A 核 MAC 地板约 61 s。要 200 必须把 30-40% 的
  MAC 分给 X100，而 X100 现在算的是不同精度（每行一个 scale），接进去 token
  就漂。正确的做法是给 X100 喂 hp 的 A 记录 + 每 K32 fp32 折叠。
- **今天修掉了 3-bit 路径上的三个 bug**（prefill 拿 4-bit 内核吃 3-bit 权重、
  反量化用错类型、embedding 行跨距写死 Q4_0）。修完 3-bit 不再出垃圾，短上下文
  20.94 tok/s，但和 llama.cpp 同一份 Q3_K 文件仍然第 0 个 token 就分叉，
  头号嫌疑是 lm_head 被从 Q6_K 压成 3-bit。
- **别再走的路**见下面 "Falsified" 一节，每条都有实测数字。最贵的三个坑：
  宽 LMUL 的 PV（慢 2.4 倍）、把 174 tok/s 的 vmadot 配置当终点（它不
  对齐）、以及 llama.cpp 的 Q4_0 当基准（它自己离 fp32 有 0.66 cosine）。

---

## 1. The hardware envelope, measured

Everything below is measured on the board, one process at a time, on a quiet
system (see the traps in section 6).

### Streaming bandwidth (`bwprobe.cpp`, 2 GB buffer, three passes)

| threads | GB/s |
|---|---|
| 1 | 6.8 |
| 8 | 17.5 |
| 12 | 19.0 |
| 16 | 19.3 |

Bandwidth saturates at 12 threads; the 16-thread number is 1.6% higher. A plain
contiguous stream with huge pages reaches **25.1 GB/s** on a synthetic buffer and
**23.6 GB/s** on the real `mega-w.bin` at 8 threads, so the 17.5 figure above is
the kernel-shaped access pattern, not the memory system.

### The stride penalty (this is the PV story, section 3)

`bwprobe.cpp MEGA_STRIDE=n` reads one 128-byte block out of every n and reports
only the bytes actually used:

| pattern | useful GB/s |
|---|---|
| contiguous | 17.3 |
| 1 of every 2 | 10.8 |
| 1 of every 4 | 9.0 |
| 1 of every 8 | 7.8 |

A 1-of-4 stride costs **48%** of the achievable rate. Any kernel whose inner loop
is over positions with a fixed dimension slice pays this.

### Compute rate per kernel (one A100 core, 4-row tile)

| kernel | blk_len | GMAC/s |
|---|---|---|
| `gemm_kernel_i8i4_m4` | 32 | 33.3 |
| `gemm_kernel_i8i8_m4` | 32 | 33.0 |
| `gemm_kernel_i8i4_hp_m4` | 256 | **47.7** |
| attention QK (int8 tiled) | - | 44 |
| attention PV (int8 tiled) | - | 42 |
| prefill vmadot int8 (aggregate, 16 cores) | 256 | 809.8 |
| IME2 i8i4 m=1 decode (aggregate) | 32 | 45 |

So IME2 is ~1 GMAC/s/core/MHz on this core and the RVV int8 dot path is an order
of magnitude higher in aggregate. That asymmetry decides the decode/prefill split
in section 2: IME2 is right where the stream hides the arithmetic, RVV is right
where arithmetic is the binding constraint.

`count_m` is **not** a scale factor for these kernels: the dispatcher only tests
`count_m >= 4`, the m4 kernel always computes exactly 4 rows, and wall time is
flat from m=4 to m=128. A benchmark that multiplies GMAC/s by count_m is
measuring nothing.

---

## 2. Decode: the number is a byte count

At 10K with Q4_0 weights and int8 KV, one token streams 1265 MB of weights plus
215 MB of K and V = **1480 MB in 89.5 ms = 16.5 GB/s**, i.e. 85-95% of what the
measured stream gives. So:

    20 tok/s  =  50 ms/token  =  850 MB/token at 17 GB/s

Q4_0 weights alone are 1265 MB. **No kernel work reaches 20 tok/s while the
weights are 4-bit and one token costs one weight pass.** There are exactly two
ways out, and both are now measured:

1. **A coarser weight format.** Q3_K: 1001.6 MB + 215 MB = 1216 MB = ~74 ms =
   13.5 tok/s at 10K (measured 12.36). Q2_K: 738 MB + 215 = 953 MB = ~58 ms =
   17.2 tok/s. Neither clears 20, and both pay quality.
2. **Fewer weight passes per token**, i.e. speculative decode. One round that
   verifies m rows still streams the weights once, so at 1.26 accepted drafts per
   round the byte cost per emitted token halves.

Measured 10K decode, one run at a time:

| config | GEMM ms | attn ms | tok/s |
|---|---|---|---|
| Q4_0, f16 KV | 52.3 | 40.3 | 10.29 |
| Q4_0, int8 KV | 53.8 | 32.9 | **11.18** |
| `-q3` homemade, int8 KV | 40.8 | 32.8 | 12.86 |
| `-q3` homemade, int8 KV, 4096 window | 42.8 | 15.1 | 16.50 |
| `-q2` homemade, int8 KV | 36.0 | 32.8 | 13.90 |
| real Q3_K, 3-bit weights, int8 KV | - | - | **12.36** |

Short context (512-token prompt, real weights) for reference: Q4_0 16.64 tok/s,
real Q3_K 3-bit **20.94 tok/s**. The gap between 20.94 and 12.36 is attention: at
10K the attention half does not shrink with the weight format.

### Why 4096-window is not a legitimate shortcut

It is the fastest decode number in the table (16.50) because it simply attends
over a quarter of the context, so the output is a different model. Recorded as a
windowed speed probe only.

### The speculator is exact by construction, and that is proven

`mega_decode -s K` runs a round as m = K+1 rows in one weight pass. With
`MEGA_REPLAY` feeding true greedy output as the drafts, acceptance is 3/3 on
every round and the token stream diffs identical to plain greedy. So the drafter
can only change the acceptance *rate*; it cannot make the output wrong.

Measured multi-row cost (FNN-gate shape, 134 MB plane, `probe_mscale.cpp`):

| m | ms | vs m=1 |
|---|---|---|
| 1 | 3.307 | 1.00 |
| 2 | 3.156 | 0.95 |
| 3 | 4.396 | 1.33 |
| 4 | 5.781 | 1.75 |
| 8 | 11.444 | 3.46 |

m=2 coming in *below* m=1 is recorded, not explained: two calls over the same
tile double the outstanding loads. The native m4 entry point is what makes
verification cheap - four rows through it cost 1.90x one row and agree
**bit-exactly** with four m=1 calls (both packings quantise identically, so the
m4 path cannot drift).

In the real round, K=3 drafts (m=4) goes from 155.1 ms via 4x m1 to 72.5 ms via
native m4, GEMM 135.7 -> 54.1 ms, essentially the m=1 stream floor.

### The drafter is ported into the four-row chain, and at 10K it loses money

`MEGA_SPEC=n` runs the whole thing end to end on `mega3.cpp`: prefill 10K, then
n rounds of draft 3 / verify 4 rows / accept the matching prefix. There is no
second binary and no replay scaffolding; the accepted prefix advances the same
KV cache the prefill wrote.

**Eight rounds is not enough to measure this.** The first rounds are cold: the
draft's cache has only prefill positions and its own committed tokens are still
few, so early acceptance is a third of steady state. The numbers below are 40
rounds, which is where acceptance stops rising.

Short context (101-token prompt), 3 drafts:

| | accepted/round | tokens/round | ms/round (mean) | tok/s |
|---|---|---|---|---|
| four rows, no drafter | - | 1.00 | 96 | 10.4 |
| dspark, 6 rounds | 0.278 | 1.28 | 106 | 17.3 |
| dspark, 40 rounds | 0.783 | 3.35 | 104 | **32.1** |
| dspark, 10K, 40 rounds | 0.308 | 1.93 | 305 | 6.3 |
| four rows, no drafter, 10K | - | 1.00 | 96 | **10.4** |

So speculation is 3.1x at a hundred tokens and 0.6x at ten thousand. The short
context decode target of 20 tok/s is cleared by 32.1. Both halves of that
disparity are measured, not modelled:

* The drafter itself is 24-26 ms per round at 101 tokens and **190-196 ms at
  10K**. Its attention is fp32, O(context), over 64 (slot, head) units, which is
  the same shape that costs the target 30 ms for four rows. This is the whole
  gap: at 40 rounds the 10K round is 305 ms against 104 ms short-context, and the
  200 ms difference is the drafter's attention.
* Acceptance itself holds up better than the round does: 0.783/round short
  context against 0.308/round at 10K, so the tokens per round only fall from 3.35
  to 1.93. At 96+190 ms that is still under one pass per weight stream.

The drafter costs ~3.4 s of a 79 s 10K prefill (4%), which is acceptable; the
per-round cost and the acceptance are not.

### The drafter's four port bugs, and the controls that found them

Each of these produced a plausible-looking draft, which is why they took a
control to pin down rather than a crash.

| symptom | cause |
|---|---|
| every position's injected K and V identical (\|K\|=26.025 at every pos), feature norms cycling with period 4 | `ds_inject4` copied `g_dshid` rows 0..3 for every group instead of `row0 + r`. The whole 10K prefix was overwritten with four positions' features. |
| occasional NaN in the markov bias, then segfault | the rank-256 gather was replicated to one row and handed to the four-row quantiser, which reads four rows. |
| segfault at round start, garbage token ids in the embedding gather | `int tok[4]` on the caller stack, clobbered by `m4_round`'s inlining -- the same class as the `g_m4out` bug. Both are file scope now. |
| drafter saw an empty prefix | `ds_setup` ran after prefill, so the encoder's five-layer capture had never been armed. It has to be up before the first chunk. |

`m4_alloc()` also had to move: the drafter's GEMMs go through the four-row kernel,
so the A-record arena has to exist before prefill, not at the first decode round.

### Draft-forward variants, scored by acceptance (all at 101 tokens, 3 drafts)

| variant | accepted/round | verdict |
|---|---|---|
| adjacent-pair rope (default) | 0.278 | keep |
| halves-rotate rope (`MEGA_DSROPE=half`) | 0.111 | falsified; it was the wrong pairing on both sides, see section 4 |
| `target_layers` read 1-based (`MEGA_DSL1`) | 0.292 | indistinguishable from 0.250; not the bug |
| encoder/output norms swapped (`MEGA_DSSWAPNORM`) | 0.000 | confirms `enc.output_norm` is the encoder's and `output_norm` is the decoder's |
| interleaved feature layout (`MEGA_DSINTER`) | 0.000 | confirms layer-major; the fused vector is 5 x 2048 in layer order |
| tanh on the rank-256 gather (`MEGA_DSTANH`) | 0.292 | no effect: this variant is the plain markov, as `dspark_joint_proj` being absent implies |
| encoder features zeroed (`MEGA_DSZERO`) | 0.125 | **the control that matters**: zeroing the entire committed prefix only halves acceptance |
| markov bias dropped, short context | 0.000 | 4 rounds only; superseded by the 40-round 10K row below |
| markov bias dropped, 10K, 40 rounds | 0.133 | the base argmax alone is *half* of base + bias (0.308); the markov chain is doing real work |

The zero-feature control is the sharpest number here. If the prefix reached the
draft properly, removing it should cost almost everything. It costs half, which
says the drafts are being driven mostly by the anchor token and the mask block's
own self-attention rather than by the injected context.

### Cosine to the target's own row, and what it settles

Acceptance is a 0/1 summary of a comparison that is really between two whole
distributions. `MEGA_DSCOS` takes that comparison directly: the draft's slot-0
base logit row answers exactly the question the target's anchor row answers (what
follows the anchor), so their cosine is the honest measure, together with where
the target's argmax lands in the draft's ranking.

| | mean cos | target argmax at draft rank 0 | within 8 |
|---|---|---|---|
| 101 tokens, 40 rounds | 0.179 | 11/40 | 28/40 |
| 10K, 40 rounds | 0.353 | 21/40 | 36/40 |

The 10K cosine is *higher* and its rank-0 rate is *higher*, while acceptance per
round is lower. Both are real: a long context makes both distributions flatter
and more alike, and it also makes the target harder to match token for token. So
"the draft is looking at the wrong thing" is not what the cosine says at 10K --
36 of 40 rounds put the right token in the draft's top eight, which is what makes
a block of 3 drafts worth more than 1 token.

The layer-shift sweep used the same metric, because cosine resolves differences
that acceptance buries in noise:

| capture shift (`MEGA_DSSL`) | mean cos |
|---|---|
| 0 (layer L's output, the default) | 0.274 |
| 1 (layer L-1's output) | 0.240 |
| 2 | 0.191 |

The capture site sits after the layer body, so it holds layer L's output; the
metadata's `target_layers` therefore means "after these layers" for this export,
not llama.cpp's `t_layer_inp[il]` (layer il's input). The sweep is what decides
that, since both readings accept drafts at about the same rate.

Also confirmed against the reference rather than by argument:

- `dflash.rope.freq_base = 5000000` and the code already uses 5e6.
- `tokenizer.ggml.mask_token_id = 75982`, which is the hard-coded default.
- The block mask: llama.cpp's `block_uniform` makes every block position reuse
  the first one's mask row, i.e. prefix plus anchor, which is what the four slots
  already do. There is no cross-attention between mask slots to add.

### What is left for decode

- **The 10K decode number is 6.3 tok/s and the drafter's attention is the entire
  gap.** 305 ms per round against 104 ms short-context, and ~195 ms of it is one
  fp32 O(context) pass over 64 (slot, head) units. Acceptance is fine (1.93
  tokens/round); the thing that produces it is too expensive.
- Cutting that attention to the target's own 30 ms would make the round 126 ms
  and the rate 15.3 tok/s. The target is reached by doing both: drafter attention
  to ~30 ms *and* 2.5 tokens/round, which is 20.8 tok/s at 120 ms.
- The drafter's cache is 31.5 MB at 12K, so a window is nearly free in memory.
  That is the one lever that changes the asymptotics instead of the constant.

### The drafter's attention window: `MEGA_DSWIN`, default 512, and it raises acceptance

`ds_attn_band` walked `g_ds_np` positions per (slot, head) unit with a vector
reduction per position, so its cost is linear in the prefix. It now starts at
`max(0, np - W)`. This was expected to be a speed-for-accuracy trade. It is not:
acceptance goes **up**, because a 324M drafter reads a recent window better than
it reads ten thousand positions.

| 2K ctx, 40 rounds | ms/round | accepted/round | tok/s |
|---|---|---|---|
| W = 0 (full prefix) | 135.1 | 0.292 | 13.88 |
| W = 1024 | 113.1 | 0.475 | 21.44 |
| W = 512 | 105.5 | 0.575 | **25.83** |
| W = 256 | 103.1 | 0.558 | 25.95 |
| W = 128 | 101.8 | 0.525 | 25.30 |

| 10K ctx, 40 rounds | ms/round | accepted/round | tok/s |
|---|---|---|---|
| W = 0 (full prefix) | 305.4 | 0.050 | 3.77 |
| W = 1024 | 147.3 | 0.092 | 8.65 |
| W = 512 | 138.0 | 0.092 | 9.24 |
| W = 256 | 135.9 | 0.100 | 9.56 |

**Forty rounds is the wrong length to measure acceptance and it read 5x low.**
The same 10K prompt over **300 rounds** accepts **484/900 drafts, 0.538 per round**,
784 tokens in 41.5 s: **18.88 tok/s at 135.89 ms best / 138.44 ms mean**. The first
forty rounds land in a stretch of the stream where the drafts miss, and acceptance
is a property of the stream, not of the configuration. Every acceptance number in
this section that came from 40 rounds is a lower bound on a bad stretch.

Over 300 rounds, with `MEGA_ATTN4P=1`, sweeping the window on the 10K prompt:
| 10K, 300 rounds | ms/round | accepted / proposed | tok/s |
|---|---|---|---|
| W = 1024 | 143.75 | 0.136 | 9.79 |
| W = 512 | 136.63 | 0.208 | 11.88 |
| W = 256 (**default**) | 133.33 | 0.198 | **11.95** |

| 2K, 300 rounds | ms/round | accepted / proposed | tok/s |
|---|---|---|---|
| W = 256 | 110.30 | 0.281 | 16.71 |

**These numbers fell when the rope was fixed, and the fall is the point.** Before
the fix the 10K prompt's greedy stream collapsed into a `944 242 39` cycle -- the
target repeated one token for tens of rounds -- and a looping stream is trivially
draftable, so acceptance read 0.617 and the rate 21.28 tok/s. With the correct rope
the same prompt produces a normal, non-repeating stream and acceptance is **0.198**,
for **11.95 tok/s at 10K** and 16.71 at 2K over 300 rounds. Every acceptance number
in this section measured before the rope fix was measured on a degenerate stream and
is not evidence of anything.

The window levers still hold their shape: 256 and 512 beat 1024 at both context
lengths, and the round time is still halved against the full prefix (305 ms). What
did not survive is the claim that the decode target was met.

The hard bound is unchanged: plain decode at 10K is 97 ms/token, and 54.3 ms of
that is streaming the weights at 16.5 GB/s. **20 tok/s is 50 ms/token, i.e. below
the weight stream alone**, so speculation is mandatory and what decides it is
acceptance. At W=256, 178/900 proposals were accepted (19.8%), or 178/300 =
0.593 accepted per round. Including the verifier token gives 478/300 = 1.593
emitted per round. At 133.33 ms, 20 tok/s requires 2.667 emitted per round:
1.667 accepted, or 55.6% of the three proposals. Alternatively, keeping current
acceptance requires a mean round time of at most 79.7 ms.

On c2k with the corrected target, 40 rounds per window, W=0 (full prefix)
and W=1024 each accepted 8/120 proposals; W=256 accepted 22/120. Mean
raw-logit cosine was 0.1326, 0.0487 and 0.3028 respectively. This comparison
supports retaining W=256 on this prompt; the target context is unchanged.

On the corrected target, W=256, 60 rounds (180 proposed tokens per run),
keeping the Markov bias accepted 65 drafts on c100 and 36 on c2k. With
`MEGA_DSNOBIAS=1` the counts fell to 39 and 11 respectively. Keep the bias.
These runs cover different numbers of emitted tokens: raw-logit cosines are
averages over different anchor positions, not a paired test of conditioning.

### Sharing the drafter's K stream across its four slots: measured, and it loses

The obvious move is the one that worked for the target: the four draft slots ask
their questions of the same prefix, so walk positions once and fan each loaded K
and V out to four accumulators (`MEGA_DSATTN4`). It cuts the unit count from 64 to
16 and the K/V traffic fourfold. Written, measured, gated off:

| 10K, 40 rounds | ms/round | tok/s |
|---|---|---|
| per-slot band (default) | 305 | 6.31 |
| four slots sharing the stream | 338 | 5.47 |

Short context is the same or a hair worse (104 -> 107 ms), and the token stream is
identical, so it is not a numerics bug: the reordering is exact (0.783 accepted per
round either way, cos 0.1784 against 0.1785).

The reason it cannot pay here is the same reason the target's own attention needed
a layout change rather than a loop change: the inner loop issues one vector
reduction per (position, slot), and it reads K in 32-byte granules across a
512-byte position stride. Cutting the *number of readers* does not reduce the bytes
those readers touch, because each of them already walked the whole line. What it
needs is dim-major tiles so a position block is contiguous and the reduction
happens once over a block of positions instead of once per position.
- Attention at 10K is 32.1 ms of the 89.5 ms token: 9 TMAC of it is QK/PV, and
  the loop stalls on 32-byte granular loads rather than hitting FMA issue.
  Running QK and PV as int8 `vmadot` GEMMs is the fix with precedent in this tree.

---

## 3. Attention: the layout, not the arithmetic

At 10K, QK 14.0 ms and PV 23.9 ms for 215 MB of f16 each, i.e. 15.4 and
**9.0 GB/s**. The PV number is the 1-of-4 stride number to the digit, and the
reason is structural: PV's inner loop is over positions with the head dimension
fixed, so it reads the 128-byte quarter of every 512-byte position stride, four
times over (two kv heads x two halves).

What was done and what it bought:

| change | result |
|---|---|
| INT8 K/V cache | attn 40.3 -> 32.1 ms; decode 10.29 -> 11.18 tok/s |
| PV-order V planes `[kv][half][pos][64]` | 10K PV 23.9 -> 15.1 ms; 4K PV 6.77 -> 5.43 ms; bit-exact |
| QK epilogue rewritten (interleaved scores, one store) | QK 128.7 -> 101.8 core-s at 10K |
| PV restructured (V block outer, row group inner, causal truncation) | PV 131 -> 54 core-s at 10K |
| int8 KV conversion vectorised (16 position-contiguous writes) | 2.36 -> 0.60 s |
| int8 conv split over 16 cores (16 even bands) | 6.41 -> 4.38 s |

The remaining PV cost is a writer-side problem: V has to be stored in the layout
PV reads. That is done for the int8 plane; the f16 plane still serves the
non-int8 path.

### The int8 `vmadot` attention ceiling, measured, and where the 16x is

`MEGA_ATTN4` drives the prefill int8 attention path (`pfa_qk_i8` / `pfa_sm_i8` /
`pfa_pv_i8`, i.e. the same `qk4` dots) after prefill, when the int8 K/V caches
already cover the whole prompt. That is exactly a multi-row verification round,
so it answers whether speculation can pay at 10K. 42 layers, `-X -K`, 64-row
tile (the prefill granularity, `I8_ROWS = I8_G * PFA_ROWS = 16 * 4`):

| rows | ctx | attention | stage split (core-s) qk / sm / pv |
|---|---|---|---|
| 64 | 1024 | 38.6 ms | - |
| 64 | 5120 | 161.6 ms | - |
| 64 | 9995 | 312.1 ms | 1.2 / 1.7 / 0.6 |
| 128 | 9995 | 642.5 ms | - |
| 64 | 9995, single cluster (`-J`) | 332.7 ms | 1.2 / 1.7 / 0.6 |

Doubling the rows doubles the time with the aggregate rate flat, and the stage
split says why: **softmax is the largest of the three** (1.7 of 3.5 core-s, 49%),
QK is 34% and PV 17%, and all three scale with the group count. The K stream is
not the limit at this size -- a 64-row tile does 16x the arithmetic of a four-row
one against the same reads.

That is the whole argument for a four-row tile: at four rows the bytes stay at
215 MB of int8 K+V (the DRAM floor is ~11 ms) while the arithmetic drops 16x.
Measured with `MEGA_ATTN4_ROWS=4`, which drives `task_pfa_i8_4` (one group, head
per thread) and sets `g_i8g = 1`:

| four rows, 9995 positions, 42 layers | attention | exp cost |
|---|---|---|
| 6-term polynomial (default) | 62.25 ms | - |
| `-e 1`, three-instruction exp | 41.54 ms | 6.15% max relative error |
| `-e 2`, table-corrected exp | 105.30 ms | the `vrgather` is trap-emulated |
| `-e 1` + softmax on the X100 cluster (`MEGA_ATTN4P=1`) | **29.34 ms** | - |

Correctness: `bench_prefill_attn` (`-x 512 -A 4 -J`) compares row 3 of head 0
against a scalar fp32 reference and gives worst |diff| 6.889e-04, identical to
what the 64-row tile produces, and `g_pfa_out` matches the 64-row run to the
last bit. The cross-cluster split runs the same `pfa_sm_i8`, so it is checked
against the non-split run: `out sum 281.160936718 sumsq 4101.074513` either way.

The split is worth 28% because softmax is the largest of the three stages at four
rows and the X100 cluster is otherwise idle for the whole decode pass. Its first
implementation gained only 8%: it routes through `pfa_pv_i8`, which stages V into
TCM so sixteen row groups can share one fetch, and at one group that staging is a
pure 0.4 ms-per-layer copy. Gating it on `g_i8g > 1` is what let the overlap
happen.

So the round is `54.3 (weights) + 29.3 (attention) + ~6 (four rows of the 2 ms
tail) ≈ 90 ms`, which clears 20 tok/s at 1.84 accepted tokens per round. The
short-context measurement was 1.83 with three drafts, so the chain sits on the
line and the next lever is acceptance, not the kernels.

### Four rows through the GEMM cost less than one

That 54.3 in the budget was an assumption: four rows sharing one weight pass.
`MEGA_M4GEMM` measures it with the exact matrix sequence of the decode loop --
qkv, o, gate+up, down, 42 layers, four rows -- using the vendor hp kernel prefill
already runs and the compact Q4_0 blob decode already has:

| configuration | GEMM for the whole round | GMAC/s |
|---|---|---|
| one row, `gemm_kernel_i8i4` (decode today) | 54.29 ms | 38 |
| four rows, `gemm_kernel_i8i4_hp_m4` | **50.59 ms** | 153 |

Four rows are *cheaper* than one. The m1 path runs the K32 legacy kernel at
38 GMAC/s; the m4 path runs the K256 hp kernel at 153 GMAC/s, and 1265 MB in
50.59 ms is 25.0 GB/s, i.e. the measured contiguous streaming ceiling. So the
round is at the DRAM floor and there is nothing left to win in the GEMM.

That also settles the round budget:

    50.6 (weights, four rows) + 29.3 (attention, four rows) + ~6 (tail) ≈ 86 ms

which is 20 tok/s at 1.80 accepted tokens per round, or 23 tok/s at 2.0. The
plan is no longer marginal.

`probe_qkload.cpp` is the check that the dots are not the problem. Same operands,
same eight `vmadot`, same 8 KiB of K per call, only the K load changes, all
L1-resident:

| K load | ns/call | GMAC/s |
|---|---|---|
| none (registers held) | 144.4 | 226.9 |
| `vl8re32.v` (what `qk4` does) | 238.1 | 137.6 |
| 8x `vle32.v` | 242.7 | 135.0 |

`vl8re32.v` is not trap-emulated (the strided-gather trap does not extend to
whole-register loads), and the dot loop reaches 137.6 GMAC/s per core with K hot.
In production QK runs at 45 GMAC/s per core, i.e. the GEMM rate, so QK is already
at the machine's dot speed and the headroom is in softmax.

---

## 4. Prefill: the A-cluster floor, and the alignment tax

10K prefill, measured, in the order they were reached:

| configuration | 10K prefill | aligned? |
|---|---|---|
| vmadot, A_BLK 256, per-row int8 | 57.6 s (178 tok/s) | no (`285`) |
| vmadot, A_BLK 128 | 115.0 s (89) | no |
| vmadot, A_BLK 64 + m4 flush | 218.2 s (47) | yes, margin 0.11 |
| vendor `i8i4` K32, Q4_0 exact | 93.8 s (109) | yes, margin 1.08 |
| vendor `i8i4_hp_m4` K256 | 75.8 s (135) | **yes** (`8181 33`) |
| + conv/quant/norm vectorisation | **73.4 s (139.4)** | yes |
| + chunk 1024 | 71.63 s (139.59) | yes, 60/64 vs golden |
| + X100 column offload (pct table) | **64.45 s (155.13)** | **yes, 60/64 vs golden** |
| real Q3_K via int8 copy (vmadot) | 100.4 s (99.6) | unverified |

| `prefill` 3-bit via the int8 copy (vmadot) | 10K 100.4 s, 99.6 tok/s | - | 40 tok/s slower than 4-bit prefill; closed by the quality result in section 7 |

**The 174-178 tok/s figure is the unaligned row.** It is the vmadot path with
per-row int8 activation scaling and A_BLK 256; it emits a different first token.
The fastest configuration that matches the reference is **155.13 tok/s** (64.45 s,
X100 offload, 60/64 against the golden set), and that is the number to compare
against 200.

The floor: GEMM 41 s + QK 20 s + PV 14 s of MAC work at the measured per-core
rates is ~61 s, plus conv, quant, norm, silu and softmax. 200 tok/s needs 51.2 s,
so **the A cluster alone cannot do it**; the X100 split is what closed 139.6 to
155.1, and the remaining 14.5 s has to come from the GEMM kernel itself, since
that is 31.7 s of the 64.45 s.

### The three-instruction exp is the softmax's whole cost, and it is now default

At 10K the softmax slice was 153.9 core-s, i.e. one element per ~6 cycles on a
32-lane thread, which is the exp-and-not-the-loads rate. `-e 1` (`exp2f_e`, the
magic-add plus mantissa estimate, 6.15% maximum relative error, always an
overestimate) takes it to **118.9 core-s** and prefill from 75.5 to **74.15 s
(134.85 tok/s)**. `-e 2` (same, then a table gather on the top three mantissa
bits) does not pay for its gather.

`g_exp3` now defaults to 1. The check that matters is that the 6.15% error is a
uniform overestimate in the same direction for every position in a row, so it
cannot reorder the softmax: ten greedy steps off a 100-token prompt are
`35 242 38 35 242 38 35 242 38 35` with exact exp and with `-e 1`, token for token.

What is left is 118.9 core-s of exp over 8e8 elements, and it does not close a
25 s gap on its own, so the remaining prefill work is the X100 MAC split below.

### A block-buffered single-pass softmax is slower: do not retry

The three passes look wasteful (max pass, exp pass that reads and writes, then a
normalise pass that reads and writes). Copying the row into a 4 KB per-thread
block and doing max/exp/normalise out of the block in f32m4 granules was measured
and is **slower than the three passes**: the row is 40 KB and stays hot between
consecutive passes, so the block copy is a fifth transfer that buys nothing. The
exp is the cost, not the traffic, which is what `-e` addresses.

### The RoPE convention was wrong on both sides, and that hid it

This section used to claim the opposite of what follows, so it is worth stating
plainly: **the megakernel, `ref_f32.py` and `layer0_ref.py` all used rotate_half
(pairs `(i, i+64)`), and llama.cpp uses `LLAMA_ROPE_TYPE_NORM` (pairs `(2i, 2i+1)`)
for `general.architecture = llama`.** The two are not interchangeable.

Why it survived three separate audits:

- All three of our implementations shared the mistake, so the layer-by-layer diff
  (0.9999 on every quantity, 0.9956 at L41 across all 42 layers) was comparing the
  kernel against a reference with the same wrong pairing. A self-consistent wrong
  model still fails against an external one.
- It looked like a quantization effect. The gap to llama.cpp was 0.87 cosine with
  an additive-looking tail lift, which is exactly the shape of an activation
  quantization floor, so it was written down as one.
- Short prompts hid it. Rotation angles are near zero at the first positions, so a
  2-token prompt agreed exactly (`1232` on both sides) and the split only started
  at 3 or more positions.

The isolation that found it: change *only* the reference's pairing, leave the
kernel alone, and re-score both against llama.cpp on the same 10-token prompt.

| reference rope | cosine vs llama.cpp Q4_0 |
|---|---|
| rotate_half `(i, i+64)` | 0.868795 |
| NORM `(2i, 2i+1)` | **0.977566** |

There were three separate rope implementations in `mega3.cpp` -- `prefill_qk_row`,
`rope_all` for single-row decode, and `m4_rope_apply` for the four-row round, each
with its own hand-written vector loop. They are now one `rope_head_norm` with three
callers, because changing only one of them was enough to build a mixed KV cache
whose logits were garbage. The deinterleave runs in registers: `vid ^ 1` swaps each
pair and `vid >> 1` duplicates each angle, so `out = x*C + swap(x)*S` is one gather
and one fma per vector. Strided access is trap-emulated on this core, so a `vlse32`
gather would have been orders of magnitude slower.

### Alignment with llama.cpp greedy, measured rather than argued

Same prompt ids into both, 40 greedy steps, token for token:

| prompt | context | matching steps |
|---|---|---|
| c100 | 100 | **40 / 40** |
| head 256 | 256 | **20 / 20** |
| head 512 | 512 | **20 / 20** |
| head 1024 | 1024 | **20 / 20** |
| head 2048 | 2048 | 4 / 24 (step 0 agrees, cos 0.992491, same top-1 702) |

So alignment is exact out to 1K context and then drifts, and the drift is
quantization noise between two different 4-bit integer pipelines rather than a
layout error: at the divergence point the two logit rows still have cosine 0.9925
and the same argmax. Our activations are int8 with a per-K32 folded scale;
llama.cpp's are Q8_0 per 32 elements inside the Q4_0 dot. Neither is more correct,
they are differently rounded.

Two things this rules out. The three-instruction exp (`-e 1`) is **not** the drift
source: at 2K, `-e 0` and `-e 1` both match 4 of 24 and both first differ at step
4, so the approximation neither helps nor hurts alignment. And the prefill and
decode paths do not disagree with each other any more, now that they share one rope:

- `PF=dc python3 layer0_ref.py mc-q4o.gguf ids.txt` diffs the decode step's layer 0
  against numpy: embedding 1.000000, post-norm 1.000000, V 0.999213, K-after-rope
  0.999999, attention 0.999467, residual 0.999943. Prefill's layer 0 is 0.999151 /
  0.999981 / 0.999531 / 0.999948.
- `ref_f32.py` against the decode step across **all 42 layers**: residual cosine
  0.999958 (L0) to 0.995609 (L41) with no step drop, which is Q4_0 quantisation
  accumulating rather than a broken layer.

### What the previous version of this section got wrong

Recorded so the same reasoning is not repeated: it claimed the megakernel and the
float32 reference agreed at 0.99790 with a matching argmax while llama.cpp was the
0.87 outlier, and explained llama.cpp's tail lift as an additive Q8 floor. The
measurements were real; the conclusion was backwards, because the reference shared
the kernel's rope bug. What actually settled it was an external agreement check --
`refgreedy` on the same ids -- rather than any amount of internal consistency.

### The X100 cluster, what it is good for

- **Conv: yes.** 16 even bands, 6.41 -> 4.38 s, bit-identical output (bands only
  change which thread quantises which position).
- **GEMM: not yet.** `-x100 P` with the `nlim` hole fixed gives GEMM 41.06 ->
  34.88 s and prefill 69.2 s (148 tok/s) at P=30, but the short-prompt chain moves
  off `8181 33` and 10K token 0 moves off with P=45. The X100 kernel applies one
  scale per row over the whole K while the hp kernel folds a scale per K32 block
  into an fp32 accumulator, so the two cannot agree. Using X100 for MACs requires
  giving it the hp A record and the same per-block fold.

---

## 5. Falsified: do not walk these again

Each of these was implemented (or measured) and rejected. The number is the
reason.

### Kernels and tiling

| attempt | result | conclusion |
|---|---|---|
| PV as one m4 pass per kv (LMUL=4) | 4K PV 9.84 -> 23.29 ms, **2.4x slower** | wide-LMUL FMA is expensive per MAC on these cores; same property already ruled out m2. Fixed the layout instead. |
| Bigger row tiles for prefill GEMM (`-M 8/16/32`) | 40.72 / 45.50 / 88.69 s vs 41.06 | at 64 rows the A record is 390 KB, over the 384 KB TCM, and A is re-read from DRAM per column block. |
| `MEGA_BHOT=1` (hold one B block for all columns) | GEMM 41.06 -> 40.60 s | B traffic is **1%** of the GEMM. It is issue-bound, not bandwidth-bound. |
| Fine activation blocks with a memory flush (A_BLK 64/32) | 181 s / 218 s vs 37.6 s at 256 | the accumulator round-trip per block dominates; rewriting the flush as m4 only got 315 -> 218 s. |
| Register-resident accumulators across the k loop | 10K GEMM 69.0 -> 37.6 s, but second 32-column pass disagrees with the legacy kernel | reverted. A_BLK 64 would cost 114 s anyway. Its two hardware findings are in section 6. |
| 32-column register-resident flush | B's second half-column layout differs from legacy | cancelled, reverted. |
| 2-bit for decode (`-q2` homemade) | 13.90 tok/s at 10K; cosine **0.0106** vs fp32 | the packer destroyed the model. See section 7 for why this measurement was also polluted. |
| 3-bit for decode (`-q3` homemade) | 12.86 tok/s; cosine **0.1766** vs fp32 | same. |
| 3-bit decode with the bugs fixed (real Q3_K) | 12.36 tok/s at 10K; cosine **0.4403**, rms 2.60 vs exact fp32 of the same file | the packer is not broken, 3 bits is. 20x Q4_0's logit error for +10% speed. Closed. |
| `-H` (hp one-row decode kernel, matches llama.cpp) | 6.3 tok/s vs 17.3 | 2.7x slower; an alignment probe, not a speed path. |
| Padding one row to four and calling `_hp_m4` | token `30930` | the one-row and four-row entry points quantise A with different record layouts. Not equivalent. |
| `gemm_kernel_i8i3k` with `count_m=4` | rows 1..3 all zero, row 0 corrupt | it is a one-row kernel. `_m4` is the four-row entry point. |
| `quantize_a_4row_i8k` + `gemm_kernel_i8i3k_m4` | garbage at every A pitch 290..300 | unresolved. Do not assume the 4-row 3-bit A record is four consecutive 292 B rows. |
| Per-chunk incremental KV int8 conversion | conv 6.41 -> 2.37 s, but 10K first token 5 -> 11127 | per-chunk scales are numerically different from whole-prefix scales even when the last chunk reconciles them. The fix is per-64-position block scales (sk/sv), not bookkeeping. |
| Weighted 4:1 A:X100 thread split for the elementwise passes | conv fine (4.39 s) but silu 1.52 -> 1.94 s | conv is a stream pass: split it evenly over 16, keep everything else on A. |
| Prompt-lookup speculation | not used | directive: use dspark. |
| Draft counts other than 3 | 2: 13.71, 4: 15.98, 5: 14.94, 6: 14.44, 7: 13.89 tok/s | only m=4 hits the native kernel exactly. m=3 falls to three m1 calls, m>=5 needs two m4 groups. |
| X100 softmax split, second half | 21.76 vs 21.26 tok/s | within noise. |

### References and baselines

| attempt | result | conclusion |
|---|---|---|
| Match llama.cpp's Q4_0 token stream | llama.cpp Q4_0 is **0.6636** cosine / 1.93 rms from exact fp32, ours is 0.9990 / 0.108 | llama.cpp Q4_0 is the wrong target. It is its own quantisation artifact. |
| `refgreedy` on `MiniCPM5-2B-F16.gguf` | `[KPACK] publish roundtrip FAILED`, 511 s, **0 tokens** | F16 cannot be run through llama.cpp on this board. `ref_f32.py` is the baseline of record. |
| `llama-simple` for 10K | silently truncates to its 512-token context | it is only valid for short prompts; 10K needs `refgreedy` with `n_ctx`. |

---

## 6. Traps: each of these cost a measurement or a day

### Harness

0. **A four-row tile in a 64-row kernel measures nothing.** `I8_ROWS = I8_G *
   PFA_ROWS = 64`, and `g_tile_n = (rows / I8_ROWS) * N_HEAD` rounds a 4-row
   request down to zero tiles. The first `MEGA_ATTN4_ROWS=4` run reported 1.2 ms
   for a full 10K attention pass that was not happening. The tell: 512, 2048,
   5120 and 9995 positions all cost the same 1.19 ms. A timing that does not move
   with the input it is supposed to read is not a timing.

0b. **The stage table's embed line reads 3.26 ms and the dequant is 0.01.** The
   per-token summary divides `g_stage[i]` by the token count, and some stages are
   also touched outside the decode loop. `MEGA_STEPS=1` prints per-step deltas:
   `embed 0.01 rmsnorm 0.62 gemm 54.29 attn 29.59` at 10K. An early budget table
   in this tree copied the 3.26 and made a 2% tail look like 3.5%. When a stage's
   number is surprising relative to the bytes it moves, read the per-step line.

1. **`-X` used to eat the next argument.** `-X` did `g_dbg = atoi(argv[++i])`, so
   `-X -K8` consumed `-K8` as `-8` and left `g_k8dec = 0`. Every "int8 KV" run
   with `-X` before this was the f16 binary. Now `-X` only consumes a digit and
   every run prints a `decode path:` banner. **Check the banner before trusting a
   delta.**
2. **`fscanf` on the prompt file returns 1 without writing.** Prompt ids were
   stack garbage and the run SIGSEGV'd. Both `mega3` and `mega_decode` had it.
3. **`ssh` timeout does not kill board processes.** A leftover process held 731%
   CPU and made every subsequent measurement wrong. Clear the board, then measure
   one run at a time.
4. **Never run two megakernels, or a megakernel and `refgreedy`, concurrently.**
   They take the same TCM blocks from `/dev/tcm_sync_mem`: a 244 s prefill and a
   first token of `100312` that evaporated once runs were serialised. Two clean
   back-to-back 10K runs hash the int8 KV identically.
5. **Repack caches are validated by byte size only.** A stale blob with the right
   size loads silently and reproduces an old bug (the Q6_K-as-Q4_0 blob, and the
   `-q2` scale plane). Any packer change must bump the cache suffix.
6. **`-F` prefills `n-1` ids** and keeps the last as the decode anchor, so the
   dumped last row belongs to the second-to-last id. A row diff that ignores this
   makes a perfect layer look like cosine 0.46.
7. **GGUF tensor offsets are relative to the data section**, not the file. Read
   them as file offsets and the f16 block scales become header bytes: every value
   NaN.
8. **`-O2` exits with stack smashing** during load (`-O1` is clean, ASan reports
   nothing, it predates the current changes). Measurement builds use
   `-fno-stack-protector`.
9. **`count_m > 4` is ignored** by the vendor m4 kernels. Benchmarking "GMAC/s at
   count_m=128" measures the same 4 rows.
10. **A stale binary looks exactly like a nondeterministic one.** The `3-bit`
   decode emitted `0`, then `71986`, then `95` for the same prompt and file, and
   each looked like a race. Three of them were different *builds*: `m3c.new` was
   built before the embedding fix and `m3c.dbg` (`-O1`) after. Rebuilding both
   from one source made it `95` three times in a row. When a number moves between
   runs, check the binary's mtime against the source's before suspecting a race.

### Hardware

10. **`vrgather.vv` with `vd == vs2` raises an illegal instruction** on A100. The
    destination must differ even when the source is dead.
11. **`vmadot vD` writes 64 bits past its own register.** With markers in v17..v31
    only v17 changed after `vmadot v16`. Accumulators need even destinations, and
    the odd register above one is unusable across the dot.
12. **`vsetvl_e16mf2` holds only eight lanes on X100** (VLEN=256 vs 128 on A100).
    The 16-byte store in `conv16` has to be one `e16m2 -> e8m1` step; chunking it
    leaves the second half of every block unwritten, and the cache still looks
    plausible - only the generated token moves.
13. **Core assignment:** 0-7 are X100 (RVV 1.0, 2.2 GHz), 8-15 are A100 (IME2,
    1.8 GHz). `/proc/set_ai_thread` selects the cluster; mis-assigning gives the
    A-cluster behaviour on the wrong core count without an error.
14. **TCM staging bug:** 16 accumulator groups were serialised into one, so every
    `-Q` result was the sum of all 16. The file path now forces `-Q` off/on
    explicitly.

---

## 7. Weight formats: the 3-bit story, corrected

The earlier verdict was "the in-tree `-q3` / `-q2` packers produce noise (cosine
0.18 / 0.01) so 3-bit is dead". **That measurement was taken through a chain with
three bugs, so it measured the bugs.** Corrected record:

### The three bugs

1. **Prefill fed a 4-bit kernel a 3-bit blob.** `g_pf_vendor == 2` unconditionally
   called `gemm_kernel_i8i4_hp_m4` and strode B by the 576-byte Q4_0 block. With
   `-q3` every prompt token was computed from misread weights, and that garbage
   went into the KV cache decode then read back. Fixed by `repack_b_i8_q3`, which
   walks the 3-bit blob back to int8 (the `-q2` precedent), and by forcing
   `g_pf_vendor = 0` for 3-bit weights.
2. **Decode's embedding fetch was hard-coded to Q4_0** - an 18-byte-per-32 row
   stride, and whatever type the last repacked tensor left in `g_src_type` (Q6_K
   for this file). Token 0's activation was garbage on every step.
3. **Stale blobs.** Both 3-bit caches got a version bump (`mega-w3-s*-3v2`,
   `mega-w8b-q3v2`) and were rebuilt.

### What is now verified about the 3-bit decode path

- **Elementwise exactness.** Single-nonzero scans over all 2 x 256 x 32 (position,
  row) pairs read back correctly: `0 of 16384` wrong. A permuted dimension index,
  shifted sign bit or mis-strided superblock would show as specific failures.
- **Group-major layout confirmed.** With per-group distinct constants, `C[g*32]`
  equals group g's own value for ng = 1, 2, 4, 8, 32 and nsup = 1, 8, 24, so the
  blob order is `[group][superblock]` and the kernel walks it that way. (The
  probe's first printout looked wrong because the check summed across groups;
  the per-group rows are the correct check.)
- **One-row A record pitch is 292 B** per K256 superblock; 288 and 290 give
  NaN/inf.
- **The reference is now able to judge it.** `ref_f32.py` gained Q2_K/Q3_K/Q4_K/
  Q5_K/Q6_K, and `check_kquants.py` validates each one against the Q4_0 file's
  twin rows of the same checkpoint (correlation 0.970 for the Q3_K tensors, 0.995
  for the Q4_K/Q5_K/Q6_K ones, norm ratios within 0.6%). llama.cpp's own Q3_K
  dequantiser is the other side of that agreement, and it agrees.

### The verdict, with the logit row this time: 3-bit is closed on both axes

64-token prompt, last position, exact float32 of **the same Q3_K file** as judge
(`ref_f32.py` + `RF_LOG`):

| stack | cosine vs fp32 | rms | top-1 |
|---|---|---|---|
| Q4_0 megakernel (control, same prompt) | **0.999002** | 0.108 | `65401` (fp32 `22354`, gap 0.13) |
| 3-bit Q3_K megakernel | **0.4403** | 2.60 | `95` (fp32 rank 27) |
| llama.cpp Q3_K, own numerics | - | - | `1219` (fp32 rank 18, 2.63 logits down) |

Neither implementation of a 3-bit file reproduces the fp32 row: llama.cpp's Q3_K
lands on the fp32 row's rank 18, and the megakernel's lands on rank 27. So
"match llama.cpp Q3_K" was never a quality target either, for the same reason
"match llama.cpp Q4_0" was not.

Why the megakernel is further off is measurable in the weight domain
(`check_q3loss.py`, which applies the packer's own arithmetic to exact-dequantised
rows):

| tensor | source | packer q3 | same scheme at q4 |
|---|---|---|---|
| token_embd | Q3_K | 0.9892 | 0.9979 |
| blk.0.attn_q | Q3_K | 0.9927 | 0.9986 |
| blk.0.ffn_up | Q3_K | 0.9898 | 0.9980 |
| blk.0.attn_v | Q5_K | 0.9835 | 0.9966 |
| output.weight | Q6_K | 0.9816 | 0.9961 |

Per row the packer is not broken - it is doing what 3 bits do. The problem is
that 0.99 per row compounds over 42 layers, while 4 bits at 0.998 does not (the
Q4_0 chain is 0.999 end to end). **3-bit weights cost 20x the logit error of
Q4_0, and deliver 12.36 tok/s at 10K against Q4_0's 11.18.** Not a trade worth
keeping: 3-bit clears neither the speed target nor a quality bar.

Consequences, so they are not re-litigated:

- Do **not** chase the residual 3-bit mismatch by keeping `output.weight` at
  native precision. The head is 0.9816 per row and the body averages 0.988; the
  body dominates, so a better head cannot recover a 0.44 cosine.
- Do **not** re-run "cosine 0.18 so 3-bit is noise" - that number was measured
  through the three bugs above. 0.44 is the number with the bugs fixed.
- The Q4_0 + int8 KV configuration stays the decode reference: 0.999 cosine
  against exact fp32 and 11.18 tok/s at 10K.

### The prefill tax for 3-bit

3-bit has no four-row vendor kernel (`i8i3k` is one-row; `i8i3k_m4`'s A record is
not four consecutive 292 B rows, see section 5), so prefill falls back to the
vmadot path on the int8 copy: 10K prefill **99.6 tok/s** against 139.4 for the
4-bit vendor path. Decode gains 11.18 -> 12.36; prefill loses 40 tok/s. Cracking
`i8i3k_m4`'s A record is the only way to get both - and given the quality result
above, it is no longer worth cracking.

---

## 8. Current standing and the next experiment for each target

| target | now | gap | next concrete step |
|---|---|---|---|
| prefill 200 tok/s @10K | **155.1 (64.45 s)**, verified 60/64 against the golden set | 14.5 s | GEMM is now 31.7 s of 64.45 s and is the only lever left: attention is 22.4 s of which QK's 12.8 s wall is the K plane being walked 8 times per 512-row chunk, and 128-row tiles were measured to gain nothing. The X100 split is at its optimum with the per-projection table; flat shares 20-50% are all worse. |
| decode 20 tok/s @10K | **11.95** over 300 rounds (`MEGA_SPEC=300`, 0.593 accepted drafts per 133.33 ms round); 16.71 at 2K | needs 2.67 emitted tokens/round at 133.33 ms (55.6% proposal acceptance) | Plain decode cannot reach 20 at all (54.3 ms of its 97 ms token is the weight stream), so speculation is mandatory and acceptance is the whole problem. The 21.28 that was recorded here before the rope fix was measured on a degenerate looping stream and is void. |
| output alignment | **exact to 1K context**: 40/40 tokens at 100, 20/20 at 256/512/1024; 2048 first differs at step 4 with cosine 0.9925 and the same argmax | the 2K+ drift is quantization noise, not layout | Do not "fix" the 2048 drift with a rope or exp change: `-e 0` and `-e 1` both first differ at the same step. The three rope implementations are now one function; keep it that way. |

Acceptance test for the rest of the work: `ref_f32.py` on the same prompt, not
llama.cpp Q4_0 (0.66 cosine from fp32) and not llama.cpp F16 (which does not run
on this board). `ref_f32.py` now handles Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, so any file on
the board can be the judge.

### The golden set: how precision is scored from here on

A change is an improvement only against something that is not itself, and the
two candidates in hand (G=512 against G=1024, `-x100` on against off) are both
ours. `refs/` now holds a frozen 10K reference produced once by llama.cpp
(`refgreedy`, 4 threads, 64 steps) with per-step logit rows, and
`check_logits.py` scores a candidate against it. `-force refs/golden_10k_tokens.txt`
pins the candidate to the reference's token chain, because one near-tie otherwise
splits the two chains and every later step compares different prefixes — at 10K
step 12 is exactly that case (three candidates within 0.05 logits).

Standing result, base configuration at 10K: **mean cosine 0.978702, top-1 60/64.**
Of the four misses, two have a margin ratio of 0.0000, meaning our argmax is the
reference's rank 1; step 57 is the one real outlier (cosine 0.9314, max|d| 5.27 on
a 26.0 span). A change that lowers the mean cosine is a regression regardless of
what it does to tok/s.

Two facts this cost a day to establish, and they are cheap to forget:

- **Chunk size is not numerical.** G=512 and G=1024 differ from step 12, and the
  cause is the int8 KV scale: `pfa_convert_kv_i8` takes the max over the prefix
  converted so far, so `sk` moves 0.315% and `sv` 2.93% between the first chunk
  and the rest. Same configuration twice is **bit-identical** (32/32 tokens), so
  this is the scale, not a race. The divergence is a 0.02-logit tie being broken
  two ways, which is why the ratio column exists.
- **Never measure while anything else runs.** Two megakernels together gave
  `12425 61485 50734 ...` where the same binary alone gives `220 5298 242 8647 ...`,
  and a megakernel beside `refgreedy` is just as bad.

### The X100 column offload works: 155.13 tok/s at 10K, 60/64 against the golden set

| configuration | 10K prefill | tok/s | mean cosine | top-1 |
|---|---|---|---|---|
| `-G 1024` base | 71.63 s | 139.59 | 0.978702 | 60/64 |
| `-G 1024 -x100` (pct table) | **64.45 s** | **155.13** | 0.977614 | 60/64 |

Teacher-forced over 64 steps against `refs/golden_10k_logits.bin`. The offload
diverges at the same steps as the base (12, 14, 15, 57) and moves the mean cosine
by 0.0011, which is inside the noise: moving 30-40% of the columns to X100 adds no
new error source. Both of the misses that are genuine ties keep a margin ratio of
0.0000.

**Five defects had to be fixed together, and each one hid the next.** The measured
speed was never wrong; the result was, which is why every one of these figures
sat unusable for a day.

1. The plain int8 copy (`g_blob8`, what `task_x8_pack_b` decodes) is skipped on
   the vendor path (`g_i8w && g_src_type != 2`), because the A100 kernel reads the
   compact i8i4 blob instead. X100 was therefore packing and reading untouched
   `hugemem`. `-x100` now forces the copy to exist.
2. The cached `mega-w8b.bin` was dated July and validated by size alone, so the
   copy that did exist had stale contents. Rebuilding it took 33 s and is part of
   why an earlier "-x100 is still wrong" reading was misleading.
3. `task_pf_gemm_x100` wrote its outputs at `J.c + nb*16`, i.e. columns
   `0..nx-1`, but the cluster owns the **tail** `nlim..n`. It overwrote the block
   the A100 kernel had just finished, corrupting every column of every matrix.
4. One `g_xchunk` counter served every job, and `x100_prepare` clears it once. Job
   0 consumed all the units, so with `nj > 1` every later job found `u >= units`
   and exited: qkv computed q and never k or v, leaving those columns holding the
   previous layer's values. Now one counter per job.
5. `-x100 N` was parsed and then ignored, because `x100_begin` read only
   `g_x100_pct_tab`. Every explicit share in the earlier sweeps (23, 40) was
   therefore running the same table, which is why they differed only by noise.

The kernel itself was never at fault: `MEGA_XCHK` recomputes one 4x16 tile in
scalar code from the same A and B records and matches the assembly 16/16. That
exonerated the A layout, the B pack and the lane map and pointed at the
integration, where all five defects lived.

The X100 path quantises activations with **one scale per row** (`g_a8xs`, four
floats per four-row group) where the A100 hp path uses **one scale per 256
k-elements**. That is a coarser step, not a byte permutation, so it is scored
rather than assumed equal — and it measures equal in practice.

### A per-projection column share beats a flat one, and both beat going higher

`g_x100_pct_tab` is qkv 35, o 33, gate+up 37, down 28. Flat shares, measured:

| share | prefill | tok/s | gate+up | down |
|---|---|---|---|---|
| 20 | 68.82 s | 145.29 | 18.94 | 9.17 |
| 25 | 67.32 s | 148.52 | 18.05 | 8.73 |
| 30 | 65.98 s | 151.54 | 17.23 | 8.77 |
| 35 | 66.35 s | 150.70 | 16.26 | **10.30** |
| 45 | 66.88 s | 149.52 | 15.08 | **12.60** |
| 50 | 69.28 s | 144.33 | 15.80 | **14.06** |
| 55 | 71.49 s | 139.87 | 16.83 | **14.80** |
| **table** | **64.45 s** | **155.13** | — | — |

`gate+up` keeps improving to 45% while `down` turns upward from 35%, so the two
want opposite shares: the table wins by 1.5 s over the best flat value. The
turnaround is at the point where X100 stops being the faster cluster for that
shape, which is what the per-shape rates predict (~198 GMAC/s at k=6144, ~287 at
k=2048, against ~486-509 on the A cluster).

### Falsified: 128-row attention tiles (`-DI8G=32`)

Half the K-plane passes and 5.5 core-s off PV sound like a win; the wall clock
says 71.37 s against 71.63 s, and attaching a step log crashes the run, so the
buffer sizing at `I8_NSLOT = 24*16/I8_G` is wrong as well. Not worth repairing for
0.26 s.

### `-V0` (vmadot) is more accurate and 29 tok/s slower

| configuration | tok/s | mean cosine | top-1 |
|---|---|---|---|
| `-V0`, no x100 | 110.62 | 0.983615 | 8/8 |
| `-x100` (vendor hp) | 155.13 | 0.977614 | 60/64 |

The vmadot path scores higher because both clusters then read the same int8 copy,
but 110.62 against 155.13 closes it as a performance option. It was still useful:
`-V0` with and without `-x100` (0.983615 against 0.303290 at the time) proved the
int8 copy's column correspondence was fine and localised the defects to the X100
path.

### llama.cpp cannot run at 16 threads on this board

`REF_TH=16` produces `[KPACK] publish roundtrip FAILED` and a degenerate chain
(`0 0 5 528 38 0 5 528 38 ...`). At 4 threads it reports no failures and a normal
chain, taking 316 s for 10K prefill plus 64 steps. The golden set therefore has to
come from the 4-thread configuration; a 16-thread reference would have graded
every candidate against garbage.

### Chunk size 1024 is the prefill chunk, and it is worth 2.5 s

`-G 1024` gives 71.63 s against `-G 512`'s 74.13 s (139.59 against 134.88 tok/s)
with the component times otherwise unchanged — the gain is all fixed cost, mostly
`conv` 4.34 -> 2.24 s. `-G 2048` does not work: it runs past 8 minutes with 4.1 GB
resident and never prints a result, so it exceeds a buffer's capacity rather than
being slower.

---

## 9. The probes, and what each one settles

Every claim above that is not a timing comes from one of these. Build them all
with the same flags as the kernel:

    g++ -O2 -fno-stack-protector -march=rv64gcv_zvfh_xsmtvdotii -o <probe> \
        <probe>.cpp ime2.o rvv.o -lm

| probe | settles |
|---|---|
| `bwprobe.cpp` | stream ceiling per thread count; contiguous vs 1-of-2/4/8 stride penalty. Confirms PV's 9.0 GB/s is the stride, not the kernel. |
| `probe_qkload.cpp` | `qk4`'s dot loop in isolation with K hot: 137.6 GMAC/s per core, and `vl8re32.v` costs the same as eight `vle32.v`, so the whole-register load is not trapped. Separates the attention gap into "K stream" and "arithmetic". |
| `probe_mscale.cpp` | cost of m rows through one weight pass (the speculation economics). |
| `probe_m4.cpp` | native m4 vs 4x m1: 1.90x faster and bit-identical. |
| `probe_q3kern.cpp` | `i8i3k` one-row A record pitch = 292 B; 288/290 give NaN/inf. Sign conventions and sparse-row mapping. |
| `probe_q3pos.cpp` | all 2x256x32 (position,row) reads exact -> the 3-bit kernel and blob have no permuted index or shifted sign bit. |
| `probe_q3grp.cpp` | blob order is `[group][superblock]` and the kernel walks it that way, up to 32 groups x 24 superblocks (the real decode shape), worst rel 0.0013. |
| `probe_q3m4.cpp` | `i8i3k` with `count_m=4` returns zeros for rows 1..3: it is a one-row kernel. |
| `probe_q3pf.cpp` | `quantize_a_4row_i8k` + `i8i3k_m4` is **not** four consecutive 292 B rows (garbage at every pitch 290..300). Unresolved; this is the blocker for 3-bit prefill. |
| `probe_q3pack.cpp` | the in-tree 3-bit packer's own layout (used by `repack_b_q3`). |
| `ref_f32.py` | exact float32 logit row via numpy, one layer at a time. The only accepted quality baseline. Handles F32/F16/Q4_0/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K; `RF_LOG=` dumps the full row. |
| `check_kquants.py` | validates each K-quant dequantiser against the Q4_0 file's twin rows of the same checkpoint. Correlation and norm ratio, so a transposed field or a wrong shift is caught by a number rather than by eyeballing. |
| `check_q3loss.py` | the packer's quantisation loss in the weight domain, per tensor, at 3 bits and at 4 bits. Separates "the packer is broken" from "3 bits is lossy". |
| `refgreedy.cpp` | llama.cpp on the *same* file with an explicit `n_ctx`; the token-level cross-check. |
| `layer0_ref.py` | layer 0 of the same GGUF in float64 against `MEGA_DUMP`; isolates a broken op. |

Two of these exist because their first version was wrong and produced a
misleading verdict, which is itself worth remembering: `probe_q3grp` first
summed across groups and printed `WRONG` for a correct walk, and `layer0_ref`'s
first version read GGUF offsets as file offsets.


### The chunk sweep reaches a plateau at 2560, and the elementwise pool is real

`-G 2048` used to hang; that was the buffer bug, not the size, so the sweep could
finally run past 1024 on the current build:

| chunk | 10K prefill | tok/s | conv | gemm | mean cosine |
|---|---|---|---|---|---|
| 1024 | 63.17 s | 158.29 | 2.46 | 32.70 | 0.977614 (60/64) |
| 1280 | 62.24 s | 160.65 | 2.10 | 32.06 | - |
| 1536 | 61.23 s | 163.31 | 2.04 | 32.00 | - |
| 2048 | 60.21 s | 166.07 | 1.55 | 31.25 | - |
| **2560** | **59.49 s** | **168.08** | **1.06** | **31.10** | **0.977893 (60/64)** |
| 3072 | 60.14 s | 166.25 | 1.30 | 31.07 | - |
| 4096 | 59.44 s | 168.21 | 0.96 | 30.59 | 0.977109 (60/64) |

The gain is entirely fixed cost: conv 2.46 -> 1.06 s and the GEMM's per-phase
setup 32.70 -> 31.10 s, while attention stays flat at 18.7-19.2 s. 2560 also
scores *higher* than 1024, because the int8 KV scale is a max over the prefix
processed so far and fewer chunks put it nearer its final value. 4096 is the same
speed as 2560 and scores worse, so 2560 is the default.

### Sixteen bands for rmsnorm and the residual add, and one quantisation per phase

`MEGA_ELN` splits rmsnorm and the residual add across all sixteen cores. An even
split wins, not the 3:1 an A100 core's f32 throughput would suggest, because the
bands are row bands and the X100 cores get through their share:

| 10K, G=2560 | norm | add | prefill |
|---|---|---|---|
| 8 bands | 1.50 s | 1.50 s | 60.2 s |
| 16, weighted 3:1 | 1.24 s | - | 60.69 s |
| 16, even | **0.85 s** | **0.84 s** | **58.9 s** |

silu does not go to sixteen bands (1.50 -> 1.85 s) and still runs on A.

The activation quantiser had its own waste: q/k/v are the same 2048 activation
rows and gate/up likewise, but the loop quantised the phase once per job, i.e.
five times per layer for three distinct buffers. `g_pfa_jslot` makes the jobs of a
phase read one copy, written once:

| 10K, G=2560 | quant | prefill |
|---|---|---|
| per-job copies | 3.10 s | 59.16 s |
| aliased | **2.57 s** | **58.24 s** |

Both changes are numerically exact by construction -- they change who writes a
byte, not what the byte is -- and the 64-step logit dump after them is
byte-identical to the one before (`md5 ac0a0fbd...`), which is the check that says
so. Final standing on the same build:

| | 10K prefill | mean cosine | top-1 |
|---|---|---|---|
| best verified | **57.58 s / 173.67 tok/s** | **0.977893** | **60/64** |

### The 10K budget now adds up, and what is left is elementwise

Full breakdown at 10K, chunk 2560, best verified build (54.59-55.0 s):

| stage | s | what sets it |
|---|---|---|
| GEMM | 30.36 | 19.8 TMAC at the measured 652 GMAC/s aggregate (A 383 + X100 269) |
| attention | 15.85 | QK 58.5 + PV 53.2 core-s over 8 = 587 GMAC/s of vmadot |
| quant | 2.56 | vendor quantiser, 252 M elem/s/core, compute bound |
| silu | 1.52 | ~20 GB/s |
| add | 1.27 | ~32 GB/s, bandwidth bound |
| conv | 1.12 | only 55 M elements, so overhead not data |
| rope | 0.92 | ~8.4 GB/s |
| norm | 0.90 | sixteen even bands |
| packq | 0.34 | quantised Q for attention |
| **total** | **54.84** | |

Two of those are at a hard limit and should be read as fixed: GEMM at the IME2
issue rate plus the X100's column share, and attention at the vmadot rate. Both
were established by the probes above, so the dot work is 46.2 s of the 54.8 --
216 tok/s would be the figure with all elementwise free.

That reframes the target. 200 tok/s is a 50.0 s budget, so elementwise has to fit
in 3.8 s where it now takes 8.6 s.

The activation quantiser looked like the place to find it and is not.
`probe_quant.cpp` measures the vendor `quantize_a_4row_i8_hp` at 1277 M elem/s per
core (K=2048) and 1244 (K=6144) against 121 for a hand-written RVV version, so the
vendor code is ten times faster than the obvious rewrite. An earlier entry here
claimed an order-of-magnitude gap from dividing its element count by its wall time
and getting 252 M/s; that ignored its traffic. 5.16 G elements at 4 B in and 1.1 B
out is 26 GB over 2.56 s, i.e. about 10 GB/s, which is why it is where it is. It
is bandwidth bound already.

The same probe settles the record layout question: the fp16 scales in an hp
subblock are not max|row|/127 (256 of 256 differ), the int8 values differ from
round(x/scale) by up to 127, and the 72-byte trailer's values reach 28046, so it
is not a sum of 32 int8 in i16 either. Reimplementing it bit-exactly is not
available by inspection, and it is not needed.

conv is real work rather than handshake overhead, measured: on sixteen bands 1.12 s
against 1.93 s on the A cluster alone, so the X100 half is doing 0.8 s of it. The
per-element cost is f16->f32 widen, scale, round, clamp, narrow, on a strided read,
not a data-movement anomaly. Everything else is within 2x of its bandwidth.

Also measured and left alone: the X100 column split is already balanced to 0.79 s
of idle across the four GEMM phases (qkv 0.19, o 0.38, gate+up 0.04, down 0.18),
which is the ceiling on what rebalancing can return, and `per-phase pack` costs
0.43 s because the same weights are repacked once per chunk.

### Pre-expanding the rotation table removes the gathers from RoPE

`rope_head_norm` rebuilt its cos and signed-sin vectors from the raw table on
every 32-lane chunk: one `vrgather` to duplicate each angle across the rotate pair,
another to swap the pairs of x, and a `vfmul` by a +-1 sign table. The angle
sequence is fixed per position, so all of it can be built once at startup into 128
cos and 128 signed sin per position, in the order the kernel consumes them:

| 10K, chunk 2560 | prefill | rope |
|---|---|---|
| gather from the raw table | 53.78 s | 0.92 s |
| pre-expanded table | **53.07 s** | **0.78 s** |

Multiplying by +-1.0f is exact, so the 64-step logit dump is byte-identical again
(`md5 ac0a0fbd...`). Verified 53.07 s / 188.4 tok/s, cosine 0.977893, 60/64.

### Fusing the residual add into the norm that reads it is exact

Every layer does `add(residual stream)`, then `rmsnorm` that reads the row straight
back. A row is 8 KB, so the norm's read is a full DRAM pass the add has just made
and left in nothing smaller than L2. Keeping the summed row in a per-thread
buffer makes the second read an L1 hit, and the second call disappears entirely:

| 10K, chunk 2560 | prefill | add + norm |
|---|---|---|
| two passes | 54.92 s | 2.89 s |
| fused | **53.78 s** | **1.95 s** |

The 64-step logit dump is byte-identical to the unfused arm (`md5 ac0a0fbd...`),
because the add is the same `vfadd`, the sum accumulates over `i` in the same order
in double, and the scale keeps `x*rn*w`. Verified: **53.78 s / 185.9 tok/s**,
cosine 0.977893, 60/64.

### The score epilogue was the QK cost, and removing it is exact

`qk4` left its eight int32 accumulators in memory: eight `vse32` into a thread
buffer, then the caller's eight `vle32`, eight `vnsra` and eight `vse16` to write
the int16 scores. The narrowing can read the accumulators where they are.
`qk4n` does `vnsra.wx` and `vse16` straight from v16..v30 inside the asm, so
sixteen vector accesses per row group disappear. Shift, shift-out and destination
address are unchanged, so this is bit-exact by construction and the 64-step logit
dump confirms it: `md5 ac0a0fbd...` before and after.

| 10K, chunk 2560 | prefill | QK core-s | QK GMAC/s/core |
|---|---|---|---|
| memory round trip | 57.53 s | 83.1 | 52.3 |
| fused narrowing store | **54.13 s** | **58.6** | **74.1** |

Verified: **54.59 s / 183.17 tok/s**, mean cosine 0.977893, 60/64. QK at 58.6
core-s against the 57.6 that the loop alone measured, i.e. the epilogue now costs
almost nothing and what remains is the dot loop under the K stream.

Two dead ends measured in the same pass, both worth recording so they are not
retried:

* X100 is not idle during GEMM. It already owns 28-37% of the columns and is
  balanced against the A cluster (37% of 201 GMAC/s against 63% of 338 gives the
  same wall time, and its wait is 0.001 ms). Raising the share makes it worse:
  pct 50 -> 34.20 s of GEMM, pct 65 -> 41.24 s. There is no spare X100 capacity
  to hide the elementwise set in, so the row-pipelined overlap is not available.
* Tile height is already optimal. `-DI8G=8` gives 60.75 s (QK 103.7 core-s) and
  `-DI8G=32` 60.95 s (QK 96.3), against 57.5 s at the default 16.

### GEMM and the attention dots share one port; softmax does not

The layer chain is strictly serial, so the only way to overlap GEMM with attention
is across rows. That pays only if the IME2 path (Q4_0 GEMM) and the vmadot path
(QK/PV) are different execution ports, so `probe_coinissue.cpp` runs both in one
thread against identical buffers:

| K=2048 | ime2 | vmadot | sum | max | both |
|---|---|---|---|---|---|
| 1 thread | 0.169 s | 0.097 s | 0.265 | 0.169 | **0.264** (2% overlap) |
| 8 threads | 0.049 | 0.050 | 0.099 | 0.050 | 0.089 (21%, at the noise floor) |
| 1 thread, N=6144 | 0.046 | 0.026 | 0.072 | 0.046 | **0.072** (0%) |

`both` is the sum, so the two dot paths contend for one issue port. Row-splitting
GEMM against QK/PV is therefore not a lever, and the comment in mega3.cpp that
calls them "the same dot units" is right. The 47.9-vs-74.6 GMAC/s gap between them
is the operand width (i8i4 against i8i8) and the K per call, not a second port.

Softmax is a different story and worth stating separately: it runs on X100 as
plain RVV f32, so it never competes with a GEMM on the A cluster. It still cannot
fill the GEMM window, for two reasons. It depends on QK, and QK is the contending
stage above. And it is not the critical path even inside attention:

| 10K attention | A cluster | X100 |
|---|---|---|
| QK | 82-87 core-s | - |
| PV | 54-57 core-s | - |
| softmax | - | 104-111 core-s (max 54, exp 56, other 42) |
| wall per cluster | 17.0 s | 13.0 s |

So A is the critical path and hiding softmax buys nothing. What is left to hide is
the elementwise set -- quant 2.57, silu 1.52, norm 0.86, conv 1.06, add 0.84, i.e.
about 6 s of RVV f32 that uses no dot unit at all -- because X100 is idle for the
whole 30 s GEMM phase.

### QK at 10K is 74.6 GMAC/s per core once both ends are removed

The same three-way isolation on the current build, chunk 2560, 10K:

| configuration | prefill | QK core-s | QK GMAC/s/core |
|---|---|---|---|
| base | 57.42 s | 82.0 | 52.3 |
| `MEGA_NOSM` (X100 skips the score reads) | 53.77 s | 73.3 | 58.6 |
| + `MEGA_QKNOST` (one score store in eight) | 46.30 s | **57.6** | **74.6** |

So QK's 82 core-s is 57.6 of dot loop, ~15.7 of score stores, ~8.7 of contention
with the consumer's reads. 74.6 GMAC/s is the production loop ceiling under the K
stream -- the 131.7 in the probe was a K plane resident in L2, which the real
prefix is not. `-D` alone only recovered 2.4 core-s because it parks the stores in
one L1 block while both the K stream and the X100 reader are still there.

### The GEMM kernel is at its engine rate, measured with the stream removed

`probe_hp_peak.cpp` feeds `gemm_kernel_i8i4_hp_m4` the same A record prefill builds
and varies only the B working set, so the number it prints is the IME2 engine's own
issue rate rather than the memory behind it:

| K | N | B size | G | threads | GMAC/s |
|---|---|---|---|---|---|
| 2048 | 32 | 36 KB (L1) | 1 | 1 | 47.0 |
| 2048 | 1024 | 1.2 MB (L2) | 1 | 1 | 47.8 |
| 2048 | 6144 | 7 MB (DRAM) | 1 | 1 | **47.9** |
| 2048 | 6144 | | 4 | 1 | 42.3 |
| 2048 | 6144 | | 1 | 8 | **338.5** |
| 2048 | 6144 | | 4 | 8 | 320.9 |
| 6144 | 6144 | | 1 | 8 | **226.9** |

47.9 GMAC/s at N=32 where B is 36 KB and L1-resident, 47.8 at N=1024, 47.9 at
N=6144: the kernel does not care how big B is. That is the engine, and the
production number matches it — 34.4 s for 19.8 TMAC of 10K prefill GEMM is
576 GMAC/s over both clusters against a measured ceiling of 338.5 (A) + ~260
(X100) = 600, i.e. **96% of what the hardware issues**. There is no kernel
rewrite inside this budget; the only way GEMM gets faster is a format with
fewer MACs, and 3-bit already measured 100.4 s on the prefill path.

The vmadot comparison in the same probe explains why `-V0` loses in production
even though its hot rate is higher: 153.5 GMAC/s per core at N=64 (B in L1),
87.1 at N=6144 with eight row groups reusing each column block, 362.2 across
eight threads, and 141.7 at K=6144. Its B copy is 128 KB per 64 columns against
Q4_0's 73.7 KB, 1.74x the bytes, so at the production shape the extra density is
paid back in traffic.

### Where the 10K time actually goes: QK streams K, and it is 19x off the dot rate

| configuration | 10K prefill | tok/s | attn | qk (core-s) | pv (core-s) |
|---|---|---|---|---|---|
| `-x100` (the standing) | 66.61 s | 150.12 | 23.15 | 103.2 | 58.7 |
| `-D` (score stores parked in one L1 block) | 64.49 s | 155.05 | 21.05 | 96.7 | 55.1 |
| `-0` (PV skipped) | **61.62 s** | 162.28 | 18.53 | 74.4 | 2.7 |
| `-h0` (legal, head-fast tiles) | 66.41 s | 150.56 | 23.32 | 105.7 | 59.5 |

Two things follow. PV's wall is only **4.6 s** (66.61 - 61.62 + the QK relief it
gives), and the score buffer's write traffic is **1.9 s** — both already small.
QK plus softmax is 18.5 s and is nearly all of attention.

The contraction is **4.30 TMAC** for QK and the same for PV at 10K
(`sum_p (p+1) x 16 heads x 128 dims x 42 layers`, counted against the real
1024-row chunking). Against the measured core-seconds that is **41.4 GMAC/s per
core for QK** and 72.6 for PV; attention as a whole is 8.60 TMAC in 185 core-s,
i.e. 46.4 GMAC/s per core.

The same `qk4` loop reaches **131.7 GMAC/s per core** in `probe_qk_real` with the
K plane hot and nothing else running, so production QK runs at 3.1x off its own
dot rate. Two of the three candidates are already excluded: parking the score
stores in L1 (`-D`) recovers 1.9 s of 23, and halving the K pass count with
128-row tiles (`-DI8G=32`) recovered 0.26 s, so neither the stores nor the K
traffic per pass is the limit.

What is left is contention, and `-0` measures it: skipping PV entirely drops QK's
own core-seconds from 103.8 to **74.4** for identical work, i.e. PV's V stream
makes QK 1.4x more expensive while both run. Even with PV gone QK is at
57.8 GMAC/s per core, 2.3x off hot, so contention is part of it and not all of it.

### Elementwise cannot hide behind GEMM either: IME2 and plain RVV share the issue port

The earlier port probe (`probe_coinissue.cpp`) settled GEMM against the attention
dots: IME2 and `vmadot` land on the same unit, so splitting rows to overlap them
returns 1-2%. That left a different bet open. Elementwise is not a dot op -- it is
`vle32`, `vfadd`, `vfmul`, `vse32` -- and if those retired on a separate pipe then
the chunk could be split by rows and norm/quant/silu/conv for rows i+1 would run
under the GEMM for rows i. That would take seconds off the critical path, so
`probe_coinissue2.cpp` measures it directly: one thread, Z IME2 calls, Z elementwise
passes, and Z of each interleaved one for one.

| working set | ime2 | rvv | serial sum | ideal overlap | both |
|---|---|---|---|---|---|
| 32 KB (L1) | 0.1091 s | 0.0061 s | 0.1151 s | 0.1091 s | **0.1161 s** |
| 128 KB (L1) | 0.0244 s rvv at 322 GB/s | | 0.1333 s | 0.1089 s | **0.1338 s** |
| 1 MB (L2/DRAM) | 0.1123 s | 0.4765 s | 0.5888 s | 0.4765 s | 0.1050 s worse |

With the working set inside L1, `both` equals the serial sum to within 1%. Ordinary
vector instructions contend for the same issue slots as IME2. The large-working-set
row is worse than serial and is a separate effect: interleaving two streams that do
not fit in L1 evicts each other, which is why the 1 MB case is 2x the sum. So the
8.28 s of elementwise rounds off the GEMM time and cannot be scheduled under it.
This closes the last structural idea for the prefill target.

### The quantiser is at 79% of the bus, not 10 GB/s

An earlier entry derived ~10 GB/s for the quantiser from dividing 26 GB by 2.56 s
across four stages. Instrumenting each stage separately (`g_qa_t`, `g_qa_bytes`)
gives the qkv phase **1.75 s at 15.3 GB/s**, against the board's 19.3 GB/s. The
vendor kernel is at 79% of the achievable bus on the phase that dominates quant, so
the "five times slower than its hot rate" reading was the probe's 32 KB array
staying in L1, not headroom in production.

### Staging the V convert block changes nothing, and is reverted

`pfa_convert_kv_i8` reads V through `conv16` with a 256-element stride, so each
16-position block issues sixteen 2-byte loads, one per cache line. Staging the 4 KB
block into a thread-local buffer first looks like it should recover most of the
1.04 s. Measured: **1.04 s -> 1.02 s**, noise. `conv` is bound by the max scan in
`task_pfa_conv_max`, not by the convert's loads. The staged version also scored
cosine 0.529916 and 15/64 top-1, so it was wrong as well as useless, and is
reverted; the standing build is restored.

### Where the prefill target stands

| | s |
|---|---|
| GEMM | 29.17 |
| attention | 15.35 |
| quant | 2.60 |
| misc | 2.62 |
| add | 1.83 |
| silu | 1.51 |
| conv | 1.02 |
| rope | 0.79 |
| packq | 0.34 |
| norm | 0.02 |
| **total** | **52.77** |

200 tok/s is a 50.0 s budget. The dot work is 44.5 s and every elementwise stage is
within 1.3x of its own bandwidth; the three ways to remove the remaining 2.8 s --
overlapping elementwise under GEMM, a faster quantiser, a faster KV convert -- are
each measured and closed above. 188.4 tok/s is where this configuration lands.

### Fusing silu into the quantiser: exact, and worth 0.26 s

The FFN product ran as two passes: `silu_mul_rows` wrote `h = silu(gate)*up` to
DRAM (10.6 GB), then `pf_quant` read it back (10.6 GB, twice for the max scan and
the quantise). Keeping the product in a thread-local row and handing that to the
same vendor kernel removes both DRAM passes.

What makes it legal was measured first, with two probes:

- `MEGA_QPROBE`: the vendor hp record for one K256 superblock is byte-identical
  whether or not the data past 256 is zeroed. The quantiser is superblock-local.
- `probe_quant3`: calling it over the full width and over 1024-wide slices writes
  the same 27840 bytes for K=6144, so the record does not depend on the call shape.

The staging buffer is 4 rows x N_FF floats = 96 KB per thread. An A core's L2 is
4 MB (from `cache/index2/size`), so eight threads hold 768 KB and every group's
copy stays resident.

The first attempt scored cosine 0.043751. The cause was not the quantiser: the
record was byte-identical to the two-pass one (`sqcmp: 0 of 27840 bytes differ`).
It was the X100 cluster. `pf_quant` builds a second A copy for the tail columns
those cores own, behind `if (g_x8_want && j == 0)`, and the fused path bypassed
that hook, so the cluster multiplied against the previous phase's activations.
Rebuilding the X100 copy inside the fused band from the same staged rows fixed it,
and the logit dump returned to `ac0a0fbd...` with cosine 0.977893 and 60/64.

Measured, same binary, `-SQ0` selecting the two-pass path:

| | two passes | fused |
|---|---|---|
| prefill | 52.73 s | 52.47 s |
| silu | 1.51 s | 1.30 s (part of fused) |
| gate quantise | 1.93 s | 1.84 s (part of fused) |
| **silu + quantise** | **3.44 s** | **3.15 s** |

Verified build: 53.06 s, `md5 ac0a0fbd89d13ad4757be8b17b8189df`, cosine 0.977893,
60/64. The gain is 0.29 s on that pair, which is inside this board's +/-0.5 s run
to run spread, so it is recorded as a small win rather than a step change.

### Rewriting the X100 A-copy store buys 2%, so that path is closed

The fused band's X100 half writes each 64-byte step block with eight 8-byte stores
and two vsetvl round-trips, which looked like the reason the fused pass was not
faster. `probe_x8store.cpp` rewrites it three ways and compares bytes and speed:

- per row, 8-byte stores (what the code does) -- 516.6 M elem/s
- four rows per step, one `vrgather` and one 64-byte store -- `vrgather.vv` on
  SEW=8 measured 100 M elem/s here, five times slower, so it was dropped
- four rows per step, `vslideup.vi`/`vslidedown.vi` transposing into two 32-byte
  stores, byte-exact against the layout spec -- **528.8 M elem/s, +2%**

So the routine is arithmetic bound at about 515 M elem/s per core, not store bound,
and the whole X100 half of the fused band measures 0.07 s wall. There is nothing
there to recover.

Two things the probe settled for later work: the A cluster's VLEN is 256 (A-core
`e8m1=32`) while `cpu8`, the first X100 core, reports `e8m1=128`, i.e. VLEN 1024,
so any index table for that cluster has to be built from its own VLMAX; and this
toolchain's `riscv_vector.h` has no `vcreate`, so assembling a fractional-LMUL tile
has to go through assembly. Also `vslideup`/`vslidedown` need `.vi` for an immediate;
`.vx` is rejected.

### The RMSNorm scale vectorises for 0.61 s, and the sum does not

Both RMSNorm sites (`atr_band` and `rn_rows_band`) did two scalar loops per row: a
dependent `double` chain of N_EMBD adds for the sum of squares, then a scalar
`dst[i] = t[i] * rn * w[i]`. The second loop is element-wise, and the two
multiplies are the same operations in the same order as the scalar version, so
vectorising it is exact: verified `md5 ac0a0fbd...`, cosine 0.977893, 60/64.

The sum is a different question, and the answer is no. It is serial within the
row, so no number of bands shortens it. `-FSUM` offers a vector-double form
(`vfwcvt` to f64, `vfmadd`, one `vfredusum`) and it measured **slower**:

| two interleaved pairs | scalar scale | vector scale |
|---|---|---|
| prefill | 52.845 / 53.127 s | 52.393 / 52.355 s |
| mean | 52.986 s | **52.374 s** |

f64m2 is eight lanes on this VLEN, so the accumulate is eight elements per
instruction and the `vfredusum` costs more than it saves. It stays scalar, which
also keeps the sum bit-identical to what the reference used.

The vectorised scale is worth 0.61 s, and unlike the fusion above it is outside
the run-to-run spread: both vector runs are below both scalar runs, in order.

### The `conv` cost is 42 layers of real work, not a broken access pattern

`conv` is 1.02-1.17 s and only moves about 1 GB, so it looked like a bad access
shape. `MEGA_TIME=1` prints the two halves per layer:

```
  conv L0: max 0.001 s, quant 0.004 s
  conv L1..L40: max 0.001 s, quant 0.002 s
```

Per layer it is milliseconds, so the total is 42 layers x 4 chunks of genuine
f16 max-scan plus convert. The earlier idea of converting only new positions is
already recorded as a numerics change: it rescales the prefix when a new maximum
appears, which moves greedy output (11127 against 5 at 10K), so the full rewrite
is what keeps the dump bit-exact.
### Residual add + rmsnorm + quantisation as one pass: 1.48 s

`add_then_rmsnorm` kept the summed row in a thread-local buffer so the norm did
not have to read the residual stream back from DRAM. The norm's own **output**
was still written out and read straight back, because `pf_quant` runs on it
immediately and nothing else ever reads it. That is 8 KB per row written and
8 KB read again, 9999 rows, twice per layer (FFN norm and next layer's attention
norm).

The three are now one pass over each four-row group. The summed row goes into a
staged buffer (`4 x 2048` floats, 32 KB, L2-resident like the silu fusion's
96 KB), the norm scales it in place, and both quantisers read it from there. What
leaves the core is the record, not the f32 activations.

Bit-exactness is structural, not lucky: the add is the same `vfadd`, the sum of
squares accumulates over `i` in the same double, the scale is the same
`x*rn*w` association, and the quantisers see the same f32 values in the same
`[4][N_EMBD]` order. The running `vfmax` for the X100 copy is order-free and
matches the separate `vfredmax` reduce it replaces.

| two interleaved pairs | three passes | fused |
|---|---|---|
| prefill | 53.096 / 53.709 s | 52.087 / 51.755 s |
| mean | 53.40 s | **51.92 s** |

Worth 1.48 s, both rounds in the same direction. `MEGA_ATRQ=0` restores the
three-pass path. Verified on the fused run: logit md5
`ac0a0fbd89d13ad4757be8b17b8189df`, cosine 0.977893, top-1 60/64.

#### The X100 A copy has to be built in one register's worth of elements

The first attempt scored cosine 0.62 despite the A100 records being byte-exact.
The `MEGA_ATRQ=2` bisect (fuse the A100 record, leave the X100 copy on the old
path) restored the golden md5 and left all twelve dump files identical, which
put the fault inside `x8_quant_stage`. Dumping its output showed every four-row
group at index 320 and above with the **second** 32-byte half of each 64-byte
block zero, and the first 320 groups correct.

The cause is the loop it inherited from `x100_quant_a`: sixteen elements per step
through an `e8mf4` store, which is eight lanes on this core. Half the source
elements therefore had no lane to land in. The rewrite steps eight elements at a
time, so every load, conversion and store is a single register:

```
(x[i/16])*64 + r*8 + ((i%16) >= 8)*32 + (i%8)
```

The record layout is unchanged, and the copy now matches the pre-existing
`x100_quant_a` byte for byte.
### The column split between the clusters is a precision dial: 51.38 s -> 50.00 s

Sweeping `-x100` over 24..48 percent and reading the per-shape `gemm split` takes
the prefill from 194.6 to 200.0 tok/s. The optimum is not one number, because the
shapes do not share it:

| share given to X100 | qkv | o | gate+up | down | prefill |
|---|---|---|---|---|---|
| 24 percent | 3.70 | 3.54 | 16.13 | 8.46 | 54.97 s |
| 27 | 3.44 | 3.54 | 15.70 | 8.30 | 54.03 s |
| 30 | 3.37 | 3.41 | 15.12 | 8.12 | 52.75 s |
| 42 | 3.07 | 3.16 | 12.84 | 11.22 | 53.84 s |
| 48 | 2.92 | 3.13 | 12.82 | 12.66 | 54.74 s |
| table `{48,48,45,28}` | 3.05 | 2.97 | 12.47 | 8.40 | **50.00 s** |

`down_proj` is the odd one out: it is the only k=6144 shape, and above about 30
percent it goes from 8.1 s to 11.2 s, which costs more than the other three gain.
The three smaller-k shapes keep improving up to nearly half.

**It is not free, and the cost is understood.** An A100 record carries one scale
per K256 superblock; the copy the X100 cluster reads carries one scale for the
whole row (`rowsc = rowmax/127`). Every column moved to X100 is therefore a dot
product over more coarsely quantised activations. The split decides how many
columns are computed that way, which is why it moves the output at all:

| split | prefill | logit md5 | cosine | top-1 |
|---|---|---|---|---|
| `{35,33,37,28}` | 51.377 s | `ac0a0fbd...` | 0.977893 | 60/64 |
| `{48,48,45,28}` | 50.001 s | `46ee3c9b...` | 0.977316 | 60/64 |

The two runs differ in 100 percent of logit elements, 0.12 mean absolute
(1.38 max against a peak of 28). That is a real numerical difference, not
last-bit noise, and it is entirely the activation-quantisation granularity: both
clusters accumulate the same integer products, they just start from different
int8 activations. Top-1 agreement with the f32 reference is unchanged.

Accepted deliberately. Two runs of the committed source: 50.001 s (199.98
tok/s) and 49.202 s (203.22 tok/s), both reproducing `46ee3c9b...` and cosine
0.977316 exactly, so the spread is the board and not the code. The bit-exact
split is `{35,33,37,28}`, which is what `ac0a0fbd89d13ad4757be8b17b8189df` was
recorded against.

Recovering the accuracy means giving the X100 copy per-K256 scales, and that is
not a local change. Building them is cheap -- `x8_quant_stage` costs well under
0.1 s and already has the row max in hand -- but consuming them is not:
`x100_gemm_4x16` accumulates the *whole* K in int32 in one call (`nsteps = k/16`,
384 steps at k=6144) and the epilogue applies a single per-row scale to that
finished accumulator. Per-superblock scales therefore mean either 24 kernel calls
per tile with an f32 accumulation between them, or rescaling 128 int32
accumulators in place, and both put 24x the current epilogue work on the hot
path. The scale is not free where it is actually read.

So the split stays the dial it is: `{48,48,45,28}` for 200+ tok/s with cosine
0.977316, `{35,33,37,28}` for 194.6 tok/s and a bit-exact md5.
### The int8 KV decode cache never stored the second half of V

Turning on `-K8` scored cosine 0.4524 with 9 of 64 top-1. That is the same
neighbourhood as the 3-bit weight result (`0.4403`), which had already been
written off as "3 bits is broken". It was not the same cause.

The V plane is `[kv][half][pos][64]`, sixty-four contiguous int8 per half, and
`v8_off(kv, half, pos)` is the mapping. The per-token append path built its
address by hand:

```
kv * 2 * MAXCTX + pos, then * 64, then + d % 64
```

With `HEAD_DIM = 128` the loop takes `d = 0, 32, 64, 96`, so `d % 64` is
`0, 32, 0, 32` -- and there is no `half` term at all. The second 64 dims of each
head were written over the first 64, and half 1 was never written, so the reader
(which does include `half * MAXCTX * 64`) read zeroes for half of every V vector.
The prefill path was already correct because it calls `v8_off`; only the decode
append was wrong, which is why the error showed up in every decode step and not
in prefill.

Routing the append through `v8_off` too:

| `-K8`, 64 tokens after 10K prefill | cosine | top-1 | decode |
|---|---|---|---|
| before | 0.4524 | 9/64 | 10.76 tok/s |
| after | **0.9121** | **55/64** | 10.70 tok/s |

The f16 path is untouched and still reproduces `46ee3c9b...` at 0.977316, 60/64.

What is left is genuine quantisation error, not a bug: step 0 is already 0.9417
and it stays in that band, so it does not compound. The cause is the scale:
`g_v8dsc[layer][kv]` is one absmax over **every position of the prefix times all
128 dims**, i.e. 1.28M values sharing one grid, so one outlier position sets the
step for the whole head. Per-position scales would fix it and cost 64 KB per
layer of side array; the same reasoning applies to K. Until then int8 KV buys
10.01 -> 10.70 tok/s for cosine 0.9773 -> 0.9121, which is not a trade worth
taking by default, so `-K8` stays off.
### Decode attention is at 12 GB/s and prefetching it ahead does not help

At 10K the f16 decode attention takes 43.8 ms and streams about 528 MB of K and V,
which is 12 GB/s against the probe's 27.5 GB/s ceiling. Two things were checked
before touching the loop:

- **The score reads are not the problem.** PV loads eight softmax weights per
  position, one from each head, 10.5 KB apart -- eight pages per step, which looks
  like the obvious suspect. `MEGA_PVPROBE=1` removes those loads entirely:
  attn 43.8 -> 42.8 ms. So the scores are L2-hot as designed and the V stream is
  what stalls.
- **Explicit prefetch makes it worse.** The V walk reads 128 B of every 512 B
  block, a stride the hardware prefetcher does not follow, so
  `__builtin_prefetch` (which does emit `prefetch.r` under `zicbop`) was added 12
  positions, i.e. 6 KB, ahead. Two interleaved pairs:

| | attn | decode |
|---|---|---|
| prefetch off | 45.5 / 45.3 ms | 9.83 / 9.87 tok/s |
| prefetch on | 48.5 / 48.4 ms | 9.59 / 9.60 tok/s |

Consistently 3 ms worse, both rounds. The prefetch instructions cost issue slots
and the stream is already partly covered, so this is reverted. Note the run-to-run
offset from the 43.8 ms figure above: absolute numbers move between binaries, which
is why every one of these is an interleaved pair.

So the remaining decode gap is not a prefetch-distance problem, and the two levers
that would actually move the bytes -- int8 KV (needs per-position scales first, see
above) and coarser weights (breaks accuracy, cosine 0.44) -- are both blocked on
numerics rather than on kernel work.
### int8 KV costs 0.04 cosine and per-position scales are not where it goes

With the half-addressing bug fixed, int8 KV sits at cosine 0.9121 and step 0 is
already 0.9394 against the f16 path's 0.9821. It is not compounding error; the
quantised prefix is simply read less accurately from the first token.

The obvious suspect was the scale granularity: `g_v8dsc[layer][kv]` is one absmax
over 128 dims times every position of the prefix, i.e. 1.28M values sharing one
grid. So V was given a scale per 64-byte block -- position-local, 64 dims each,
no history rewrite needed, and folded into the softmax weights inside PV so it
costs eight scalar multiplies per position instead of vector work:

| `-K8`, 64 tokens at 10K | cosine | top-1 | attn |
|---|---|---|---|
| one scale per head | 0.9121 | 55/64 | 35.9 ms |
| one scale per 64-dim block | 0.9078 | 54/64 | 37.7 ms |

**No improvement, and slightly slower** (the per-block absmax costs a second pass
over the block). So V's error is not a granularity problem: within a head, V is
already close to uniform in magnitude, so the global scale was near-optimal. It
is reverted.

That leaves K, and K cannot be fixed the same way. V's scale multiplies the
output, so a per-position scale folds into the weight for free. K's scale enters
`exp(S*ks - max)` *before* the softmax, so a per-position scale becomes a
per-position vector multiply and a per-position max in the hottest loop. It would
also barely help: the error term is `(max_k / k_rms) / 254`, and that ratio is
about 4 both within one position and across the whole prefix, so position-local
scales leave it unchanged. What would actually cut it is a scale per 32 dims --
and that breaks the single vmadot accumulation that makes the QK dot fast in the
first place.

So int8 KV buys 10.01 -> 10.7 tok/s, about 5 percent, for cosine 0.977 -> 0.91.
By default it stays off.

## What 10K decode is actually limited by

Decode at 10K is 10 tok/s, i.e. 100 ms a token, and the split is nearly all of it:

| stage | ms per token | share |
|---|---|---|
| gemm | 53.2 | 53% |
| attn | 43.8 | 44% |
| rmsnorm | 0.63 | 0.6% |
| silu | 0.38 | 0.4% |
| rope | 0.35 | 0.4% |
| lm_head + argmax | 0.16 | 0.2% |
| embed + misc | 0.41 | 0.4% |

Two numbers from `bwprobe` on this board set the frame. A contiguous
full-width vector read:

| threads | buffer | measured |
|---|---|---|
| 8 | 2048 MB | 27.5 GB/s |
| 8 | 1024 MB | 27.4 GB/s |
| 16 | 1024 MB | 25.8 GB/s |

Sixteen threads is no better than eight, so **27.5 GB/s is the ceiling** and the
older 19.3 figure came from a strided pattern, not from the bus.

### The GEMM is already at the wall

Decode streams the whole weight set once per token: 1265 MB of Q4_0. At 53.2 ms
that is **23.8 GB/s, or 87 percent of the ceiling**. There is at most 7 ms of
headroom in the GEMM and it would take restructuring the kernel for a stream it
is already mostly saturating. The `MEGA_BHOT` isolation (pin one B column block,
results garbage) confirms it is not a scheduling or reuse problem: holding B
entirely in cache saves only 4 percent of prefill GEMM.

### Attention is at 12 GB/s and that is a kernel problem

At 10K the f16 KV caches are 528 MB and attention takes 43.8 ms, which is
**12 GB/s, 44 percent of the ceiling**. Unlike the GEMM this has room. Two
explanations were ruled out with measurements:

- **Not the score reads.** PV loads eight softmax weights per position from eight
  different head rows 10.5 KB apart. `MEGA_PVPROBE=1` deletes those loads:
  attn 43.8 -> 42.8 ms. The scores are L2-hot as designed.
- **Not a prefetch-distance problem.** The V walk reads 128 B of every 512 B
  block. `__builtin_prefetch` (`prefetch.r`, verified in the assembly) 12
  positions ahead is consistently **3 ms worse** over two interleaved pairs
  (45.3 -> 48.4 ms) because the instructions cost issue slots. Reverted.

### So the arithmetic ceiling for this configuration

```
weights          1265 MB   (Q4_0, streamed once per token)
f16 KV            528 MB   (42 layers x 2 kv x 128 dims x 10000 pos x 2 B)
total            1793 MB / 27.5 GB/s = 65.2 ms  ->  15.3 tok/s
```

Reaching 20 tok/s means 50 ms a token, i.e. at most 1375 MB, and the weights
alone are 1265 MB. KV would have to fit in 110 MB -- a 4.8x reduction -- while
attention is simultaneously fixed to go from 12 to 27.5 GB/s.

The three ways to shrink the bytes, and why each is closed:

| lever | bytes | ceiling | measured | blocked on |
|---|---|---|---|---|
| Q4_0 + f16 KV (today) | 1793 MB | 15.3 tok/s | 10.0 | attention at 12 GB/s |
| Q4_0 + int8 KV | 1529 MB | 18.0 tok/s | 10.7 | cosine 0.977 -> 0.91 |
| Q3_K + int8 KV | 1266 MB | 21.7 tok/s | 12.4 | cosine 0.44 |
| Q3_K + int8 KV, 4096 window | 1266 MB | 21.7 tok/s | 16.5 | same accuracy, plus truncated context |

The int8 KV accuracy loss is on the K side and does not respond to scale
granularity (see the section above), and the 3-bit loss is 20x the Q4_0 logit
error -- not a dial that can be turned a little.

**Conclusion: 20 tok/s at 10K is not reachable on this board with a 4-bit weight
format.** The honest target for Q4_0 with accuracy held is about 15 tok/s, and
getting there means reworking the attention kernel from 12 to ~27 GB/s, which is
a larger change than anything in the prefill work above. Anything above that
requires either a coarser weight format (blocked on accuracy) or hardware with
more bandwidth.

## Decode PV: the KV layout, not the byte count

The bandwidth ceiling above said attention ran at 12 GB/s against a 27.5 GB/s
bus. That was the wrong denominator: it divided all of attention's 43.8 ms by
the KV bytes, but attention also reads K and runs exp. Split per stage, at 10K:

| stage | ms/token | bytes | effective |
|---|---|---|---|
| qk | 18.0 | K 264 MB | 14.7 GB/s |
| exp | 1.3 | -- | -- |
| pv | 24.0 | V 264 MB | 11.0 GB/s |
| fin | 0.6 | -- | -- |

So PV moved 264 MB in 24 ms. The question was why 11 and not 27.

V is stored pos-major, `[pos][kv][dim]`, so one position is 512 bytes and PV
takes 128 of them per pass (one kv, one 64-dim half). `bwprobe` models exactly
that pattern with `MEGA_STRIDE`, reading N*32 bytes out of every N*128:

| stride | useful bytes per block | measured |
|---|---|---|
| 1 | 128 of 128 | 27.2 GB/s |
| 2 | 64 of 128 | 13.5 GB/s |
| 4 | 32 of 128 | 5.7 GB/s |

Useful bandwidth falls with the fraction of the block used, so the hardware
prefetcher pulls the whole block in and three quarters of it is evicted unused.
The stride, not the byte count, is what PV pays for.

The fix is a second f16 V copy in PV order, `[kv][half][pos][64]`. A position's
64 dims are then 128 contiguous bytes and consecutive positions are consecutive
in memory, so one (kv, half) pass is a single sequential run. It is the layout
the int8 V plane already used.

**A layout change alone does not fix it, because the loop cannot be reordered.**
Making the read contiguous in place would mean processing all four (kv, half)
quarters of a position in one iteration, which keeps 16 heads x 128 dims of
accumulator live -- 32 registers before the V values are even loaded. Splitting
by half is what makes the 16 accumulators fit, and splitting by half is what
creates the stride. So the copy is the only way.

Costs and results:

- The copy is built once after prefill, 264 MB read + 264 MB written, measured
  at 0.107 s. Allocated only after `prefill_release()` so the 264 MB never
  overlaps the prefill working set, and collapsed to 2 MB pages (`huge_collapse`)
  because fresh pages start on 4 KB and the pass is a single 264 MB run.
- The per-token append writes the same 128 bytes twice, to both layouts. It runs
  once per layer per token, next to a 6144-wide GEMM.
- Only decode PV reads the new copy. Prefill attention, the int8 conversion and
  every debug path keep the pos-major cache.

| | pv | decode at 10K |
|---|---|---|
| pos-major | 23.9 / 24.1 ms | 9.98 / 9.98 tok/s |
| PV order | 13.4 / 13.6 ms | 10.98 / 10.93 tok/s |

Interleaved, same thermal state. Prefill is unchanged (50.14 / 50.18 s against
50.12 / 50.41 s for the previous commit; the board was hot for both and the
earlier 49.2 s was measured cooler). Logits are bit-exact with the previous
commit: md5 `46ee3c9b108fb40ee63678f632e1b465`, mean cosine 0.977316, top-1
60/64. A reorder that moves the same values cannot change the arithmetic, and it
does not.

Two things this cost time on, both worth writing down:

- `MEGA_VPO` was first read with `getenv` from inside the PV loops. That made the
  build miscompile: `ngen` in `main` came back as a different garbage value on
  every run, and the token loop then segfaulted. `-O2` was correct, `-O1` was
  not, and ASan saw nothing, which is the signature of a compiler-level problem
  rather than a real overrun. Reading the environment once in `main` into a plain
  global fixed it. Environment reads do not belong in a worker thread.
- The first precision run after the change came back at cosine 0.55 with the
  first 14 steps matching. That was the run missing `-force`, so decode fed its
  own tokens from step 15 on and the comparison was against a different prefix.
  With `-force` the md5 is identical. Logits are only comparable step by step
  when every step is fed the same input.

## What is left in decode attention, and what is not

PV is now 13.5 ms for 215 MB of V (15.9 GB/s) and QK is 18.0 ms for 215 MB of K
(11.9 GB/s), against 27.2 GB/s for a plain sequential read. Three attempts to
close that gap, all measured at 10K, interleaved, with logits bit-exact:

**Load depth (QK).** The inner loop kept four K loads in flight and then ran 32
FMAs. Double-buffering -- load the next four dims before the current four are
consumed, eight loads outstanding -- changed nothing: qk 18.07 / 18.03 ms against
17.97 / 18.00 ms without it. Outstanding load count is not the limit, and the
change was reverted rather than left in.

**The q broadcasts (QK).** Per iteration the loop also issues 32 scalar loads for
`qb[head*HEAD_DIM + d + j]`, one broadcast per FMA. Replacing them with a
constant made qk 2.031 -> 1.670 ms on a short prompt, 18% faster. So they cost
18%, and the cost is structural: the scalar for a (head, dim) pair is reused
across every position block, but with `d` in the inner loop it has to be reloaded
per block, and 8 heads x 128 dims of scalars do not fit in registers. Cutting it
means either 2x the positions per FMA (see below) or re-reading K per dim.

**Wider LMUL (PV).** PV issues two loads plus 16 FMAs per 128 bytes, which is
close to its measured time at 1.8 GHz -- it is issue-bound, not bandwidth-bound.
If that is right, LMUL=2 should halve the FMAs and win. It does not: with the new
contiguous layout, the existing m2 path measures pv 14.85 / 14.88 ms against
13.50 / 13.52 ms for m1. This core's wide-LMUL FMA costs more per MAC than the
instruction saving is worth, which is the same result the m4 PV path already
recorded, now confirmed with a layout that no longer penalises m1.

**The scattered score reads (PV).** PV reads eight weights per position from
eight rows 40 KB apart, i.e. eight pages. Dropping them entirely (`MEGA_PVPROBE`)
leaves pv at 11.43 / 11.38 ms against 11.43 / 11.47 ms -- no effect at all on the
contiguous layout. The score rows stay L2-hot and are not the limiter.

So the remaining 33 ms of attention sits at 1.4x to 2.3x the bandwidth floor, and
none of the four obvious levers moves it. The next thing to try is not another
scheduling change but a different K tile width: KT=32 means each K load is 64
bytes, and widening the tile to 64 would make every load a full 128-byte line and
halve the q broadcasts per MAC at the same time. That changes the K layout, the
per-token append and the prefill store, so it is a separate piece of work.

## Decode QK: 64 positions per K tile

The previous section left QK at 18.0 ms and named a 64-wide K tile as the next
thing to try. Three measurements decided it, and the first one inverted the
question.

**QK is not memory bound.** Pointing every position block at tile 0, so the whole
K stream is one 8 KB tile resident in L1, changes nothing:

| | qk |
|---|---|
| real tiles | 17.605 / 17.529 ms |
| tile 0 every time | 17.584 / 17.562 ms |

215 MB of K per token is fully hidden behind the arithmetic. That removed layout,
prefetch and load-width as explanations and left instruction issue.

**A wider LMUL is free, not faster.** `fmaprobe` runs the same 8-accumulator loop
at f32m1 and f32m2 with nothing else live: 70.10 and 70.22 GMAC/s. The vector
unit's throughput is per MAC, so m2 does not shorten an FMA -- but it also does
not cost anything.

**What a wider tile actually buys.** `qkprobe` runs the real QK inner loop (four
K loads, four widening converts, 32 FMA and 32 broadcast scalars per group of four
dims) at both tile widths, data in L1:

| tile | time | MACs | rate |
|---|---|---|---|
| KT=32 | 3.498 s | 1.07 G | 18.74 GMAC/s |
| KT=64 | 3.498 s | 2.15 G | 37.47 GMAC/s |

Identical wall time. The loop is dominated by the fixed per-iteration cost, so
doubling the lanes doubles the MACs for free. This is why PV's m2 lost while QK's
should win: PV is stream limited and has nothing to amortise, QK is issue limited
and everything to amortise.

Implemented as a second f16 K copy tiled 64 wide, built in 0.050 s after prefill,
freeing prefill from any change to the shared 32-wide cache:

- `build_k64` merges each pair of adjacent 32-wide tiles; one band item is one
  (block, dim) and both sides are 64 contiguous bytes.
- Thread position ranges now split on the tile width in use, because QK reads a
  tile from its position 0.
- The per-token append writes the new position to both tilings.
- Only decode QK reads the 64-wide copy; `-K8` still takes the 32-wide path.

| | qk | decode at 10K |
|---|---|---|
| 32-wide | 17.96 / 17.89 ms | 10.88 / 10.90 tok/s |
| 64-wide | 14.53 / 14.46 ms | 11.16 / 11.15 tok/s |

Bit-exact again: md5 `46ee3c9b108fb40ee63678f632e1b465`, cosine 0.977316, top-1
60/64. The gain is 19%, not the 2x the loop-shape probe implied, so the remaining
4.5 ms is not the dot loop -- it is the score stores, the per-head max pass and
the partial last block.

One bug worth recording: the builder first enumerated blocks against `NKT64`
instead of the number of blocks actually in use. A 9999-position prefix ends
mid-tile, so that walked past the last real block of kv 0 and never built the tail
of kv 1. Cosine came back 0.648. Two things to note about how it surfaced -- the
timings looked fine, and the second flag bug hid the A/B that would have caught
it: `getenv("MEGA_K64")` returns the string `"0"`, which is truthy, so both arms
of the first A/B ran the new path and both read `qk 14.4`. Flags that can be
switched off must compare the value, not test for presence.

## Split-KV, flash decoding, and what the X100 cluster is worth

Decode attention is already a split-KV with a combine: `attn_ranges` divides the
position range across the eight participants and `attn_combine` reduces the
per-thread partials with the log-sum-exp rescale. That is the flash-decoding
structure. The question is not whether to adopt it but whether the split is deep
enough, and that is measurable by changing the participant count:

| participants | qk | pv | decode |
|---|---|---|---|
| 4 | 28.66 ms | 23.41 ms | 7.55 tok/s |
| 8 | 14.54 ms | 13.67 ms | 10.29 tok/s |

Doubling the participants roughly halves both stages, so the position split is
still paying at eight. That invites the obvious next step -- the other eight cores
on the die.

**The two clusters are not interchangeable, and the cores that were idle are the
slow ones.** Measuring `vlenb` per core:

| cores | cluster | vlenb | VLEN | e32m2 lanes |
|---|---|---|---|---|
| 0..7 | X100 | 32 | 256 | 16 |
| 8..15 | A100 | 128 | 1024 | 64 |

The A100s need the `/proc/set_ai_thread` handshake before they accept a bind,
which is why the direct measurement sees only cores 0..7. Decode attention binds
the A100s and never touches the X100s.

Running the QK inner loop on both, one thread each:

| cluster | per core | 8 cores |
|---|---|---|
| A100 | 37.47 GMAC/s | 299.8 |
| X100 | 7.44 GMAC/s | 59.5 |

An X100 core is worth one fifth of an A100 on this loop, so all eight of them add
20% to QK's issue throughput. QK would go from 14.5 to 12.1 ms, and since QK is
14.5 ms of an 89.6 ms token, the end-to-end gain is about 2.4 ms, or 3%. PV cannot
follow them usefully: at e32m2 = 16 lanes a 64-dim accumulator needs four
registers, so eight heads no longer fit and the head count would have to drop.

**Where the 89.6 ms actually goes, measured at 10K:**

| stage | ms | bytes | effective |
|---|---|---|---|
| gemm | 52.1 | 1265 MB | 24.3 GB/s |
| qk | 14.5 | 215 MB | 14.8 GB/s |
| pv | 13.6 | 215 MB | 15.8 GB/s |
| exp + fin + embed + norm + lm_head | 9.4 | -- | -- |

GEMM is 58% of the token and already runs at 88% of the 27.5 GB/s bus. Taking
attention all the way to the streaming ceiling would give 52.1 + 15.6 + 9.4 =
77 ms, or 13.0 tok/s, against 11.16 today. That is the realistic ceiling for Q4_0
on this board, and it needs no new cluster: it needs PV to reach 27.5 GB/s, which
the layout change already moved from 11.0 to 15.8.

So flash decoding is in place, its split is deep enough that adding the X100s is
worth 3%, and the remaining gap is a per-core streaming rate on a bus that GEMM
has already saturated. Going past ~13 tok/s means fewer bytes per token, which is
the weight-format question the previous section closed on accuracy.

### The 3-bit accuracy loss is the bit budget, not the quantizer

The 3-bit arm reached 12.37 tok/s but scored cosine 0.6744 against the Q4_0
reference. The working hypothesis was that `repack_b_q3` lost range: it stores
one int8 scale per 16-weight sub-block under a per-row fp16 `fs = smax / 127`,
so a row whose sub-blocks differ by more than 127:1 would have to clamp the small
ones. Two measurements falsified that.

First, instrumenting the repack over all 140,574,720 sub-blocks of the model:
the largest intra-row step ratio is **90:1**, 0.000% of rows exceed 127:1, and
only **0.0032%** of sub-blocks round below `fs/2` and get clamped to scale 1. The
int8 intermediate never runs out of range.

Second, accumulating squared error during the same repack, once against the
stored `isc * fs` grid and once against an exact float per-sub-block step:

| grid | relative RMS weight error |
|---|---|
| stored `isc * fs` (what ships) | 19.36% |
| exact float step (upper bound) | 19.43% |

Quantizing the scale into int8 costs **-0.07%** — it is not measurable. The grid
itself is the loss.

A 3-bit, 8-level, 16-weight-group quantizer is worth about 18% relative RMS on
Gaussian weights, and the shipped rule already sits at that ceiling:

| rule | relative RMS on N(0,1), groups of 16 |
|---|---|
| shipped `max(mp/3, mn/4)`, codes -4..3 | 18.01% |
| symmetric `max(mp,mn)/3`, codes -3..3 | 19.89% |
| `std/2` step | 20.45% |
| least-squares-optimal step | 18.87% |

18.01% is the best of the four. There is no scale rule to fix.

19.4% independent weight noise accumulating over 28 layers predicts
`1/sqrt(1 + 28 * 0.194^2) = 0.71` output cosine, which is the 0.674 observed.
The path is behaving correctly and is simply too lossy for the accuracy bar. A
mixed-precision split does not rescue it either: protecting attention and
compressing only the per-layer FFN halves the injected variance and still lands
near 0.73.

So the 12.37 tok/s arm is closed on accuracy, and the ~13.0 tok/s Q4_0 ceiling
above — bit-exact, no new format — is the target that remains.

### PV was one position in flight; batching four is bit-exact

`attn_pv_m2` issued a position's two vector loads and then consumed them
immediately, so a single iteration's requests were the only ones on the bus. Four
positions are now loaded before any is used, which puts four independent 64-byte
streams in flight without holding anything extra:

| arm | pv ms | decode tok/s | logits |
|---|---|---|---|
| m1 accumulators, U=1 | 14.39 | 11.28 | `614c5ac3...` |
| m2 accumulators, U=1 | 11.77 | 11.62 | `614c5ac3...` |
| m2 accumulators, U=4 | **9.82** | **11.89** | `614c5ac3...` |

U=8 spills (pv 14.4 ms). All three arms write byte-identical logits, and the
binary built before the change writes the same `614c5ac3...`, so the reordering
moves the same values.

A zicbop prefetch at distance 8 or 16 positions made it slower (10.4 and 10.4 ms
against 9.9), so the hardware prefetcher already has this stream; the gain came
from issuing loads earlier inside the loop, not from more of them outstanding.

**Batching does nothing on the f16 path** (pv 13.5 ms before and after): a
position there is 128 bytes and one `vle16` already covers it, so the loop was
never short of in-flight bytes. The win is int8-only, 14.2 to 9.8 ms.

### The int8 KV cache now costs more accuracy than speed it buys

Verifying the arms turned up a bookkeeping error worth recording: the
`0.977316` cosine in earlier sections belongs to the **f16** KV path. Scored
against the same golden logits, the two KV dtypes are

| KV | qk ms | pv ms | decode tok/s | cosine | top-1 |
|---|---|---|---|---|---|
| f16 | 12.37 | 13.49 | 11.65 | **0.9783** | **60/64** |
| int8 (`-K8`) | 13.23 | 9.82 | 11.89 | 0.9075 | 54/64 |

int8 KV reads 107 MB instead of 215 MB and saves 3.7 ms of PV, but it costs 0.07
of cosine and six top-1 slots, and after the PV batching the whole net speed
advantage is 2%. Per byte it is now the slower path (10.9 GB/s against 15.9),
because halving the bytes per position turns PV into an issue-bound loop rather
than a bandwidth one. The precision-preserving configuration is therefore f16 KV
at 11.65 tok/s, and `-K8` is a speed experiment, not a default.

### Correctness record for the shipped decode configuration

Two things were open, and both are now pinned.

**The exit fault is fixed.** Every decode run faulted after printing its last line,
with all logits already on disk. The cause is shutdown, not arithmetic: main
returns to run static destructors while sixteen worker threads are still parked in
`worker()`'s loop and every buffer is a raw huge-page mapping that nothing joins
or tears down in order. main now flushes stdout, stderr and the step log, closes
the step log, and calls `_exit(0)`. Exit status is 0 on every arm, and the logits
are unchanged by the fix -- `d532fda4...` before and after on the same arm, so it
touched no arithmetic.

**The default configuration is the precision-preserving one.** Teacher-forced over
64 steps at 10K, against the frozen llama.cpp golden:

| arm | exit | logits md5 | qk | pv | gemm | decode | cosine | top-1 |
|---|---|---|---|---|---|---|---|---|
| m1 accumulators, U=1 | 0 | `d532fda4...` (n=32) | 12.94 | 13.74 | -- | 11.41 | -- | -- |
| m2 accumulators, U=4 (default) | 0 | `d532fda4...` (n=32) | 12.94 | 13.26 | -- | 11.33 | -- | -- |
| default, 64 steps | 0 | `c016e612...` | 12.88 | 13.36 | 52.95 | **11.62** | **0.9783** | **60/64** |

The default and the unbatched m1 arm write byte-identical logits, so the PV
batching is a pure reordering. The default keeps f16 KV: the `-K8` int8 cache is
2% faster and 0.07 cosine worse, so it stays an experiment.

One bookkeeping correction to the earlier sections: `c016e612...` is what this
source produces, not the recorded `46ee3c9b...`. The recorded hash belongs to the
`{48,48,45,28}` X100 column split, and the cosine is the same quality either way
(0.9783 against 0.9773, both 60/64). The comparison that matters -- same source,
same split, before and after a change -- is bit-exact in every arm above.

### Where decode QK's 12.9 ms goes, measured on the path that runs

The earlier section said the remaining QK time was "the score stores, the per-head
max pass and the partial last block". Timing those three separately inside the
32-wide decode path, summed over all threads per token, says otherwise:

| section | thread-ms/token | share of qk work |
|---|---|---|
| dot loop | 68.2 | 70% |
| score stores | 0.27 | 0.3% |
| per-head max sweep | 8.4 | 9% |

The dot loop dominates. Stores are already free; the max sweep is 9% and is a
serial `vfmax` chain over the score matrix.

**A correction.** The claim that decode QK is not memory bound came from
`MEGA_QKSAME`, but that run's output was grepped for `qk` and picked up the
*prefill* line (`attn work: qk 58.0 ...`), so decode was never tested. Pinning
every position block to tile 0 of each layer (1.3 MB, L2-resident) on the decode
path gives dot 68.3 -> 63.0 thread-ms and qk 13.78 -> 12.29 ms, i.e. **11%**, not
"nothing". Decode QK does have a memory component, it is just small.

**The three cheap levers are exhausted**, all bit-identical:

| change | effect on qk |
|---|---|
| lanes doubled (m2, vl=64 through `MEGA_K64=1`) | 13.83 -> 12.65 ms (+8.5%) |
| position blocks pinned to L2 | 13.78 -> 12.29 ms (+11%) |
| K loads software-pipelined one d-group ahead | 68.23 vs 68.48 dot thread-ms (0%) |

The pipeline is neutral because the compiler already hoists those loads; the
register pressure of holding eight K vectors alongside eight accumulators buys
nothing. That variant was removed rather than shipped behind a flag.

**Per-core rate, and a stale number.** `probe_cores` (same inner loop, `vsetvl`
sized) gives **25.0 GMAC/s on an X100 core**, identical whether one X100 core or
all eight run, so there is no occupancy clock drop on that cluster. The recorded
"X100 = 7.44 GMAC/s, one fifth of an A100" is wrong and should not be used to size
a port: at 25 GMAC/s an X100 core is in the same class as the 13.6 GMAC/s a
production A100 thread sustains. The A100 side of that probe faulted on this board
(`/proc/set_ai_thread` and thread creation interact), so its number here comes
from production, not from the probe.

So the remaining lever for attention looked like the idle X100 cluster, and more
than the recorded 3% estimate suggested -- but it needs VLEN-agnostic QK/PV kernels
and a second thread pool, since every int8 kernel in this file assumes 32 e32 lanes
(VLEN=1024) and would compute garbage on an X100 core. That lever was pulled: the
port is built, it is correct, and it is **2.1 to 2.5 tok/s slower**, for the reason
in the next section. Attention stays on the A cluster.
### The X100 cluster cannot take decode QK yet: the score slab is owned by a thread

The port is half done and the half that is done is inert. The default path is
byte-identical to the shipped one, so nothing regressed:

| arm | exit | logits md5 | qk | pv | decode | cosine | top-1 |
|---|---|---|---|---|---|---|---|
| `c016e612...` before this change, 64 steps | 0 | `c016e612...` | 12.77 | 13.35 | 11.61 | 0.978269 | 60/64 |
| after the refactor below, 64 steps | 0 | `c016e612...` | 12.77 | 13.35 | 11.61 | 0.978269 | 60/64 |
| `MEGA_XATTN=1`, 64 steps | 0 | `bba293f9...` | 24.05 | 13.73 | 10.19 | 0.171912 | 1/64 |

**What was built.** QK's inner loop now advances by the achieved `vl` instead of
by the tile width, and carries the intra-tile offset into the K pointer, so one
body serves 32-lane and 8-lane cores. `attn_exp_sum` has the same treatment. The
per-thread attention state grew from eight slots to sixteen (`N_ATTN`), the X100
worker has an attention branch behind `g_ax_gen`, and `ax_run` posts QK/exp to all
sixteen threads with tile weights taken from each thread's measured `vsetvl`.

**Why it is wrong, and why the timing says so.** QK got *slower* going from eight
threads to sixteen (12.77 -> 24.05 ms) while producing garbage. Both are the same
defect: PV reads `g_sa[tid]`, so a score slab belongs to the thread whose PV band
covers it. Sixteen-way QK splits positions across two sets of slabs, and PV then
reads every position from the A cluster's own slab -- the second half of the
context is read from memory nobody wrote, and the ten X100 threads' QK work is
paid for and used by nothing. The fix is to have X100 thread `t` write into A
thread `t`'s slab at an offset past that thread's own band, which leaves PV's
addressing untouched; the eight-slot width of `g_sa` then also bounds the split.

**One correction to the port's premise.** An X100 core reaches 8 lanes at
`e32m1`, not the 32 the tile loop assumes, so its share of a KT=32 tile is four
steps of arithmetic and four load/store pairs instead of one of each. Weighting
the split by measured lanes keeps both clusters at the same finish line but does
not make the X100 tile cheap: the load/store overhead per element is unchanged,
which is why the port is worth doing only if the arithmetic it removes from the A
cluster is what the A cluster was waiting on.
### The X100 cluster cannot take decode attention: it is correct now, and 2.1 tok/s slower

The port works. Sixteen threads fill the score slabs, the A cluster's PV reads them,
and the result is bit-for-bit as accurate as the eight-thread path (cosine 0.978228
against 0.978269, 60/64 both). It is also slower in every weighting tried, and the
reason is not the one the 25 GMAC/s probe suggested.

| arm, 64 forced steps | exit | qk | exp | pv | decode | cosine | top-1 |
|---|---|---|---|---|---|---|---|
| 8-way A cluster (shipped) | 0 | 13.10 | 1.05 | 13.04 | **11.62** | 0.978269 | 60/64 |
| XATTN, X weight = its lanes | 0 | 34.26 | 3.01 | 12.75 | 9.20 | 0.978228 | 60/64 |
| XATTN, weight 31% | 0 | 30.56 | 3.19 | 12.89 | 9.57 | -- | -- |
| XATTN, weight 200% | 0 | 34.07 | 3.21 | 13.11 | 9.16 | -- | -- |

**Three defects had to be fixed to get a correct answer at all, and each one is a
fact about this code that will bite again.**

1. *A score slab belongs to the thread whose PV band covers it.* PV reads
   `g_sa[tid]` and indexes it from its own `p0`, so a split over a wider thread
   count reads positions nobody wrote. Sixteen-way QK with eight-way PV gave cosine
   0.172 and made QK *slower* (24.05 ms): the A threads waited on the barrier while
   ten of the sixteen threads' work was thrown away. The fix is to pair A thread `t`
   with X thread `t`, give the pair PV band `t`, and have the X thread write the
   second part of that same band at the offset the first part ends on. PV is then
   untouched, and the split can only divide a band, never cross one.
2. *A layout fix that is arithmetically neutral can cost 25 ms.* Indexing the slab
   by absolute position removes the ownership question entirely -- and it is
   bit-identical (`c016e612...`, so the addressing change alone moves no arithmetic)
   -- but PV went from 13.04 to **38.41 ms** and the token to 9.08. A PV thread reads
   eight head rows; per-thread slabs put those rows 12.5 KB apart over a 48 KB band,
   absolute indexing puts them 49 KB apart over 393 KB. The compact per-thread slab
   is worth more than the ownership rule costs.
3. *A reduction must reduce the register, not the last chunk.* The exp sum for a
   narrow core reduced with the final chunk's `vl`, dropping every lane the earlier
   chunks had accumulated: cosine 0.684, 28/64. Lanes past a chunk's `vl` were never
   written, so they are still the accumulator's initial zero and reducing the full
   register is correct.

**Why it still loses, measured rather than assumed.** The X cluster's own reported
work time is 127.66 thread-ms of QK against the eight-thread path's 75.46 while it
holds 6% of the positions: about 7 ms per X thread per token regardless of how much
it is given. Fitting the two weighting points gives ~27.5 ms of per-call cost plus a
slope of 0.0272 ms/position against the A cluster's 0.0103, so a third of the loss is
the narrower vector and two thirds is paid per stage.

`probe_ops` says that cost is not an emulated instruction, which was the hypothesis
worth testing: on an X100 core at vl=8, `fma` is 0.45 ns (17.8 GMAC/s per core), a
f32 load+add 1.82, an f16 load plus widening convert 2.27, a store 1.89, `vfredusum`
1.36 and `vrgather` 1.36. Nothing is trap-emulated; the vector unit is fine at the
width it has. The cost is in dispatching a second cluster once per attention stage,
and the ceiling without it is still small: even at zero fixed cost the slope alone
takes QK from 12.66 to ~9.3 ms, 4% of a token.

**What was kept, and what was taken back out.** The port itself is removed -- a flag
that turns a 11.62 tok/s path into 9.20 is not a variant worth carrying, and this
file already has the precedent. What stays is the kernel change it forced: QK's tile
loop steps by the vector length it is granted and carries the offset inside the tile.
That is the same loop at 32 lanes (step is KT, offset is zero, output unchanged --
`c016e612...` and 11.62 tok/s before and after), and it removes an assumption that
was silently false: the loop claimed to cover a tile while storing only the lanes one
op produced.

The one number this section does not have is the A cluster's own `probe_ops` row.
That arm dies in the `/proc/set_ai_thread` handshake (exit 255, single-threaded, no
thread creation), so every A100 figure quoted here is from production rather than
from the probe.

### Measuring the speculative round, and 5% out of the drafter's bias pass

The round was never broken down. `m4_report` existed and nothing called it, so the
four-row timers it fills were write-only. Wired up at 10K over 300 rounds:

| where a 128 ms round goes | ms | what it is |
|---|---|---|
| verify: gemm | 52.0 | 42 layers x Q4_0 through the hp_m4 kernel for four rows |
| verify: attention | 25.2 | int8 4-row QK+PV+softmax over 215 MB of K/V |
| verify: lm_head | 7.4 | 150 MB of the shared output head |
| verify: norm/rope/quant/silu | 4.6 | -- |
| draft: 5 layers | 13.4 | 1.7 MB of weights, so it is not bandwidth |
| draft: attention | 2.1 | 256-position window, 64 units |
| draft: shared lm_head | 6.5 | the 150 MB head again, for the draft's four rows |
| draft: markov bias | 6.2 | 3 x 308 KB of weights plus an argmax |

The draft's own forward is 1.7 MB of weights in 13.4 ms, i.e. 0.13 GB/s: the
drafter is latency, not bandwidth, and the same is true of its markov head.

**The one clear win there is the bias pass's argmax.** It was three scalar scans of
130,560 entries on one core, 6.2 ms of a 128 ms round. Bands over eight threads
with the same first-wins tie rule:

| 10K, 300 rounds | accepted/round | ms/round | tok/s |
|---|---|---|---|
| baseline | 0.637 (191/900) | 134.25 | 12.19 |
| bias argmax over eight threads | 0.637 (191/900) | 127.91 | **12.80** |

Identical acceptance, so the reordering is exact. Two measurements cost an hour
before it was right, and both are the same mistake:

* **A parallel argmax must carry the row it is scanning.** The first version
  indexed `g_dslog` from zero instead of `i * VOCAB`, so draft slots 1 and 2 read
  slot 0's logits. Acceptance "rose" to 0.793 (238/900) and tok/s to 14.10. That
  is the shape of the error to watch for: a *better* acceptance number from a
  change that was supposed to be a pure reordering means the change moved tokens.
* **A stale process on the board invalidates everything.** Those runs also
  reported 4.43 tok/s, because two `m3c.xd` instances from earlier measurements
  were still resident at 786% and 668% CPU. This is the same trap the earlier
  section recorded; `ps` before a measurement is cheaper than the run.

### The drafter's rope recomputed 23,040 transcendentals a round

The drafter's prep work was 7.0 ms a round for 4 x 2048 elements a layer, which is
not a bandwidth number. `ds_rope` called `powf`, `cosf` and `sinf` once per element
pair, per head, per row, per layer: 64 freqs x 4 rows x 18 vectors x 5 layers =
23,040 transcendental calls. The angle depends only on `(pos, i)`, and `g_rope_tab`
already holds cos and sin for every position, built by the target's setup from the
identical expression -- so the fix is a table lookup, and it is bit-exact.

It made things **worse** at first, which is the interesting half. Removing the
function calls left a pure load/FMA body over interleaved pairs, and GCC vectorised
it into stride-2 accesses. This core trap-emulates strided vector access at ~600 ns
an instruction, so prep went to 9.8 ms and the round to 133.4 ms. `compute_gmax`
already carries the `no-tree-vectorize` attribute for exactly this reason; `ds_rope`
needed it too.

| 10K, 300 rounds | draft layers | prep (excl. attn) | ms/round | tok/s |
|---|---|---|---|---|
| powf/cosf/sinf per element | 14.72 | 6.97 | 127.91 | 12.80 |
| table lookup, auto-vectorised | 17.44 | 9.78 | 133.36 | 12.27 |
| table lookup, scalar | **11.48** | **3.79** | **123.81** | **13.22** |

Acceptance is 191/900 in all three, so the table is exact and the only thing that
changed is time. The injection path also benefits: 3.5 -> 2.8 s, because it ropes
every prefix position and was paying the same tax there.

**13.22 tok/s is past the 13.0 tok/s single-row ceiling** this file derived for
Q4_0: four rows amortise the 52 ms of weights over 1.637 tokens instead of one.
The round is now verify 101.3 + draft 21.3 + 1.2. Inside the draft, the shared
lm_head is 6.55 ms for 150 MB (23 GB/s, at the bus) and the five layers' 20 GEMM
calls are 5.56 ms for 1.7 MB (0.3 GB/s), so the layers are still latency, not
bytes, which is where the next 3 ms is.

### The drafter is at the bus, and my "1.7 MB, latency bound" reading was wrong

The byte counter that printed the drafter's traffic left out the `n/32` factor of a
Q4_0 matrix, so it reported 1.7 MB for what is really 153.4 MB and I spent a stretch
treating the drafter as latency bound. Its real shapes are
`wq[2048x2048] wg[2048x6144] wd[6144x2048] wo[2048x2048]`, 576 B per block, i.e.
30.7 MB a layer and 153.4 MB for five.

With the arithmetic right, the picture inverts:

| pass | bytes | ms | effective |
|---|---|---|---|
| target layers, 42 | 1421 MB | 52.0 | 27.3 GB/s |
| drafter layers, 5 | 153.4 MB | 5.66 | **27.1 GB/s** |
| shared lm_head, verify | 150.4 MB | 6.55 | 23.0 GB/s |
| verify attention, int8 K+V | 220 MB | 25.2 | **8.7 GB/s** |

Three of those four are at the bus and the one that is not is attention, by a
factor of three.

**kstep does not matter, which is the experiment that settled it.** The draft's
matrices are small (n of 2048 or 6144, so 8 to 24 n-blocks a thread), so 4-row
by 128-column calls looked suspiciously small and a per-call cost seemed likely:

| 10K, 120 rounds | drafter gemm | draft layers | round |
|---|---|---|---|
| `-k 4` (default) | 5.66 | 11.92 | 123.89 |
| `-k 8` | 5.65 | 10.80 | 125.41 |
| `-k 16` | 5.70 | 11.98 | 123.63 |

The GEMM time is flat to 1% across a 4x change in call size, so there is no per-call
cost to recover: the pass is streaming 153 MB at the bus already. The layer total
moves by more than the GEMM does, which is the prep work's variance, not a trend.

So the draft's 21.8 ms is gemm 5.66 (at the bus), shared head 6.54 (23 GB/s), prep
6.26 (of which attention 2.14) and bias 3.28. Nothing in it is three times off.

**Where the round's headroom is not.** A round moves 1421 + 153 + 150 + 150 =
1874 MB, which at 27.5 GB/s is 68 ms; the round took 123.8. I read the 55 ms above
that floor as verify attention running at 8.7 GB/s, and projected 8 ms and 15.3
tok/s from tiling it dim-major. That projection was wrong, and the measurement
below is what corrected it.

## Four-row verify attention is MAC bound, not byte bound

`MEGA_QKNOST=1` keeps one int16 score store in eight and drops the other seven.
Across 60 rounds at 10K, QK went 57.95 -> 58.49 core-s, i.e. **nowhere**. The
score write-back is not the cost, so the 8.7 GB/s figure was the wrong lens: the
kernel is not waiting on memory at all.

What it is doing is arithmetic. Four rows mean four queries against every cached
position, so a round asks for 4 x 16 x 10240 x 128 = 83.9 MMAC a layer for QK and
the same again for PV, 3.52 GMAC each. Against the measured 12.1 ms of QK and
10.8 ms of PV that is 291 and 326 GMAC/s of aggregate `vmadot` throughput on eight
A100 cores -- 36 to 41 GMAC/s a core, above the 25 GMAC/s a core that a f32 FMA
pass reaches on this chip. The int8 QK/PV pair is at the dot-instruction ceiling.
There is no 17 ms sitting in the KV layout, and dim-major tiling would buy nothing
that is not already there.

The consequence for the round: 25.2 ms of verify attention is the price of
checking four rows, and the only way to move it is to stop asking four rows to
read the whole context, or to put the second cluster's `vmadot` on half the heads.
It is not a layout problem.

## The X100 elementwise pool is wrong for four rows

`parallel_for_w` gives rmsnorm a sixteen-band split, which was measured as a win on
prefill's sixty-four rows (`60.2 -> 58.9 s`). A four-row decode norm has 8192
floats in total, so each A core gets a quarter of a row and then the whole cluster
waits on the X100 wake path. Per round:

| 10K, four rows | rmsnorm | draft prep | round |
|---|---|---|---|
| `MEGA_ELN=2` (was the default) | 1.54-2.58 | 5.31-7.41 | 125.23 |
| `MEGA_ELN=0` | **0.12** | 3.88 | 121.26 |

A dispatch of the A pool costs 3.1 us (`MEGA_BARTIME`), so the 200 us a call this
path was paying is the cross-cluster handshake, not the work. `parallel_for_w` now
falls back to the eight-way A split below eight bands, which keeps prefill's
sixteen bands and gives decode the plain one. Whole rows move between threads, and
`rn_rows_band` reduces each row on one thread, so the result is unchanged bit for
bit -- the golden check below is the same 60/64 top-1.

## The yield in the X100 idle loop is a rate limiter, and must stay one

The idle loop polled five generation counters and called `sched_yield` between
polls, which looked like an obvious waste: eight syscalls every few hundred
nanoseconds on cores that are otherwise parked. Both replacements were worse.

| 40 rounds, 10K | attn mean | attn min | attn max |
|---|---|---|---|
| `sched_yield` | 28.3 | 24.9 | **40.3** |
| `pause` spin | 37.0 | 24.8 | 86.0 |
| `pause` then 200 us sleep | 36.3 | 24.8 | **143.0** |

A bare spin polls `g_a4_gen` at full rate from eight cores and invalidates the
line under the A cluster's QK; a sleeping thread is late to the layer handshake,
which comes every 600 us. The yield is what paces both. Reverted, with the numbers
in the comment so the next reader does not repeat it.

The same table is also the four-row softmax's whole story: the X100 handshake has a
24.8 ms floor against 35.9 ms for doing the softmax on the A cluster, but it
averages out to the same round (118.46 against 118.92 over 100 rounds) because of
the tail. Keeping it -- the floor is real headroom if the handshake can be
shortened -- and it emits the identical 49/300 tokens either way.

## Three small items, and what each was worth

The drafter's per-layer prologue was eighty `ds_rmsvec`/`ds_rope` calls and four
512-element scalar f16 stores on a single core: 1.09 ms a round. It is now one
(slot, head) unit per pool thread, the same shape as the target's `m4_rope_band`,
and costs **0.19 ms**. Every element is independent, so the split changes no
result.

The draft's markov argmax carried `best` through 16320 iterations of a
compare-and-branch, 3.28 ms for three slots. A lane-wise running max with the same
strict `>` gives each lane its own chain and the same earliest-index tie-break:
**3.00 ms**. Only 0.3 ms, which says the scan was already at the memory floor of
reading a 130560-wide row plus the bias.

The prologue fix also had to resolve `g_ds_rope_half` before the band runs, since
eight threads reading and lazily writing the same flag is a race even when they
all write the same value.

## Where the speculative round stands

Steady-state per-round budget on the current build, `/tmp/p10k.txt` at 10K:

| stage | ms | note |
|---|---|---|
| verify GEMM, 42 layers | 52.0 | 1265 MB at 24.3 GB/s, 88% of the stream peak |
| verify attention, 4 rows | 25.2 | at the `vmadot` ceiling, see above |
| verify lm_head | 7.4 | 150 MB at 23 GB/s |
| draft, 5 layers | 18.2 | gemm 5.66 (at the bus), prep 3.09, head 6.50, bias 3.00 |
| round | **119.8** | 17.89 tok/s at 1.143 accepted per round (see the bias section below) |

The remaining measurable items are 2 ms in the two lm_head passes (23 against 27.5
GB/s) and 6 ms in the verify GEMM (24.3 against 27.5), both of them the vendor
IME2 kernel. Everything else in the round is either at the bus, at the dot
instruction, or is the four-row attention's arithmetic. Acceptance, not the round,
is what moved last: per-slot markov weights took the same 119.8 ms round from
12.58 to 17.89 tok/s.

## Decode V in int8, K left in f16

`-K8` quantises both caches to one scale per (layer, kv head) and was off by
default because it costs accuracy: cosine **0.9109**, top-1 58/64, against
0.978266. Splitting the two halves of that measurement shows the loss is all in K
and the speed is all in V:

| arm, 64 forced steps | qk | pv | decode | cosine | top-1 |
|---|---|---|---|---|---|
| f16 K, f16 V (shipped before) | 12.39 | 13.41 | 11.65 | 0.978266 | 60/64 |
| int8 K, int8 V (`-K8`) | 13.12 | 9.82 | 12.22 | 0.9109 | 58/64 |
| **f16 K, int8 V** (now default) | 12.37 | **9.73** | **12.28** | 0.978304 | 60/64 |

Two things had to be true for the mixed arm to exist, and the measurements say
both are:

- **A K error is amplified, a V error averages out.** K enters through the dot
  before the exponent, so one wrong element moves the whole score row's softmax.
  V enters after the weights, so its error is a weighted sum of per-position
  errors: 0.4% per element over ten thousand positions lands at the fourth
  decimal of the output, which is what the cosine column shows.
- **The two stages gain from int8 in opposite directions.** PV is stream limited
  at 16 GB/s for the f16 plane, so halving its bytes takes 13.41 to 9.73 ms. QK
  is issue limited, and int8 K adds two conversions per load, so int8 K is
  *slower* (12.39 to 13.12) even though it halves the bytes. Splitting the flag is
  what lets each stage keep its better format.

`g_v8dec` (`MEGA_V8=0` turns it off, `-V8` forces it on) now owns the int8 V plane
alone: the plane, its per-(layer, kv) scales from prefill's max-abs, the append in
`rope_all`, and a matching append in `m4_rope_item` so a run that emits through
four-row rounds and then falls back to single-row decode does not read V for those
positions out of a plane nobody wrote. The PV-order f16 copy is not built when V
is int8, which is 264 MB back.

**One bug worth recording.** `rope_all` wrote the PV-order f16 copy under
`if (vpo_on())`, and `vpo_on()` returned `g_vpo`. With the copy no longer built,
that store landed on a null base and took the process down on the first decode
token. `vpo_on()` now asks the buffer, and `build_vpv_order` asks the flag, so a
"read PV order" answer cannot outlive the buffer it describes.

The cost of the change on the 10K prompt is 4 ms: `/tmp/p10k.txt`, 64 forced
steps, md5 of the new logits `0.978304` mean cosine and 60/64 top-1, same as the
f16 arm's top-1 and 4e-4 below its cosine. Speculative rounds are untouched: the
same 30 rounds at 10K give the same drafts and the same 10/90 acceptance with V
in int8 and with `MEGA_V8=0`.

**A trap for the next measurement.** Two rounds of this A/B were run with the
command duplicated, so two processes shared the eight A cores and the eight X100
workers. Both came back with 0/300 drafts accepted instead of 49/300, at a
*lower* verify time. A contended board does not just slow the numbers down, it
changes what the speculative path computes, because the four-row verify's softmax
runs on that pool. `/tmp/p10k.txt` takes 150 s of prefill, so it is tempting to
run two arms at once; don't.

## The markov bias weight, and why slot 0 is the only draft that matters

`ds_markov_bias` projects the token into a rank-256 space and back out to the
vocabulary, and the result was added to the drafter's own logits at weight 1. That
weight asserts that the bigram prior and the drafter's 324M forward have been
calibrated against each other, which nothing ever measured. It is a free knob: the
bias reorders drafts and the emitted tokens are the verify's own argmax chain, so
the target's output cannot change whatever it does. On the 10K prompt:

| weight | accepted/round | where it was measured |
|---|---|---|
| 1.00 (the old constant) | 0.33 | 30 rounds |
| 0.55 | 0.505 | 100 rounds |
| 0.40 | 0.590 / 0.527 | two runs, 100 and 150 rounds |
| 0.35 | 0.690 | 100 rounds |
| 0.30 | 0.490 | 100 rounds |
| 0.25 | 0.420 | 100 rounds |
| 0.20 | 0.280 | 100 rounds |

**That table cannot decide the weight.** The emitted stream is the verify's chain,
so the tokens are the same for every weight; what a weight changes is the stride
`p0 += n + 1`, and two weights therefore sample different stretches of the same
text. This prompt's accept rate varies by more than 2x between stretches, which is
why a neighbouring weight reads 0.35 and 0.40 and 0.55 in that order. Two 600-round
runs settle the same question with numbers that do not overlap at all.

The measurement that does decide it is paired, and it exists because of one
asymmetry: slot 0's question is "what follows the anchor", and the verify's row 0
answers exactly that question under the same prefix whatever the draft said. So a
round can run its draft twice at two weights and score both against one set of
target logits, on identical positions. `MEGA_DSBW2=a,b,c` up to seven extra
weights; the extra drafts cost 18 ms each and are for measurement only.

| paired slot-0 hits, 300 rounds | w=0.50 | 0.80 | 1.00 | 1.30 | 1.60 | 2.20 | 3.00 | 4.00 |
|---|---|---|---|---|---|---|---|---|
| target argmax at draft rank 0 | **218** | **221** | 214 | 186 | 154 | 78 | 37 | 23 |

The plateau runs 0.5-1.0 and the cliff is past 1.6, so the old constant of 1 sat
near the top of the plateau by luck and the earlier 0.2-0.55 readings were sampling
noise.

**The larger finding is that the slots are not the same question.** Every draft
slot attends the prefix plus slot 0 and nothing else, so slots 1-3 predict
positions t+1..t+3 from one context and differ from slot 0 only by their bias. At
300 rounds, slot 0 alone accounts for 221 of 297 accepted drafts: slots 1-3 deliver
0.25 per round between them, 0.08 each, and at weight 0.40 they delivered 0.135.

That makes a per-slot weight the right shape. Only slot 0's predecessor is a token
the target actually emitted, so it can carry a strong prior; slots 1-3 condition on
a hypothesis. `MEGA_DSBVW=w0,w1,w2,w3`, and two 600-round runs on the same prompt:

| 600 rounds at 10K | accepted/round | ms/round | tok/s |
|---|---|---|---|
| uniform 0.80 | 0.638 (383/1800) | 119.46 | 13.71 |
| **0.80 / 0.25 x 3** | **1.143 (686/1800)** | 119.78 | **17.89** |

Same round cost, 79% more accepted tokens. That is the default now. The round
budget is unchanged: draft 18.4 ms (layers 8.6, head 6.5, bias 3.2), verify 100.4,
round 119.8.

**What this does not fix.** tokens/round is `1 + E[accepted]`, so 20 tok/s at
119.8 ms needs 1.40 accepted per round, and slot 0 alone caps out near
`E[n] = P(slot 0) + P(slot 0) * P(slot 1) + ...`. With slot 0 at 0.74 and slots 1-3
at 0.08 the sum is 1.14, which is where the line above sits. Closing the last 2
tok/s needs the drafter to condition slot 1 on slot 0's token, which it cannot do:
the slots are parallel mask queries over one anchor. A tree over the four rows
would let the four rows cover 4 positions with alternatives instead of one
position each.

**Also in this round.** The speculative path ended in the same exit fault the
single-row path documents -- workers parked on raw huge-page buffers, static
destructors run underneath them -- so every spec run printed `rc=139` after its
last line of output. It is `_exit(0)` now, and the numbers above come from runs
that exit 0. `MEGA_TOKD=path` writes one int32 per emitted token from either decode
path so the two can be diffed.

### What speculative decode emits, and where it differs from plain greedy

The emitted tokens are the four-row verify's own argmax chain, so the draft
weights cannot move them. The four-row verify and the single-row path are still
different arithmetic: verify reads the int8 K plane `g_k8i`, single-row reads f16
K. `MEGA_TOKD` on both paths, same 10K prompt:

| | |
|---|---|
| identical tokens | 21 of the first 300 |
| first difference | token 12 (spec 3009, single 367) |

Divergence compounds, so 21/300 is really "agreed to 12, then the chains split".
The size of the underlying difference, at position 10000, row 0 of the verify
against the same position from the single-row path:

| | |
|---|---|
| cosine | 0.9883 |
| top-1 | 220 both, 15.43 against 16.00 |
| top-2 | 1207 both |
| max abs difference | 1.909 on a 26-logit span |

That is smaller than this build's own deviation from the frozen llama.cpp
reference, which is 0.978 cosine with `max|d|` 1.7-5.2 at the steps in the golden
table. So the speculative stream is the same model at int8-K resolution rather
than a different model, and a near-tie at token 12 is what a 1.9-logit difference
flips. Anyone quoting output equality between the two paths has to say which
position they compared: step 0 agrees, step 12 does not.

## The backup softmax inherited a loop bound from the stage that came after it

The neighbour-backup pass took the round from 119.9 to 113.3 ms, and the first
determinism checks -- 100 rounds at 2K and 150 at 10K, comparing token md5 -- passed.
They passed because they were too short to cross a window that opens about once in
400 rounds.

The audit that finds it hashes the round's own numbers instead of its tokens: an FNV
fold over the four rows of `g_logits4` and over the drafter's slot-0 logits, printed
per round under `MEGA_SPECLOG=1`, 400 rounds at 10K, backup on against backup off.

| 400 rounds, 10K | verify-logit hash differs | drafter-logit hash differs | token dump |
|---|---|---|---|
| backup on, before the fix, run 1 | 63/400, first at round 229 | 0/400 | `ec0b9e52...`, 843 tokens |
| backup on, before the fix, run 2 | -- | -- | `490e5fc0...`, 836 tokens |
| backup off | -- | -- | `490e5fc0...`, 836 tokens |
| backup on, after the fix (2 runs) | **0/400** | **0/400** | **`490e5fc0...`**, 436/1200 |

The drafter's logits never moved, so the proposal side was clean. The target side
moved, and at round 336 it moved enough to fork the emitted stream. Round 229 is the
useful one: same position, same drafts, same accepted count, same next token, and a
different hash over 4 x VOCAB logits. A tolerance would have hidden it.

**Cause.** `pfa_sm_i8` looped `for (gg = 0; gg < g_i8g; ++gg)` over the shared
`g_i8g`. `pfa_run` sets it to 1 for the four-row round and puts it back to `I8_G`
(16) the moment `run(task_attn4_a)` returns; the X100 pool is not a participant in
that barrier, by design. The window is therefore: the A thread takes the backup
plane, the layer finishes, the stage flips the count to 16, and the owning worker is
still inside the call. The remaining fifteen iterations write fifteen more group
blocks starting at `g_w4 + h*WBLK`: heads h+1..h+15, each fed from that head's own
score plane but with `L` shifted by 4..60 positions and the causal-tail zeroing
computed from the shifted length. A plane read inside the window hands a correct
head a wrong tail, and the PV sum carries it into every later layer.

Without the backup the window cannot open: the A thread cannot leave a head before
its owner has published, so no owner is ever inside the call when the count flips.
That is why the arm that had been running for weeks was clean and the new one was
not -- not a missing barrier, but a new path around one.

**Fix.** The count is an argument now, `pfa_sm_i8(..., int ng)`. The four decode call
sites (owner, backup, the A-side fallback, and the `MEGA_A4S=2` in-place softmax)
pass 1; the two prefill sites pass `g_i8g` at the call. Nothing in the decode path
reads a global that the next stage owns. Cost: nothing measurable -- the four arms
of the 400-round A/B above emit the same 836 tokens with the same 436/1200
acceptance.

**The general shape**, because this is what a working overlap hides. A generation
tag protects a *flag*: a stale publish is ignored because it carries an older
number. A loop bound has no generation to carry. Anything a late thread reads to
decide *how much* to do has to arrive as an argument or be snapshotted while the
caller still owns the stage.

## The neighbour backup charged the cluster's publish skew to the owner

With the fix in, the 400-round means at 10K are 121.5 and 121.7 ms with the backup
off and 115.5 and 114.9 with it on. The stage timers (`MEGA_M4TIME=1`) say where the
backup's own cost sits: in a quiet 60-round window the attention layer wall is
26.4 ms with the backup off and 32.6 ms with it at 80 us, and the takeover counter
fires 78 times a round -- while the A side's own wait is 0.011 ms. The backup was
scoring heads nobody was waiting for.

The deadline was measured from the pass's entry, and that timestamp is taken
*before* the wait for the neighbour's qk flag. A worker that arrives at the loop
before the A thread has published spends the whole 80 us on the flag wait and then
takes the head over with no lateness to point at. Starting the deadline at the
first sight of the flag -- the moment the owning worker could also have started --
cuts takeovers from 78 to 1.5 a round and the steady-state attention wall from 32.6
to 29.4 ms.

The right semantics is not the right schedule. Rescue latency is worth more than
duplicate work, and the deadline curve is flat only below 80 us:

| 400 rounds, 10K, backup deadline | mean ms/round | tok/s |
|---|---|---|
| off | 121.49 / 121.71 | 17.20 / 17.17 |
| 20 us | 113.81 | 18.36 |
| 40 us | 113.48 | 18.42 |
| **80 us (default)** | **114.29 / 113.64** | **18.29 / 18.39** |
| 150 us | 116.75 / 115.95 | 17.90 / 18.03 |
| 300 us | 119.43 / 120.36 | 17.50 / 17.37 |

20, 40 and 80 us are inside one round's noise of each other; 150 and up lose more to
rescue latency than they save on duplicate work. The default stays 80 us, now
measured from the flag. The deadline change alone is worth about 1.2 ms at the same
nominal value (115.5/114.9 before against 114.3/113.6 after).

Where the round stands at 10K, four rows, 1.09 accepted per round: draft 18.5,
verify GEMM 52.4, verify attention 29-33 of which 23.1 is QK+PV at the `vmadot`
ceiling, lm_head 7.4, elementwise 4.2, round 113.6-114.3 at 18.3-18.4 tok/s. The
6-9 ms between the attention's compute and its wall is now the largest identified
slack, and the GEMM's 24.2 against 27.5 GB/s is 6 ms behind it.

## Where the attention stage's 6 ms went: the pool was polling the wrong line

The stage decomposition added to `attn4p_run` measures four things per layer and
prints their sums per round: `ent` (release of the entry barrier, the latest stamp
at the top of the body), `acc` (qk + wait + pv inside the body), `body` (the largest
thread's whole body) and `skew` (stage wall minus body). At 10K with the backup
pass on, `ent 0.32 body 26.1 acc 26.1 skew 0.62`: the barriers are 0.9 ms of a
29 ms stage and the rest is inside the threads. The sum of qk and pv over all eight
threads is 22.5 ms, so the gap is not compute and not a barrier.

A per-layer hand-to-hand trace (`MEGA_A4TRACE=L`) prints, for one layer, the time
each head's QK was published, when its A thread started waiting, when its owner
published the weights, and when the PV ran. Layer 20 at 10K, 726 us wall:

| head group | QK published | weights published | A-side wait began |
|---|---|---|---|
| h0-3 | 113.7 us | 211.3 | 257.7 |
| h4-7 | 192.7 | 297.5 | 326.7 |
| h8-11 | 257.2 | 356.1 | 432.9 |
| h12-15 | 326.5 | 430.2 | 494.1 |
| **h5** | 192.7 | **530.6** | 326.7 |

Head 5's weights arrive 233 us after its own QK against 98 us of softmax work, and
its A thread waits 204 us of a 726 us layer. Its owner's own flag wait was under
1 us, so the delay is entirely *before* the worker notices: the pool's idle loop
polled `g_a4_gen`, which is written once per layer, while the two head flags that
worker is responsible for are written by its own A thread. A worker woken through
the scheduler had already lost a third of the layer.

Each worker now polls its own two head flags in the idle loop (bounded pause spin,
then one `sched_yield`, so the core still comes back to the board's daemons), and
dispatches on "my heads have unscored work" rather than on "a layer started". A
worker can no longer be a generation behind for a head that is waiting on it.

400 rounds at 10K, paired against the backup-on build before it:

| 400 rounds, 10K | mean ms/round | tok/s | tokens |
|---|---|---|---|
| before | 113.6 / 114.3 | 18.3 / 18.4 | 836 |
| **after** | **111.3 / 111.5** | **18.76** | 836, same md5 |

The bounded spin is not where the gain is: with `MEGA_A4WSPIN=0` (plain yield, the
old idle loop) the round is unchanged. What pays is the dispatch key. The spin is
kept because it costs nothing measurable and removes the scheduler from the wake.

## Two rescuers per head is not a plane the code can publish into

The residual 90 us a layer of A-side wait after the wake fix is a layer where the
owner and its one rescuer both lost their core. Giving each head two rescuers
(four backup slots instead of two, `MEGA_A4BKNB=2`) brought the stage to its
arithmetic floor -- `body 24.1` against a `qk + pv` sum of 24.4 -- and it is
**wrong**:

| 400 rounds, 10K | accepted | tokens | md5 |
|---|---|---|---|
| one rescuer | 436/1200 | 836 | `490e5fc0...` |
| two rescuers | **197/1200** | 597 | **`5ee8e190...`** |

The streams fork at token 41. `pfa_sm_i8` is not a pure writer: it clears a
sixteen-position tail block with `memset` before filling it, and both rescuers
write the same `g_w4b`. One rescuer's clear can land in a block the other has
already published, so a reader inside that window gets zeros. The window is also
widened by the pass itself: four backup slots at up to 300 us of flag wait plus
80 us of deadline can still be running when the next layer's QK starts overwriting
`g_sc4 + h*SC_GSTRIDE`.

The speed was not worth fixing: the two-rescuer round is 109.6 ms against the one
rescuer's 110.2 ms in the same A/B, i.e. **0.6 ms**, and the walk back to a correct
version needs a plane and a flag per rescuer rank. Reverted to one rescuer, which
is the invariant the original comment states: each head has exactly one rescuer, so
each plane has exactly one writer.

## 200 tok/s needs `-x100`, and the board hides it twice

A prefill run without `-x100` is 60.8 s (164.5 tok/s) and with it 49.8 s (200.65),
in the same build, back to back. Nothing regressed; the flag was simply missing, and
the X100 cluster is what takes GEMM from 39.2 s to 26.6 s:

| prefill at 10K | qkv | o | gate+up | down | GEMM | total |
|---|---|---|---|---|---|---|
| no `-x100` | 4.36 | 3.75 | 20.74 | 10.36 | 39.2 s | 60.78 s |
| `-x100` | 2.90 | 3.11 | 12.24 | 8.32 | 26.6 s | **49.83 s** |

The first `-x100` run of this build read 101 tok/s, which is a contended board and
not the code: a previous invocation's decode loop was still alive, so two processes
shared the sixteen cores. `pgrep -x m3c.dv` is the check, and it has to be empty
before any number is recorded. Every benchmark note in this file that quotes a
regression should be read as "unless a stray run was alive", because that failure
mode has now produced a false regression twice.

## The 200 tok/s prefill is unchanged by two weeks of decode work, and the

## 3e-4 cosine it lost is the decode V plane

The decode line went from 10.87 to 19.0 tok/s after the 200 tok/s prefill was
recorded, and the two share one X100 idle loop, so the prefill number needs a check
across that boundary rather than an assumption. Reference-locked logits make the
check exact: `-force refs/golden_10k_tokens.txt` walks the reference chain so every
step scores the prefix the reference scored, and `MEGA_STEPLOG` writes one 130560-wide
row per step.

| 10K, 64 forced steps, `-x100` | prefill | cosine | top-1 | rows md5 |
|---|---|---|---|---|
| recorded 2026-11-25, before the decode work | 49.907-50.148 s | 0.977316 | 60/64 | `46ee3c9b...` |
| today, default (int8 V on) | 49.839 s / 200.63 | 0.977015 | 60/64 | `4e0f7d81...` |
| today, `MEGA_V8=0` | 49.299 s / 202.83 | **0.977316** | 60/64 | `29724e42...` |

Prefill is inside the old spread and slightly faster at its best run, and turning
the decode V plane off restores the recorded cosine to its last digit. So the
3.0e-4 gap is the int8 V quantisation from the decode commit, matching the 4e-4
that entry measured, and nothing in the decode work moved the prefill.

Two things that look like findings and are not:

- The two rows md5s do not match because the dump is 64 separate rows and the
  historical `46ee3c9b` was recorded over the whole 64-step comparison, not one row.
  Cosine and top-1 are the comparable quantities.
- `MEGA_V8=0` reads 202.83 tok/s against 200.63, which is not the V plane making
  prefill faster: prefill does not read the decode V plane at all. It is the same
  board noise as the 49.91-50.15 s spread, and the run order.
