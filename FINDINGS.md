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

Short context (101-token prompt), 3 drafts:

| | accepted/round | ms/round (mean) | tok/s |
|---|---|---|---|
| four rows, no drafter | - | 96 | 10.4 |
| dspark | 0.278 | 106 | **17.3** |
| dspark, 10K | 0.111 | 310 | 4.3 |
| dspark, 10K (8 rounds) | 0.250 | 299 | 5.6 |
| four rows, no drafter, 10K | - | 96 | **10.4** |

So speculation is worth 1.7x at a hundred tokens and costs 2x at ten thousand.
Both halves of that are measured, not modelled:

* The drafter itself is 24-26 ms per round at 101 tokens and **190-196 ms at
  10K**. Its attention is fp32, O(context), over 64 (slot, head) units, which is
  the same shape that costs the target 30 ms for four rows.
* Acceptance collapses from 0.28 to 0.11-0.25 per round. The 1.26/round that
  `mega_decode` measures at short context does not survive to 10K.

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
| halves-rotate rope (`MEGA_DSROPE=half`) | 0.111 | falsified; the draft does not use the target's rope layout |
| `target_layers` read 1-based (`MEGA_DSL1`) | 0.292 | indistinguishable from 0.250; not the bug |
| encoder/output norms swapped (`MEGA_DSSWAPNORM`) | 0.000 | confirms `enc.output_norm` is the encoder's and `output_norm` is the decoder's |
| interleaved feature layout (`MEGA_DSINTER`) | 0.000 | confirms layer-major; the fused vector is 5 x 2048 in layer order |
| tanh on the rank-256 gather (`MEGA_DSTANH`) | 0.292 | no effect: this variant is the plain markov, as `dspark_joint_proj` being absent implies |
| encoder features zeroed (`MEGA_DSZERO`) | 0.125 | **the control that matters**: zeroing the entire committed prefix only halves acceptance |
| markov bias dropped (`MEGA_DSNOBIAS`) | 0.000 | the base logits alone are worse than base + bias, so the markov head is helping |

The zero-feature control is the sharpest number here. If the prefix reached the
draft properly, removing it should cost almost everything. It costs half, which
says the drafts are being driven mostly by the anchor token and the mask block's
own self-attention rather than by the injected context.

### What is left for decode

- The primary decode number is **0.11-0.25 accepted/round at 10K, not 1.8**.
  Nothing about the arithmetic changes until that is understood: at 96 ms per
  verification round even 2.5 accepted tokens per round only reaches 25 tok/s,
  and the round would have to pay for a drafter that currently costs twice what
  it saves.
- The sharper probe than acceptance is the distribution itself: the draft's base
  logit row at a position where the target's own row is known, correlated
  against it. Acceptance is a 0/1 summary of a comparison that should be a
  cosine.
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
| real Q3_K via int8 copy (vmadot) | 100.4 s (99.6) | unverified |

| `prefill` 3-bit via the int8 copy (vmadot) | 10K 100.4 s, 99.6 tok/s | - | 40 tok/s slower than 4-bit prefill; closed by the quality result in section 7 |

**The 174-178 tok/s figure is the unaligned row.** It is the vmadot path with
per-row int8 activation scaling and A_BLK 256; it emits a different first token.
The fastest configuration that matches the reference is 139.4 tok/s, and that is
the number to compare against 200.

The floor: GEMM 41 s + QK 20 s + PV 14 s of MAC work at the measured per-core
rates is ~61 s, plus conv, quant, norm, silu and softmax. 200 tok/s needs 51.2 s,
so **the A cluster alone cannot do it**; roughly 30-40% of the MACs have to move
to X100.

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
| prefill 200 tok/s @10K | 139.4 (73.4 s) | 51.2 s | X100 with the hp A record and per-K32 fold. Nothing else reaches it: the A-cluster MAC floor is ~61 s. |
| decode 20 tok/s @10K | 11.47 single row; four-row round now ~90 ms | 50 ms | 54.3 GEMM (DRAM floor) + 29.3 attention (`MEGA_ATTN4_ROWS=4 -e 1 MEGA_ATTN4P=1`) + ~6 four rows of tail. That clears 20 tok/s at 1.84 accepted tokens per round and short context measured 1.83, so the chain sits on the line. Remaining work is integration, not kernels: m=4 in mega3's decode GEMM, the dspark drafter, and the four-row attention hooked to the live decode KV. Then measure 10K acceptance, and it is acceptance rather than kernel speed that decides. Byte arithmetic still rules out weight format alone: 3-bit only gets to 12.36 and costs 0.44 cosine. |
| output alignment | Q4_0 matches exact fp32 (0.9990 cosine, ids `35 242 38 35`) | - | keep Q4_0 as the reference configuration. No 3-bit variant is a candidate. |

Acceptance test for the rest of the work: `ref_f32.py` on the same prompt, not
llama.cpp Q4_0 (0.66 cosine from fp32) and not llama.cpp F16 (which does not run
on this board). `ref_f32.py` now handles Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, so any file on
the board can be the judge.

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

