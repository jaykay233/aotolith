# Where 200 tok/s actually stands: measured rates and the corrected budget

All numbers below are measured on the board, cores 8-15 (all A100, `vlenb=128`).
The file named next to each figure reproduces it.

## Correction to the earlier budget

I previously wrote that the 1.265 GB of Q4_0 weights take "39.5 s" to stream at
32 GB/s. That was a unit slip: 1.32 GB / 32 GB/s is **41 milliseconds**. The real
model file is 1,421,419,488 bytes. So at pp10240 the weight stream is
**irrelevant** (44 ms) and prefill is purely **compute bound**. Decode is the
opposite: one token streams every weight, so 44 ms is the whole budget there.

## The int8 dot beats the int4 dot on arithmetic density

The 413.8 GMAC/s int4 figure from `bench_dot_peak.cpp` is raw `vmadotsu i4` on a
single operand pair. A real int8-activation x int4-weight product needs
`vupack` + `vsll` + two macs per 1024 products, i.e. ~341 products per
instruction. int8 x int8 does **512 products per instruction** with one
`vmadot`. Since prefill is compute bound, density wins over compactness, so the
GEMM should run int8 x int8.

## GEMM: 680 GMAC/s aggregate, and loop order is worth 2x

4 rows x 64 cols per pass, `vl8re32.v` for a 1024-byte weight load,
8 accumulators (`vmadot` requires even destinations, so 8 is the hard cap).
`bench_int8_gemm_real_shapes.cpp`:

| configuration | rate |
|---|---|
| 1 core, M=2048 N=2048 K=2048, weights resident | **151.4 GMAC/s** |
| 1 core, same but `for mb { for nb }` (weights restreamed 64x) | 74.0 GMAC/s |
| 8 processes, M=2048 N=256 K=2048 | **85 GMAC/s each, 680 aggregate** |

The `for nb { for mb }` order is the whole difference: it keeps a 64-column
weight tile resident across every row group instead of re-reading it per group.
Getting this backwards halves-to-quarters the rate, which is very likely what
the vendor kernel's 45 GMAC/s is suffering from.

## Attention: both halves measured

`int8_dot_attention.md` has the details; the operand map is bit exact over all
512 one-hot pairs.

| stage | rate |
|---|---|
| QK dot loop | 101-122 GMAC/s |
| PV | 115 GMAC/s |
| softmax / repack glue | ~3.4 s at 10K |

## Budget with those rates

MiniCPM5-2B is 1.98 GMAC/token (47.2 MMAC per layer x 42), so 10K prefill is
20.3 TMAC of GEMM plus 9.01 TMAC of attention.

```
GEMM   20.3 TMAC / 680 GMAC/s = 29.8 s
attn   QK 5.6 + PV 4.9 + softmax 3.4 = 13.9 s
total  43.7 s  ->  234 tok/s        (target 200)
```

Decode at 10K, 50 ms/token budget:

```
Q4_0 (1.42 GB): 44.4 stream + 5.7 attn + 7.1 other = 57.2 ms -> 17.5 tok/s
3-bit (1.10 GB): 34.4 + 5.7 + 7.1 = 47.2 ms -> 21.2 tok/s
```

So prefill 200 is reachable with what is already measured, but decode needs the
weights below about 4 bit/param. That is a separate, independent requirement.

## What this changes about priorities

The GEMM rewrite is now the single biggest lever (57.7 s with the vendor kernel
at 45 GMAC/s, versus 29.8 s at 680 GMAC/s aggregate). Attention is second at
13.9 s. Neither depends on the `vupack` semantics I was blocked on, because
int8 x int8 needs no unpacking at all.
