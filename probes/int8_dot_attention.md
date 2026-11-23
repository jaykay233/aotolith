# int8 dot attention: what is proven and what is left

Everything below is measured on the board (cores 8-15, all A100, `vlenb=128`),
not estimated. Reproduce with the file named next to each number.

## The operand map is solved (bit exact)

`pmap = {A: row*16 + d, B: col*16 + d}` with `lane = row*8 + col`, for
`vmadot v_acc, vA, vB` in `e32,m1`.

- A uses only bytes `[0,64)` (4 rows x 16); bytes `[64,128)` are ignored.
- B uses all 128 bytes (8 cols x 16).
- 16 products per lane, so **512 MAC per instruction**.
- Destination must be an **even** register, so only 8 accumulators are live
  (`v16..v30`). Odd destinations are rejected by the assembler.
- Checked against all 4x8x16 = 512 one-hot pairs: **512/512 exact**
  (`resolve_int8_dot_map.cpp`).
- One dot costs 0.628 ns, i.e. one instruction per cycle at 1.6 GHz.

## Kernel rates, single core, bit exact

| shape | rate | note |
|---|---|---|
| 4 rows x 32 cols | 94.8 GMAC/s | `bench_int8_gemm_4x32.cpp` |
| 8 rows x 32 cols | 93.1 GMAC/s | doubling rows does not help |
| 4 rows x 64 cols | **121.9 GMAC/s** | `bench_int8_gemm_4x64.cpp` |

All three are `0/256` or `0/512` wrong against a scalar reference, which also
confirms that `vl8r.v` needs an 8-register-aligned base (`v8`, not `v4`).

The loop is **load-issue bound, not dot bound**. Per k-step it pays two vector
loads at ~21 cycles each (~62 cycles) and only ~8 cycles of dots. That is why
8x32 (three loads) gains nothing over 4x32, and why 4x64 (two loads, 1024-byte
`vl8r`) is the best shape. 121.9 GMAC/s is the practical ceiling of this form
because 8 accumulators is the hard cap.

## DRAM is ~32 GB/s, and it sets the prefill floor

`bench_dram_bandwidth.cpp`, 512-byte `e8,m4` loads with four independent
accumulators, one process per core:

```
cpu 9 6.3  cpu 10 4.1  cpu 11 3.9  cpu 8 3.9
cpu 12 3.5 cpu 13 3.6  cpu 14 3.6 cpu 15 3.3   -> ~32 GB/s aggregate
```

An earlier version that put a dependent add after every 32-byte load reported a
flat 2.5 GB/s per thread. That number was an ALU chain, not memory; ignore it.

Q4_0 weights are 1.265 GB, and prefill streams them once, so the GEMM floor is
**39.5 s**. The whole 200 tok/s budget at 10K is 51.2 s, which leaves **11.7 s
for attention** (9.01 TMAC), i.e. 770 GMAC/s on 8 cores.

## The attention prototype works, and one hotspot remains

`proto_int8_dot_attention.cpp` runs full int8 QK + softmax + PV for a 16-query
chunk against a scalar reference.

- PV: **115.2 GMAC/s/core** (reaches the GEMM ceiling).
- QK: only **28.6 GMAC/s/core**.
- Softmax: negligible once vectorised.

The QK gap is an epilogue problem, not a dot problem. On this core **every
vector instruction costs 8-11 cycles regardless of `vl`**, because the register
file is 1024 bits but the ALU datapath is 128 bits. The prototype's epilogue
uses `vl=8` convert+scale+store per (accumulator, row), i.e. 32 groups of four
instructions, ~1000 cycles per 32768 MAC — more than the dot loop itself.

Fix, not yet applied: store each accumulator with one 32-lane `vrgather` (to
`[pos][row]`) plus one 32-lane `vse32` (16 instructions per block instead of
128), then let softmax read row `r` with a single `vlse32` stride-4 load. That
should put QK near the PV rate.

## Consequences for the target

With QK and PV both near 115 GMAC/s, attention is 9.01 TMAC / (115 x 8) ~
9.8 s plus ~1.5 s of softmax, i.e. ~11.3 s, against an 11.7 s budget. Total
prefill ~= 39.5 + 11.3 = 50.8 s ~= 202 tok/s. That clears 200 with no margin,
so the weight format has to shrink to make it robust: 3-bit weights (1.001 GB)
move the floor to 31.3 s and give ~250 tok/s.

## Quantisation accuracy caveat

In the prototype the scores are quantised with a *global* K scale and the output
error is ~1.1e-2 absolute against a peak of 6e-3. That looks alarming but the
peak is tiny because the test Q/K are random and the softmax averages thousands
of near-cancelling V values. What matters for greedy alignment is absolute error
in the residual stream, so this needs checking against the real model, not the
synthetic one. Per-position K scales would improve it if needed.
