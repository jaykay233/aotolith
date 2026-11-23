# checkpoint: 174.4 tok/s prefill @ pp10240 (2026-11-22)

build: g++ -O1 -std=c++20 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba_xsmtvdotii
       mega3.cpp ime2.o rvv.o -lpthread -lm          (-O2 miscompiles RVV intrinsics)

run:   ./mega3 -A 10240 -J -K      attention  31.31 s  [qk 132.8 + pv 111.6 core-s]
       ./mega3 -P 10240 -I      gemm       27.41 s  (gemm 24.73 + quant 2.68, 821 GMAC/s)
       total 58.72 s = 174.4 tok/s

numerics: worst |diff| 4.554e-03 vs f32 reference, out[0]=1.001167, repeatable

WHAT 58.72 s IS, AND IS NOT. It is the sum of two separate phase benchmarks in
two separate processes, both driven by synthetic activations: -A builds Q and a
42-layer KV from sinf/cosf, -P streams the real repacked weights against
synthetic activations. Both are honest throughput measurements of the kernels at
the right shapes, and the real weights do flow through the GEMM path. But no
single process has ever run 10240 real tokens through 42 layers and been timed
end to end, so 174.4 tok/s is a model, not a measurement. A true e2e number is
still owed for the "measured with a reproducible command" requirement, and the
gap could go either way: a real run adds norms, activation quantisation, layer
transitions and residual traffic that this sum never pays, but it also keeps the
weights and the int8 planes hot across layers instead of re-warming them per
process. mega_decode.cpp is the shape of the missing runner -- its layer body is
already the full model for M=1.

flags: -J int8 attention  -K cross-cluster softmax pipeline  -I int8 GEMM weights
what this state contains, on top of the 131 tok/s O1 baseline:
  - -K: 8 X100 cores run softmax (e16m4 max / e32m2 exp, 123 core-s) behind the
    A100 cores' QK+PV; workers are created before the main thread writes
    /proc/set_ai_thread, and bind_cpu verifies vlenb==128 or aborts
  - madvise(MADV_HUGEPAGE) on the int8 KV planes (g_k8i/g_v8i), the pipeline
    slots and g_psa16: physical page colouring was making PV bimodal
    (110 vs 195 core-s on the same binary), now pinned at ~112
  - per-layer KV requantise parallelised over 8 cores (was scalar, single
    threaded): 6.3 s -> 1.4 s

known wall: attention dots run at 26-33% of the dot unit while the GEMM reaches 78%.
  GEMM (gemm_i8d_4x64)   102.6 GMAC/s/core = 9.0 cycles/vmadot  (peak 131 = 7)
  attention QK            34.0 GMAC/s/core = 27.1 cycles/vmadot
  attention PV            40.4 GMAC/s/core = 22.7 cycles/vmadot
  Closing that 2.4-3x puts attention at ~11 s and the end-to-end at ~40 s = 258 tok/s.

  BUILD BUG THAT FAKED THREE RESULTS (2026-11-23). Build steps were run as
    g++ ... -o mega3p mega3.cpp ime2.o rvv.o 2>&1 | grep error | head -5
    echo BUILT
  The pipeline returns head's exit status, and the assembler writes "Error:"
  with a capital E, so every assembly failure vanished into the pipe while
  BUILT still printed. The binary stayed at its previous version, and a binary
  that does not know a flag ignores it -- so each flagged experiment measured
  the control and reported "no change". Three entries below were produced that
  way; two of the three were real effects, one with the opposite sign:
    -c 8  chunking : reported "no effect 110"  -> actually PV 164.65 (+44%)
    -H    head-major: reported "no effective 112" -> actually 28.98 s (-8.5%)
    -L    vle32    : reported "no effect 110"  -> actually PV 110.88 (-3%)
  Rule: check $? and grep case-insensitively, or a change that never compiled
  is indistinguishable from a change that did nothing.

  Also never compiled, now deleted:
    pv8x2  used `vmadot v17/v19/...` -- odd vmadot destinations are rejected.
    pv4_split (-s) used `vl1re32.v v9,128(t5)` -- this assembler accepts a vector
      load only with zero displacement (verified with a one-instruction probe:
      vle32.v v9,(t5) ok, vle32.v v9,128(t5) error, for vle32/vle8/vl8re32).
  Any number previously quoted for those two is void.

  What did NOT explain it (all measured, all no-ops):
    - data source: V read from the weight blob instead of g_v8i   106.5 vs 113
    - cache: alias every V block onto one 1 KB L1-resident block  110 vs 111
    - kernel text: pv4 rewritten as a line-for-line copy of
      gemm_i8d_4x64 (chained "+r" pointers, unique label, addi)   109.8 vs 111
    - memory type: stage the V range into TCM (per-core scratchpad) 110 vs 111
    - dataflow inversion: V resident / w8 streaming, 3 variants   110/110/113
    - clock: scaling_cur_freq during each phase                   1.8 GHz throughout
  Six real no-ops, each within +-3%. probe_dot2.cpp was written to hunt this and
  is NOT trustworthy: it tops out at 20.9 GMAC/s/core, below the real kernel's
  40.4, so it measures a pathological path and cannot arbitrate.

  What DID move it:
    - head-major tile order (-H): attention 31.67 -> 28.98/29.13 s, PV 114 -> 93.7
      Head-fast made a thread cycle all sixteen heads, so eight threads held
      every head's KV live (21 MB). Head-major keeps each thread inside one
      head's KV and the threads on a kv plane share those lines. -H now holds the
      best measured attention time.
    - plain vle32.v instead of vl1re32.v (-L): PV 114 -> 110.88 (-3%), small but
      repeatable, and now that the instruction form is fixed it is a real effect.
    - chunking the PV loop (-c 8) is 44% WORSE: the prologue and epilogue are
      real and the loop body is the part that is insensitive to everything above.

  The one invalid control: "hoist both loads out of the loop" appeared to give
  13.5x, but it completes only ~500 of the 4500 GMAC of PV work and its answer is
  wrong by 3.0, so it is not a like-for-like comparison. Do not cite it.

## 2026-11-23 update: 182.8 tok/s, and what the build bug cost

After the build bug above was found and the two dead asm functions deleted, the
three "no effect" experiments were re-run for real. Two of them were wins:

    head-major tile order (-H, now default)   attention 31.70 -> 28.94 s
                                              PV 114.8 -> 93.4 core-s
    vle32.v loads (-L, now default)           PV 114.8 -> 110.9 on its own
    PV loop chunking (-c 8)                   PV 114.8 -> 164.7, 44% WORSE

Clean serial runs, one process at a time (concurrent runs pollute each other
badly on a 16-core board and were discarded):

    flags        attention      pv        numerics
    none         31.704 s      114.78   4.554e-03
    -H           29.604 s       96.22   4.554e-03
    -H -L        29.195 s       93.59   4.554e-03
    default now  28.938 s       93.41   4.554e-03  repeatable=yes
    -h0 -l0      31.584 s      113.07   4.554e-03

    GEMM (-P 10240 -I)  27.082 s  (gemm 24.395 + quant 2.687)  831.9 GMAC/s

    end to end: 28.938 + 27.082 = 56.02 s = 182.8 tok/s  (was 173.8)
    200 tok/s needs 51.2 s, so the remaining gap is 4.8 s.

Why head-major works, since every other cache hypothesis had been excluded:
the tile order was head-fast, so a thread cycled through all sixteen heads and
the eight threads between them kept every head's KV live (21 MB). Head-major
lets a thread work inside one head's KV, and the threads sharing a kv plane hit
the same lines. It is a working-set effect, and it was dismissed earlier only
because the build never compiled the flag that tested it.

Both wins are now defaults with opt-outs (-h0, -l0). Neither changes a reduction
order, so numerics are unchanged: tile order only permutes which tile runs when,
and vle32.v moves the same bytes as vl1re32.v.

Still owed: a true end-to-end prefill run (one process, 10240 real tokens, all 42
layers). 182.8 is the sum of two phase benchmarks in two processes.
