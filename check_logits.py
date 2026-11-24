#!/usr/bin/env python3
"""Score a teacher-forced megakernel run against the frozen llama.cpp golden.

Both sides are flat float32 arrays of [step][VOCAB]:

  golden   llama.cpp (`refgreedy`, REF_STEPLOG) walking its own greedy chain
  ours     `m3c -force refs/golden_10k.txt` (MEGA_STEPLOG) walking the same chain

Teacher forcing is what makes this a precision measurement rather than a
divergence hunt: a single near-tie would otherwise split the two chains and every
later step would be comparing different prefixes. With the chain pinned, each row
scores the same position of the same prefix.

Reported per step: cosine, top-1 agreement, and the top-1 margin ratio, so a
near-tie resolved the other way is distinguishable from lost accuracy.
"""
import argparse
import struct
import sys

VOCAB = 130560


def load(path, n, vocab=VOCAB):
    with open(path, "rb") as f:
        raw = f.read()
    want = n * vocab * 4
    if len(raw) < want:
        sys.exit(f"{path}: {len(raw)} bytes, need {want} for {n} rows of {vocab}")
    return [struct.unpack(f"<{vocab}f", raw[i * vocab * 4:(i + 1) * vocab * 4])
            for i in range(n)]


def top2(row):
    i0 = 0
    for i in range(1, len(row)):
        if row[i] > row[i0]:
            i0 = i
    i1 = -1
    for i in range(len(row)):
        if i != i0 and (i1 < 0 or row[i] > row[i1]):
            i1 = i
    return i0, row[i0], i1, row[i1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("ours")
    ap.add_argument("-n", type=int, default=0, help="rows to score (0 = all in golden)")
    a = ap.parse_args()

    import os
    n = a.n or (os.path.getsize(a.golden) // (VOCAB * 4))
    g = load(a.golden, n)
    o = load(a.ours, n)

    print(f"{'step':>4} {'cosine':>9} {'top1':>6} {'margin_ratio':>12} "
          f"{'max|d|':>8} {'logit_range':>11}")
    tot_cos = 0.0
    agree = 0
    for s in range(n):
        gr, orow = g[s], o[s]
        dot = sum(x * y for x, y in zip(gr, orow))
        ng = sum(x * x for x in gr) ** 0.5
        no = sum(y * y for y in orow) ** 0.5
        cos = dot / (ng * no) if ng > 0 and no > 0 else 0.0
        g0, gv0, g1, gv1 = top2(gr)
        o0, ov0, o1, ov1 = top2(orow)
        # How much of the reference's own top-2 gap survives. 1.0 means our
        # ordering matches with the same separation; <0 means we inverted it.
        g_gap = gv0 - gv1
        ratio = (gv0 - gr[o1]) / g_gap if g_gap > 0 else float("nan")
        if o0 == g0:
            agree += 1
        rng = max(gr) - min(gr)
        md = max(abs(x - y) for x, y in zip(gr, orow))
        if s < 16 or o0 != g0:
            print(f"{s:>4} {cos:>9.6f} {'OK' if o0 == g0 else 'MISS':>6} "
                  f"{ratio:>12.4f} {md:>8.3f} {rng:>11.3f}")
        tot_cos += cos
    print(f"\nmean cosine {tot_cos / n:.6f}   top-1 agreement {agree}/{n}")


if __name__ == "__main__":
    main()
