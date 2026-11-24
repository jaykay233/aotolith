#!/usr/bin/env python3
"""Layer-0 reference in numpy, for diffing against the megakernel's dumps.

With MEGA_DUMP=1 the megakernel writes /tmp/pf_00v.bin (V projection, last
row), /tmp/pf_q.bin (Q after rope, every row), /tmp/pf_00o.bin (attention
output, last row) and /tmp/pf_00a.bin (post-attention residual, last row).
This script recomputes those four quantities from the same GGUF in float32 and
prints both sides, so a mismatch names the operation instead of the layer.
"""
import os
import struct
import sys

import numpy as np

N_EMBD, N_HEAD, N_KV, HEAD_DIM, N_FF = 2048, 16, 2, 128, 6144
ROPE_BASE, EPS = 5000000.0, 1e-6

FIXED = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
         6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


def load_gguf(path):
    f = open(path, 'rb')

    def rd(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def rs():
        n = rd('<Q')
        return f.read(n).decode(errors='replace')

    def val(t):
        if t == 8:
            return rs()
        if t == 9:
            at = rd('<I')
            n = rd('<Q')
            for _ in range(n):
                val(at)
            return None
        return rd(FIXED[t])

    rd('<I')
    rd('<I')
    nt, nkv = rd('<Q'), rd('<Q')
    for _ in range(nkv):
        rs()
        val(rd('<I'))
    tensors = {}
    for _ in range(nt):
        name = rs()
        nd = rd('<I')
        dims = [rd('<Q') for _ in range(nd)]
        ty = rd('<I')
        off = rd('<Q')
        tensors[name] = (ty, dims, off)
    # Tensor offsets are relative to the data section, which starts at the first
    # 32-byte boundary after the info table; using them as file offsets reads
    # header bytes as f16 scales and every downstream value turns into NaN.
    f.seek((f.tell() + 31) // 32 * 32)
    data = f.tell()
    return f, tensors, data


def deq_q4_0(f, off, nrows, k):
    f.seek(off)
    raw = np.frombuffer(f.read(nrows * (k // 32) * 18), dtype=np.uint8)
    blk = raw.reshape(nrows, k // 32, 18)
    sc = blk[:, :, :2].copy().view('<f2').astype(np.float32).reshape(nrows, k // 32, 1)
    q = blk[:, :, 2:]
    lo = (q & 0x0F).astype(np.float32) - 8.0
    hi = (q >> 4).astype(np.float32) - 8.0
    out = np.concatenate([lo, hi], axis=2) * sc
    return out.reshape(nrows, k)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/dev/shm/mc-q4o.gguf'
    ids = [int(x) for x in open(sys.argv[2] if len(sys.argv) > 2 else '/tmp/p32.txt').read().split()]
    # The megakernel prefills all but the last id and keeps the last as the decode
    # anchor, so row cm-1 is the second-to-last id; drop it or every comparison is
    # shifted by one position.
    # PF=dc compares the decode step, which does consume the last id (as the
    # anchor), so only the prefill comparison drops it.
    if os.environ.get('PF') != 'dc':
        ids = ids[:-1]
    f, T, data = load_gguf(path)
    T = {k: (v[0], v[1], data + v[2]) for k, v in T.items()}
    n = len(ids)

    def get(name, rows, k):
        ty, dims, off = T[name]
        assert ty == 2, (name, ty)
        return deq_q4_0(f, off, rows, k)

    def get_f32(name, cnt):
        ty, dims, off = T[name]
        assert ty == 0, (name, ty)
        f.seek(off)
        return np.frombuffer(f.read(cnt * 4), dtype=np.float32).copy()

    emb = get('token_embd.weight', 130560, N_EMBD)
    x = emb[ids].astype(np.float32)

    an = get_f32('blk.0.attn_norm.weight', N_EMBD)
    rms = np.sqrt((x * x).mean(axis=1, keepdims=True) + EPS)
    xn = x / rms * an

    wq = get('blk.0.attn_q.weight', N_HEAD * HEAD_DIM, N_EMBD)
    wk = get('blk.0.attn_k.weight', N_KV * HEAD_DIM, N_EMBD)
    wv = get('blk.0.attn_v.weight', N_KV * HEAD_DIM, N_EMBD)
    q = xn @ wq.T
    k = xn @ wk.T
    v = xn @ wv.T

    pos = np.arange(n, dtype=np.float32)[:, None]
    i = np.arange(HEAD_DIM // 2, dtype=np.float32)
    th = pos * (1.0 / np.power(ROPE_BASE, 2.0 * i / HEAD_DIM))
    c, s = np.cos(th), np.sin(th)

    def rope(t):
        t = t.reshape(n, -1, HEAD_DIM)
        a, b = t[..., :64], t[..., 64:]
        return np.concatenate([a * c[:, None, :] - b * s[:, None, :],
                               a * s[:, None, :] + b * c[:, None, :]], axis=-1)

    q = rope(q) / np.sqrt(HEAD_DIM)
    k = rope(k)

    qh = q.reshape(n, N_HEAD, HEAD_DIM)
    kh = k.reshape(n, N_KV, HEAD_DIM)
    vh = v.reshape(n, N_KV, HEAD_DIM)
    rep = N_HEAD // N_KV
    kk = np.repeat(kh, rep, axis=1)
    vv = np.repeat(vh, rep, axis=1)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    out = np.zeros((n, N_HEAD, HEAD_DIM), dtype=np.float32)
    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
    for h in range(N_HEAD):
        # q already carries 1/sqrt(head_dim) (the megakernel pre-scales it so the
        # int8 attention operand is ready to dot), so the score needs no second
        # factor; applying one here flattens the softmax and blends positions.
        sc = (qh[:, h, :] @ kk[:, h, :].T) + mask
        sc -= sc.max(axis=1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(axis=1, keepdims=True)
        out[:, h, :] = p @ vv[:, h, :]
    ao = out.reshape(n, N_EMBD)

    wov = get('blk.0.attn_output.weight', N_EMBD, N_EMBD)
    attn_out = ao @ wov.T
    resid = x + attn_out

    def cmp(tag, ref, ours_path, row=None):
        """Compare one dumped vector against the numpy reference.

        Everything here is compared for the last real row of the chunk, which is
        what the megakernel's dump_row writes (row cm-1).
        """
        name = os.path.basename(ours_path)
        try:
            a = np.fromfile(ours_path, dtype=np.float32).astype(np.float64)
        except OSError:
            print('  %-10s missing %s' % (tag, name))
            return
        if a.size % ref.size == 0 and a.size != ref.size:
            a = a.reshape(-1, ref.size)
            a = a[-1] if row is None else a[row]
        a = a.reshape(-1)
        r = ref.reshape(-1).astype(np.float64)
        if a.shape != r.shape:
            print('  %-10s shape %s vs ref %s' % (tag, a.shape, r.shape))
            return
        d = np.abs(a - r)
        na, nr = np.linalg.norm(a), np.linalg.norm(r)
        print('  %-10s ours|%.4f| ref|%.4f| cos %.6f  max|d| %.5f  mean|d| %.6f'
              % (tag, na, nr, float(a @ r) / (na * nr), d.max(), d.mean()))

    print('tokens %d' % n)
    # The V dump is the first row of the chunk (dump_n writes g_pfv from index 0)
    # while the o/a dumps are row cm-1, so the two families take different rows.
    # MEGA_DUMP=1 on the decode path writes the same quantities under dc_*, for
    # the single row the decode step runs. Point PF at the prefix to compare the
    # decode kernel against the same numpy reference as prefill's.
    pf = os.environ.get('PF', 'pf')
    if pf == 'dc':
        cmp('emb', x[-1], '/tmp/dc_00e.bin')
        cmp('norm', xn[-1], '/tmp/dc_00n.bin')
        cmp('V(proj)', v[-1], '/tmp/dc_00v.bin')
        cmp('K(rope)', k[-1], '/tmp/dc_00k.bin')
        cmp('attn-raw', ao[-1], '/tmp/dc_00i.bin')
        cmp('resid', resid[-1], '/tmp/dc_00a.bin')
        return
    cmp('V(proj)', v[0], '/tmp/pf_00v.bin')
    cmp('Q(rope)', q[0], '/tmp/pf_q.bin', row=0)
    cmp('attn-raw', ao[-1], '/tmp/pf_00o.bin')
    cmp('attn+o', attn_out[-1], '/tmp/pf_00o.bin')
    cmp('resid', resid[-1], '/tmp/pf_00a.bin')


if __name__ == '__main__':
    main()
