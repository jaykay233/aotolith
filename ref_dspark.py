#!/usr/bin/env python3
"""Float32 reference for the DSpark drafter's forward pass.

The drafter decides the decode rate, and its own accuracy is only observable
through acceptance, which mixes the draft's forward with the markov head, the
four-row verifier, and the target's quantisation. This script removes all of that
by taking the kernel's own captured target features (MEGA_DSHID) and computing the
draft forward in numpy, so the two logit rows answer the same question: does the
draft's 5-layer block, given identical inputs, produce the kernel's logits?

Inputs
  ref_dspark.py dspark-q4.gguf mc-q4o.gguf /tmp/ds_hid.bin
  env DRAFT_K (default 3), MASK_TOK (default 75982)

Writes /tmp/ref_dslog.bin -- DS_SLOT x VOCAB float32, raw logits before the markov
bias -- and prints the top few. MEGA_DSLOGD writes the kernel's side of the same.
"""
import struct, sys, os
import numpy as np

DRAFT = sys.argv[1] if len(sys.argv) > 1 else "dspark-q4.gguf"
TARGET = sys.argv[2] if len(sys.argv) > 2 else "mc-q4o.gguf"
HID = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ds_hid.bin"

VOCAB = 130560
N_EMBD, N_FF, NH, NKV, HD = 2048, 6144, 16, 2, 128
DS_LAYER, DS_SLOT, RANK = 5, 4, 256
ROPE_BASE, EPS = 5e6, 1e-6
MASK_TOK = int(os.environ.get("MASK_TOK", "75982"))

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.f.read(4)
        self.ver = self.u32()
        nt, nkv = self.u64(), self.u64()
        for _ in range(nkv):
            self.st(); self.val(self.u32())
        self.T = {}
        for _ in range(nt):
            name = self.st(); nd = self.u32()
            dims = [self.u64() for _ in range(nd)]
            ty = self.u32(); off = self.u64()
            self.T[name] = (dims, ty, off)
        base = self.f.tell()
        self.base = base + (-base) % 32

    def rd(self, n):
        return self.f.read(n)

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def st(self):
        n = self.u64(); return self.f.read(n).decode("utf8", "replace")

    def val(self, t):
        if t == 8:
            return self.st()
        if t == 9:
            et = self.u32(); n = self.u64()
            return [self.val(et) for _ in range(n)]
        return struct.unpack("<" + FMT[t], self.rd(SZ[t]))[0]

    def row(self, name, r0, r1):
        """Dequantise output rows r0..r1-1 into (n, in) float32."""
        dims, ty, off = self.T[name]
        nin = dims[0]
        n = r1 - r0
        if ty == 0:
            self.f.seek(self.base + off + r0 * nin * 4)
            return np.frombuffer(self.rd(n * nin * 4), "<f4").reshape(n, nin).astype(np.float32)
        if ty == 1:
            self.f.seek(self.base + off + r0 * nin * 2)
            return np.frombuffer(self.rd(n * nin * 2), "<f2").reshape(n, nin).astype(np.float32)
        if ty == 2:
            nblk = nin // 32
            self.f.seek(self.base + off + r0 * nblk * 18)
            b = np.frombuffer(self.rd(n * nblk * 18), np.uint8).reshape(n, nblk, 18)
            d = b[:, :, :2].copy().view("<f2").astype(np.float32)
            qs = b[:, :, 2:]
            lo = (qs & 0xF).astype(np.float32) - 8.0
            hi = (qs >> 4).astype(np.float32) - 8.0
            return (np.concatenate([lo, hi], axis=2) * d).reshape(n, nin)
        raise SystemExit("type %d not handled for %s" % (ty, name))

    def full(self, name):
        dims = self.T[name][0]
        return self.row(name, 0, dims[1] if len(dims) > 1 else 1)


def rmsnorm(x, w):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS)
    return x * r * w


def rope(v, pos):
    """LLAMA_ROPE_TYPE_NORM: pairs (2i, 2i+1)."""
    inv = ROPE_BASE ** (-(np.arange(HD // 2) * 2.0) / HD)
    th = pos * inv
    c, s = np.cos(th), np.sin(th)
    x = v.reshape(v.shape[:-1] + (HD // 2, 2))
    a, b = x[..., 0], x[..., 1]
    return np.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(v.shape)


raw = open(HID, "rb").read()
hdr = np.frombuffer(raw[:24], np.int32)
nlay, npref, width, anchor, pos0, np2 = [int(x) for x in hdr]
assert nlay == DS_LAYER and width == N_EMBD and npref == np2 and pos0 == npref, (hdr,)
hid = np.frombuffer(raw[24:24 + 4 * DS_LAYER * npref * N_EMBD], np.float32).reshape(
    DS_LAYER, npref, N_EMBD).astype(np.float64)
print("features: %d rows, anchor %d, pos0 %d" % (npref, anchor, pos0))

d = GGUF(DRAFT)
t = GGUF(TARGET)

w = {}
def W(name):
    if name not in w:
        w[name] = d.full(name).astype(np.float64).T          # (in, out)
    return w[name]

# ---------------------------------------------------------------- 1. encoder
feat = np.concatenate([hid[L] for L in range(DS_LAYER)], axis=1)   # (npref, 10240) layer-major
fc = d.full("fc.weight").astype(np.float64).T                      # (10240, 2048)
g = rmsnorm(feat @ fc, d.full("enc.output_norm.weight").astype(np.float64))
print("g: |g| mean %.4f" % np.linalg.norm(g, axis=1).mean())

# ------------------------------------------------------- 2. inject prefix K/V
kc = np.zeros((DS_LAYER, npref + DS_SLOT, NKV, HD))
vc = np.zeros((DS_LAYER, npref + DS_SLOT, NKV, HD))
for L in range(DS_LAYER):
    B = "blk.%d." % L
    k = (g @ W(B + "attn_k.weight")).reshape(npref, NKV, HD)
    v = (g @ W(B + "attn_v.weight")).reshape(npref, NKV, HD)
    knorm = d.full(B + "attn_k_norm.weight").astype(np.float64)
    for p in range(npref):
        for kv in range(NKV):
            kc[L, p, kv] = rope(rmsnorm(k[p, kv], knorm), p)
            vc[L, p, kv] = v[p, kv]

# -------------------------------------------------- 3. decoder over the block
toks = [anchor] + [MASK_TOK] * (DS_SLOT - 1)
# one row at a time: the embedding table is 1 GB in f32 if taken whole
x = np.zeros((DS_SLOT, N_EMBD))
for i, r in enumerate(toks):
    x[i] = t.row("token_embd.weight", r, r + 1)[0]

for L in range(DS_LAYER):
    B = "blk.%d." % L
    anorm = d.full(B + "attn_norm.weight").astype(np.float64)
    qn = d.full(B + "attn_q_norm.weight").astype(np.float64)
    kn = d.full(B + "attn_k_norm.weight").astype(np.float64)
    h = rmsnorm(x, anorm)
    q = (h @ W(B + "attn_q.weight")).reshape(DS_SLOT, NH, HD)
    k = (h @ W(B + "attn_k.weight")).reshape(DS_SLOT, NKV, HD)
    v = (h @ W(B + "attn_v.weight")).reshape(DS_SLOT, NKV, HD)
    for i in range(DS_SLOT):
        pos = pos0 + i
        for hh in range(NH):
            q[i, hh] = rope(rmsnorm(q[i, hh], qn), pos)
        for kv in range(NKV):
            kk = rope(rmsnorm(k[i, kv], kn), pos)
            kc[L, pos, kv] = kk
            vc[L, pos, kv] = v[i, kv]
    # block_uniform: every block query sees the prefix plus slot 0, and nothing
    # else of the block.
    vis = pos0 + 1
    out = np.zeros((DS_SLOT, N_EMBD))
    for i in range(DS_SLOT):
        for hh in range(NH):
            kv = hh // (NH // NKV)
            K = kc[L, :vis, kv]                      # (vis, HD)
            V = vc[L, :vis, kv]
            sc = (K @ q[i, hh]) / np.sqrt(HD)
            sc = sc - sc.max()
            p = np.exp(sc); p /= p.sum()
            out[i, hh * HD:(hh + 1) * HD] = p @ V
    x = x + out @ W(B + "attn_output.weight")
    hn = rmsnorm(x, d.full(B + "ffn_norm.weight").astype(np.float64))
    gate = hn @ W(B + "ffn_gate.weight")
    up = hn @ W(B + "ffn_up.weight")
    act = (gate / (1.0 + np.exp(-gate))) * up
    x = x + act @ W(B + "ffn_down.weight")

x = rmsnorm(x, d.full("output_norm.weight").astype(np.float64))

# logits through the target's lm_head, in vocabulary blocks so the dequantised
# 130560 x 2048 table never has to exist at once.
logits = np.zeros((DS_SLOT, VOCAB))
CH = 8192
for v0 in range(0, VOCAB, CH):
    wh = t.row("output.weight", v0, min(v0 + CH, VOCAB)).astype(np.float64)
    logits[:, v0:v0 + CH] = x @ wh.T
logits.astype(np.float32).tofile("/tmp/ref_dslog.bin")

# ------------------------------------------------------------ 4. markov bias
prev = anchor
drafts = []
for i in range(DS_SLOT):
    w1p = d.row("markov_w1.weight", prev, prev + 1)[0].astype(np.float64)
    bias = w1p @ d.full("markov_w2.weight").astype(np.float64).T
    lg = logits[i] + bias
    best = int(np.argmax(lg))
    drafts.append(best)
    prev = best

for i in range(DS_SLOT):
    top = np.argsort(-logits[i])[:5]
    print("  row %d base top5 %s" % (i, [(int(v), round(float(logits[i][v]), 3)) for v in top]))
print("  markov drafts %s" % drafts)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


kern = os.environ.get("KERN_LOG")
if kern and os.path.exists(kern):
    kl = np.fromfile(kern, np.float32).reshape(DS_SLOT, VOCAB)
    for i in range(DS_SLOT):
        print("  row %d  cos(ref, kernel) %.6f   argmax %d vs %d"
              % (i, cos(logits[i], kl[i].astype(np.float64)), int(np.argmax(logits[i])),
                 int(np.argmax(kl[i]))))
