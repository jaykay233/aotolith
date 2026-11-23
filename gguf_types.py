import struct, sys

p = sys.argv[1] if len(sys.argv) > 1 else "mc-q4o.gguf"
f = open(p, "rb")


def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


# GGUF value types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string
# 9 array 10 u64 11 i64 12 f64. Getting 7 or 12 wrong silently desynchronises the
# whole stream, which is exactly what the first version of this script did.
FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
         6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

T = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
     8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}


def rs():
    n, = rd("<Q")
    return f.read(n).decode(errors="replace")


def value(t):
    if t == 8:
        return rs()
    if t == 9:
        at, = rd("<I")
        n, = rd("<Q")
        for _ in range(n):
            value(at)
        return "<array>"
    if t in FIXED:
        v, = rd(FIXED[t])
        return v
    raise ValueError("unknown gguf type %d" % t)


magic, = rd("<I")
ver, = rd("<I")
nt, = rd("<Q")
nkv, = rd("<Q")
print("magic", hex(magic), "ver", ver, "tensors", nt, "kv", nkv)

kv = {}
for _ in range(nkv):
    k = rs()
    t, = rd("<I")
    kv[k] = value(t)

arch = kv.get("general.architecture")
print("arch", arch)
for k in sorted(kv):
    if any(s in k for s in ("embedding_length", "block_count", "head_count",
                            "feed_forward_length", "rope", "vocab_size",
                            "context_length")):
        print("  ", k, "=", kv[k])

types = {}
for _ in range(nt):
    name = rs()
    nd, = rd("<I")
    dims = rd("<" + "Q" * nd)
    t, = rd("<I")
    rd("<Q")
    types.setdefault(T.get(t, t), []).append((name, list(dims)))

for ty, lst in types.items():
    print(ty, len(lst), "e.g.", lst[:3])
    if ty == "Q4_0":
        for nm, dm in lst:
            if nm.startswith(("token_embd", "output.")):
                print("   ", nm, dm)
