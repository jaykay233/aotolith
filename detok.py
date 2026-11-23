#!/usr/bin/env python3
"""Detokenise token ids with the vocabulary stored in a GGUF file.

The chain emits ids; the reference implementation emits text. To compare them the
ids have to come back out as text, and the only authority for that mapping is the
token table in the model itself.
"""
import struct
import sys


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_vocab(path):
    f = open(path, "rb")
    _, _, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf8", "replace")

    tokens = None
    for _ in range(n_kv):
        key = rstr()
        t, = struct.unpack("<I", f.read(4))
        if t == 8:
            val = rstr()
        elif t in (4, 5, 10):
            val = struct.unpack("<I", f.read(4))[0]
        elif t == 6:
            val = struct.unpack("<f", f.read(4))[0]
        elif t == 7:
            val = bool(struct.unpack("<B", f.read(1))[0])
        elif t == 9:
            at, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            if at == 8:
                val = [rstr() for _ in range(n)]
            else:
                size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at]
                f.read(n * size)
                val = None
        else:
            val = None
        if key == "tokenizer.ggml.tokens":
            tokens = val
    return tokens


def main():
    gguf, ids_file = sys.argv[1], sys.argv[2]
    tokens = load_vocab(gguf)
    rev = {c: b for b, c in bytes_to_unicode().items()}
    ids = [int(x) for x in open(ids_file).read().split()]
    out = bytearray()
    for i in ids:
        for ch in tokens[i]:
            out.append(rev.get(ch, ord("?")))
    sys.stdout.write(out.decode("utf-8", "replace"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
