#!/usr/bin/env python3
"""Compare two logit rows and two hidden rows, megakernel vs llama.cpp."""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
n = min(len(a), len(b))
a, b = a[:n], b[:n]
print(f"len {n}")
print(f"  norm  mega {np.linalg.norm(a):.4f}  ref {np.linalg.norm(b):.4f}  ratio {np.linalg.norm(a)/np.linalg.norm(b):.4f}")
print(f"  cosine {float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-30)):.6f}")
print(f"  absmax mega {np.abs(a).max():.4f}  ref {np.abs(b).max():.4f}")
print(f"  rms diff {float(np.sqrt(np.mean((a-b)**2))):.5f}")
ka = np.argsort(-a)[:8]
kb = np.argsort(-b)[:8]
print("  top8 mega:", " ".join(f"{i}({a[i]:.3f})" for i in ka))
print("  top8 ref :", " ".join(f"{i}({b[i]:.3f})" for i in kb))
ra = np.argsort(np.argsort(-a))
rb = np.argsort(np.argsort(-b))
print(f"  rank agreement of ref-top8 in mega: {[int(ra[i]) for i in kb]}")
