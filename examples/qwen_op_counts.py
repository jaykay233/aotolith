#!/usr/bin/env python3
"""Reproduce the op counts `docs/performance.md` quotes for a decode step.

The prose makes two kinds of claim, and they age differently. A per-pass figure
- "the fold is worth 245 operators" - is a property of the pass and holds until
the pass changes. An absolute figure - "1239 ops to 994" - is a property of the
graph the frontend traced, which is a property of the installed transformers
version. When the tracing changes, every absolute number in the section goes
stale at once while the per-pass figures stay true. This prints both so the
difference is visible.

    python examples/qwen_op_counts.py
    python examples/qwen_op_counts.py --quantize 64

Needs the Qwen2.5-0.5B-Instruct weights. Not a CI test: the counts need the real
model, and the point of the absolute column is to record one environment, not to
gate on it.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

import aotolith.passes.driver as driver
from aotolith.models.qwen2 import compile_qwen2_static_cache

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

#: The passes whose net contribution the docs quote by name.
QUOTED = ("fuse_kv_cache_update", "fuse_norm_scale")


def _net_contribution(graph, name: str) -> int:
    """Ops the whole pipeline emits without ``name``, minus ops it emits with it.

    This is not the same as the pass's own delta where it sits in the pipeline:
    removing a fusion also removes what the later fusions would have folded into
    its output. The docs quote this number, so this is the number to measure.
    """

    saved = driver.PIPELINE
    full = len(driver.rewrite(graph, native_matmul=True).ops)
    try:
        driver.PIPELINE = tuple(step for step in saved if step[0] != name)
        without = len(driver.rewrite(graph, native_matmul=True).ops)
    finally:
        driver.PIPELINE = saved
    return without - full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--quantize",
        type=int,
        default=None,
        metavar="GROUP",
        help="weight-only int4 with this group size, as the benchmark's --quantize",
    )
    args = parser.parse_args()

    # The round trip helper lives in the benchmark script so both read the same
    # weights; importing it keeps one definition of the quantization grid.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen_static_cache_benchmark import _round_trip_weights

    path = snapshot_download(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    if args.quantize is not None:
        _round_trip_weights(model, args.quantize)

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=128,
        max_new_tokens=8,
        quantize=args.quantize,
    )
    graph = runtime.decode.input_graph
    kinds = collections.Counter(op.op for op in graph.ops)

    variant = "bf16" if args.quantize is None else f"int4 group {args.quantize}"
    print(f"model: {args.model}, {variant}")
    print(f"decode step as traced: {len(graph.ops)} ops "
          f"({kinds['cast']} casts, {kinds['reshape']} reshapes)")
    print()
    print(f"{'pass':32s} {'before':>7} {'after':>7} {'at position':>12}")

    cursor = graph
    for name, step in driver.PIPELINE:
        before = len(cursor.ops)
        cursor = (
            step(cursor, native_matmul=True)
            if step is driver.fuse_rms_norm_epilogue
            else step(cursor)
        )
        print(f"{name:32s} {before:7d} {len(cursor.ops):7d} {len(cursor.ops) - before:+12d}")

    print(f"{'after every pass':32s} {len(graph.ops):7d} {len(cursor.ops):7d}"
          f" {len(cursor.ops) - len(graph.ops):+12d}")
    lowered = len(runtime.decode.lowered_graph.ops)
    print(f"{'lowered':32s} {'':7s} {lowered:7d} {'':>12s}")
    print()
    for name in QUOTED:
        print(f"net contribution of {name}: {_net_contribution(graph, name)} ops")
    print()
    print(
        "Quote the net contribution. The absolute columns move with the\n"
        "transformers version that traced the graph; the nets move when the pass\n"
        "changes, which is what a reader can check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
