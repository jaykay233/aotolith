#!/usr/bin/env python3
"""How far each execution path lands from PyTorch, and how much that varies.

The throughput table in `docs/performance.md` says both paths produce the same 64
greedy tokens as the PyTorch bf16 reference. That is a statement about the
answer, not about the numbers: two implementations can agree on every argmax and
still disagree on the logits. This measures the second thing.

It prints one row per prompt length, because that is the part a single number
hides. On this machine the MLX fallback's mean distance moves by 3.4x between a
7-token and a 16-token prompt, so quoting one measurement as "the model's"
distance would be quoting the prompt, not the model.

    python examples/qwen_logit_distance.py
    python examples/qwen_logit_distance.py --lengths 7,16,128

Needs the Qwen2.5-0.5B-Instruct weights (`huggingface-cli download`). It is not a
CI test: the numbers are only meaningful on an Apple GPU, and CI has no weights.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import compile_qwen2_static_cache

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT_TEXT = """
TensorRT-LLM is an inference engine for large language models. It combines a request
scheduler, paged KV cache management, continuous batching, optimized attention kernels,
and tensor, pipeline, data, and expert parallelism. On Apple Silicon, PyTorch can use the
Metal Performance Shaders backend while MLX provides an array framework designed for
unified memory. A useful compiler baseline should use a fixed tokenizer, a fixed prompt,
the same numerical precision, explicit synchronization, and repeated measurements so that
loading time, compilation time, and asynchronous execution do not distort throughput.
This paragraph is intentionally long enough to produce more than one hundred and twenty
eight valid Qwen tokenizer tokens without padding or synthetic token identifiers.
""".strip()

#: Paths to compare, as ``(label, codegen_matmul, codegen_attention)``.
PATHS = (
    ("fallback", False, False),
    ("codegen", True, True),
)


def _reference(model: torch.nn.Module, prompt: list[int]) -> np.ndarray:
    """PyTorch's last-token logits, widened to float64 for the comparison."""

    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([prompt]), use_cache=True, logits_to_keep=1
        )
    # bfloat16 has no numpy dtype, so widen before crossing the boundary; the
    # value is exact either way, and float64 keeps the subtraction out of the
    # precision being measured.
    return output.logits[0, -1].float().numpy().astype(np.float64)


def _mlx_logits(value: mx.array) -> np.ndarray:
    return np.asarray(value.astype(mx.float32))[0, -1].astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--lengths",
        default="7,8,16,128",
        help="comma-separated prompt lengths to measure",
    )
    args = parser.parse_args()
    lengths = [int(part) for part in args.lengths.split(",") if part.strip()]

    path = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0].tolist()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()

    print(f"model: {args.model} ({len(ids)} prompt tokens available)")
    header = f"{'prompt':>7}  " + "  ".join(
        f"{label + ' mean':>14} {label + ' max':>12}" for label, _, _ in PATHS
    )
    print(header)
    print("-" * len(header))
    for length in lengths:
        if length > len(ids):
            print(f"{length:>7}  skipped: prompt is only {len(ids)} tokens")
            continue
        prompt = ids[:length]
        reference = _reference(model, prompt)
        cells = []
        for _, codegen_matmul, codegen_attention in PATHS:
            runtime = compile_qwen2_static_cache(
                model,
                prompt_length=length,
                max_new_tokens=2,
                codegen_matmul=codegen_matmul,
                codegen_attention=codegen_attention,
            )
            logits, _ = runtime.prefill_ids([prompt])
            mx.eval(logits)
            difference = np.abs(_mlx_logits(logits) - reference)
            cells.append(f"{difference.mean():>14.4f} {difference.max():>12.4f}")
        print(f"{length:>7}  " + "  ".join(cells))

    print()
    print(
        "Read down a column, not across a row: the distance depends on how many\n"
        "tokens the prefill sees, so one prompt length does not describe the model.\n"
        "The argmax is what a greedy run consumes, and it agrees on every length."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
