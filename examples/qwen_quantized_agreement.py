#!/usr/bin/env python3
"""Do a quantized plan and its PyTorch reference agree on what can be decided?

A greedy walk cannot answer that on its own. The moment one token differs, the
two runs are reading different text, so every later token compares two different
prefixes and the resulting fraction measures how fast two sequences decorrelate,
not whether the implementations compute the same thing. At four bits the
benchmark reports 13 of 64 while the implementations pick the same token on 62 of
64 decidable steps - the same plan, described two ways.

So both sides here are teacher-forced: at every step the runtime is fed PyTorch's
own token, which keeps the two KV caches holding the same prefix, and the step's
logits are compared where that is still a like-for-like comparison. Each row also
reports the gap between the top two logits, because that gap decides whether an
argmax difference means anything: a step whose top two logits differ by 0.0000
can go either way in any implementation, and a step whose gap is far above the
measured logit distance should not.

    python examples/qwen_quantized_agreement.py
    python examples/qwen_quantized_agreement.py --quantize-bits 8 --decode 64
    python examples/qwen_quantized_agreement.py --all-steps

Needs the Qwen2.5-0.5B-Instruct weights. Not a CI test: the numbers only mean
something on an Apple GPU and CI has no weights.
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import compile_qwen2_static_cache
from aotolith.quantize import dequantize_int4, dequantize_int8
from aotolith.quantize import quantize_int4, quantize_int8

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


def _round_trip_weights(model: torch.nn.Module, group_size: int, bits: int) -> None:
    """Put every projection on the quantization grid, the way the runtime sees it.

    This is the same transformation `examples/qwen_static_cache_benchmark.py`
    applies before its reference walk, so the words the runtime packs are the
    words this reference consumed. The embedding stays in bf16 because the
    runtime leaves it there too.
    """

    quantize = quantize_int4 if bits == 4 else quantize_int8
    dequantize = dequantize_int4 if bits == 4 else dequantize_int8
    targets = [model.lm_head]
    for layer in model.model.layers:
        attention = layer.self_attn
        targets.extend((attention.q_proj, attention.k_proj, attention.v_proj))
        targets.extend((attention.o_proj, layer.mlp.gate_proj))
        targets.extend((layer.mlp.up_proj, layer.mlp.down_proj))
    with torch.no_grad():
        for module in targets:
            weight = module.weight
            packed, scales = quantize(weight.detach().float().cpu().numpy(), group_size)
            restored = dequantize(packed, scales, group_size)
            weight.copy_(torch.from_numpy(restored).to(weight.dtype))


def _reference_walk(
    model: torch.nn.Module, prompt: list[int], steps: int
) -> tuple[list[np.ndarray], list[int]]:
    """PyTorch's logits and greedy choices, walking its own cache."""

    logits: list[np.ndarray] = []
    tokens: list[int] = []
    with torch.no_grad():
        past = None
        ids = torch.tensor([prompt], dtype=torch.long)
        for _ in range(steps):
            output = model(ids if past is None else ids[:, -1:], past_key_values=past)
            past = output.past_key_values
            row = output.logits[0, -1].float().cpu().numpy()
            logits.append(row)
            token = int(row.argmax())
            tokens.append(token)
            ids = torch.tensor([[token]], dtype=torch.long)
    return logits, tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode", type=int, default=64)
    parser.add_argument("--quantize", type=int, default=64, metavar="GROUP")
    parser.add_argument(
        "--quantize-bits", type=int, default=4, choices=(4, 8), metavar="BITS"
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="print a row per decode step instead of only the disagreements",
    )
    args = parser.parse_args()

    path = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0].tolist()
    if args.prompt_tokens > len(ids):
        parser.error(
            f"prompt needs {args.prompt_tokens} tokens but the sample has {len(ids)}"
        )
    prompt = ids[: args.prompt_tokens]

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    _round_trip_weights(model, args.quantize, args.quantize_bits)
    reference_logits, reference_tokens = _reference_walk(model, prompt, args.decode)

    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=args.prompt_tokens,
        max_new_tokens=args.decode,
        codegen_matmul=True,
        codegen_attention=True,
        quantize=args.quantize,
        quantize_bits=args.quantize_bits,
    )

    logits, compact = runtime.prefill_ids([prompt])
    mx.eval(logits, compact)
    cache = runtime.pad_cache(compact)
    mx.eval(cache)

    print(
        f"{args.model}, int{args.quantize_bits}-group{args.quantize},"
        f" {args.prompt_tokens} prompt tokens, {args.decode} steps"
    )
    print(f"{'step':>5} {'mean|d|':>9} {'max|d|':>9} {'top gap':>9} {'agree':>6}")

    agreements = 0
    valid_length = args.prompt_tokens
    for step in range(args.decode):
        row = np.asarray(logits[0, -1, :].astype(mx.float32)).reshape(-1)
        reference = reference_logits[step]
        difference = np.abs(row - reference)
        ordered = np.sort(reference)
        gap = float(ordered[-1] - ordered[-2])
        agree = int(row.argmax()) == int(reference.argmax())
        agreements += int(agree)
        if args.all_steps or not agree:
            print(
                f"{step:>5} {difference.mean():>9.4f} {difference.max():>9.4f}"
                f" {gap:>9.4f} {'yes' if agree else 'NO':>6}"
            )
        if step + 1 == args.decode:
            break
        valid_length += 1
        # Teacher forcing: PyTorch's token, so both caches hold one prefix.
        token = mx.array([[reference_tokens[step]]], dtype=mx.int64)
        logits, cache = runtime.decode_token(
            token, cache, valid_length, verify_token=False
        )
        mx.eval(logits)

    print()
    print(f"agreement on aligned prefixes: {agreements}/{args.decode}")
    print(
        "A disagreement only counts against the plan when the top two logits are\n"
        "further apart than the distance between the two paths; read the gap column\n"
        "next to each NO row to see whether that step was decidable."
    )
    if not args.all_steps and agreements == args.decode:
        print("every step agreed, so no disagreement rows were printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
