#!/usr/bin/env python3
"""Decode workload for Metal System Trace / gputrace capture.

Runs enough warm decode steps that Instruments can see a steady GPU queue,
then a timed window of async decode. Prefer launching under::

    xctrace record --template 'Metal System Trace' --time-limit 15s \\
        --output /tmp/aoto_decode.trace --launch -- \\
        .venv/bin/python scripts/metal_profile_decode.py --prompt 4096

Or for an Xcode Metal debugger capture::

    MTL_CAPTURE_ENABLED=1 .venv/bin/python scripts/metal_profile_decode.py \\
        --prompt 4096 --gputrace /tmp/aoto_decode.gputrace
"""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import compile_qwen2_static_cache


def _prompt_ids(tokenizer: object, count: int) -> list[int]:
    text = ("TensorRT-LLM is an inference engine for large language models. ") * 400
    ids = tokenizer.encode(text, add_special_tokens=False)
    return (ids * ((count + len(ids) - 1) // len(ids)))[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument(
        "--gputrace",
        default="",
        help="If set, wrap one decode window in mx.metal.start_capture",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=0.5,
        help="Sleep after printing ready so xctrace can attach before work",
    )
    args = parser.parse_args()

    print(f"device={mx.device_info().get('device_name')}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).eval()
    prompt = _prompt_ids(tokenizer, args.prompt)

    print(f"compile prompt={len(prompt)} …", flush=True)
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(prompt),
        max_new_tokens=max(64, args.steps + 8),
        codegen_matmul=True,
        codegen_attention=True,
    )
    print(
        f"ready capacity={runtime.cache_capacity} hold={args.hold_s}s",
        flush=True,
    )
    time.sleep(args.hold_s)

    def one_window(*, capture: bool) -> float:
        logits, compact = runtime.prefill_ids([prompt])
        mx.eval(logits, compact)
        cache = runtime.pad_cache(compact)
        token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int64)
        mx.eval(cache, token)
        valid = len(prompt)
        if capture and args.gputrace:
            os.makedirs(os.path.dirname(args.gputrace) or ".", exist_ok=True)
            mx.metal.start_capture(args.gputrace)
        mx.synchronize()
        started = time.perf_counter()
        steps = 0
        for _ in range(args.steps):
            valid += 1
            if valid >= runtime.cache_capacity:
                break
            logits, cache = runtime.decode_token(
                token, cache, valid, verify_token=False
            )
            token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int64)
            steps += 1
        mx.eval(token, cache)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if capture and args.gputrace:
            mx.metal.stop_capture()
            print(f"wrote {args.gputrace}", flush=True)
        return elapsed / max(steps, 1)

    for i in range(args.warmups):
        ms = one_window(capture=False) * 1000
        print(f"warmup {i}: {ms:.2f} ms/tok", flush=True)

    ms = one_window(capture=bool(args.gputrace)) * 1000
    print(f"timed: {ms:.2f} ms/tok ({1000 / ms:.1f} tok/s)", flush=True)
    # Keep the process alive briefly so late GPU work is still in the trace.
    time.sleep(0.25)


if __name__ == "__main__":
    main()
