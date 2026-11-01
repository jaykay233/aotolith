#!/usr/bin/env python3
"""Time aotolith and mlx_lm on the same prompt, for prefill and for decode.

`examples/qwen_static_cache_benchmark.py --mlx-baseline` compares decode rates
only, at one prompt length. That leaves the other half of a serving workload
unmeasured: prefill is a different regime (compute-bound, wide, latency to first
token) and a decoder that wins on decode can lose there. This measures both, at
several prompt lengths, with each side doing the same greedy work.

    python examples/compare_mlx_lm.py
    python examples/compare_mlx_lm.py --lengths 32,128,512 --decode 64

aotolith's plans are static, so each prompt length gets its own compile; the
compile time is reported separately and excluded from the rates. mlx_lm is
dynamic and needs none, which is the trade the bucketing work exists to soften.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from aotolith.models.qwen2 import compile_qwen2_static_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

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
eight valid Qwen tokenizer tokens without padding or synthetic token identifiers. Adding
one more sentence gives the longest prompt length something to read past the halfway
point of the context window, and keeps every measurement below inside one bucket.
""".strip()


def _median(values: list[float]) -> float:
    return float(np.median(values))


def _time_aotolith(
    model, prompt: list[int], decode_tokens: int, codegen_matmul: bool
) -> tuple[float, float, float, float]:
    """Return ``(compile_s, prefill, decode_sync, decode_async)`` rates."""

    started = time.perf_counter()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=len(prompt),
        max_new_tokens=decode_tokens,
        codegen_matmul=codegen_matmul,
    )
    compile_s = time.perf_counter() - started

    # Warm up twice - the first call after a compile pays kernel construction,
    # not arithmetic - and then take the median of several passes. A single
    # timed pass is not enough here: the same configuration measured 6 200 and
    # 10 200 tok/s one after the other while the machine was busy, which is a
    # 60% spread on the number this comparison is supposed to report.
    for _ in range(2):
        logits, compact = runtime.prefill_ids([prompt])
        mx.eval(logits, compact)
    passes = []
    for _ in range(5):
        start = time.perf_counter()
        logits, compact = runtime.prefill_ids([prompt])
        mx.eval(logits, compact)
        passes.append(len(prompt) / (time.perf_counter() - start))
    prefill_rate = _median(passes)

    cache = runtime.pad_cache(compact)
    mx.eval(cache)
    # The token stays on the device. Reading it back with ``.item()`` would
    # synchronize every step, which is exactly the overlap the decode path is
    # built for, so a rate measured that way describes the harness rather than
    # the compiler. ``argmax`` stays lazy until the final sync.
    token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int64)
    # Same convention as the benchmark: the prefill above produced the first
    # token, so N-1 steps generate N tokens.
    def decode_steps(*, sync_each_step: bool) -> float:
        nonlocal token, cache
        mx.synchronize()
        start = time.perf_counter()
        valid_length = len(prompt)
        for _ in range(decode_tokens - 1):
            valid_length += 1
            logits, cache = runtime.decode_token(
                token, cache, valid_length, verify_token=False
            )
            token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int64)
            if sync_each_step:
                mx.eval(token)
        mx.synchronize()
        return (decode_tokens - 1) / (time.perf_counter() - start)

    decode_async = decode_steps(sync_each_step=False)
    decode_sync = decode_steps(sync_each_step=True)
    return compile_s, prefill_rate, decode_sync, decode_async


def _time_mlx_lm(model, prompt: list[int], decode_tokens: int) -> tuple[float, float, float]:
    """Return ``(prefill, decode_sync, decode_async)`` rates for mlx_lm."""

    from mlx_lm.models import cache as mlx_cache

    ids = mx.array([prompt])

    def prefill_once() -> tuple[float, mx.array]:
        prompt_cache = mlx_cache.make_prompt_cache(model)
        start = time.perf_counter()
        logits = model(ids, cache=prompt_cache)
        mx.eval(logits)
        return time.perf_counter() - start, logits

    for _ in range(2):
        _, logits = prefill_once()
    passes = []
    for _ in range(5):
        elapsed, logits = prefill_once()
        passes.append(len(prompt) / elapsed)
    prefill_rate = _median(passes)

    def decode_steps(*, sync_each_step: bool) -> float:
        prompt_cache = mlx_cache.make_prompt_cache(model)
        mx.eval(model(ids, cache=prompt_cache))
        step_token = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(decode_tokens - 1):
            out = model(step_token, cache=prompt_cache)
            step_token = mx.argmax(out[:, -1, :], axis=-1).reshape(1, 1)
            # Async submission is what a serving loop does, and it is what makes
            # the comparison fair: aotolith's decode path submits each step with
            # ``mx.async_eval`` internally. Without this line the whole sequence
            # builds one graph whose only evaluation is the final sync, which
            # reads as 2600 tok/s on a model whose weights alone take 5 ms per
            # step to stream - a number that describes the harness, not mlx_lm.
            if sync_each_step:
                mx.eval(step_token)
            else:
                mx.async_eval(step_token)
        mx.synchronize()
        return (decode_tokens - 1) / (time.perf_counter() - start)

    decode_async = decode_steps(sync_each_step=False)
    decode_sync = decode_steps(sync_each_step=True)
    return prefill_rate, decode_sync, decode_async


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", default="32,128,512")
    parser.add_argument("--decode", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    lengths = [int(part) for part in args.lengths.split(",") if part.strip()]

    path = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0].tolist()
    torch_model = AutoModelForCausalLM.from_pretrained(path, dtype="auto").eval()

    from mlx_lm import load as mlx_load

    mlx_model, _ = mlx_load(path)

    print(f"model: {args.model}, {len(ids)} prompt tokens available, "
          f"{args.decode} generated, median of {args.repeats}\n")
    header = (
        f"{'prompt':>6} | {'prefill tok/s':>15} | {'decode sync tok/s':>19} | "
        f"{'decode async tok/s':>19} | compile"
    )
    print(header)
    sub = f"{'':>6} | {'aoto':>7} {'mlx_lm':>7} | {'aoto':>8} {'mlx_lm':>8} | "
    sub += f"{'aoto':>8} {'mlx_lm':>8} | "
    print(sub)
    print("-" * len(header))

    for length in lengths:
        if length > len(ids):
            print(f"{length:>6} | skipped: prompt is only {len(ids)} tokens")
            continue
        prompt = ids[:length]
        a_pre, a_sync, a_async, compiles = [], [], [], []
        m_pre, m_sync, m_async = [], [], []
        for _ in range(args.repeats):
            c, ap, asy, aas = _time_aotolith(
                torch_model, prompt, args.decode, codegen_matmul=True
            )
            compiles.append(c)
            a_pre.append(ap)
            a_sync.append(asy)
            a_async.append(aas)
            mp, msy, mas = _time_mlx_lm(mlx_model, prompt, args.decode)
            m_pre.append(mp)
            m_sync.append(msy)
            m_async.append(mas)
        print(
            f"{length:>6} | {_median(a_pre):>7,.0f} {_median(m_pre):>7,.0f} | "
            f"{_median(a_sync):>8.1f} {_median(m_sync):>8.1f} | "
            f"{_median(a_async):>8.1f} {_median(m_async):>8.1f} | "
            f"{_median(compiles):.1f} s"
        )

    print()
    print(
        "Prefill is a single timed pass after a warmup; decode is N-1 steps after\n"
        "the prefill that produced the first token. Both sides run greedy argmax.\n"
        "The sync column waits for every step and the async column submits each\n"
        "step and waits once, so read a column against itself, not across: pairing\n"
        "aotolith's async rate with mlx_lm's sync rate flatters the compiler by the\n"
        "cost of a device round trip per token. aotolith's compile time is excluded\n"
        "from every rate and shown separately; mlx_lm needs none, which is the trade\n"
        "the bucketing work exists to soften."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
