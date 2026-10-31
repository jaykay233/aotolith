#!/usr/bin/env python3
"""Benchmark Qwen2.5 with Aotolith's fixed-capacity static KV-cache graph.

The defaults match the PyTorch MPS and MLX baselines used for this machine:
batch 1, 128 prompt tokens, 64 greedy decode tokens, BF16, two warmups, and
five measured runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import (
    Qwen2StaticCacheRuntime,
    compile_qwen2_static_cache,
)

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


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _rate_summary(
    latencies: list[float],
    tokens: int,
) -> dict[str, float]:
    return _summary([tokens / latency for latency in latencies])


def _prompt_ids(tokenizer: object, count: int) -> list[int]:
    ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    if not ids:
        raise RuntimeError("tokenizer produced an empty prompt")
    if len(ids) < count:
        ids = ids * ((count + len(ids) - 1) // len(ids))
    return ids[:count]


def _reference_tokens(
    model: torch.nn.Module,
    prompt: list[int],
    decode_tokens: int,
) -> list[int]:
    input_ids = torch.tensor([prompt], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [int(token.item())]
        for _ in range(decode_tokens - 1):
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=torch.int64)),
                dim=1,
            )
            output = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
    return generated


def _round_trip_weights(model: torch.nn.Module, group_size: int) -> None:
    """Replace each projection weight by its own int4 round trip, in place.

    The runtime packs these same tensors, and quantizing a value that is already
    on the int4 grid returns the same words - the group's widest element rounds
    to +-7, so the scale recomputes unchanged. So the words the runtime streams
    are byte-identical to the ones this reference consumed, which is the only
    way to tell a decode bug from a rounding difference. The embedding stays
    untouched because the runtime leaves it in bf16 too.
    """

    from aotolith.quantize import dequantize_int4, quantize_int4

    targets = [model.lm_head]
    for layer in model.model.layers:
        attention = layer.self_attn
        targets.extend((attention.q_proj, attention.k_proj, attention.v_proj))
        targets.extend((attention.o_proj, layer.mlp.gate_proj))
        targets.extend((layer.mlp.up_proj, layer.mlp.down_proj))
    with torch.no_grad():
        for module in targets:
            weight = module.weight
            packed, scales = quantize_int4(weight.float().cpu().numpy(), group_size)
            restored = dequantize_int4(packed, scales, group_size)
            weight.copy_(torch.from_numpy(restored).to(weight.dtype))


def _prefill_once(
    runtime: Qwen2StaticCacheRuntime,
    prompt: list[int],
) -> float:
    start = time.perf_counter()
    logits, compact_cache = runtime.prefill_ids(prompt)
    cache = runtime.pad_cache(compact_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
    mx.eval(token, cache)
    return time.perf_counter() - start


def _decode_once(
    runtime: Qwen2StaticCacheRuntime,
    prompt: list[int],
    decode_tokens: int,
    *,
    sync_each_step: bool,
) -> tuple[float, mx.array]:
    logits, compact_cache = runtime.prefill_ids(prompt)
    cache = runtime.pad_cache(compact_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
    mx.eval(token, cache)
    generated = [token.reshape(-1)]
    valid_length = runtime.prompt_length

    start = time.perf_counter()
    for _ in range(decode_tokens - 1):
        valid_length += 1
        # ``token`` is always an ``argmax`` over the logits, so it indexes a real
        # embedding row; the strict per-step bounds check would cost a device
        # sync per token for a condition that cannot be violated here.
        logits, cache = runtime.decode_token(
            token, cache, valid_length, verify_token=False
        )
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(mx.int64)
        if sync_each_step:
            mx.eval(token, cache)
        generated.append(token.reshape(-1))
    tokens = mx.stack(generated, axis=1)
    mx.eval(tokens, cache)
    return time.perf_counter() - start, tokens


def _mlx_lm_decode_rate(
    model_path: str,
    prompt: list[int],
    decode_tokens: int,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, object] | None:
    """Time mlx_lm's own greedy decode on the same prompt for comparison.

    The native baseline is the number this project is measured against, so it
    has to be produced on the machine running the benchmark rather than quoted
    from elsewhere. ``mlx_lm`` is optional: without it the key is omitted.
    """

    try:
        from mlx_lm import load
        from mlx_lm.models import cache as mlx_cache
    except ImportError:
        return None

    model, _ = load(model_path)
    ids = mx.array([prompt])

    def once() -> tuple[float, list[int]]:
        prompt_cache = mlx_cache.make_prompt_cache(model)
        logits = model(ids, cache=prompt_cache)
        mx.eval(logits)
        tokens = [int(mx.argmax(logits[:, -1, :], axis=-1).item())]
        start = time.perf_counter()
        for _ in range(decode_tokens - 1):
            logits = model(mx.array([[tokens[-1]]]), cache=prompt_cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tokens.append(int(token.item()))
        return time.perf_counter() - start, tokens

    for _ in range(warmups):
        once()
    latencies: list[float] = []
    tokens: list[int] = []
    for _ in range(repeats):
        elapsed, tokens = once()
        latencies.append(elapsed)
    return {
        "backend": "mlx_lm",
        "decode": {
            "latency_ms": _summary([value * 1000 for value in latencies]),
            "tokens_per_s": _rate_summary(latencies, decode_tokens - 1),
        },
        "generated_tokens": tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model id or local directory",
    )
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codegen-matmul", action="store_true")
    parser.add_argument("--codegen-attention", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument(
        "--quantize",
        type=int,
        default=None,
        metavar="GROUP",
        help=(
            "weight-only int4 with this group size (for example 64); the "
            "streaming kernel is selected whether or not --codegen-matmul is on"
        ),
    )
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--mlx-baseline",
        action="store_true",
        help="also time mlx_lm's greedy decode on the same prompt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-prefix", type=Path)
    args = parser.parse_args()

    model_path = snapshot_download(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = _prompt_ids(tokenizer, args.prompt_tokens)

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_start

    reference_tokens: list[int] | None = None
    if not args.skip_reference:
        reference_tokens = _reference_tokens(
            model,
            prompt,
            args.decode_tokens,
        )
    # The bf16 reference is taken first, while the weights are still pristine;
    # then the same weights are rounded onto the int4 grid, so the second
    # reference is what a checkpoint that arrived already quantized would say.
    quantized_reference_tokens: list[int] | None = None
    if args.quantize is not None:
        _round_trip_weights(model, args.quantize)
        quantized_reference_tokens = _reference_tokens(
            model,
            prompt,
            args.decode_tokens,
        )

    compile_start = time.perf_counter()
    runtime = compile_qwen2_static_cache(
        model,
        prompt_length=args.prompt_tokens,
        max_new_tokens=args.decode_tokens,
        codegen_matmul=args.codegen_matmul,
        codegen_attention=args.codegen_attention,
        autotune=args.autotune,
        quantize=args.quantize,
    )
    compile_seconds = time.perf_counter() - compile_start

    artifact_paths: list[str] = []
    if args.artifact_prefix is not None:
        artifact_paths = [str(path) for path in runtime.save(args.artifact_prefix)]

    # Warm up both graph executions and the host greedy loop before recording.
    _prefill_once(runtime, prompt)
    _decode_once(
        runtime,
        prompt,
        args.decode_tokens,
        sync_each_step=True,
    )
    _decode_once(
        runtime,
        prompt,
        args.decode_tokens,
        sync_each_step=False,
    )

    prefill_latencies: list[float] = []
    for _ in range(args.repeats):
        prefill_latencies.append(_prefill_once(runtime, prompt))

    sync_decode_latencies: list[float] = []
    async_decode_latencies: list[float] = []
    actual_tokens: mx.array | None = None
    for _ in range(args.repeats):
        latency, actual_tokens = _decode_once(
            runtime,
            prompt,
            args.decode_tokens,
            sync_each_step=True,
        )
        sync_decode_latencies.append(latency)
    for _ in range(args.repeats):
        latency, actual_tokens = _decode_once(
            runtime,
            prompt,
            args.decode_tokens,
            sync_each_step=False,
        )
        async_decode_latencies.append(latency)

    if actual_tokens is None:
        raise RuntimeError("benchmark did not execute a decode run")
    mx.eval(actual_tokens)
    generated = actual_tokens.tolist()[0]
    first_reference_mismatch = (
        None
        if reference_tokens is None
        else next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(generated, reference_tokens, strict=True)
                )
                if actual != expected
            ),
            None,
        )
    )
    quantized_reference_match = (
        None
        if quantized_reference_tokens is None
        else generated == quantized_reference_tokens
    )
    quantized_reference_agreement = (
        None
        if quantized_reference_tokens is None
        else sum(
            int(actual == expected)
            for actual, expected in zip(
                generated, quantized_reference_tokens, strict=True
            )
        )
        / len(generated)
    )
    first_quantized_mismatch = (
        None
        if quantized_reference_tokens is None
        else next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(generated, quantized_reference_tokens, strict=True)
                )
                if actual != expected
            ),
            None,
        )
    )
    sync_e2e = [
        prefill + decode
        for prefill, decode in zip(
            prefill_latencies,
            sync_decode_latencies,
            strict=True,
        )
    ]
    async_e2e = [
        prefill + decode
        for prefill, decode in zip(
            prefill_latencies,
            async_decode_latencies,
            strict=True,
        )
    ]

    result = {
        "backend": "aotolith-apple-metal",
        "model_path": model_path,
        "model_dtype": "bfloat16",
        "weight_quantization": (
            None if args.quantize is None else f"int4-group{args.quantize}"
        ),
        "device": str(mx.default_device()),
        "load_s": load_seconds,
        "compile_s": compile_seconds,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "batch_size": 1,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "greedy": True,
        "codegen_matmul": args.codegen_matmul,
        "codegen_attention": args.codegen_attention,
        "autotune": args.autotune,
        "prefill": {
            "latency_ms": _summary([value * 1000 for value in prefill_latencies]),
            "tokens_per_s": _rate_summary(prefill_latencies, args.prompt_tokens),
        },
        "decode_sync_each_step": {
            "latency_ms": _summary([value * 1000 for value in sync_decode_latencies]),
            "tokens_per_s": _rate_summary(
                sync_decode_latencies, args.decode_tokens - 1
            ),
        },
        "decode_async": {
            "latency_ms": _summary([value * 1000 for value in async_decode_latencies]),
            "tokens_per_s": _rate_summary(
                async_decode_latencies, args.decode_tokens - 1
            ),
        },
        "end_to_end_sync_each_step": {
            "latency_ms": _summary([value * 1000 for value in sync_e2e]),
            "tokens_per_s": _rate_summary(sync_e2e, args.decode_tokens),
        },
        "end_to_end_async": {
            "latency_ms": _summary([value * 1000 for value in async_e2e]),
            "tokens_per_s": _rate_summary(async_e2e, args.decode_tokens),
        },
        "prefill_graph_ops": len(runtime.prefill.input_graph.ops),
        "prefill_lowered_ops": len(runtime.prefill.lowered_graph.ops),
        "decode_graph_ops": len(runtime.decode.input_graph.ops),
        "decode_lowered_ops": len(runtime.decode.lowered_graph.ops),
        "active_memory_bytes": mx.get_active_memory(),
        "peak_memory_bytes": mx.get_peak_memory(),
        "prompt_ids_sha256": hashlib.sha256(
            np.asarray(prompt, dtype=np.int32).tobytes()
        ).hexdigest(),
        "generated_tokens": generated,
        "generated_tokens_sha256": hashlib.sha256(
            np.asarray(generated, dtype=np.int32).tobytes()
        ).hexdigest(),
        "reference_tokens": reference_tokens,
        "reference_tokens_match": (
            None if reference_tokens is None else generated == reference_tokens
        ),
        "first_reference_mismatch": first_reference_mismatch,
        "quantized_reference_tokens": quantized_reference_tokens,
        "quantized_reference_tokens_match": quantized_reference_match,
        "quantized_reference_agreement": quantized_reference_agreement,
        "first_quantized_reference_mismatch": first_quantized_mismatch,
        "artifact_paths": artifact_paths,
        "artifact_bytes": [Path(path).stat().st_size for path in artifact_paths],
    }
    if args.mlx_baseline:
        baseline = _mlx_lm_decode_rate(
            model_path,
            prompt,
            args.decode_tokens,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        if baseline is None:
            print(
                "mlx_lm is not installed; skipping the native baseline", file=sys.stderr
            )
        else:
            result["mlx_lm_baseline"] = baseline
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
