from __future__ import annotations

import random
import time

import mlx.core as mx
import torch
import torch.nn.functional as F

import aotolith as aoto

_WARMUP_ELEMENTS = 2048


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True)


def _spin_gpu(milliseconds: float = 350.0) -> None:
    """Raise the GPU clock before timing so a case is not penalised.

    Compiling a Metal kernel takes long enough for the clock to drop again, so
    this runs immediately before every measurement instead of once per process.
    """

    left = mx.random.normal((_WARMUP_ELEMENTS, _WARMUP_ELEMENTS)).astype(mx.float16)
    right = mx.random.normal((_WARMUP_ELEMENTS, _WARMUP_ELEMENTS)).astype(mx.float16)
    mx.eval(left, right)
    deadline = time.perf_counter() + milliseconds / 1000.0
    while time.perf_counter() < deadline:
        for _ in range(20):
            mx.eval(left @ right)
    mx.synchronize()


def _timed(fn, inner: int) -> float:
    start = time.perf_counter()
    outputs = [fn() for _ in range(inner)]
    mx.eval(*outputs)
    return (time.perf_counter() - start) / inner * 1000


def benchmark(candidates, inner: int, rounds: int = 7) -> dict[str, float]:
    """Paired timing: round-robin every candidate in shuffled order, keep the best.

    Interleaving matters because a single long sweep lets the GPU clock drift;
    paired rounds keep all candidates in the same thermal state.
    """

    _spin_gpu()
    for _ in range(3):
        for fn in candidates.values():
            _timed(fn, inner)
    best = {name: float("inf") for name in candidates}
    order = list(candidates)
    for _ in range(rounds):
        random.shuffle(order)
        for name in order:
            best[name] = min(best[name], _timed(candidates[name], inner))
    return best


def run_case(batch: int, heads: int, sequence: int, head_dim: int) -> None:
    torch.manual_seed(0)
    torch_inputs = tuple(
        torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            dtype=torch.float16,
        )
        for _ in range(3)
    )
    native = aoto.compile_callable(attention, torch_inputs, codegen_attention=True)
    kernel = native.plan.kernels[0]
    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mx.float16) for value in torch_inputs
    )
    actual = native.plan(*inputs)[0]
    flops = 4 * batch * heads * sequence * (sequence // 2) * head_dim
    scale = head_dim**-0.5
    if flops < 5e10:
        expected = mx.array(attention(*torch_inputs).float().numpy(), dtype=mx.float16)
        reference = "PyTorch CPU"
    else:
        # A CPU reference this large would dominate the run and heat the machine
        # up right before timing, so large shapes check against MLX instead.
        expected = mx.fast.scaled_dot_product_attention(
            *inputs, scale=scale, mask="causal"
        )
        reference = "MLX SDPA"
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))

    inner = 12 if flops < 2e9 else 4
    timings = benchmark(
        {
            "custom": lambda: native.plan(*inputs)[0],
            "mlx": lambda: mx.fast.scaled_dot_product_attention(
                *inputs, scale=scale, mask="causal"
            ),
        },
        inner,
    )
    shape = f"{batch}x{heads}x{sequence}x{head_dim}"
    print(f"\nattention {shape}: {kernel.name}")
    print(f"max abs error against {reference}: {max_error:.3e}")
    print(
        f"custom kernel:         {timings['custom'] * 1e3:8.1f} us "
        f"({flops / timings['custom'] * 1e-9:.1f} TFLOPS)"
    )
    print(
        f"MLX SDPA:              {timings['mlx'] * 1e3:8.1f} us "
        f"({flops / timings['mlx'] * 1e-9:.1f} TFLOPS)"
    )
    ratio = timings["mlx"] / timings["custom"]
    verdict = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    print(f"custom vs MLX:         {verdict}")


def main() -> None:
    run_case(1, 8, 512, 64)
    run_case(1, 8, 2048, 128)
    run_case(1, 64, 4096, 128)


if __name__ == "__main__":
    main()
