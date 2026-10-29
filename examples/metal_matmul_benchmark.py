from __future__ import annotations

import random
import time

import mlx.core as mx
import torch

import aotolith as aoto

_WARMUP_ELEMENTS = 2048


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


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


def benchmark(candidates, inner: int, rounds: int = 9) -> dict[str, float]:
    """Paired timing: round-robin every candidate in shuffled order, keep the best.

    Interleaving matters because a single long sweep lets the GPU clock drift;
    paired rounds keep all candidates in the same thermal state.  Every launch
    stays alive until the group is evaluated, because MLX prunes lazy
    operations whose result is never consumed.
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


def main() -> None:
    torch.manual_seed(0)
    run_case("f32", torch.float32, mx.float32, 512, 2048, 512)
    run_case("f16", torch.float16, mx.float16, 512, 2048, 512)
    run_case("bf16", torch.bfloat16, mx.bfloat16, 512, 2048, 512)
    run_case("f16", torch.float16, mx.float16, 1024, 4864, 896)
    run_case("f16", torch.float16, mx.float16, 4096, 4096, 4096)


def run_case(
    label: str,
    torch_dtype,
    mlx_dtype,
    rows: int = 512,
    reduction: int = 2048,
    columns: int = 512,
) -> None:
    torch_inputs = (
        torch.randn(rows, reduction, dtype=torch_dtype),
        torch.randn(reduction, columns, dtype=torch_dtype),
    )
    compiled = aoto.compile_callable(matmul, torch_inputs, codegen_matmul=True)
    kernel = compiled.plan.kernels[0]
    print(f"\n{label} matmul {rows}x{reduction}x{columns}: {kernel.name}")

    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mlx_dtype) for value in torch_inputs
    )
    actual = compiled.plan(*inputs)[0]
    flops = 2 * rows * reduction * columns
    if flops < 5e10:
        expected = mx.array(matmul(*torch_inputs).float().numpy(), dtype=mlx_dtype)
        reference = "PyTorch CPU"
    else:
        # A 137 GFLOP CPU matmul would dominate the run and heat the machine up
        # right before timing, so large shapes check against MLX instead.
        expected = inputs[0] @ inputs[1]
        reference = "MLX"
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"max abs error against {reference}: {max_error:.3e}")

    inner = 12 if flops < 3e10 else 4
    timings = benchmark(
        {
            "custom": lambda: compiled.plan(*inputs)[0],
            "mlx": lambda: inputs[0] @ inputs[1],
        },
        inner,
    )
    print(f"custom kernel:         {timings['custom'] * 1e3:8.1f} us "
          f"({flops / timings['custom'] * 1e-9:.1f} TFLOPS)")
    print(f"MLX matmul:            {timings['mlx'] * 1e3:8.1f} us "
          f"({flops / timings['mlx'] * 1e-9:.1f} TFLOPS)")
    ratio = timings["mlx"] / timings["custom"]
    verdict = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    print(f"custom vs MLX:         {verdict}")


if __name__ == "__main__":
    main()
