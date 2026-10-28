from __future__ import annotations

import statistics
import time

import mlx.core as mx
import torch

import aotolith as aoto


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def benchmark(fn, rounds: int = 21) -> float:
    for _ in range(5):
        mx.eval(fn())
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(0)
    run_case("f32", torch.float32, mx.float32)
    run_case("f16", torch.float16, mx.float16)


def run_case(label: str, torch_dtype, mlx_dtype) -> None:
    torch_inputs = (
        torch.randn(512, 2048, dtype=torch_dtype),
        torch.randn(2048, 512, dtype=torch_dtype),
    )
    compiled = aoto.compile_callable(matmul, torch_inputs, codegen_matmul=True)
    fallback = aoto.compile_callable(matmul, torch_inputs)
    kernel = compiled.plan.kernels[0]
    print(f"\n{label} matmul: {kernel.name}")

    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mlx_dtype) for value in torch_inputs
    )
    actual = compiled.plan(*inputs)[0]
    expected = mx.array(matmul(*torch_inputs).float().numpy(), dtype=mlx_dtype)
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")

    custom_ms = benchmark(lambda: compiled.plan(*inputs)[0])
    mlx_ms = benchmark(lambda: fallback.plan(*inputs)[0])
    print(f"custom kernel:         {custom_ms:.3f} ms")
    print(f"MLX matmul:            {mlx_ms:.3f} ms")
    print(f"custom / MLX:          {custom_ms / mlx_ms:.2f}x")


if __name__ == "__main__":
    main()
