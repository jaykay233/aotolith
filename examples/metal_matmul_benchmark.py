from __future__ import annotations

import statistics
import time

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def benchmark(fn, rounds: int = 9) -> float:
    for _ in range(3):
        mx.eval(fn())
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(0)
    torch_inputs = (torch.randn(512, 2048), torch.randn(2048, 512))
    compiled = compile_callable(matmul, torch_inputs, codegen_matmul=True)
    fallback = compile_callable(matmul, torch_inputs)
    kernel = compiled.plan.kernels[0]
    print("custom kernel:", kernel.name)
    print("shared-memory tile: 64x64x32")
    print("register microtile:", "8x4" if "8x4" in kernel.name else "4x4")

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = compiled.plan(*inputs)[0]
    expected = mx.array(matmul(*torch_inputs).numpy())
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")

    custom_ms = benchmark(lambda: compiled.plan(*inputs)[0])
    mlx_ms = benchmark(lambda: fallback.plan(*inputs)[0])
    print(f"custom tiled:          {custom_ms:.3f} ms")
    print(f"MLX matmul:            {mlx_ms:.3f} ms")
    print(f"custom / MLX:          {custom_ms / mlx_ms:.2f}x")


if __name__ == "__main__":
    main()
