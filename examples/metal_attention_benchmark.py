import time

import mlx.core as mx
import torch
import torch.nn.functional as F

import aotolith as aoto


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True)


def benchmark(fn, rounds: int = 21) -> float:
    for _ in range(5):
        mx.eval(fn())
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


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
    fallback = aoto.compile_callable(attention, torch_inputs)
    kernel = native.plan.kernels[0]
    inputs = tuple(
        mx.array(value.float().numpy(), dtype=mx.float16) for value in torch_inputs
    )
    actual = native.plan(*inputs)[0]
    expected = mx.array(attention(*torch_inputs).float().numpy(), dtype=mx.float16)
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))

    native_ms = benchmark(lambda: native.plan(*inputs)[0])
    mlx_ms = benchmark(lambda: fallback.plan(*inputs)[0])
    shape = f"{batch}x{heads}x{sequence}x{head_dim}"
    print(f"\nattention {shape}: {kernel.name}")
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")
    print(f"custom kernel:         {native_ms:.3f} ms")
    print(f"MLX SDPA:              {mlx_ms:.3f} ms")
    print(f"custom / MLX:          {native_ms / mlx_ms:.2f}x")


def main() -> None:
    run_case(1, 8, 512, 64)
    run_case(1, 8, 1024, 128)


if __name__ == "__main__":
    main()
