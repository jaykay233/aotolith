from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)

    def reductions(value):
        return (
            torch.sum(value, dim=-1),
            torch.mean(value, dim=(1, 2), keepdim=True),
        )

    result = compile_callable(reductions, (x,))
    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    for kernel in result.plan.kernels:
        print("kernel:", kernel.name, kernel.kind)
        print(kernel.source)

    actual = result.plan(mx.array(x.numpy()))
    expected = tuple(mx.array(value.numpy()) for value in reductions(x))
    mx.eval(*actual, *expected)
    for name, actual_value, expected_value in zip(
        ("sum", "mean"), actual, expected, strict=True
    ):
        max_error = float(mx.max(mx.abs(actual_value - expected_value)))
        print(f"{name} max abs error against PyTorch CPU: {max_error:.3e}")


if __name__ == "__main__":
    main()
