from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 512)
    result = compile_callable(
        lambda value: (
            torch.nn.functional.softmax(value, dim=-1),
            torch.nn.functional.softmax(value, dim=1),
        ),
        (x,),
    )

    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    for kernel in result.plan.kernels:
        print("kernel:", kernel.name, kernel.kind)
        print(kernel.source)

    actual = result.plan(mx.array(x.numpy()))
    expected = (
        mx.array(torch.nn.functional.softmax(x, dim=-1).numpy()),
        mx.array(torch.nn.functional.softmax(x, dim=1).numpy()),
    )
    mx.eval(*actual, *expected)
    for name, output, expected_output in zip(
        ("last_dim", "middle_dim"), actual, expected, strict=True
    ):
        max_error = float(mx.max(mx.abs(output - expected_output)))
        print(f"{name} max abs error against PyTorch CPU: {max_error:.3e}")


if __name__ == "__main__":
    main()
