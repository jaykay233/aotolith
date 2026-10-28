from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 512)
    weight = torch.randn(512)
    bias = torch.randn(512)
    result = compile_callable(
        lambda value, scale, shift: torch.nn.functional.layer_norm(
            value, (512,), scale, shift, eps=1e-5
        ),
        (x, weight, bias),
    )
    kernel = result.plan.kernels[0]

    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    print("kernel:", kernel.name, kernel.kind)
    print(kernel.source)

    actual = result.plan(
        mx.array(x.numpy()), mx.array(weight.numpy()), mx.array(bias.numpy())
    )[0]
    expected = mx.array(
        torch.nn.functional.layer_norm(x, (512,), weight, bias, eps=1e-5).numpy()
    )
    mx.eval(actual, expected)
    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")


if __name__ == "__main__":
    main()
