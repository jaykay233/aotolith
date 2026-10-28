from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def mlp(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor):
    gate = x @ gate_w
    up = x @ up_w
    return (torch.nn.functional.silu(gate) * up) @ down_w


def main() -> None:
    torch.manual_seed(0)
    torch_inputs = (
        torch.randn(2, 16),
        torch.randn(16, 32),
        torch.randn(16, 32),
        torch.randn(32, 16),
    )
    result = compile_callable(mlp, torch_inputs, codegen_matmul=True)
    print("kernels:", [kernel.kind for kernel in result.plan.kernels])

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    actual = result.plan(*inputs)[0]
    expected = mx.array(mlp(*torch_inputs).numpy())
    mx.eval(actual, expected)

    max_error = float(mx.max(mx.abs(actual - expected)))
    print(f"output shape: {actual.shape}")
    print(f"max abs error against PyTorch CPU: {max_error:.3e}")


if __name__ == "__main__":
    main()
