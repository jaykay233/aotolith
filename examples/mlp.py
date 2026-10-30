from __future__ import annotations

import time

import mlx.core as mx
import torch

import aotolith as aoto


def mlp(
    x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
):
    gate = x @ gate_w
    up = x @ up_w
    return (torch.nn.functional.silu(gate) * up) @ down_w


def main() -> None:
    torch.manual_seed(0)
    hidden = 1024
    intermediate = 4096
    batch = 1
    torch_inputs = (
        torch.randn(batch, hidden),
        torch.randn(hidden, intermediate),
        torch.randn(hidden, intermediate),
        torch.randn(intermediate, hidden),
    )
    result = aoto.compile_callable(mlp, torch_inputs)
    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    for kernel in result.plan.kernels:
        print(f"\n--- {kernel.name} ---\n{kernel.source}")

    inputs = tuple(mx.array(value.detach().numpy()) for value in torch_inputs)
    start = time.perf_counter()
    output = result.plan(*inputs)[0]
    mx.eval(output)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\noutput shape: {output.shape}, first call: {elapsed:.2f} ms")


if __name__ == "__main__":
    main()
