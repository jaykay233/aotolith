from __future__ import annotations

import time

import mlx.core as mx
import torch

import aotolith as aoto


def embedding_projection(
    token_ids: torch.Tensor,
    embedding: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    hidden = torch.nn.functional.embedding(token_ids, embedding)
    hidden = hidden.reshape(-1, hidden.shape[-1])
    return torch.nn.functional.silu(hidden @ projection).reshape(*token_ids.shape, -1)


def main() -> None:
    torch.manual_seed(0)
    torch_inputs = (
        torch.tensor([[1, 5, 2, 7], [3, 0, 4, 2]], dtype=torch.int64),
        torch.randn(8, 256),
        torch.randn(256, 512),
    )
    result = aoto.compile_callable(embedding_projection, torch_inputs)
    print("lowered ops:", [op.op for op in result.lowered_graph.ops])
    for kernel in result.plan.kernels:
        print(f"\n--- {kernel.name} ---\n{kernel.source}")

    inputs = tuple(mx.array(value.numpy()) for value in torch_inputs)
    start = time.perf_counter()
    output = result.plan(*inputs)[0]
    mx.eval(output)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"\noutput shape: {output.shape}, first call: {elapsed:.2f} ms")


if __name__ == "__main__":
    main()
