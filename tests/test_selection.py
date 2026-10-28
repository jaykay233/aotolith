from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def test_kernel_selection_records_native_and_fallback_ops() -> None:
    def fn(query, key, value):
        scores = query @ key.transpose(-1, -2)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        return attended + scores.mean(dim=-1, keepdim=True)

    torch.manual_seed(5)
    inputs = tuple(torch.randn(1, 2, 4, 8) for _ in range(3))
    compiled = compile_callable(fn, inputs)
    selections = compiled.plan.selections

    assert len(selections) == len(compiled.lowered_graph.ops)
    assert [selection.op_index for selection in selections] == list(
        range(len(selections))
    )
    assert all(selection.reason for selection in selections)
    assert any(
        selection.op == "attention" and not selection.is_native
        for selection in selections
    )
    assert any(
        selection.op == "matmul" and not selection.is_native
        for selection in selections
    )

    input_arrays = tuple(mx.array(value.numpy()) for value in inputs)
    actual = compiled.plan(*input_arrays)[0]
    mx.eval(actual)


def test_native_kernel_selection_names_the_artifact() -> None:
    value = torch.randn(4, 8)
    compiled = compile_callable(
        lambda current: torch.relu(current + 1.0),
        (value,),
        codegen_matmul=True,
    )
    native = [item for item in compiled.plan.selections if item.is_native]
    assert len(native) == 1
    assert native[0].implementation == "metal_kernel"
    assert native[0].artifact_name == compiled.plan.kernels[0].name
