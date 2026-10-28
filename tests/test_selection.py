from __future__ import annotations

import mlx.core as mx
import torch

from aotolith import compile_callable


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


def test_autotune_measures_native_and_mlx_candidates() -> None:
    torch.manual_seed(29)
    inputs = (torch.randn(8, 16), torch.randn(16, 12))
    compiled = compile_callable(
        lambda left, right: torch.relu(left @ right),
        inputs,
        autotune=True,
        autotune_warmup=1,
        autotune_iterations=3,
    )

    assert compiled.autotune
    assert compiled.plan.selections
    for selection in compiled.plan.selections:
        assert selection.autotuned
        assert selection.device_name
        assert {item.candidate for item in selection.benchmarks} == {
            "metal_kernel",
            "mlx",
        }
        assert "autotuned:" in selection.reason
        for benchmark in selection.benchmarks:
            assert benchmark.warmup_iterations == 1
            assert benchmark.iterations == 3
            assert benchmark.error is not None or (
                benchmark.median_seconds is not None
                and benchmark.min_seconds is not None
                and len(benchmark.samples_seconds) == 3
            )

    native_or_mlx = {selection.implementation for selection in compiled.plan.selections}
    assert native_or_mlx <= {"metal_kernel", "mlx"}
