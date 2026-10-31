from __future__ import annotations

import mlx.core as mx
import pytest
import torch

from aotolith import compile_callable, plan_storage
from aotolith.backends import AppleMetalBackend
from aotolith.ir import DType, Graph, Operation, TensorType, Value


def test_storage_plan_tracks_lifetimes_and_reuses_dead_buffer() -> None:
    tensor_type = TensorType((4,), DType.F32)
    values = {
        value_id: Value(value_id, tensor_type)
        for value_id in ("x", "y", "z", "dead", "live", "out")
    }
    graph = Graph(
        ("x", "y", "z"),
        values,
        (
            Operation("add", ("x", "y"), ("dead",)),
            Operation("neg", ("x",), ("live",)),
            Operation("add", ("live", "z"), ("out",)),
        ),
        ("out",),
    )

    plan = plan_storage(graph)
    lifetime = {item.value_id: item for item in plan.lifetimes}
    assert (lifetime["dead"].start, lifetime["dead"].end) == (0, 1)
    assert (lifetime["live"].start, lifetime["live"].end) == (1, 3)
    assert lifetime["out"].is_output
    assert plan.peak_live_bytes >= 3 * 4 * 4
    assert "dead" in plan.release_after_op[0]
    assert "x" in plan.release_after_op[1]
    assert {"live", "z"} <= set(plan.release_after_op[2])
    reusable_slots = [slot for slot in plan.slots if "dead" in slot.value_ids]
    assert len(reusable_slots) == 1
    assert reusable_slots[0].value_ids == ("dead", "live")
    assert plan.arena_size_bytes >= reusable_slots[0].size_bytes


def test_compiled_plan_exposes_storage_plan_and_releases_intermediates() -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(value + 1.0)
        return hidden * 2.0

    value = torch.randn(8, 4)
    result = compile_callable(fn, (value,))
    assert result.plan.storage_plan.peak_live_bytes > 0
    assert result.plan.storage_plan.release_after_op == (("arg0",),)

    source = mx.array(value.numpy())
    actual = result.plan(source)[0]
    expected = mx.array(fn(value).numpy())
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected)


def test_unrolled_runner_agrees_with_the_interpreter() -> None:
    """The unrolled fast path must match the interpreted execution exactly."""

    def fn(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(value + 1.0)
        normalized = hidden / hidden.sum(dim=-1, keepdim=True)
        return torch.cat((normalized, hidden), dim=-1) * scale

    value = torch.randn(4, 6)
    scale = torch.randn(12)
    result = compile_callable(fn, (value, scale))
    assert result.plan._runner is None

    inputs = (mx.array(value.numpy()), mx.array(scale.numpy()))
    unrolled = result.plan(*inputs)[0]
    interpreted, _ = result.plan.call_with_stats(*inputs)
    mx.eval(unrolled, *interpreted)
    assert result.plan._runner is not None
    assert mx.array_equal(unrolled, interpreted[0])
    assert mx.allclose(unrolled, mx.array(fn(value, scale).numpy()))


def test_unrolled_runner_keeps_the_index_bounds_contract() -> None:
    def fn(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indices, source)

    table = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    indices = torch.tensor([0, 5], dtype=torch.int64)
    result = compile_callable(fn, (table, indices))

    # Build the unrolled runner on valid indices first.
    valid = result.plan(mx.array(table.numpy()), mx.array(indices.numpy()))[0]
    mx.eval(valid)
    assert result.plan._runner is not None

    with pytest.raises(IndexError, match="index out of range"):
        result.plan(
            mx.array(table.numpy()),
            mx.array([0, 6], dtype=mx.int64),
        )
    with pytest.raises(IndexError, match="index out of range"):
        result.plan(
            mx.array(table.numpy()),
            mx.array([-1, 0], dtype=mx.int64),
        )
    skipped = result.plan(
        mx.array(table.numpy()),
        mx.array([0, 6], dtype=mx.int64),
        verify_index_bounds=False,
    )[0]
    mx.eval(skipped)


def test_runtime_arena_reuses_a_planned_slot() -> None:
    tensor_type = TensorType((4,), DType.F32)
    values = {
        value_id: Value(value_id, tensor_type)
        for value_id in ("x", "y", "first", "out0", "second", "out1")
    }
    graph = Graph(
        ("x", "y"),
        values,
        (
            Operation("add", ("x", "y"), ("first",)),
            Operation("neg", ("first",), ("out0",)),
            Operation("add", ("x", "y"), ("second",)),
            Operation("neg", ("second",), ("out1",)),
        ),
        ("out0", "out1"),
    )

    plan = AppleMetalBackend().compile(graph)
    assert len(plan.storage_plan.slots) == 1
    assert plan.storage_plan.slots[0].value_ids == ("first", "second")

    x = mx.array([1.0, 2.0, 3.0, 4.0])
    y = mx.array([10.0, 20.0, 30.0, 40.0])
    actual, stats = plan.call_with_stats(x, y)
    mx.eval(*actual)

    expected = -(x + y)
    assert mx.allclose(actual[0], expected)
    assert mx.allclose(actual[1], expected)
    assert stats.arena_size_bytes == 16
    assert stats.arena_slot_count == 1
    assert [(item.value_id, item.reused) for item in stats.assignments] == [
        ("first", False),
        ("second", True),
    ]
    assert stats.reuse_count == 1


def test_every_operator_the_backend_executes_has_a_contract() -> None:
    """The registry is only a guarantee if nothing can bypass it.

    A new pointwise or layout operator reaches the runtime through the shared
    sets in ``passes``, so those names are where an operator could slip past
    verification. This test fails as soon as that happens.

    The legalizer's own table is the second way in: ``_SUPPORTED_OPS`` names
    the core op every ATen target becomes, and an operator that appears there
    but not in the registry is skipped by the backend's
    ``if op.op in OP_CONTRACTS`` guard - it still runs, just unvalidated.
    ``erfinv`` sat in exactly that hole, so both sets are checked.
    """

    from aotolith.ir.ops import OP_CONTRACTS
    from aotolith.legalize import targets
    from aotolith.passes import LAYOUT_OPS, POINTWISE_OPS

    registered = set(OP_CONTRACTS)
    for group in (POINTWISE_OPS, LAYOUT_OPS, set(targets._SUPPORTED_OPS.values())):
        missing = sorted(group - registered)
        assert not missing, f"no contract registered for {missing}"


def test_contracts_reject_unknown_operators_by_name() -> None:
    from aotolith.ir.ops import contract_for

    with pytest.raises(KeyError, match="not_really_an_op"):
        contract_for("not_really_an_op")


def test_contract_reports_the_number_of_values_an_operator_writes() -> None:
    """A multi-output operator is pinned to its own arity of results."""

    from aotolith.ir.ops import check_operation

    tensor_type = TensorType((2, 4), DType.F32)
    values = {
        "x": Value("x", tensor_type),
        "out": Value("out", tensor_type),
        "second": Value("second", tensor_type),
        "third": Value("third", tensor_type),
    }
    attrs = {
        "normalized_shape": (4,),
        "eps": 1e-5,
        "has_weight": False,
        "has_bias": False,
    }
    one_output = Graph(
        ("x",),
        values,
        (Operation("layer_norm", ("x",), ("out",), attrs),),
        ("out",),
    )
    check_operation(one_output, one_output.ops[0])

    too_many = Graph(
        ("x",),
        values,
        (Operation("neg", ("x",), ("out", "second")),),
        ("out",),
    )
    with pytest.raises(ValueError, match="neg must produce 1 results, got 2"):
        check_operation(too_many, too_many.ops[0])
