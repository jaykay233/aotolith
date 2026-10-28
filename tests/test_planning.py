from __future__ import annotations

import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable, plan_storage
from transformer_subgraph_compiler.backends import AppleMetalBackend
from transformer_subgraph_compiler.ir import DType, Graph, Operation, TensorType, Value


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

