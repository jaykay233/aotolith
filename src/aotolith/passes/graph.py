"""The graph as a structure: indexes, rewiring, and shape facts.

Nothing here knows what a pass is. Every matcher and every rewrite needs to ask
who produces a value, who consumes it, what shape an operator has, or how to
build the rewritten graph; those questions live here so the matchers do not each
answer them differently.
"""

from __future__ import annotations

from collections.abc import Callable

from ..ir import Graph, Operation, Value


def _producers(graph: Graph) -> dict[str, int]:
    return {
        value_id: index for index, op in enumerate(graph.ops) for value_id in op.outputs
    }


def _consumers(graph: Graph) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(graph.ops):
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(index)
    return consumers


def _replace_graph(
    graph: Graph, ops: tuple[Operation, ...], outputs: tuple[str, ...]
) -> Graph:
    result = Graph(graph.inputs, graph.values, ops, outputs)
    result.validate()
    return result


def _fresh_value_id(prefix: str, graph: Graph, extra_values: dict[str, Value]) -> str:
    candidate = prefix
    suffix = 0
    while candidate in graph.values or candidate in extra_values:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _retarget_input(op: Operation, previous: str, replacement: str) -> Operation:
    inputs = tuple(
        replacement if value_id == previous else value_id for value_id in op.inputs
    )
    return Operation(op.op, inputs, op.outputs, op.attrs)


def _rewrite_fused_region_inputs(
    op: Operation, inputs: tuple[str, ...]
) -> dict[str, object]:
    renames = dict(zip(op.inputs, inputs, strict=True))
    attrs = dict(op.attrs)
    body = op.attrs.get("body")
    if isinstance(body, (tuple, list)) and all(
        isinstance(body_op, Operation) for body_op in body
    ):
        attrs["body"] = tuple(
            Operation(
                body_op.op,
                tuple(renames.get(value_id, value_id) for value_id in body_op.inputs),
                body_op.outputs,
                body_op.attrs,
            )
            for body_op in body
        )
    return attrs


def _operation_attribute_key(
    graph: Graph,
    op: Operation,
    resolve: Callable[[str], str],
) -> tuple[object, ...]:
    if op.op == "fused_elementwise":
        normalized = _normalized_fused_region(graph, op, resolve)
        if normalized is not None:
            return ("fused_elementwise", normalized)
    return _freeze_attribute(op.attrs)


def _normalized_fused_region(
    graph: Graph,
    op: Operation,
    resolve: Callable[[str], str],
) -> tuple[object, ...] | None:
    body = op.attrs.get("body")
    if not isinstance(body, (tuple, list)) or not body:
        return None
    if not all(isinstance(body_op, Operation) for body_op in body):
        return None

    boundary = {
        value_id: ("argument", index) for index, value_id in enumerate(op.inputs)
    }
    internal: dict[str, object] = {}
    normalized_ops: list[object] = []
    for index, body_op in enumerate(body):
        normalized_inputs: list[object] = []
        for value_id in body_op.inputs:
            if value_id in internal:
                normalized_inputs.append(internal[value_id])
            elif value_id in boundary:
                normalized_inputs.append(boundary[value_id])
            else:
                normalized_inputs.append(("external", resolve(value_id)))

        output_name = ("value", index)
        output_type = graph.value(body_op.outputs[0]).type
        normalized_ops.append(
            (
                body_op.op,
                tuple(normalized_inputs),
                (output_type.shape, output_type.dtype.value),
                _freeze_attribute(body_op.attrs),
            )
        )
        internal[body_op.outputs[0]] = output_name

    root = internal.get(op.outputs[0])
    if root is None:
        return None
    return (tuple(normalized_ops), root)


def _freeze_attribute(value: object) -> tuple[object, ...]:
    if isinstance(value, Operation):
        return (
            "operation",
            value.op,
            value.inputs,
            value.outputs,
            _freeze_attribute(value.attrs),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (str(key), _freeze_attribute(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_freeze_attribute(item) for item in value))
    if isinstance(value, float):
        return ("float", repr(value))
    return (type(value).__name__, repr(value))


def _matmul_output_shape(graph: Graph, op: Operation) -> tuple[int, ...] | None:
    if op.op == "addmm":
        left_position, right_position = 1, 2
    else:
        left_position, right_position = 0, 1
    left_shape = graph.value(op.inputs[left_position]).type.shape
    right_shape = graph.value(op.inputs[right_position]).type.shape
    if len(left_shape) < 2 or len(right_shape) < 2:
        return None
    if op.attrs.get("transpose_a", False):
        left_shape = left_shape[:-2] + left_shape[-2:][::-1]
    if op.attrs.get("transpose_b", False):
        right_shape = right_shape[:-2] + right_shape[-2:][::-1]
    if left_shape[-1] != right_shape[-2]:
        return None

    batch_shape = _broadcast_shape(left_shape[:-2], right_shape[:-2])
    if batch_shape is None:
        return None
    return batch_shape + (left_shape[-2], right_shape[-1])


def _broadcast_shape(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...] | None:
    rank = max(len(left), len(right))
    result: list[int] = []
    for axis in range(rank):
        left_dim = left[axis - (rank - len(left))] if axis >= rank - len(left) else 1
        right_dim = (
            right[axis - (rank - len(right))] if axis >= rank - len(right) else 1
        )
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            return None
        result.append(max(left_dim, right_dim))
    return tuple(result)
