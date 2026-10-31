"""Layout folding: drop identity views, cancel inverse pairs.

A traced model is full of transposes and reshapes that exist only because the
frontend thought in another layout. These passes remove the ones that do nothing
and collapse the ones that undo each other.
"""

from __future__ import annotations

from ..ir import Graph, Operation
from ..op_sets import LAYOUT_OPS
from .graph import (
    _consumers,
    _matmul_output_shape,
    _producers,
    _replace_graph,
)
from .matchers.view import (
    _is_identity_layout,
    _is_last_two_transpose,
    _normalized_dims,
    _normalized_flip_dims,
    _roll_shift_is_identity,
    _roll_shifts_by_axis,
    _transpose_axes,
)


def simplify_layouts(graph: Graph) -> Graph:
    """Remove identity views and fold adjacent inverse layout operations."""

    graph.validate()
    graph = _remove_identity_layouts(graph)
    while True:
        graph, changed = _fold_inverse_layouts(graph)
        if not changed:
            return graph


def fold_matmul_input_layouts(graph: Graph) -> Graph:
    """Fold single-use transpose and batch-expand views into matmul operands."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    removed: set[int] = set()
    replacements: dict[int, Operation] = {}

    for index, op in enumerate(graph.ops):
        if op.op not in {"matmul", "addmm"}:
            continue

        positions = (
            ((1, "transpose_a"), (2, "transpose_b"))
            if op.op == "addmm"
            else ((0, "transpose_a"), (1, "transpose_b"))
        )
        output_shape = graph.value(op.outputs[0]).type.shape
        for allow_expand in (True, False):
            inputs, attrs, local_removed = _fold_matmul_operand_layouts(
                graph,
                op,
                positions,
                producers,
                consumers,
                output_set,
                removed,
                allow_expand,
            )
            if not local_removed:
                break
            candidate = Operation(op.op, tuple(inputs), op.outputs, attrs)
            if _matmul_output_shape(graph, candidate) != output_shape:
                continue
            replacements[index] = candidate
            removed.update(local_removed)
            break

    if not removed:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = replacements.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in removed:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _fold_matmul_operand_layouts(
    graph: Graph,
    op: Operation,
    positions: tuple[tuple[int, str], ...],
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    output_set: set[str],
    already_removed: set[int],
    allow_expand: bool,
) -> tuple[list[str], dict[str, object], set[int]]:
    inputs = list(op.inputs)
    attrs = dict(op.attrs)
    removed: set[int] = set()

    for position, attribute in positions:
        while True:
            operand_id = inputs[position]
            producer_index = producers.get(operand_id)
            if producer_index is None:
                break
            if (
                producer_index in already_removed
                or producer_index in removed
                or operand_id in output_set
                or len(consumers.get(operand_id, [])) != 1
            ):
                break

            producer = graph.ops[producer_index]
            if producer.outputs != (operand_id,):
                break
            if not producer.inputs:
                # A constant producer (``zeros``, ``ones``, ``scalar_tensor``)
                # has no layout to fold; reading an input would fail. RNNCell
                # and every other cell built from a zero hidden state hit this.
                break
            operand_type = graph.value(operand_id).type
            source_type = graph.value(producer.inputs[0]).type
            if operand_type.dtype != source_type.dtype:
                break

            if producer.op in {"transpose", "permute"}:
                if not _is_last_two_transpose(producer, len(operand_type.shape)):
                    break
                expected_shape = source_type.shape[:-2] + (
                    source_type.shape[-1],
                    source_type.shape[-2],
                )
                if operand_type.shape != expected_shape:
                    break
                current = attrs.get(attribute, False)
                if not isinstance(current, bool):
                    raise ValueError(f"matmul {attribute} must be a boolean")
                inputs[position] = producer.inputs[0]
                attrs[attribute] = not current
            elif (
                allow_expand
                and producer.op == "expand"
                and len(source_type.shape) >= 2
                and len(operand_type.shape) >= 2
                and source_type.shape[-2:] == operand_type.shape[-2:]
            ):
                inputs[position] = producer.inputs[0]
            else:
                break

            removed.add(producer_index)

    return inputs, attrs, removed


def _remove_identity_layouts(graph: Graph) -> Graph:
    aliases: dict[str, str] = {}
    rewritten: list[Operation] = []

    def resolve(value_id: str) -> str:
        while value_id in aliases:
            value_id = aliases[value_id]
        return value_id

    for op in graph.ops:
        inputs = tuple(resolve(value_id) for value_id in op.inputs)
        current = (
            op
            if inputs == op.inputs
            else Operation(op.op, inputs, op.outputs, op.attrs)
        )
        if (current.op in LAYOUT_OPS or current.op == "pad") and (
            _is_identity_layout(graph, current)
        ):
            aliases[current.outputs[0]] = current.inputs[0]
            continue
        rewritten.append(current)

    outputs = tuple(resolve(value_id) for value_id in graph.outputs)
    return _replace_graph(graph, tuple(rewritten), outputs)


def _fold_inverse_layouts(graph: Graph) -> tuple[Graph, bool]:
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    aliases: dict[str, str] = {}
    removed: set[int] = set()
    replacements: dict[int, Operation] = {}

    for index, op in enumerate(graph.ops):
        if op.op not in {
            "reshape",
            "transpose",
            "permute",
            "flip",
            "roll",
            "unsqueeze",
            "squeeze",
        }:
            continue
        producer_index = producers.get(op.inputs[0])
        if producer_index is None:
            continue
        producer = graph.ops[producer_index]
        if (
            producer_index in removed
            or producer_index in replacements
            or len(consumers.get(op.inputs[0], [])) != 1
            or not producer.inputs
        ):
            continue
        if op.inputs[0] in output_set:
            continue
        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        producer_input_type = graph.value(producer.inputs[0]).type
        if input_type.dtype != output_type.dtype:
            continue
        if producer_input_type.dtype != graph.value(producer.outputs[0]).type.dtype:
            continue

        rank = len(input_type.shape)
        if op.op == "transpose" and producer.op == "transpose":
            outer_axes = _transpose_axes(op, rank)
            inner_axes = _transpose_axes(producer, rank)
            if outer_axes is not None and outer_axes == inner_axes:
                aliases[op.outputs[0]] = producer.inputs[0]
                removed.update({producer_index, index})
                continue

        if op.op == "permute" and producer.op == "permute":
            outer = _normalized_dims(op, rank)
            inner = _normalized_dims(producer, rank)
            if outer is None or inner is None:
                continue
            composed = tuple(inner[axis] for axis in outer)
            if composed == tuple(range(rank)):
                aliases[op.outputs[0]] = producer.inputs[0]
                removed.update({producer_index, index})
            continue

        if op.op == "flip" and producer.op == "flip":
            outer = _normalized_flip_dims(op, rank)
            inner = _normalized_flip_dims(producer, rank)
            if outer is not None and inner is not None and set(outer) == set(inner):
                aliases[op.outputs[0]] = producer.inputs[0]
                removed.update({producer_index, index})
            continue

        if op.op == "roll" and producer.op == "roll":
            outer = _roll_shifts_by_axis(op, input_type.shape)
            inner = _roll_shifts_by_axis(producer, input_type.shape)
            if outer is None or inner is None:
                continue
            combined = dict(inner)
            for axis, shift in outer.items():
                combined[axis] = combined.get(axis, 0) + shift
            if _roll_shift_is_identity(combined, input_type.shape):
                aliases[op.outputs[0]] = producer.inputs[0]
                removed.update({producer_index, index})
            continue

        if op.op == "reshape" and producer.op == "reshape":
            replacements[index] = Operation(
                "reshape", (producer.inputs[0],), op.outputs, op.attrs
            )
            removed.add(producer_index)
            continue

        if {op.op, producer.op} == {"unsqueeze", "squeeze"}:
            original_type = graph.value(producer.inputs[0]).type
            if output_type.shape == original_type.shape:
                aliases[op.outputs[0]] = producer.inputs[0]
                removed.update({producer_index, index})

    if not removed:
        return graph, False

    def resolve(value_id: str) -> str:
        while value_id in aliases:
            value_id = aliases[value_id]
        return value_id

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = replacements.get(index)
        if replacement is not None:
            rewritten.append(
                Operation(
                    replacement.op,
                    tuple(resolve(value_id) for value_id in replacement.inputs),
                    replacement.outputs,
                    replacement.attrs,
                )
            )
            continue
        if index in removed:
            continue
        rewritten.append(
            Operation(
                op.op,
                tuple(resolve(value_id) for value_id in op.inputs),
                op.outputs,
                op.attrs,
            )
        )
    outputs = tuple(resolve(value_id) for value_id in graph.outputs)
    return _replace_graph(graph, tuple(rewritten), outputs), True
