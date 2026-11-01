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
    graph = _collapse_unit_axis_permutes(graph)
    while True:
        # Folding a pair can leave an identity behind - ``reshape`` composed out
        # of two ``reshape``s is often the source shape again - so the identity
        # sweep has to repeat until the graph stops changing rather than run
        # once up front.
        before = len(graph.ops)
        graph = _remove_identity_layouts(graph)
        graph, changed = _fold_inverse_layouts(graph)
        if not changed and len(graph.ops) == before:
            return graph


#: Views that cannot break C contiguity. A reshape copies when it has to, and
#: adding or removing a singleton axis renames strides without moving data.
_CONTIGUITY_PRESERVING_VIEWS = frozenset(
    {"unsqueeze", "squeeze", "alias", "contiguous"}
)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _permutation_axes(op: Operation, rank: int) -> tuple[int, ...] | None:
    if op.op == "transpose":
        axes = _transpose_axes(op, rank)
        if axes is None:
            return None
        first, second = axes
        order = list(range(rank))
        order[first], order[second] = order[second], order[first]
        return tuple(order)
    if op.op == "permute":
        return _normalized_dims(op, rank)
    return None


def _permute_preserves_memory(
    op: Operation, shape: tuple[int, ...], source_is_contiguous: bool
) -> bool:
    """Say whether a permute leaves the array in the same C order it arrived in.

    The test is strides, not shapes. Swapping two axes whose sizes differ does
    *not* commute with flattening: ``(1, 5, 2, 2, 8)`` transposed on axes 0 and
    3 is ``(2, 5, 2, 1, 8)`` in that index order, but the stride that reaches
    each element is the one the old axis had, and flattening that view reads a
    different sequence than reshaping the original does. So this compares the
    permuted strides against the strides a contiguous array of the output shape
    would have, axis by axis, skipping singleton axes - which contribute no
    offset - and only rewrites when every axis that carries data already agrees.

    That makes ``(1, 1, 14, 64) -> (1, 14, 1, 64)`` fold (the reachable axes
    keep their strides) and leaves the cross-attention case alone.
    """

    if not source_is_contiguous:
        return False
    rank = len(shape)
    order = _permutation_axes(op, rank)
    if order is None or len(order) != rank or sorted(order) != list(range(rank)):
        return False
    output_shape = tuple(shape[axis] for axis in order)
    source_strides = _contiguous_strides(shape)
    permuted_strides = tuple(source_strides[axis] for axis in order)
    target_strides = _contiguous_strides(output_shape)
    return all(
        size <= 1 or permuted == expected
        for size, permuted, expected in zip(
            output_shape, permuted_strides, target_strides, strict=True
        )
    )


def _produces_contiguous(graph: Graph, producers: dict[str, int], value_id: str) -> bool:
    """Say whether ``value_id`` is known to arrive C-contiguous.

    A reshape either passes contiguity through or copies into a fresh buffer, so
    it always ends contiguous; squeeze and unsqueeze rename axes without moving
    data, so they inherit it; and a slice keeps it exactly when the axes it
    leaves behind sit at the strides the shorter shape expects - which is the
    same stride test the permute collapse uses, applied to the source's strides.

    Operators that materialize a buffer are contiguous by construction. A graph
    input is deliberately not: nothing here proves the caller's array is
    contiguous, and rewriting a view of a strided array is the mistake this
    guard exists to prevent.
    """

    producer = producers.get(value_id)
    if producer is None:
        return False
    op = graph.ops[producer]
    if op.op == "reshape" or op.op not in LAYOUT_OPS:
        return True
    if op.op in _CONTIGUITY_PRESERVING_VIEWS:
        return _produces_contiguous(graph, producers, op.inputs[0])
    if op.op == "slice" and op.inputs:
        source_id = op.inputs[0]
        if not _produces_contiguous(graph, producers, source_id):
            return False
        output_shape = graph.value(op.outputs[0]).type.shape
        source_strides = _contiguous_strides(graph.value(source_id).type.shape)
        expected = _contiguous_strides(output_shape)
        return all(
            size <= 1 or stride == want
            for size, stride, want in zip(
                output_shape, source_strides, expected, strict=True
            )
        )
    return False


def _collapse_unit_axis_permutes(graph: Graph) -> Graph:
    """Rewrite permutes that only relabel axes into ``reshape``.

    Both produce the same values, but a ``reshape`` is metadata-only while a
    ``transpose`` hands its consumer a non-contiguous array. The reshape is
    also the spelling the inverse-pair folder understands, so writing it this
    way lets the rest of the pass cancel it against a neighbouring reshape.
    """

    producers = _producers(graph)
    rewritten: list[Operation] = []
    changed = False
    for op in graph.ops:
        if op.op in {"transpose", "permute"}:
            shape = graph.value(op.inputs[0]).type.shape
            if _permute_preserves_memory(
                op, shape, _produces_contiguous(graph, producers, op.inputs[0])
            ):
                rewritten.append(
                    Operation(
                        "reshape",
                        op.inputs,
                        op.outputs,
                        {"shape": graph.value(op.outputs[0]).type.shape},
                    )
                )
                changed = True
                continue
        rewritten.append(op)
    if not changed:
        return graph
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def collapse_trailing_layouts(graph: Graph) -> Graph:
    """Run :func:`simplify_layouts` to a fixpoint after every fusion has landed.

    The pipeline simplifies layouts once, early, because the matchers that
    follow read shapes through whatever the frontend left behind. But the
    fusions themselves emit views - a fused attention region reshapes its
    output back to the residual stream, a fused rope region transposes to the
    head layout - and nothing sweeps those up afterwards. They are not free:
    the decode runner launches one node per view, and a node costs about the
    same to dispatch whether it moves a byte or not.
    """

    graph.validate()
    while True:
        before = len(graph.ops)
        graph = simplify_layouts(graph)
        if len(graph.ops) == before:
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
            or not producer.inputs
        ):
            continue
        exclusive = len(consumers.get(op.inputs[0], [])) == 1
        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        producer_input_type = graph.value(producer.inputs[0]).type
        if input_type.dtype != output_type.dtype:
            continue
        if producer_input_type.dtype != graph.value(producer.outputs[0]).type.dtype:
            continue

        rank = len(input_type.shape)

        # Reshape composition needs no exclusivity. ``reshape(reshape(x))`` is
        # ``reshape(x)`` whatever else reads the middle value, so a probe, a
        # second head or a residual path sharing it does not stop the fold.
        # When the composed shape is the shape the middle value started from,
        # the whole chain is an alias and only the outer op goes away - the
        # usual decode shape, ``(1, 896)`` out to ``(1, 1, 896)`` and back.
        if op.op == "reshape" and producer.op == "reshape":
            source_id = producer.inputs[0]
            if exclusive and op.inputs[0] not in output_set:
                replacements[index] = Operation(
                    "reshape", (source_id,), op.outputs, op.attrs
                )
                removed.add(producer_index)
            elif graph.value(source_id).type.shape == output_type.shape:
                aliases[op.outputs[0]] = source_id
                removed.add(index)
            continue

        if not exclusive:
            continue
        if op.inputs[0] in output_set:
            continue

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
