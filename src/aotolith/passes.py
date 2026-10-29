from __future__ import annotations

from collections.abc import Callable
from math import prod

from .ir import DType, Graph, Operation, TensorType, Value

POINTWISE_OPS = {
    "silu",
    "gelu",
    "softplus",
    "mish",
    "logsigmoid",
    "mul",
    "add",
    "sub",
    "rsub",
    "div",
    "floor_divide",
    "remainder",
    "pow",
    "neg",
    "abs",
    "sigmoid",
    "tanh",
    "relu",
    "exp",
    "expm1",
    "exp2",
    "log",
    "log2",
    "log10",
    "log1p",
    "erf",
    "sin",
    "cos",
    "floor",
    "ceil",
    "round",
    "trunc",
    "sign",
    "square",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "clamp",
    "clamp_min",
    "clamp_max",
    "leaky_relu",
    "elu",
    "hardswish",
    "hardsigmoid",
    "threshold",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "isnan",
    "isinf",
    "where",
    "masked_fill",
    "maximum",
    "minimum",
    "logaddexp",
    "atan2",
    "scalar_tensor",
    "zeros",
    "ones",
    "full",
    "new_zeros",
    "new_ones",
    "new_full",
    "zeros_like",
    "ones_like",
    "full_like",
}

LAYOUT_OPS = {
    "reshape",
    "transpose",
    "permute",
    "expand",
    "slice",
    "flip",
    "roll",
    "unsqueeze",
    "squeeze",
    "contiguous",
    "alias",
}


def simplify_layouts(graph: Graph) -> Graph:
    """Remove identity views and fold adjacent inverse layout operations."""

    graph.validate()
    graph = _remove_identity_layouts(graph)
    while True:
        graph, changed = _fold_inverse_layouts(graph)
        if not changed:
            return graph


def canonicalize_batched_matmul(graph: Graph) -> Graph:
    """Recover rank-N matmul from PyTorch's fold-bmm-unfold decomposition."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    replaced: set[int] = set()
    replacement_at: dict[int, Operation] = {}

    for index, op in enumerate(graph.ops):
        if index in replaced or op.op != "matmul":
            continue
        canonical = _match_folded_batched_matmul(
            graph, index, producers, consumers, replaced
        )
        if canonical is None:
            continue
        left, right, output_id = canonical
        replacement_at[index] = Operation(
            "matmul", (left, right), (output_id,), op.attrs
        )

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = replacement_at.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in replaced:
            rewritten.append(op)

    result = _replace_graph(graph, tuple(rewritten), graph.outputs)
    return simplify_layouts(result)


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


def fuse_attention(graph: Graph) -> Graph:
    """Fold QK^T, scale/mask, softmax and PV into one attention operation."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    matched: set[int] = set()
    replacement_at: dict[int, Operation] = {}

    for pv_index, pv in enumerate(graph.ops):
        if pv_index in matched or pv.op != "matmul":
            continue
        match = _match_attention(graph, pv_index, producers, consumers, matched)
        if match is None:
            continue
        replacement_index, internal, attention = match
        matched.update(internal)
        replacement_at[replacement_index] = attention

    if not replacement_at:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = replacement_at.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in matched:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def fuse_rope(graph: Graph) -> Graph:
    """Fold rotary positional embedding into one ``rope`` operation.

    PyTorch writes RoPE as ``x * cos + rotate_half(x) * sin``, and
    ``rotate_half`` alone is two slices, a negation and a concatenation. For one
    decode token each of those operates on a few hundred elements, so every one
    of them costs a kernel launch and nothing else. Rewriting the whole pattern
    into a single op removes the launches and the intermediate tensors.
    """

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    removed: set[int] = set()

    for index, op in enumerate(graph.ops):
        if op.op != "add" or len(op.inputs) != 2 or index in removed:
            continue
        match = _match_rope(graph, index, producers, consumers, output_set)
        if match is None:
            continue
        internal, rope = match
        rewrites[index] = rope
        removed.update(internal)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in removed:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _match_rope(
    graph: Graph,
    index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    output_set: set[str],
) -> tuple[set[int], Operation] | None:
    """Match ``x * cos + rotate_half(x) * sin`` rooted at one ``add``."""

    add = graph.ops[index]
    if add.attrs.get("alpha", 1.0) != 1.0:
        return None
    output_id = add.outputs[0]
    if len(add.outputs) != 1:
        return None
    output_type = graph.value(output_id).type
    rank = len(output_type.shape)
    if rank == 0 or output_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        return None

    products: list[tuple[int, Operation]] = []
    for value_id in add.inputs:
        producer = producers.get(value_id)
        if producer is None:
            return None
        product = graph.ops[producer]
        if product.op != "mul" or len(product.inputs) != 2:
            return None
        if consumers.get(value_id) != [index]:
            return None
        if graph.value(value_id).type != output_type:
            return None
        products.append((producer, product))

    rotate_products = [
        entry
        for entry in products
        if any(
            producers.get(value_id) is not None
            and graph.ops[producers[value_id]].op == "cat"
            for value_id in entry[1].inputs
        )
    ]
    if len(rotate_products) != 1:
        return None
    rotate_index, rotate_product = rotate_products[0]
    data_index, data_product = next(
        entry for entry in products if entry[0] != rotate_index
    )

    rotated_id = next(
        value_id
        for value_id in rotate_product.inputs
        if graph.ops[producers[value_id]].op == "cat"
    )
    sine_id = next(
        value_id for value_id in rotate_product.inputs if value_id != rotated_id
    )
    cat_index = producers[rotated_id]
    cat = graph.ops[cat_index]
    if len(cat.inputs) != 2 or len(cat.outputs) != 1:
        return None
    cat_dim = cat.attrs.get("dim")
    if not isinstance(cat_dim, int) or cat_dim % rank != rank - 1:
        return None
    if graph.value(rotated_id).type != output_type:
        return None
    if consumers.get(rotated_id) != [rotate_index]:
        return None

    columns = output_type.shape[-1]
    if columns % 2 != 0:
        return None
    half = columns // 2

    sliced: list[int] = []
    sources: set[str] = set()
    neg_index: int | None = None
    for position, (expected_start, expected_end) in enumerate(
        ((half, columns), (0, half))
    ):
        piece_id = cat.inputs[position]
        producer = producers.get(piece_id)
        if producer is None:
            return None
        piece = graph.ops[producer]
        if len(piece.outputs) != 1:
            return None
        piece_type = graph.value(piece_id).type
        if piece_type.shape != output_type.shape[:-1] + (half,):
            return None
        if piece_type.dtype != output_type.dtype:
            return None
        if position == 0:
            if piece.op != "neg" or len(piece.inputs) != 1:
                return None
            if consumers.get(piece_id) != [cat_index]:
                return None
            neg_index = producer
            slice_id = piece.inputs[0]
            slice_index = producers.get(slice_id)
            if slice_index is None:
                return None
            if consumers.get(slice_id) != [neg_index]:
                return None
        else:
            slice_id = piece_id
            slice_index = producer
            if consumers.get(slice_id) != [cat_index]:
                return None
        piece_slice = graph.ops[slice_index]
        if piece_slice.op != "slice" or len(piece_slice.inputs) != 1:
            return None
        sliced.append(slice_index)
        if piece_slice.attrs.get("dim", 0) % rank != rank - 1:
            return None
        if piece_slice.attrs.get("step", 1) != 1:
            return None
        start = piece_slice.attrs.get("start", 0)
        end = piece_slice.attrs.get("end")
        if start != expected_start:
            return None
        if not isinstance(end, int) or end < expected_end:
            return None
        sources.add(piece_slice.inputs[0])

    if len(sources) != 1:
        return None
    source_id = sources.pop()
    if source_id not in data_product.inputs:
        return None
    if source_id in output_set:
        return None
    source_type = graph.value(source_id).type
    if source_type != output_type:
        return None

    cosine_id = next(
        value_id for value_id in data_product.inputs if value_id != source_id
    )
    for table_id in (cosine_id, sine_id):
        table_type = graph.value(table_id).type
        if table_type.dtype != output_type.dtype:
            return None
        if not _can_broadcast_to(table_type.shape, output_type.shape):
            return None
        if table_type.shape[-1] not in (1, columns):
            return None

    internal = {
        data_index,
        rotate_index,
        cat_index,
    }
    if neg_index is not None:
        internal.add(neg_index)
    internal.update(sliced)
    rope = Operation(
        "rope",
        (source_id, cosine_id, sine_id),
        (output_id,),
        {},
    )
    return internal, rope


def fuse_residual_epilogue(graph: Graph) -> Graph:
    """Fold a residual add into the matrix multiply that feeds it.

    ``hidden = hidden + delta``, where ``delta`` comes from a projection, costs a
    whole kernel launch for a single output row and reads the result back from
    memory only to write it again. Expressing it as ``addmm`` with the residual
    as the bias lets the projection accumulate it in the epilogue, and every
    matmul path already supports a broadcast bias. The projection is usually
    followed by a shape-only view, so the match walks back through layout
    operations and re-points the last one at the add's output value.
    """

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    retargets: dict[int, tuple[str, ...]] = {}
    removed: set[int] = set()
    inserted: dict[int, list[Operation]] = {}
    extra_values: dict[str, Value] = {}

    for index, op in enumerate(graph.ops):
        if op.op != "add" or len(op.inputs) != 2:
            continue
        if op.attrs.get("alpha", 1.0) != 1.0:
            continue
        output_id = op.outputs[0]
        output_type = graph.value(output_id).type
        match = _trace_add_operand_to_matmul(
            graph, index, producers, consumers, output_set
        )
        if match is None:
            continue
        product_index, chain, residual_id = match
        if graph.value(residual_id).type != output_type:
            continue
        # The residual has to be available before the projection runs; otherwise
        # replacing the projection would delete its own input.
        residual_producer = producers.get(residual_id)
        if residual_producer is not None and residual_producer >= product_index:
            continue
        product = graph.ops[product_index]
        # The epilogue adds the residual to the projection's own output, so the
        # residual has to line up with it. Rank differences are singleton-only
        # here, and a reshape is a free view, so one is inserted when needed.
        product_shape = graph.value(product.outputs[0]).type.shape
        residual_type = graph.value(residual_id).type
        if not _can_broadcast_to(residual_type.shape, product_shape):
            if prod(residual_type.shape) != prod(product_shape):
                continue
            view_id = _fresh_value_id(f"{residual_id}_epilogue", graph, extra_values)
            extra_values[view_id] = Value(
                view_id, TensorType(product_shape, residual_type.dtype)
            )
            inserted.setdefault(product_index, []).append(
                Operation("reshape", (residual_id,), (view_id,), {"shape": product_shape})
            )
            residual_id = view_id
        left_id, right_id = product.inputs
        attrs = dict(product.attrs)
        attrs["alpha"] = 1.0
        attrs["beta"] = 1.0
        created = Operation(
            "addmm",
            (residual_id, left_id, right_id),
            (product.outputs[0] if chain else output_id,),
            attrs,
        )
        rewrites[product_index] = created
        if chain:
            retargets[chain[-1]] = (output_id,)
        removed.add(index)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.extend(inserted.get(index, ()))
            rewritten.append(replacement)
            continue
        retarget = retargets.get(index)
        if retarget is not None:
            rewritten.append(Operation(op.op, op.inputs, retarget, op.attrs))
            continue
        if index not in removed:
            rewritten.append(op)
    values = dict(graph.values)
    values.update(extra_values)
    result = Graph(graph.inputs, values, tuple(rewritten), graph.outputs)
    result.validate()
    return result


def _fresh_value_id(
    prefix: str, graph: Graph, extra_values: dict[str, Value]
) -> str:
    candidate = prefix
    suffix = 0
    while candidate in graph.values or candidate in extra_values:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate



def _trace_add_operand_to_matmul(
    graph: Graph,
    index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    output_set: set[str],
) -> tuple[int, tuple[int, ...], str] | None:
    """Return ``(matmul, layout_chain, residual)`` for one residual ``add``."""

    add = graph.ops[index]
    for position in (0, 1):
        chain_reversed: list[int] = []
        current = add.inputs[position]
        while True:
            producer = producers.get(current)
            if producer is None:
                break
            candidate = graph.ops[producer]
            if candidate.op == "matmul":
                if current in output_set:
                    break
                if len(chain_reversed) > 2:
                    break
                chain = tuple(reversed(chain_reversed))
                return producer, chain, add.inputs[1 - position]
            if candidate.op not in LAYOUT_OPS or len(candidate.inputs) != 1:
                break
            if consumers.get(current) != [index] and not (
                chain_reversed and consumers.get(current) == [chain_reversed[-1]]
            ):
                break
            chain_reversed.append(producer)
            current = candidate.inputs[0]
    return None


def fuse_gate_up_projection(graph: Graph) -> Graph:
    """Project ``gate`` and ``up`` from one launch instead of two.

    A gated MLP feeds the same normalized hidden state to both halves of its
    projection, so a single-token step streams two weight matrices that are read
    against one activation and immediately multiplied. Emitting both rows from
    one kernel removes a launch and a barrier from the per-layer chain without
    changing how many bytes are read.

    This only fires for the single-token form. A taller left operand goes to the
    tiled matrix multiply, which has no paired epilogue, so the two projections
    stay separate and the result is unchanged.
    """

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    removed: set[int] = set()

    for index, op in enumerate(graph.ops):
        if op.op != "fused_elementwise" or len(op.inputs) != 2:
            continue
        body = op.attrs.get("body")
        if not isinstance(body, tuple) or len(body) != 2:
            continue
        activation, multiply = body
        if not isinstance(activation, Operation) or activation.op != "silu":
            continue
        if not isinstance(multiply, Operation) or multiply.op != "mul":
            continue
        if multiply.outputs[0] != op.outputs[0]:
            continue
        gate_id = activation.inputs[0]
        other_id = next(
            (
                value_id
                for value_id in multiply.inputs
                if value_id != activation.outputs[0]
            ),
            None,
        )
        if other_id is None:
            continue

        gate = _trace_operand_to_matmul(
            graph, gate_id, index, producers, consumers, output_set
        )
        product = _trace_operand_to_matmul(
            graph, other_id, index, producers, consumers, output_set
        )
        if gate is None or product is None:
            continue
        gate_index, gate_chain, gate_source = gate
        product_index, product_chain, _ = product
        if gate_index == product_index:
            continue
        gate_op = graph.ops[gate_index]
        product_op = graph.ops[product_index]
        # A bias term would need its own slot in the paired epilogue, and a
        # fused normalization is attached after this point, so only the plain
        # two-operand form qualifies.
        if gate_op.op != "matmul" or product_op.op != "matmul":
            continue
        if len(gate_op.inputs) != 2 or len(product_op.inputs) != 2:
            continue
        if gate_op.attrs != product_op.attrs:
            continue
        gate_weight = graph.value(gate_op.inputs[1]).type
        if gate_weight != graph.value(product_op.inputs[1]).type:
            continue
        if not _chain_preserves_order(graph, gate_chain):
            continue
        if not _chain_preserves_order(graph, product_chain):
            continue
        output_type = graph.value(product_op.outputs[0]).type
        if graph.value(gate_op.outputs[0]).type != output_type:
            continue
        consumed_id = (
            graph.ops[product_chain[-1]].outputs[0]
            if product_chain
            else product_op.outputs[0]
        )
        if graph.value(op.outputs[0]).type != graph.value(consumed_id).type:
            continue

        # The pair has to read the same activation, and the transposed layout has
        # to put the projection columns first for the two rows to be neighbours.
        gate_left = gate_op.inputs[0]
        left_type = graph.value(gate_left).type
        if prod(left_type.shape[:-1]) != 1:
            continue
        reduction = left_type.shape[-1]
        if reduction % 4 != 0:
            continue
        if gate_weight.shape != (output_type.shape[-1], reduction):
            continue
        if gate_op.attrs.get("transpose_b") is not True:
            continue
        # A transposed left operand would be routed to the tiled kernel, which
        # has no paired form.
        if gate_op.attrs.get("transpose_a") is True:
            continue
        # Both operands are reshapes of one tensor, so either can feed the fused
        # projection; the product's copy is dropped and the gate's is kept.
        gate_root, _ = _single_consumer_reshape_chain(
            graph, gate_left, gate_index, producers, consumers
        )
        product_root, product_left_chain = _single_consumer_reshape_chain(
            graph, product_op.inputs[0], product_index, producers, consumers
        )
        if gate_root != product_root:
            continue
        if graph.value(product_op.inputs[0]).type != left_type:
            continue
        # Dropping the gate projection is only sound when nothing else reads it.
        gate_readers = [gate_chain[0]] if gate_chain else [index]
        if consumers.get(gate_source) != gate_readers:
            continue

        attrs = dict(gate_op.attrs)
        attrs["epilogue"] = "silu_mul_paired"
        rewrites[product_index] = Operation(
            "matmul",
            (gate_left, gate_op.inputs[1], product_op.inputs[1]),
            product_op.outputs if product_chain else (op.outputs[0],),
            attrs,
        )
        if product_chain:
            last = product_chain[-1]
            rewrites[last] = Operation(
                "reshape",
                (graph.ops[last].inputs[0],),
                (op.outputs[0],),
                {"shape": graph.value(op.outputs[0]).type.shape},
            )
        removed.add(index)
        removed.add(gate_index)
        removed.update(gate_chain)
        removed.update(product_left_chain)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in removed:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _single_consumer_reshape_chain(
    graph: Graph,
    value_id: str,
    consumer_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
) -> tuple[str, tuple[int, ...]]:
    """Walk back through reshapes that exist only to feed one consumer."""

    chain: list[int] = []
    current = value_id
    while True:
        producer = producers.get(current)
        if producer is None:
            break
        candidate = graph.ops[producer]
        if candidate.op != "reshape" or len(candidate.inputs) != 1:
            break
        if consumers.get(current) != [consumer_index]:
            break
        chain.append(producer)
        consumer_index = producer
        current = candidate.inputs[0]
    return current, tuple(chain)


def fuse_activation_epilogue(graph: Graph) -> Graph:
    """Fold ``silu(gate) * up`` into the projection that produces ``up``.

    The gated MLP activation runs on a wide but short tensor once per layer, so a
    standalone pass over it is another launch and another round trip through
    memory. The projection already has both operands live, and ``silu`` of the
    gate is elementwise, so the multiply moves into its epilogue.
    """

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    removed: set[int] = set()

    for index, op in enumerate(graph.ops):
        if op.op != "fused_elementwise" or len(op.inputs) != 2:
            continue
        body = op.attrs.get("body")
        if not isinstance(body, tuple) or len(body) != 2:
            continue
        activation, multiply = body
        if not isinstance(activation, Operation) or activation.op != "silu":
            continue
        if not isinstance(multiply, Operation) or multiply.op != "mul":
            continue
        if multiply.outputs[0] != op.outputs[0]:
            continue
        gate_id = activation.inputs[0]
        other_id = next(
            (value_id for value_id in multiply.inputs if value_id != activation.outputs[0]),
            None,
        )
        if other_id is None:
            continue
        gate = _trace_operand_to_matmul(
            graph, gate_id, index, producers, consumers, output_set
        )
        product = _trace_operand_to_matmul(
            graph, other_id, index, producers, consumers, output_set
        )
        if gate is None or product is None:
            continue
        gate_index, gate_chain, gate_source = gate
        product_index, product_chain, _ = product
        gate_op = graph.ops[gate_index]
        product_op = graph.ops[product_index]
        if graph.value(gate_op.outputs[0]).type != graph.value(
            product_op.outputs[0]
        ).type:
            continue
        # The epilogue reads the gate at the projection's own output index, so
        # the two chains must be pure reshapes that preserve element order.
        if not _chain_preserves_order(graph, gate_chain):
            continue
        if not _chain_preserves_order(graph, product_chain):
            continue
        attrs = dict(product_op.attrs)
        attrs["epilogue"] = "silu_mul"
        rewrites[product_index] = Operation(
            product_op.op,
            product_op.inputs + (gate_source,),
            (op.outputs[0],) if not product_chain else product_op.outputs,
            attrs,
        )
        if product_chain:
            removed.add(index)
            rewrites[product_chain[-1]] = Operation(
                "reshape",
                (graph.ops[product_chain[-1]].inputs[0],),
                (op.outputs[0],),
                {"shape": graph.value(op.outputs[0]).type.shape},
            )
        else:
            removed.add(index)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in removed:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _trace_operand_to_matmul(
    graph: Graph,
    value_id: str,
    consumer_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    output_set: set[str],
) -> tuple[int, tuple[int, ...], str] | None:
    """Walk back from ``value_id`` through shape-only views to a matrix multiply."""

    chain_reversed: list[int] = []
    current = value_id
    while True:
        producer = producers.get(current)
        if producer is None:
            return None
        candidate = graph.ops[producer]
        if candidate.op in {"matmul", "addmm"}:
            if current in output_set:
                return None
            return producer, tuple(reversed(chain_reversed)), current
        if candidate.op not in LAYOUT_OPS or len(candidate.inputs) != 1:
            return None
        downstream = [consumer_index] if not chain_reversed else [chain_reversed[-1]]
        if consumers.get(current) != downstream:
            return None
        chain_reversed.append(producer)
        current = candidate.inputs[0]


def _chain_preserves_order(graph: Graph, chain: tuple[int, ...]) -> bool:
    for index in chain:
        if graph.ops[index].op != "reshape":
            return False
    return True


def fuse_rms_norm_epilogue(graph: Graph) -> Graph:
    """Fold ``rms_norm(x) @ w`` into the projection's GEMV prologue.

    A single-token projection already streams the whole weight matrix and
    reads the normalized activation once per output column, so recomputing the
    RMS scale inside that kernel costs one extra pass over a cache-resident
    vector and removes a launch, a barrier, and a memory round trip.
    """

    graph.validate()
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    removable: set[int] = set()

    for norm_index, norm in enumerate(graph.ops):
        if norm.op != "rms_norm":
            continue
        plan = _match_rms_norm_consumers(
            graph, norm_index, norm, consumers, output_set
        )
        if plan is None:
            continue
        projection_rewrites, layout_rewrites = plan
        rewrites.update(layout_rewrites)
        rewrites.update(projection_rewrites)
        removable.add(norm_index)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index not in removable:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _match_rms_norm_consumers(
    graph: Graph,
    norm_index: int,
    norm: Operation,
    consumers: dict[str, list[int]],
    output_set: set[str],
) -> tuple[dict[int, Operation], dict[int, Operation]] | None:
    """Return projection rewrites plus layout rewrites fed by one norm."""

    if norm.outputs[0] in output_set:
        return None
    if norm.attrs.get("has_weight") is not True:
        return None
    normalized_shape = norm.attrs.get("normalized_shape")
    if not isinstance(normalized_shape, tuple) or len(normalized_shape) != 1:
        return None
    if len(norm.inputs) != 2:
        return None
    eps = norm.attrs.get("eps")
    if not isinstance(eps, float):
        return None
    source_id, weight_id = norm.inputs
    source_type = graph.value(source_id).type
    weight_type = graph.value(weight_id).type
    reduction = normalized_shape[0]
    if weight_type.shape != normalized_shape:
        return None
    if source_type.shape[-1] != reduction or prod(source_type.shape) != reduction:
        return None
    if reduction % 8 != 0:
        return None

    # Walk the layout chain; every leaf must be a projection that can absorb
    # the normalization, otherwise the standalone kernel has to stay.
    layout_rewrites: dict[int, Operation] = {}
    projections: list[int] = []
    frontier = [norm.outputs[0]]
    while frontier:
        next_frontier: list[str] = []
        for value_id in frontier:
            for consumer in consumers.get(value_id, []):
                candidate = graph.ops[consumer]
                if candidate.op in {"matmul", "addmm"}:
                    projections.append(consumer)
                elif candidate.op == "reshape":
                    layout_rewrites[consumer] = _retarget_input(
                        candidate, value_id, source_id
                    )
                    next_frontier.extend(candidate.outputs)
                else:
                    return None
        frontier = next_frontier

    if not projections:
        return None

    rewrites: dict[int, Operation] = {}
    for projection_index in projections:
        projection = graph.ops[projection_index]
        left_position = 1 if projection.op == "addmm" else 0
        left_id = projection.inputs[left_position]
        left_type = graph.value(left_id).type
        output_type = graph.value(projection.outputs[0]).type
        if left_type.dtype != source_type.dtype:
            return None
        if weight_type.dtype != source_type.dtype:
            return None
        if prod(left_type.shape[:-1]) != 1:
            return None
        if projection.attrs.get("rms_norm_eps") is not None:
            return None
        if output_type.shape[:-2] != left_type.shape[:-2]:
            return None
        attrs = dict(projection.attrs)
        attrs["rms_norm_eps"] = eps
        inputs = projection.inputs
        if left_id == norm.outputs[0]:
            # ``rms_norm`` preserves shape, so the raw source can stand in.
            inputs = (
                inputs[:left_position] + (source_id,) + inputs[left_position + 1 :]
            )
        rewrites[projection_index] = Operation(
            projection.op, inputs + (weight_id,), projection.outputs, attrs
        )
    return rewrites, layout_rewrites


def _retarget_input(op: Operation, previous: str, replacement: str) -> Operation:
    inputs = tuple(
        replacement if value_id == previous else value_id for value_id in op.inputs
    )
    return Operation(op.op, inputs, op.outputs, op.attrs)


def fuse_pointwise(graph: Graph) -> Graph:
    """Fuse adjacent single-use pointwise operations into one Metal region."""

    graph.validate()
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(graph.ops):
        for value_id in op.outputs:
            producers[value_id] = index
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(index)

    pointwise = {
        index
        for index, op in enumerate(graph.ops)
        if _is_pointwise_fusion_candidate(graph, op)
    }
    roots: list[int] = []
    for index in pointwise:
        output_id = graph.ops[index].outputs[0]
        node_consumers = consumers.get(output_id, [])
        if output_id in graph.outputs or any(
            consumer not in pointwise for consumer in node_consumers
        ):
            roots.append(index)
    roots.sort()

    fused_at: dict[int, Operation] = {}
    removed_members: set[int] = set()
    assigned: set[int] = set()
    output_set = set(graph.outputs)

    for root in roots:
        if root in assigned:
            continue
        members = {root}
        changed = True
        while changed:
            changed = False
            for index in sorted(members):
                op = graph.ops[index]
                for value_id in op.inputs:
                    producer = producers.get(value_id)
                    if producer is None or producer in members or producer not in pointwise:
                        continue
                    producer_output = graph.ops[producer].outputs[0]
                    if producer_output in output_set:
                        continue
                    producer_consumers = set(consumers.get(producer_output, []))
                    if producer_consumers.issubset(members):
                        members.add(producer)
                        changed = True
        if len(members) < 2:
            continue

        ordered = sorted(members)
        body = tuple(graph.ops[index] for index in ordered)
        member_outputs = {
            output_id for index in members for output_id in graph.ops[index].outputs
        }
        boundary_inputs: list[str] = []
        for op in body:
            for value_id in op.inputs:
                if value_id not in member_outputs and value_id not in boundary_inputs:
                    boundary_inputs.append(value_id)

        output_id = graph.ops[root].outputs[0]
        fused = Operation(
            "fused_elementwise",
            tuple(boundary_inputs),
            (output_id,),
            {"body": body},
        )
        fused_at[root] = fused
        removed_members.update(members)
        assigned.update(members)

    if not fused_at:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        fused = fused_at.get(index)
        if fused is not None:
            rewritten.append(fused)
        elif index not in removed_members:
            rewritten.append(op)

    return graph.replace_ops(tuple(rewritten))


def _is_pointwise_fusion_candidate(graph: Graph, op: Operation) -> bool:
    return op.op in POINTWISE_OPS


def eliminate_common_subexpressions(graph: Graph) -> Graph:
    """Reuse identical pure operations with the same inputs and attributes."""

    graph.validate()
    aliases: dict[str, str] = {}

    def resolve(value_id: str) -> str:
        while value_id in aliases:
            value_id = aliases[value_id]
        return value_id

    common: dict[tuple[object, ...], str] = {}
    rewritten: list[Operation] = []
    for op in graph.ops:
        inputs = tuple(resolve(value_id) for value_id in op.inputs)
        attrs = op.attrs
        if op.op == "fused_elementwise" and inputs != op.inputs:
            attrs = _rewrite_fused_region_inputs(op, inputs)
        current = (
            op
            if inputs == op.inputs and attrs is op.attrs
            else Operation(op.op, inputs, op.outputs, attrs)
        )
        output_type = graph.value(op.outputs[0]).type
        key = (
            current.op,
            current.inputs,
            (output_type.shape, output_type.dtype.value),
            _operation_attribute_key(graph, current, resolve),
        )
        existing = common.get(key)
        if existing is not None:
            aliases[current.outputs[0]] = existing
            continue
        common[key] = current.outputs[0]
        rewritten.append(current)

    outputs = tuple(resolve(value_id) for value_id in graph.outputs)
    return _replace_graph(graph, tuple(rewritten), outputs)


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
        if current.op in LAYOUT_OPS and _is_identity_layout(graph, current):
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


def _is_identity_layout(graph: Graph, op: Operation) -> bool:
    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_type.shape)
    if input_type.dtype != output_type.dtype:
        return False

    if op.op in {"contiguous", "alias"}:
        return input_type.shape == output_type.shape
    if op.op in {"reshape", "expand"}:
        return input_type.shape == output_type.shape
    if op.op == "squeeze":
        return input_type.shape == output_type.shape
    if op.op == "transpose":
        axes = _transpose_axes(op, rank)
        return axes is not None and axes[0] == axes[1]
    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or len(dims) != rank:
            return False
        normalized = _normalized_dims(op, rank)
        return normalized == tuple(range(rank))
    if op.op == "flip":
        if input_type.shape != output_type.shape:
            return False
        dims = _normalized_flip_dims(op, rank)
        return dims is not None and all(input_type.shape[dim] == 1 for dim in dims)
    if op.op == "roll":
        if input_type.shape != output_type.shape:
            return False
        shifts = _roll_shifts_by_axis(op, input_type.shape)
        return shifts is not None and _roll_shift_is_identity(
            shifts, input_type.shape
        )
    if op.op == "slice":
        dim = op.attrs.get("dim")
        if not _valid_int(dim) or rank == 0 or int(dim) < -rank or int(dim) >= rank:
            return False
        axis = int(dim) + rank if int(dim) < 0 else int(dim)
        start = op.attrs.get("start")
        end = op.attrs.get("end")
        step = op.attrs.get("step")
        if start is not None and not _valid_int(start):
            return False
        if end is not None and not _valid_int(end):
            return False
        if not _valid_int(step) or int(step) <= 0:
            return False
        return slice(start, end, step).indices(input_type.shape[axis]) == (
            0,
            input_type.shape[axis],
            1,
        )
    return False


def _match_folded_batched_matmul(
    graph: Graph,
    index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    replaced: set[int],
) -> tuple[str, str, str] | None:
    matmul = graph.ops[index]
    if len(matmul.inputs) != 2:
        return None
    left_type = graph.value(matmul.inputs[0]).type
    right_type = graph.value(matmul.inputs[1]).type
    if len(left_type.shape) != 3 or len(right_type.shape) != 3:
        return None

    left_index = producers.get(matmul.inputs[0])
    right_index = producers.get(matmul.inputs[1])
    if left_index is None or right_index is None:
        return None
    left_reshape = graph.ops[left_index]
    right_reshape = graph.ops[right_index]
    if left_reshape.op != "reshape" or right_reshape.op != "reshape":
        return None
    if len(consumers.get(matmul.inputs[0], [])) != 1:
        return None
    if len(consumers.get(matmul.inputs[1], [])) != 1:
        return None

    left_base = graph.value(left_reshape.inputs[0]).type
    right_base = graph.value(right_reshape.inputs[0]).type
    if len(left_base.shape) < 3 or left_base.shape[:-2] != right_base.shape[:-2]:
        return None
    if left_type.shape != (
        prod(left_base.shape[:-2]),
        left_base.shape[-2],
        left_base.shape[-1],
    ):
        return None
    if right_type.shape != (
        prod(right_base.shape[:-2]),
        right_base.shape[-2],
        right_base.shape[-1],
    ):
        return None

    out_consumers = consumers.get(matmul.outputs[0], [])
    if len(out_consumers) != 1:
        return None
    output_index = out_consumers[0]
    output_reshape = graph.ops[output_index]
    if output_reshape.op != "reshape" or output_reshape.inputs[0] != matmul.outputs[0]:
        return None
    expected_output = left_base.shape[:-2] + (left_base.shape[-2], right_base.shape[-1])
    if graph.value(output_reshape.outputs[0]).type.shape != expected_output:
        return None

    replaced.update({left_index, right_index, index, output_index})
    return left_reshape.inputs[0], right_reshape.inputs[0], output_reshape.outputs[0]


def _match_attention(
    graph: Graph,
    pv_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    matched: set[int],
) -> tuple[int, set[int], Operation] | None:
    pv_match = _match_attention_product(
        graph, pv_index, producers, consumers, matched
    )
    if pv_match is None:
        return None
    replacement_index, probability_id, value_id, internal = pv_match
    probability_index = producers.get(probability_id)
    if probability_index is None or probability_index in matched:
        return None
    probability = graph.ops[probability_index]
    if probability.op not in {"softmax", "safe_softmax"} or len(probability.inputs) != 1:
        return None
    probability_type = graph.value(probability.outputs[0]).type
    rank = len(probability_type.shape)
    dim = probability.attrs.get("dim")
    if not _valid_int(dim) or int(dim) % rank != rank - 1:
        return None

    stage = probability.inputs[0]
    scale = 1.0
    mask: str | None = None
    has_scale = False
    has_mask = False
    intermediate = set(internal)
    intermediate.add(probability_index)

    for _ in range(2):
        index = producers.get(stage)
        if index is None:
            break
        op = graph.ops[index]
        parsed_scale = _parse_scale(op)
        if parsed_scale is not None and not has_scale:
            scale = parsed_scale
            stage = op.inputs[0]
            has_scale = True
            intermediate.add(index)
            continue
        if op.op == "add" and not has_mask and float(op.attrs.get("alpha", 1.0)) == 1.0:
            score_sides = [
                value_id
                for value_id in op.inputs
                if _looks_like_score_value(graph, value_id, producers)
            ]
            if len(score_sides) == 1:
                stage = score_sides[0]
                mask = next(value_id for value_id in op.inputs if value_id != stage)
                has_mask = True
                intermediate.add(index)
                continue
        break

    parsed_scale = _parse_scale(graph.ops[producers[stage]]) if stage in producers else None
    if parsed_scale is not None and not has_scale:
        scale_index = producers[stage]
        scale = parsed_scale
        stage = graph.ops[scale_index].inputs[0]
        has_scale = True
        intermediate.add(scale_index)

    score_index = producers.get(stage)
    if score_index is None or score_index in intermediate or score_index in matched:
        return None
    score = graph.ops[score_index]
    if score.op != "matmul" or len(score.inputs) != 2:
        return None
    intermediate.add(score_index)

    key_transform_id = score.inputs[1]
    key_transform_index = producers.get(key_transform_id)
    if key_transform_index is None or key_transform_index in matched:
        return None
    key_transform = graph.ops[key_transform_index]
    parsed_key_transform_scale = _parse_scale(key_transform)
    if parsed_key_transform_scale is not None:
        scale *= parsed_key_transform_scale
        key_transform_id = key_transform.inputs[0]
        intermediate.add(key_transform_index)
        key_transform_index = producers.get(key_transform_id)
        if key_transform_index is None or key_transform_index in matched:
            return None
        key_transform = graph.ops[key_transform_index]
    if not _is_last_two_transpose(key_transform, rank):
        return None
    intermediate.add(key_transform_index)

    query_id = score.inputs[0]
    key_id = key_transform.inputs[0]
    query_scale_index = producers.get(query_id)
    if query_scale_index is not None:
        query_scale = graph.ops[query_scale_index]
        parsed_query_scale = _parse_scale(query_scale)
        if parsed_query_scale is not None:
            scale *= parsed_query_scale
            query_id = query_scale.inputs[0]
            intermediate.add(query_scale_index)

    key_scale_index = producers.get(key_id)
    if key_scale_index is not None:
        key_scale = graph.ops[key_scale_index]
        parsed_key_scale = _parse_scale(key_scale)
        if parsed_key_scale is not None:
            scale *= parsed_key_scale
            key_id = key_scale.inputs[0]
            intermediate.add(key_scale_index)

    key_group = _match_grouped_kv(graph, key_id, producers, matched)
    value_group = _match_grouped_kv(graph, value_id, producers, matched)
    if (key_group is None) != (value_group is None):
        return None
    if key_group is not None and value_group is not None:
        key_id, key_internal = key_group
        value_id, value_internal = value_group
        intermediate.update(key_internal)
        intermediate.update(value_internal)

    original_replacement_index = replacement_index
    low_precision = _match_internal_float32_casts(
        graph,
        query_id,
        key_id,
        value_id,
        mask,
        replacement_index,
        producers,
        consumers,
        matched,
    )
    if low_precision is not None:
        query_id, key_id, value_id, mask, replacement_index, casts = low_precision
        intermediate.add(original_replacement_index)
        intermediate.update(casts)

    replacement_output = graph.ops[replacement_index].outputs[0]
    internal_values = {
        graph.ops[index].outputs[0] for index in intermediate
    } - {replacement_output}
    if internal_values.intersection(graph.outputs):
        return None
    for internal_value_id in internal_values:
        if len(consumers.get(internal_value_id, [])) != 1:
            return None

    query_type = graph.value(query_id).type
    key_type = graph.value(key_id).type
    value_type = graph.value(value_id).type
    if not _valid_attention_shapes(query_type.shape, key_type.shape, value_type.shape):
        return None
    if not (query_type.dtype == key_type.dtype == value_type.dtype):
        return None
    if scale == 0.0:
        return None
    if mask is not None:
        mask_type = graph.value(mask).type
        if mask_type.dtype not in {query_type.dtype}:
            return None
        if not _can_broadcast_to(mask_type.shape, probability_type.shape):
            return None

    inputs = (query_id, key_id, value_id) + ((mask,) if mask is not None else ())
    attention = Operation(
        "attention",
        inputs,
        (graph.ops[replacement_index].outputs[0],),
        {
            "scale": scale,
            "has_mask": mask is not None,
            "causal": False,
            "safe_softmax": probability.op == "safe_softmax",
        },
    )
    return replacement_index, intermediate, attention


def _match_attention_product(
    graph: Graph,
    pv_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    matched: set[int],
) -> tuple[int, str, str, set[int]] | None:
    """Recover ``softmax @ V`` when PyTorch flattened it into an ``mm``."""

    pv = graph.ops[pv_index]
    if len(pv.inputs) != 2:
        return None
    probability_id, value_id = pv.inputs
    probability_type = graph.value(probability_id).type
    value_type = graph.value(value_id).type
    if len(probability_type.shape) == 4 and len(value_type.shape) == 4:
        return pv_index, probability_id, value_id, set()
    if len(probability_type.shape) != 3 or len(value_type.shape) != 3:
        return None

    probability_index = producers.get(probability_id)
    value_index = producers.get(value_id)
    if probability_index is None or value_index is None:
        return pv_index, probability_id, value_id, set()
    probability_reshape = graph.ops[probability_index]
    value_reshape = graph.ops[value_index]
    if probability_reshape.op != "reshape" or value_reshape.op != "reshape":
        return pv_index, probability_id, value_id, set()
    if len(consumers.get(probability_id, [])) != 1:
        return None
    if len(consumers.get(value_id, [])) != 1:
        return None

    output_consumers = consumers.get(pv.outputs[0], [])
    if len(output_consumers) != 1:
        return None
    output_index = output_consumers[0]
    output_reshape = graph.ops[output_index]
    if output_reshape.op != "reshape" or output_reshape.inputs[0] != pv.outputs[0]:
        return None

    probability_base_id = probability_reshape.inputs[0]
    value_base_id = value_reshape.inputs[0]
    probability_base = graph.value(probability_base_id).type
    value_base = graph.value(value_base_id).type
    if len(probability_base.shape) != 4 or len(value_base.shape) not in {4, 5}:
        return None
    batch_heads = probability_base.shape[:2]
    if probability_base.shape != (
        batch_heads[0],
        batch_heads[1],
        probability_type.shape[-2],
        value_type.shape[-2],
    ):
        return None
    if value_base.shape[-2:] != value_type.shape[-2:]:
        return None
    if prod(value_base.shape[:-2]) != prod(batch_heads):
        return None
    if probability_type.shape != (
        prod(batch_heads),
        probability_base.shape[-2],
        value_base.shape[-2],
    ):
        return None
    if value_type.shape != (prod(batch_heads), value_base.shape[-2], value_base.shape[-1]):
        return None

    expected_output_shape = probability_base.shape[:-1] + (value_base.shape[-1],)
    if graph.value(output_reshape.outputs[0]).type.shape != expected_output_shape:
        return None

    internal = {pv_index, probability_index, value_index, output_index}
    if internal.intersection(matched):
        return None
    return output_index, probability_base_id, value_base_id, internal


def _match_grouped_kv(
    graph: Graph,
    value_id: str,
    producers: dict[str, int],
    matched: set[int],
) -> tuple[str, set[int]] | None:
    """Strip PyTorch's unsqueeze-expand-reshape K/V head replication."""

    value_type = graph.value(value_id).type
    current_id = value_id
    internal: set[int] = set()
    if len(value_type.shape) == 4:
        reshape_index = producers.get(current_id)
        if reshape_index is None or reshape_index in matched:
            return None
        reshape = graph.ops[reshape_index]
        if reshape.op != "reshape":
            return None
        current_id = reshape.inputs[0]
        expanded_type = graph.value(current_id).type
        if len(expanded_type.shape) != 5:
            return None
        if value_type.shape != (
            expanded_type.shape[0],
            expanded_type.shape[1] * expanded_type.shape[2],
            expanded_type.shape[3],
            expanded_type.shape[4],
        ):
            return None
        internal.add(reshape_index)
    elif len(value_type.shape) == 5:
        pass
    else:
        return None

    expand_index = producers.get(current_id)
    if expand_index is None or expand_index in matched:
        return None
    expand = graph.ops[expand_index]
    if expand.op != "expand":
        return None
    unsqueeze_id = expand.inputs[0]
    unsqueeze_index = producers.get(unsqueeze_id)
    if unsqueeze_index is None or unsqueeze_index in matched:
        return None
    unsqueeze = graph.ops[unsqueeze_index]
    if unsqueeze.op != "unsqueeze":
        return None
    original_id = unsqueeze.inputs[0]
    original_type = graph.value(original_id).type
    unsqueezed_type = graph.value(unsqueeze_id).type
    expanded_type = graph.value(expand.outputs[0]).type
    if len(original_type.shape) != 4 or len(unsqueezed_type.shape) != 5:
        return None
    if unsqueezed_type.shape != original_type.shape[:2] + (1,) + original_type.shape[2:]:
        return None
    if expanded_type.shape != original_type.shape[:2] + (
        expanded_type.shape[2],
    ) + original_type.shape[2:]:
        return None
    if expanded_type.shape[2] < 2:
        return None
    internal.update({expand_index, unsqueeze_index})
    return original_id, internal


def _match_internal_float32_casts(
    graph: Graph,
    query_id: str,
    key_id: str,
    value_id: str,
    mask: str | None,
    replacement_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    matched: set[int],
) -> tuple[str, str, str, str | None, int, set[int]] | None:
    """Absorb PyTorch's low-precision attention upcast/round-trip."""

    replacement_output = graph.ops[replacement_index].outputs[0]
    replacement_type = graph.value(replacement_output).type
    if replacement_type.dtype != DType.F32:
        return None

    output_consumers = consumers.get(replacement_output, [])
    if len(output_consumers) != 1:
        return None
    cast_index = output_consumers[0]
    if cast_index in matched:
        return None
    output_cast = graph.ops[cast_index]
    if output_cast.op != "cast" or output_cast.inputs != (replacement_output,):
        return None
    output_type = graph.value(output_cast.outputs[0]).type
    if (
        output_type.shape != replacement_type.shape
        or output_type.dtype not in {DType.F16, DType.BF16}
    ):
        return None

    casts: set[int] = set()
    stripped_ids: list[str] = []
    source_dtypes: set[DType] = set()
    for input_id in (query_id, key_id, value_id):
        stripped = _strip_float32_input_cast(
            input_id, graph, producers, matched, casts
        )
        if stripped is None:
            return None
        stripped_id, source_dtype = stripped
        stripped_ids.append(stripped_id)
        source_dtypes.add(source_dtype)

    if source_dtypes != {output_type.dtype}:
        return None

    stripped_mask = mask
    if mask is not None:
        mask_type = graph.value(mask).type
        if mask_type.dtype == output_type.dtype:
            pass
        elif mask_type.dtype == DType.F32:
            stripped = _strip_float32_input_cast(
                mask, graph, producers, matched, casts
            )
            if stripped is None or stripped[1] != output_type.dtype:
                return None
            stripped_mask = stripped[0]
        else:
            return None

    casts.add(cast_index)
    return (
        stripped_ids[0],
        stripped_ids[1],
        stripped_ids[2],
        stripped_mask,
        cast_index,
        casts,
    )


def _strip_float32_input_cast(
    value_id: str,
    graph: Graph,
    producers: dict[str, int],
    matched: set[int],
    casts: set[int],
) -> tuple[str, DType] | None:
    index = producers.get(value_id)
    if index is None or index in matched:
        return None
    op = graph.ops[index]
    if op.op != "cast":
        return None

    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    if (
        output_type.dtype != DType.F32
        or input_type.dtype not in {DType.F16, DType.BF16}
        or input_type.shape != output_type.shape
    ):
        return None
    casts.add(index)
    return op.inputs[0], input_type.dtype


def _parse_scale(op: Operation) -> float | None:
    scalar = op.attrs.get("scalar")
    if op.op == "mul" and _valid_number(scalar):
        return float(scalar)
    if op.op == "div" and _valid_number(scalar):
        position = op.attrs.get("scalar_position")
        if position == "rhs" and float(scalar) != 0.0:
            return 1.0 / float(scalar)
    return None


def _looks_like_score_value(
    graph: Graph, value_id: str, producers: dict[str, int]
) -> bool:
    index = producers.get(value_id)
    if index is None:
        return False
    op = graph.ops[index]
    if op.op == "matmul":
        return True
    if _parse_scale(op) is None:
        return False
    producer_index = producers.get(op.inputs[0])
    return (
        producer_index is not None
        and graph.ops[producer_index].op == "matmul"
    )


def _valid_attention_shapes(
    query: tuple[int, ...], key: tuple[int, ...], value: tuple[int, ...]
) -> bool:
    if len(query) not in {3, 4} or len(key) != len(query) or len(value) != len(query):
        return False
    if query[:1] != key[:1]:
        return False
    if key[:1] != value[:1]:
        return False
    if query[-1] != key[-1] or key[-2] != value[-2]:
        return False
    if len(query) == 3:
        return True
    query_heads, key_value_heads = query[-3], key[-3]
    return query_heads >= key_value_heads and query_heads % key_value_heads == 0


def _is_last_two_transpose(op: Operation, rank: int) -> bool:
    if op.op == "transpose":
        dim0, dim1 = op.attrs.get("dim0"), op.attrs.get("dim1")
        if not _valid_ints((dim0, dim1)):
            return False
        return {int(dim0) % rank, int(dim1) % rank} == {rank - 2, rank - 1}
    if op.op == "permute":
        dims = op.attrs.get("dims")
        if not isinstance(dims, (tuple, list)) or len(dims) != rank:
            return False
        expected = list(range(rank))
        expected[-2], expected[-1] = expected[-1], expected[-2]
        normalized = _normalized_dims(op, rank)
        return normalized == tuple(expected)
    return False


def _can_broadcast_to(shape: tuple[int, ...], output: tuple[int, ...]) -> bool:
    if len(shape) > len(output):
        return False
    offset = len(output) - len(shape)
    return all(dim in {1, output[offset + index]} for index, dim in enumerate(shape))


def _producers(graph: Graph) -> dict[str, int]:
    return {value_id: index for index, op in enumerate(graph.ops) for value_id in op.outputs}


def _consumers(graph: Graph) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(graph.ops):
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(index)
    return consumers


def _transpose_axes(op: Operation, rank: int) -> tuple[int, int] | None:
    dim0, dim1 = op.attrs.get("dim0"), op.attrs.get("dim1")
    if not _valid_ints((dim0, dim1)):
        return None
    normalized = tuple(int(dim) + rank if int(dim) < 0 else int(dim) for dim in (dim0, dim1))
    if any(dim < 0 or dim >= rank for dim in normalized):
        return None
    return normalized


def _normalized_dims(op: Operation, rank: int) -> tuple[int, ...] | None:
    dims = op.attrs.get("dims")
    if not isinstance(dims, (tuple, list)) or not all(_valid_int(dim) for dim in dims):
        return None
    normalized = tuple(
        int(dim) + rank if int(dim) < 0 else int(dim) for dim in dims
    )
    if any(dim < 0 or dim >= rank for dim in normalized):
        return None
    return normalized


def _normalized_flip_dims(op: Operation, rank: int) -> tuple[int, ...] | None:
    dims = _normalized_dims(op, rank)
    if dims is None or not dims or len(set(dims)) != len(dims):
        return None
    return dims


def _roll_shifts_by_axis(
    op: Operation, shape: tuple[int, ...]
) -> dict[int | None, int] | None:
    shifts = op.attrs.get("shifts")
    if not isinstance(shifts, (tuple, list)) or not shifts:
        return None
    if not all(_valid_int(shift) for shift in shifts):
        return None

    dims = op.attrs.get("dims")
    if dims is None:
        if len(shifts) != 1:
            return None
        return {0 if len(shape) == 1 else None: int(shifts[0])}
    if not isinstance(dims, (tuple, list)) or len(dims) != len(shifts):
        return None
    if not all(_valid_int(dim) for dim in dims):
        return None

    rank = len(shape)
    shifts_by_axis: dict[int | None, int] = {}
    for shift, dim in zip(shifts, dims, strict=True):
        axis = int(dim) + rank if int(dim) < 0 else int(dim)
        if axis < 0 or axis >= rank:
            return None
        shifts_by_axis[axis] = shifts_by_axis.get(axis, 0) + int(shift)
    return shifts_by_axis


def _roll_shift_is_identity(
    shifts_by_axis: dict[int | None, int], shape: tuple[int, ...]
) -> bool:
    element_count = prod(shape)
    for axis, shift in shifts_by_axis.items():
        dimension = element_count if axis is None else shape[axis]
        if dimension <= 0 or shift % dimension != 0:
            return False
    return True


def _valid_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_ints(values: tuple[object, ...]) -> bool:
    return all(_valid_int(value) for value in values)


def _valid_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
        value_id: ("argument", index)
        for index, value_id in enumerate(op.inputs)
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


def _replace_graph(
    graph: Graph, ops: tuple[Operation, ...], outputs: tuple[str, ...]
) -> Graph:
    result = Graph(graph.inputs, graph.values, ops, outputs)
    result.validate()
    return result
