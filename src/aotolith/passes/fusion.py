"""Fusion: replace a matched region with the single operator it means.

Each pass matches a region, proves it is the whole region, and emits one kernel
for it. They are separate functions rather than one table because the proofs
differ, but they share the same shape: find, verify, replace.
"""

from __future__ import annotations

from math import prod

from ..ir import Graph, Operation, TensorType, Value
from ..op_sets import POINTWISE_OPS
from .checks import (
    _can_broadcast_to,
)
from .graph import (
    _consumers,
    _fresh_value_id,
    _producers,
    _replace_graph,
)
from .matchers.attention import (
    _match_attention,
)
from .matchers.matmul import (
    _chain_preserves_order,
    _single_consumer_reshape_chain,
    _trace_add_operand_to_matmul,
    _trace_operand_to_matmul,
)
from .matchers.norm import (
    _match_rms_norm_consumers,
)
from .matchers.rope import (
    _match_rope,
)


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
                Operation(
                    "reshape", (residual_id,), (view_id,), {"shape": product_shape}
                )
            )
            residual_id = view_id
        left_id = product.inputs[0]
        # Everything after the left operand travels untouched: for a dense
        # matmul that is the weight, and for a packed one it is the weight plus
        # its group scales.
        operands = product.inputs[1:]
        attrs = dict(product.attrs)
        attrs["alpha"] = 1.0
        attrs["beta"] = 1.0
        created = Operation(
            "addmm",
            (residual_id, left_id) + operands,
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
        gate_op = graph.ops[gate_index]
        product_op = graph.ops[product_index]
        if (
            graph.value(gate_op.outputs[0]).type
            != graph.value(product_op.outputs[0]).type
        ):
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
        plan = _match_rms_norm_consumers(graph, norm_index, norm, consumers, output_set)
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
                    if (
                        producer is None
                        or producer in members
                        or producer not in pointwise
                    ):
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
