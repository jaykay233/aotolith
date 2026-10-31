"""Fold a quantized weight into the matrix multiply that reads it.

A quantized projection reaches the IR as two operators: ``dequantize_i4`` turns
the packed words and group scales into a weight matrix, and the ``matmul`` right
after it consumes that matrix. Run as written, the unpack materializes the whole
weight - which is exactly the traffic the 4-bit format existed to avoid, and
worse than leaving the model in bf16.

So this pass proves the unpack has one consumer and that the consumer is a
matmul that wants the weight transposed (the shape ``Linear`` stores), then
rewrites the pair into a single matmul whose weight operand is the *packed*
tensor. The scales ride along in the slot right after it. Nothing is
materialized, and the kernel that reads the result streams 4 bits per weight.

The rewrite is deliberately narrow: a weight with a second consumer stays an
explicit unpack, because folding would then have to produce the matrix anyway.
"""

from __future__ import annotations

from ..ir import Graph, Operation
from ..ir.ops.kernels import _matmul_weight_group_size
from .graph import _consumers, _producers, _replace_graph

#: Where each matmul spelling keeps its operands.
_RIGHT_POSITION = {"matmul": 1, "addmm": 2}


def _quantized_weight_positions(op: Operation) -> tuple[int, int] | None:
    """The right-operand and scales slots of an already folded matmul."""

    right_position = _RIGHT_POSITION.get(op.op)
    if right_position is None or _matmul_weight_group_size(op) is None:
        return None
    return right_position, right_position + 1


def fold_quantized_weights(graph: Graph) -> Graph:
    """Replace ``matmul(x, dequantize(w))`` with one packed-weight matmul."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    replacement_at: dict[int, Operation] = {}
    folded: set[int] = set()

    for index, op in enumerate(graph.ops):
        right_position = _RIGHT_POSITION.get(op.op)
        if right_position is None or _quantized_weight_positions(op) is not None:
            continue
        if len(op.inputs) <= right_position:
            continue
        if op.attrs.get("transpose_b") is not True:
            # A matmul reading the weight untransposed wants ``[in, out]``, and
            # the packed rows are output units; there is no kernel for it.
            continue
        left_position = 0
        if op.op == "addmm":
            left_position = 1
        left_shape = graph.value(op.inputs[left_position]).type.shape
        if len(left_shape) < 2 or left_shape[-2] != 1:
            # A wider left operand makes the weight stream the *cheap* part of
            # the multiply, so folding buys nothing and the int4 kernel would
            # add decoding work to a compute-bound pass. The unpack stays, and
            # the streaming dequantize kernel runs it in one pass.
            continue
        unpacked = op.inputs[right_position]
        unpack_index = producers.get(unpacked)
        if unpack_index is None:
            continue
        unpack = graph.ops[unpack_index]
        if unpack.op != "dequantize_i4":
            continue
        if consumers.get(unpacked) != [index]:
            # Another reader needs the matrix itself, so folding here would
            # only move the unpack, not remove it.
            continue
        if unpacked in graph.outputs:
            # The unpacked matrix is also a result; dropping the unpack would
            # leave an output with no producer.
            continue
        packed_id, scales_id = unpack.inputs
        attrs = dict(op.attrs)
        del attrs["transpose_b"]
        attrs["weight_group_size"] = int(unpack.attrs["group_size"])
        inputs = (
            op.inputs[:right_position]
            + (packed_id, scales_id)
            + op.inputs[right_position + 1 :]
        )
        replacement_at[index] = Operation(op.op, inputs, op.outputs, attrs)
        folded.add(unpack_index)

    if not replacement_at:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        if index in folded:
            continue
        rewritten.append(replacement_at.get(index, op))
    return _replace_graph(graph, tuple(rewritten), graph.outputs)
