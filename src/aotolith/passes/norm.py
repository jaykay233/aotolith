"""Fold a normalization's scale multiply into the normalization itself.

PyTorch's RMSNorm spells itself out: cast the row to fp32, normalize, cast the
weight up to fp32, multiply, cast the product back to the compute dtype. That is
five operators where one suffices, and the casts are not free - they are extra
kernel launches per layer per token, each moving a row through the command
buffer for no arithmetic.

The five operators are exactly equivalent to a normalization that takes the
weight as an operand, which the Metal kernel already implements: the row is
upcast losslessly, the product is formed in fp32, and the result is rounded once
on the way out. Folding removes four launches without changing a single bit of
the result.

The rewrite is deliberately narrow. It only fires when the intermediate values
have one consumer each and when the scale arrives through a cast from the dtype
the row will be narrowed back to, so a graph that really does want a standalone
fp32 normalized row keeps it.
"""

from __future__ import annotations

from ..ir import Graph, Operation
from .graph import _consumers, _producers, _replace_graph

#: The normalizations whose scale this pass may absorb.
_NORMALIZATIONS = ("rms_norm", "layer_norm")


def fuse_norm_scale(graph: Graph) -> Graph:
    """Replace ``cast(norm(cast(x)) * cast(w))`` with ``norm(x, w)``."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    output_set = set(graph.outputs)
    rewrites: dict[int, Operation] = {}
    removable: set[int] = set()

    for index, narrowing in enumerate(graph.ops):
        if narrowing.op != "cast":
            continue
        scaled_id = narrowing.inputs[0]
        narrowed_type = graph.value(narrowing.outputs[0]).type

        multiply_index = producers.get(scaled_id)
        if multiply_index is None or consumers.get(scaled_id) != [index]:
            continue
        multiply = graph.ops[multiply_index]
        if multiply.op != "mul" or len(multiply.inputs) != 2:
            continue
        if multiply.outputs[0] in output_set:
            continue

        matched = _match_norm_and_scale(
            graph, multiply_index, multiply, producers, consumers, output_set
        )
        if matched is None:
            continue
        norm_index, norm, source_id, weight_id, weight_cast_index = matched

        # The row arrives as an fp32 upcast of the tensor the product is
        # narrowed back to. Look through that upcast so the fused operator can
        # read the original dtype; the upcast itself then has no readers left.
        removable_here: set[int] = set()
        row_upcast = producers.get(source_id)
        if row_upcast is not None and consumers.get(source_id) == [norm_index]:
            upcast = graph.ops[row_upcast]
            if upcast.op == "cast" and len(upcast.inputs) == 1:
                candidate = upcast.inputs[0]
                candidate_type = graph.value(candidate).type
                if (
                    candidate_type.dtype == narrowed_type.dtype
                    and candidate_type.shape == narrowed_type.shape
                ):
                    source_id = candidate
                    removable_here.add(row_upcast)

        # Both the row and the scale have to arrive in the dtype the product is
        # narrowed back to: the fused kernel reads one dtype and writes the same
        # one, so a mixed chain has no equivalent single operator.
        if graph.value(source_id).type.dtype != narrowed_type.dtype:
            continue
        if graph.value(weight_id).type.dtype != narrowed_type.dtype:
            continue
        if graph.value(source_id).type.shape != narrowed_type.shape:
            continue

        attrs = dict(norm.attrs)
        attrs["has_weight"] = True
        attrs.pop("has_bias", None)
        rewrites[index] = Operation(
            norm.op, (source_id, weight_id), narrowing.outputs, attrs
        )
        removable.update({multiply_index, norm_index})
        removable.update(removable_here)

        # The scale upcast only ever fed this multiply; drop it when nothing
        # else reads it, which the consumer map proves rather than assumes.
        weight_upcast = graph.ops[weight_cast_index].outputs[0]
        if consumers.get(weight_upcast) == [multiply_index]:
            removable.add(weight_cast_index)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index in removable:
            continue
        else:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def _match_norm_and_scale(
    graph: Graph,
    multiply_index: int,
    multiply: Operation,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    output_set: set[str],
) -> tuple[int, Operation, str, str, int] | None:
    """Return ``(norm index, norm, source, weight, weight cast index)``."""

    left_id, right_id = multiply.inputs
    for scale_id, scaled_id in ((left_id, right_id), (right_id, left_id)):
        norm_index = producers.get(scaled_id)
        if norm_index is None:
            continue
        norm = graph.ops[norm_index]
        if norm.op not in _NORMALIZATIONS:
            continue
        if len(norm.inputs) != 1 or len(norm.outputs) != 1:
            continue
        if norm.attrs.get("has_weight") is True or norm.attrs.get("has_bias") is True:
            continue
        normalized_shape = norm.attrs.get("normalized_shape")
        if not isinstance(normalized_shape, tuple) or not normalized_shape:
            continue
        if consumers.get(norm.outputs[0]) != [multiply_index]:
            continue
        if norm.outputs[0] in output_set or norm.inputs[0] in output_set:
            continue
        scale_cast_index = producers.get(scale_id)
        if scale_cast_index is None:
            continue
        scale_cast = graph.ops[scale_cast_index]
        if scale_cast.op != "cast" or len(scale_cast.inputs) != 1:
            continue
        weight_id = scale_cast.inputs[0]
        if graph.value(weight_id).type.shape != normalized_shape:
            continue
        if graph.value(scale_id).type.shape != normalized_shape:
            continue
        return norm_index, norm, norm.inputs[0], weight_id, scale_cast_index
    return None
