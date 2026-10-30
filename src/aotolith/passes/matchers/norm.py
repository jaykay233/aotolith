"""Recognising the reductions that make up a normalisation.

RMS norm is a mean-square, a rsqrt, and a scale, and the fusion only applies
when the reduction really feeds that scale and nothing else.
"""

from __future__ import annotations

from math import prod

from ..graph import (
    _retarget_input,
)
from ...ir import Graph, Operation


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
            inputs = inputs[:left_position] + (source_id,) + inputs[left_position + 1 :]
        rewrites[projection_index] = Operation(
            projection.op, inputs + (weight_id,), projection.outputs, attrs
        )
    return rewrites, layout_rewrites
