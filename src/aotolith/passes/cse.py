"""Drop operators that compute something already computed."""

from __future__ import annotations

from ..ir import Graph, Operation
from .graph import (
    _operation_attribute_key,
    _replace_graph,
    _rewrite_fused_region_inputs,
)


def eliminate_common_subexpressions(graph: Graph) -> Graph:
    """Reuse identical pure operations with the same inputs and attributes."""

    graph.validate()
    aliases: dict[str, str] = {}

    def resolve(value_id: str) -> str:
        while value_id in aliases:
            value_id = aliases[value_id]
        return value_id

    common: dict[tuple[object, ...], tuple[str, ...]] = {}
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
        output_types = tuple(graph.value(value_id).type for value_id in op.outputs)
        key = (
            current.op,
            current.inputs,
            tuple(
                (output_type.shape, output_type.dtype.value)
                for output_type in output_types
            ),
            _operation_attribute_key(graph, current, resolve),
        )
        existing = common.get(key)
        if existing is not None:
            for value_id, replacement in zip(current.outputs, existing, strict=True):
                aliases[value_id] = replacement
            continue
        common[key] = current.outputs
        rewritten.append(current)

    outputs = tuple(resolve(value_id) for value_id in graph.outputs)
    return _replace_graph(graph, tuple(rewritten), outputs)
