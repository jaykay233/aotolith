from __future__ import annotations

from .ir import Graph, Operation

POINTWISE_OPS = {
    "silu",
    "gelu",
    "mul",
    "add",
    "sub",
    "rsub",
    "div",
    "neg",
    "abs",
    "sigmoid",
    "tanh",
    "relu",
    "exp",
    "erf",
    "sqrt",
    "rsqrt",
    "reciprocal",
}


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
        index for index, op in enumerate(graph.ops) if op.op in POINTWISE_OPS
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

    region_for_op: dict[int, tuple[int, Operation]] = {}
    assigned: set[int] = set()
    next_region = 0

    for root in roots:
        if root in assigned:
            continue
        members: set[int] = set()

        def collect(index: int) -> None:
            if index in members or index not in pointwise:
                return
            op = graph.ops[index]
            if index != root and len(consumers.get(op.outputs[0], [])) != 1:
                return
            members.add(index)
            for value_id in op.inputs:
                producer = producers.get(value_id)
                if producer is not None:
                    collect(producer)

        collect(root)
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
        start = ordered[0]
        region_for_op.update({index: (next_region, fused) for index in members})
        next_region += 1
        assigned.update(members)
        region_for_op[start] = (next_region - 1, fused)

    if not region_for_op:
        return graph

    rewritten: list[Operation] = []
    emitted_regions: set[int] = set()
    for index, op in enumerate(graph.ops):
        region = region_for_op.get(index)
        if region is None:
            rewritten.append(op)
            continue
        region_id, fused = region
        if region_id not in emitted_regions:
            rewritten.append(fused)
            emitted_regions.add(region_id)

    fused_graph = graph.replace_ops(tuple(rewritten))
    return fused_graph
