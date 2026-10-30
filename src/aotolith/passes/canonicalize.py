"""Put batched matmuls into the one shape the backend expects."""

from __future__ import annotations

from ..ir import Graph, Operation
from .graph import (
    _consumers,
    _producers,
    _replace_graph,
)
from .layout import (
    simplify_layouts,
)
from .matchers.matmul import (
    _match_folded_batched_matmul,
)


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
