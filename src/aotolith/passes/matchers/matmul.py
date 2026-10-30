"""Finding the matmul behind a chain of values.

The epilogue fusions start from a pointwise region and walk backwards to the
matmul that produced it, which means proving that every step in between is a
single-consumer reshape or a transpose the matmul can absorb.
"""

from __future__ import annotations

from math import prod

from ...ir import Graph
from ...op_sets import LAYOUT_OPS


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
