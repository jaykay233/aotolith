"""Recognising the rotary embedding pattern.

RoPE appears as a specific arrangement of slicing, multiplication by cos/sin and
re-assembly; the matcher proves the arrangement matches before any rewrite
happens.
"""

from __future__ import annotations

from ..checks import (
    _can_broadcast_to,
)
from ...ir import DType, Graph, Operation


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
