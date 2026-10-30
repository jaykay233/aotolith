"""Recognising an attention subgraph, in both the trained and cached shape.

This is the one matcher that is allowed to be long: SDPA-like patterns come in
several spellings, and every one of them has to be accepted without accepting a
near miss.
"""

from __future__ import annotations

from math import prod

from ..checks import (
    _can_broadcast_to,
    _valid_int,
    _valid_number,
)
from ...ir import DType, Graph, Operation
from .view import (
    _is_last_two_transpose,
)


def _match_attention(
    graph: Graph,
    pv_index: int,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    matched: set[int],
) -> tuple[int, set[int], Operation] | None:
    pv_match = _match_attention_product(graph, pv_index, producers, consumers, matched)
    if pv_match is None:
        return None
    replacement_index, probability_id, value_id, internal = pv_match
    probability_index = producers.get(probability_id)
    if probability_index is None or probability_index in matched:
        return None
    probability = graph.ops[probability_index]
    if (
        probability.op not in {"softmax", "safe_softmax"}
        or len(probability.inputs) != 1
    ):
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

    parsed_scale = (
        _parse_scale(graph.ops[producers[stage]]) if stage in producers else None
    )
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
    internal_values = {graph.ops[index].outputs[0] for index in intermediate} - {
        replacement_output
    }
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
    if value_type.shape != (
        prod(batch_heads),
        value_base.shape[-2],
        value_base.shape[-1],
    ):
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
    if (
        unsqueezed_type.shape
        != original_type.shape[:2] + (1,) + original_type.shape[2:]
    ):
        return None
    if (
        expanded_type.shape
        != original_type.shape[:2] + (expanded_type.shape[2],) + original_type.shape[2:]
    ):
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
    if output_type.shape != replacement_type.shape or output_type.dtype not in {
        DType.F16,
        DType.BF16,
    }:
        return None

    casts: set[int] = set()
    stripped_ids: list[str] = []
    source_dtypes: set[DType] = set()
    for input_id in (query_id, key_id, value_id):
        stripped = _strip_float32_input_cast(input_id, graph, producers, matched, casts)
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
            stripped = _strip_float32_input_cast(mask, graph, producers, matched, casts)
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
    return producer_index is not None and graph.ops[producer_index].op == "matmul"


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
