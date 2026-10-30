"""Operators that own a hand-written kernel.

Matmul, the fused normalizations, and attention are the operations the backend
implements as dedicated Metal source rather than as an elementwise expression or
an MLX call. Their contracts are therefore mostly attribute declarations: the
shape and dtype promises are what the kernel itself relies on, and the hooks
check the constraints the source assumes (a head dimension width, a rope pair
layout, an attention mask rank).
"""

from __future__ import annotations

from .. import DType, Graph, Operation
from .rules import (
    AttrKind,
    AttrSpec,
    DtypeRule,
    OpContract,
    ShapeRule,
)
from ...shapes import can_broadcast


def _check_rope(graph: Graph, op: Operation) -> None:
    output_type = graph.value(op.outputs[0]).type
    if output_type.shape[-1] % 2 != 0:
        raise ValueError(
            f"rope head dimension must be even, got {output_type.shape[-1]}"
        )
    columns = output_type.shape[-1]
    for value_id in op.inputs[1:]:
        table_type = graph.value(value_id).type
        if table_type.dtype != output_type.dtype:
            raise ValueError("rope table dtype must match the source dtype")
        if not can_broadcast(table_type.shape, output_type.shape):
            raise ValueError(
                f"rope table {table_type.shape} cannot broadcast to {output_type.shape}"
            )
        if table_type.shape[-1] not in (1, columns):
            raise ValueError(
                f"rope table last dimension must be 1 or {columns}, "
                f"got {table_type.shape[-1]}"
            )


def _check_attention(graph: Graph, op: Operation) -> None:
    query_type, key_type, value_type = (
        graph.value(value_id).type for value_id in op.inputs[:3]
    )
    output_type = graph.value(op.outputs[0]).type
    query_shape = query_type.shape
    key_shape = key_type.shape
    value_shape = value_type.shape
    if len(key_shape) != len(query_shape) or len(value_shape) != len(query_shape):
        raise ValueError("attention Q/K/V ranks must match")
    if query_shape[:1] != key_shape[:1]:
        raise ValueError("attention Q/K batch dimensions must match")
    if key_shape[:1] != value_shape[:1]:
        raise ValueError("attention K/V batch dimensions must match")
    if query_shape[-1] != key_shape[-1]:
        raise ValueError("attention Q/K head dimensions must match")
    if key_shape[-2] != value_shape[-2]:
        raise ValueError("attention K/V sequence dimensions must match")
    if len(query_shape) == 4:
        query_heads, key_value_heads = query_shape[-3], key_shape[-3]
        if query_heads < key_value_heads or query_heads % key_value_heads:
            raise ValueError(
                "attention query heads must be a multiple of key/value heads, "
                f"got {query_heads} and {key_value_heads}"
            )
    expected_output_shape = query_shape[:-1] + (value_shape[-1],)
    if output_type.shape != expected_output_shape:
        raise ValueError(
            f"attention output shape {output_type.shape} != {expected_output_shape}"
        )
    if not (
        query_type.dtype == key_type.dtype == value_type.dtype == output_type.dtype
    ):
        raise NotImplementedError("attention requires identical Q/K/V/output dtypes")
    if op.attrs.get("causal") and len(op.inputs) == 4:
        raise ValueError("attention cannot combine an explicit mask with causal mode")
    if len(op.inputs) == 4:
        mask_type = graph.value(op.inputs[3]).type
        score_shape = (
            (query_shape[0], query_shape[-2], key_shape[-2])
            if len(query_shape) == 3
            else (query_shape[:1] + (query_shape[-3], query_shape[-2], key_shape[-2]))
        )
        if mask_type.dtype not in {query_type.dtype, DType.BOOL}:
            raise NotImplementedError(
                "attention mask must be boolean or match the Q/K/V dtype"
            )
        if not can_broadcast(mask_type.shape, score_shape):
            raise ValueError(
                f"attention mask shape {mask_type.shape} cannot broadcast "
                f"to {score_shape}"
            )


CONTRACTS: dict[str, OpContract] = {
    # Their compilers check the operands against the kernel they emit, so the
    # table only pins the arity and the number of results.
    "matmul": OpContract(
        arity=(2, 5), dtype=DtypeRule.ANY, shape=ShapeRule.FREE, native=True
    ),
    "addmm": OpContract(
        arity=(3, 4), dtype=DtypeRule.ANY, shape=ShapeRule.FREE, native=True
    ),
    "fused_elementwise": OpContract(
        arity=(1, 8),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        native=True,
    ),
    "layer_norm": OpContract(
        arity=(1, 3),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        outputs=(1, 3),
        native=True,
    ),
    "rms_norm": OpContract(
        arity=(1, 2), dtype=DtypeRule.ANY, shape=ShapeRule.FREE, native=True
    ),
    "rope": OpContract(
        arity=(3, 3),
        dtype=DtypeRule.ANY,
        attrs=(),
        native=True,
        extra=_check_rope,
    ),
    "attention": OpContract(
        arity=(3, 4),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("scale", AttrKind.NUMBER),),
        native=True,
        extra=_check_attention,
    ),
}
