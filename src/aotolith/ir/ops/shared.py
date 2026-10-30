"""Small helpers the per-family hooks share.

These parse the shapes and attributes an ``extra`` hook needs, so that each
family file states what its constraint *is* rather than re-deriving indices
from raw attribute dicts.
"""

from __future__ import annotations

from .. import DType, Graph, Operation, TensorType
from .rules import INDEX_DTYPES


def _no_boolean_operand(graph: Graph, op: Operation) -> None:
    if graph.value(op.inputs[0]).type.dtype == DType.BOOL:
        raise NotImplementedError(f"{op.op} does not accept boolean tensor inputs")


def _unsupported_attrs(op: Operation, allowed: set[str]) -> None:
    unknown = set(op.attrs) - allowed
    if unknown:
        raise ValueError(f"{op.op} has unsupported attributes: {sorted(unknown)}")


def _only_attrs(op: Operation, allowed: set[str]) -> None:
    extra = set(op.attrs) - allowed
    if extra:
        raise ValueError(f"{op.op} has unexpected attributes: {sorted(extra)}")


def _index_tensor(
    graph: Graph, value_id: str, label: str, rank: int | None = None
) -> TensorType:
    index_type = graph.value(value_id).type
    if index_type.dtype not in INDEX_DTYPES:
        raise ValueError(
            f"{label} must be int32 or int64, got {index_type.dtype.value}"
        )
    if rank is not None and len(index_type.shape) != rank:
        raise ValueError(f"{label} must have rank {rank}, got {index_type.shape}")
    return index_type


def _positive_dim(graph: Graph, op: Operation) -> tuple[int, int]:
    """The checked ``dim`` attribute and the ranked operand's rank."""

    source_type = graph.value(op.inputs[0]).type
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{op.op} requires a tensor with rank >= 1")
    dim = op.attrs["dim"]
    if dim < 0 or dim >= rank:
        raise ValueError(f"{op.op}.dim {dim} is outside rank {rank}")
    return dim, rank


def _require_output(
    graph: Graph, op: Operation, expected: TensorType, label: str
) -> None:
    output_type = graph.value(op.outputs[0]).type
    if output_type != expected:
        raise ValueError(
            f"{op.op} {label} "
            f"{(output_type.shape, output_type.dtype.value)} does not match "
            f"expected {(expected.shape, expected.dtype.value)}"
        )


def _index_output_type(graph: Graph, op: Operation) -> TensorType:
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype != DType.I64:
        raise ValueError(
            f"{op.op} must return I64 indices, got {output_type.dtype.value}"
        )
    return output_type
