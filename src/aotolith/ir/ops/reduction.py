"""Reductions, scans and orderings.

All three families share the shape rule: the output drops or keeps the axes an
attribute names. What differs is what ``extra`` has to verify - that a scan has
a supportable axis, that a variance's correction leaves a positive count, that
``topk``'s window fits inside the axis - so each hook lives here beside its
rows.
"""

from __future__ import annotations

from .. import DType, Graph, Operation
from .rules import (
    AttrKind,
    AttrSpec,
    DtypeRule,
    OpContract,
    ShapeRule,
    keepdim,
    normalize_axis,
    reduced_shape,
)
from .shared import (
    _index_output_type,
    _no_boolean_operand,
)
from math import isfinite


def _check_scan_shape(graph: Graph, op: Operation) -> None:
    """A scan keeps the input shape; its dtype may follow PyTorch's promotion."""

    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    if output_shape != input_shape:
        raise ValueError(f"{op.op} output shape {output_shape} != {input_shape}")


def _check_correction(graph: Graph, op: Operation) -> None:
    """``var``/``std`` carry an optional Bessel correction."""

    correction = op.attrs.get("correction", 1)
    if isinstance(correction, bool) or not isinstance(correction, (int, float)):
        raise NotImplementedError(
            f"{op.op}.correction must be a number, got {correction!r}"
        )
    if isinstance(correction, float) and not isfinite(correction):
        raise ValueError(f"{op.op}.correction must be finite, got {correction!r}")


def _check_arg_reduction(graph: Graph, op: Operation) -> None:
    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_type.shape)
    dim = op.attrs.get("dim")
    if dim is None:
        if keepdim(op):
            raise ValueError(f"{op.op} cannot keep a dimension without a dim")
        expected: tuple[int, ...] = ()
    else:
        axis = normalize_axis(int(dim), rank, f"{op.op}.dim")
        expected = reduced_shape(input_type.shape, (axis,), keepdim(op))
    if output_type.shape != expected:
        raise ValueError(
            f"{op.op} output shape {output_type.shape} does not match {expected}"
        )
    if output_type.dtype != DType.I64:
        raise ValueError(
            f"{op.op} must return I64 indices, got {output_type.dtype.value}"
        )
    if input_type.dtype == DType.BOOL:
        raise ValueError(f"{op.op} does not accept boolean tensor inputs")


def _check_argsort(graph: Graph, op: Operation) -> None:
    _no_boolean_operand(graph, op)
    output_type = _index_output_type(graph, op)
    input_type = graph.value(op.inputs[0]).type
    if output_type.shape != input_type.shape:
        raise ValueError(
            f"{op.op} output shape {output_type.shape} does not match "
            f"{input_type.shape}"
        )


def _check_topk(graph: Graph, op: Operation) -> None:
    _no_boolean_operand(graph, op)
    input_type = graph.value(op.inputs[0]).type
    output_type = graph.value(op.outputs[0]).type
    expected_dtype = DType.I64 if op.op == "topk_indices" else input_type.dtype
    if output_type.dtype != expected_dtype:
        raise ValueError(
            f"{op.op} must return {expected_dtype.value}, got {output_type.dtype.value}"
        )
    _check_topk_window(graph, op, output_type.shape)


def _check_topk_window(graph: Graph, op: Operation, shape: tuple[int, ...]) -> None:
    input_type = graph.value(op.inputs[0]).type
    rank = len(input_type.shape)
    axis = normalize_axis(op.attrs["dim"], rank, f"{op.op}.dim")
    k = op.attrs["k"]
    if k > input_type.shape[axis]:
        raise ValueError(
            f"{op.op}.k {k} exceeds dimension {axis} of size {input_type.shape[axis]}"
        )
    expected = input_type.shape[:axis] + (k,) + input_type.shape[axis + 1 :]
    if shape != expected:
        raise ValueError(f"{op.op} output shape {shape} does not match {expected}")


CONTRACTS: dict[str, OpContract] = {
    "sum": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        native=True,
    ),
    "mean": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        native=True,
    ),
    "prod": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_no_boolean_operand,
    ),
    "amax": OpContract(
        arity=(1, 1),
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
    ),
    "amin": OpContract(
        arity=(1, 1),
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
    ),
    "vector_norm": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(
            AttrSpec("dims", AttrKind.AXES),
            AttrSpec("ord", AttrKind.NUMBER),
        ),
    ),
    "var": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_correction,
    ),
    "std": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.REDUCED,
        axes="dims",
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_correction,
    ),
    "log_softmax": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        attrs=(AttrSpec("dim", AttrKind.INT),),
    ),
    "softmax": OpContract(arity=(1, 1), native=True),
    "safe_softmax": OpContract(arity=(1, 1), native=True),
    "cumsum": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_scan_shape,
    ),
    "cumprod": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_scan_shape,
    ),
    "logcumsumexp": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_scan_shape,
    ),
    "sort": OpContract(
        arity=(1, 1),
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("descending", AttrKind.BOOL),
        ),
        extra=_no_boolean_operand,
    ),
    "argsort": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("descending", AttrKind.BOOL),
        ),
        extra=_check_argsort,
    ),
    "topk": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("k", AttrKind.NONNEGATIVE_INT),
            AttrSpec("largest", AttrKind.BOOL),
        ),
        extra=_check_topk,
    ),
    "topk_indices": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("k", AttrKind.NONNEGATIVE_INT),
            AttrSpec("largest", AttrKind.BOOL),
        ),
        extra=_check_topk,
    ),
    "argmax": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.DIM_OR_NONE),),
        extra=_check_arg_reduction,
    ),
    "argmin": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.DIM_OR_NONE),),
        extra=_check_arg_reduction,
    ),
}
