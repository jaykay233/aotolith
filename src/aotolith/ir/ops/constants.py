"""Operators that build a tensor from attributes alone.

They read no graph values, which is why their contract permits zero operands and
why the backend has a separate path that writes a buffer without reading one.
``arange`` also needs its own length arithmetic: the element count comes from
start/end/step, not from any input shape.
"""

from __future__ import annotations

from .. import DType, Graph, Operation
from .rules import (
    AttrKind,
    AttrSpec,
    FLOATING_DTYPES,
    INTEGRAL_DTYPES,
    OpContract,
    ShapeRule,
)
from math import ceil, isfinite


def arange_length(start: int | float, end: int | float, step: int | float) -> int:
    """The number of elements ``arange(start, end, step)`` produces."""

    if all(isinstance(value, int) for value in (start, end, step)):
        start, end, step = int(start), int(end), int(step)
        if step > 0:
            if start >= end:
                return 0
            return (end - start + step - 1) // step
        if start <= end:
            return 0
        stride = -step
        return (start - end + stride - 1) // stride
    # PyTorch derives the length from the same double quotient and then clamps
    # negative results to zero; keeping the exact expression means a fractional
    # bound either matches or is rejected by the shape check.
    length = ceil((float(end) - float(start)) / float(step))
    return max(length, 0)


_ARANGE_ATTRS = (
    AttrSpec("start", AttrKind.FINITE_NUMBER),
    AttrSpec("end", AttrKind.FINITE_NUMBER),
    AttrSpec("step", AttrKind.FINITE_NUMBER),
)


def _check_arange(graph: Graph, op: Operation) -> None:
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype not in INTEGRAL_DTYPES | FLOATING_DTYPES:
        raise NotImplementedError(
            "arange supports int32, int64, float16, bfloat16 and float32 "
            f"outputs, got {output_type.dtype.value}"
        )
    if len(output_type.shape) != 1:
        raise ValueError(
            f"arange must produce a rank-1 tensor, got {output_type.shape}"
        )
    bounds = tuple(op.attrs[name] for name in ("start", "end", "step"))
    if bounds[2] == 0:
        raise ValueError("arange.step must be nonzero")
    if output_type.dtype in INTEGRAL_DTYPES and not all(
        isinstance(bound, int) for bound in bounds
    ):
        # PyTorch would round every element back to an integer here, which
        # makes the fractional bounds meaningless.
        raise NotImplementedError(
            "arange with integral output dtype requires integer start, end and "
            f"step values, got {list(bounds)}"
        )
    expected_length = arange_length(*bounds)
    if output_type.shape[0] != expected_length:
        raise ValueError(
            f"arange output shape {output_type.shape} does not match expected "
            f"length {expected_length}"
        )


def integer_constant_value(op: Operation, dtype: DType) -> int:
    """The integer an integer-constant operation stands for, checked to fit."""

    value = op.attrs["value"]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{op.op}.value cannot be converted to an integer")
    converted = int(value)
    bits = 32 if dtype == DType.I32 else 64
    if not -(1 << (bits - 1)) <= converted <= (1 << (bits - 1)) - 1:
        raise ValueError(f"{op.op}.value {value!r} overflows {dtype.value}")
    return converted


def check_integer_constant(graph: Graph, op: Operation) -> None:
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype not in INTEGRAL_DTYPES:
        raise ValueError(f"{op.op} is not an integer constant")
    integer_constant_value(op, output_type.dtype)


CONTRACTS: dict[str, OpContract] = {
    "arange": OpContract(
        arity=(0, 0),
        shape=ShapeRule.FREE,
        attrs=_ARANGE_ATTRS,
        extra=_check_arange,
    ),
}
