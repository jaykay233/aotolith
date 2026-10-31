"""The rules a contract is written in, and the generic check that applies them.

``OpContract`` describes one operator with six fields. ``check_operation`` is
the single entry point that runs them: arity, operand rank, dtype, output count,
attribute shape, and then ``extra`` for whatever fields cannot say.

The separation matters because the fields cover the common cases - a pointwise
operator, a reduction over named axes, a view that preserves type - and
``extra`` exists only for the inter-operator constraints (a scatter's indices
and source must agree) that no field can express. Adding an operator is
therefore one row, and only a genuinely new kind of constraint needs code.
"""

from __future__ import annotations

from .. import DType, Graph, Operation, TensorType
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from math import isfinite

FLOATING_DTYPES = frozenset({DType.F16, DType.BF16, DType.F32})


INTEGRAL_DTYPES = frozenset({DType.I32, DType.I64})


INDEX_DTYPES = INTEGRAL_DTYPES


SCATTER_REDUCES = frozenset({"add", "prod", "amax", "amin", "mean"})


class DtypeRule(str, Enum):
    """What an operator requires of, and promises about, its dtypes."""

    PRESERVE = "preserve"
    """The output keeps the first operand's dtype."""
    FLOATING_IN = "floating_in"
    """The first operand must be floating point; the output keeps its dtype."""
    ANY = "any"
    """No dtype requirement; ``extra`` owns what is left."""


class ShapeRule(str, Enum):
    """What an operator promises about its output shape."""

    SAME = "same"
    """The output has exactly the first operand's tensor type."""
    REDUCED = "reduced"
    """The output reduces the axes an attribute names, honouring keepdim."""
    FREE = "free"
    """No generic rule; ``extra``, or the operator's own compiler, checks it."""


class AttrKind(str, Enum):
    """The value shape an attribute must have."""

    INT = "int"
    BOOL = "bool"
    NUMBER = "number"
    FINITE_NUMBER = "finite_number"
    NONNEGATIVE_INT = "nonnegative_int"
    AXES = "axes"
    """A non-empty tuple of integers naming distinct axes."""
    INT_TUPLE = "int_tuple"
    """A tuple of integers, possibly empty."""
    NUMBER_TUPLE = "number_tuple"
    """A tuple of numbers, possibly empty."""
    INT_OR_NONE = "int_or_none"
    DIM_OR_NONE = "dim_or_none"
    LITERAL = "literal"
    """A bool, int or float scalar the operator uses as data."""


@dataclass(frozen=True, slots=True)
class AttrSpec:
    """One required attribute and the value shape it must have."""

    name: str
    kind: AttrKind

    def check(self, op: Operation, rank: int) -> object:
        """Validate this attribute and return its normalized value."""

        if self.name not in op.attrs:
            raise ValueError(f"{op.op} requires a {self.name!r} attribute")
        value = op.attrs[self.name]
        label = f"{op.op}.{self.name}"
        kind = self.kind
        if kind is AttrKind.INT:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label} must be an integer, got {value!r}")
        elif kind is AttrKind.BOOL:
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean, got {value!r}")
        elif kind in {AttrKind.NUMBER, AttrKind.FINITE_NUMBER}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number, got {value!r}")
            if kind is AttrKind.FINITE_NUMBER and not isfinite(float(value)):
                raise ValueError(f"{label} must be finite, got {value!r}")
        elif kind is AttrKind.NONNEGATIVE_INT:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{label} must be a non-negative integer, got {value!r}"
                )
        elif kind is AttrKind.DIM_OR_NONE:
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label} must be an integer or None, got {value!r}")
            return normalize_axis(value, rank, label)
        elif kind is AttrKind.AXES:
            return normalize_axes(value, rank, label)
        elif kind is AttrKind.INT_TUPLE:
            if not isinstance(value, tuple) or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in value
            ):
                raise ValueError(f"{label} must be a tuple of integers, got {value!r}")
        elif kind is AttrKind.NUMBER_TUPLE:
            if not isinstance(value, tuple) or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                raise ValueError(f"{label} must be a tuple of numbers, got {value!r}")
        elif kind is AttrKind.INT_OR_NONE:
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label} must be an integer or None, got {value!r}")
        elif kind is AttrKind.LITERAL:
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"{label} must be a scalar, got {value!r}")
        return value


def normalize_axis(value: int, rank: int, label: str) -> int:
    if not -rank <= value < rank:
        raise ValueError(f"{label} {value} exceeds rank {rank}")
    return value % rank


def normalize_axes(value: object, rank: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} must be a non-empty tuple, got {value!r}")
    if not all(isinstance(axis, int) and not isinstance(axis, bool) for axis in value):
        raise ValueError(f"{label} must contain integers, got {value!r}")
    axes = tuple(normalize_axis(int(axis), rank, label) for axis in value)
    if len(set(axes)) != len(axes):
        raise ValueError(f"{label} repeats an axis: {value!r}")
    return axes


def reduced_shape(
    shape: tuple[int, ...], axes: tuple[int, ...], keepdim: bool
) -> tuple[int, ...]:
    """The shape a reduction over ``axes`` produces."""

    if keepdim:
        return tuple(
            1 if position in axes else size for position, size in enumerate(shape)
        )
    return tuple(size for position, size in enumerate(shape) if position not in axes)


def normalized_axes(graph: Graph, op: Operation, attribute: str) -> tuple[int, ...]:
    """The axes ``attribute`` names, resolved against the first operand."""

    rank = len(graph.value(op.inputs[0]).type.shape)
    return normalize_axes(op.attrs[attribute], rank, f"{op.op}.{attribute}")


def keepdim(op: Operation) -> bool:
    """The checked ``keepdim`` attribute, defaulting to False."""

    value = op.attrs.get("keepdim", False)
    if not isinstance(value, bool):
        raise ValueError(f"{op.op}.keepdim must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class OpContract:
    """One IR operation's operands, attributes and result contract.

    ``arity`` is an inclusive operand range. ``outputs`` is the number of SSA
    values the operation writes; more than one only when a single kernel
    produces a tuple that ATen returns together. ``rank`` bounds the first
    operand's rank, ``axes`` names the attribute carrying reduction axes, and
    ``extra`` holds whatever the generic fields cannot express.
    """

    arity: tuple[int, int]
    outputs: tuple[int, int] = (1, 1)
    rank: tuple[int, int] = (1, 8)
    dtype: DtypeRule = DtypeRule.PRESERVE
    shape: ShapeRule = ShapeRule.SAME
    axes: str | None = None
    attrs: tuple[AttrSpec, ...] = ()
    native: bool = False
    """Whether this build may generate a Metal kernel for the operation."""
    extra: Callable[[Graph, Operation], None] | None = None

    def allows(self, operand_count: int) -> bool:
        low, high = self.arity
        return low <= operand_count <= high


def _check_arity(op: Operation, contract: OpContract) -> None:
    if contract.allows(len(op.inputs)):
        return
    low, high = contract.arity
    expected = f"{low}" if low == high else f"{low} to {high}"
    raise ValueError(f"{op.op} expects {expected} inputs, got {len(op.inputs)}")


def _check_rank(op: Operation, contract: OpContract, input_type: TensorType) -> int:
    rank = len(input_type.shape)
    low, high = contract.rank
    if low <= rank <= high:
        return rank
    if rank == 0:
        raise ValueError(f"{op.op} does not support rank-0 tensors")
    raise ValueError(f"{op.op} requires rank {low} to {high}, got {rank}")


def _check_reduction_shape(
    graph: Graph, op: Operation, contract: OpContract, input_type: TensorType
) -> None:
    shape = input_type.shape
    rank = len(shape)
    assert contract.axes is not None
    label = f"{op.op}.{contract.axes}"
    if contract.axes == "dim":
        dim = op.attrs["dim"]
        if dim is None:
            expected: tuple[int, ...] = ()
        else:
            axis = normalize_axis(int(dim), rank, label)
            expected = reduced_shape(shape, (axis,), keepdim(op))
    else:
        axes = normalize_axes(op.attrs[contract.axes], rank, label)
        expected = reduced_shape(shape, axes, keepdim(op))
    output_type = graph.value(op.outputs[0]).type
    if output_type.shape != expected:
        raise ValueError(
            f"{op.op} output shape {output_type.shape} does not match the "
            f"reduced input {expected}"
        )


def check_contract(graph: Graph, op: Operation, contract: OpContract) -> None:
    """Validate one operation against one contract.

    The caller supplies the contract because deciding *which* contract applies
    is a property of the table, not of the rules: this module knows how to check
    a contract, and ``table.check_operation`` knows which one an operator has.
    Splitting it that way keeps the rules below the table instead of in a cycle
    with it.
    """

    _check_arity(op, contract)
    low, high = contract.outputs
    if not low <= len(op.outputs) <= high:
        expected = f"{low}" if low == high else f"{low} to {high}"
        raise ValueError(
            f"{op.op} must produce {expected} results, got {len(op.outputs)}"
        )

    if not op.inputs:
        for attribute in contract.attrs:
            attribute.check(op, 0)
        if contract.extra is not None:
            contract.extra(graph, op)
        return

    input_type = graph.value(op.inputs[0]).type
    rank = _check_rank(op, contract, input_type)
    for attribute in contract.attrs:
        attribute.check(op, rank)

    output_type = graph.value(op.outputs[0]).type
    if contract.dtype is DtypeRule.FLOATING_IN:
        if input_type.dtype not in FLOATING_DTYPES:
            raise NotImplementedError(
                f"{op.op} needs a floating-point input, got {input_type.dtype.value}"
            )
    elif contract.dtype is DtypeRule.PRESERVE:
        if output_type.dtype != input_type.dtype:
            raise ValueError(
                f"{op.op} must preserve the input dtype, got "
                f"{output_type.dtype.value} for a {input_type.dtype.value} input"
            )

    if contract.shape is ShapeRule.SAME:
        if output_type != input_type:
            raise ValueError(
                f"{op.op} must preserve the input tensor type, got "
                f"{(output_type.shape, output_type.dtype.value)}"
            )
    elif contract.shape is ShapeRule.REDUCED:
        _check_reduction_shape(graph, op, contract, input_type)

    if contract.extra is not None:
        contract.extra(graph, op)
