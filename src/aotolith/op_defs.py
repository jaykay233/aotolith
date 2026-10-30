"""Declarative definitions for the compiler's IR operations.

Every IR operation the backend executes is described once, here: how many
operands it takes, which attributes it must carry, what its output shape and
dtype must be, and how many values it writes. Whatever those fields cannot
say - a scatter's index/source compatibility, an embedding's float weight and
integer indices - lives in an ``extra`` hook beside the row instead of in a
separate branch of the backend.

The table exists so a new operator is one row plus, at most, one hook, rather
than another ``if op.op == ...`` arm with its own hand-written rank, dtype and
shape preamble. The generic checks in ``check_operation`` are the ones the
operators were each repeating; the table says which apply and with what
parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, prod

from .ir import DType, Graph, Operation, TensorType
from .op_sets import CONSTANT_OPS as _CONSTANT_OPS
from .op_sets import POINTWISE_OPS as _POINTWISE_OPS
from .shapes import broadcast_shapes, can_broadcast

FLOATING_DTYPES = frozenset({DType.F16, DType.BF16, DType.F32})
INTEGRAL_DTYPES = frozenset({DType.I32, DType.I64})
INDEX_DTYPES = INTEGRAL_DTYPES
# `mean` is not a single MLX reduce: it accumulates the values and the number
# of writes, then divides, so it is not one of the ``array.at`` methods.
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
    INT_OR_NONE = "int_or_none"
    DIM_OR_NONE = "dim_or_none"
    LITERAL = "literal"
    """A bool, int or float scalar the operator uses as data."""


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
        elif kind is AttrKind.INT_OR_NONE:
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{label} must be an integer or None, got {value!r}")
        elif kind is AttrKind.LITERAL:
            if not isinstance(value, (bool, int, float)):
                raise ValueError(f"{label} must be a scalar, got {value!r}")
        return value


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


def check_operation(graph: Graph, op: Operation) -> None:
    """Validate one operation against its contract.

    Raises ``ValueError`` for a malformed operation and ``KeyError`` for one
    this build has no contract for, so an unexpected operator fails by name
    instead of reaching a runtime that cannot execute it.
    """

    contract = contract_for(op.op)
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


def _no_boolean_operand(graph: Graph, op: Operation) -> None:
    if graph.value(op.inputs[0]).type.dtype == DType.BOOL:
        raise NotImplementedError(f"{op.op} does not accept boolean tensor inputs")


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


def _check_cat(graph: Graph, op: Operation) -> None:
    input_types = [graph.value(value_id).type for value_id in op.inputs]
    output_type = graph.value(op.outputs[0]).type
    rank = len(input_types[0].shape)
    axis = normalize_axis(op.attrs["dim"], rank, "cat.dim")
    if any(len(input_type.shape) != rank for input_type in input_types):
        raise ValueError("cat inputs must have the same rank")
    if any(input_type.dtype != output_type.dtype for input_type in input_types):
        raise ValueError("cat must preserve input dtype")
    for position, input_type in enumerate(input_types):
        for input_axis, (input_dim, output_dim) in enumerate(
            zip(input_type.shape, output_type.shape, strict=True)
        ):
            if input_axis != axis and input_dim != output_dim:
                raise ValueError(
                    f"cat input {position} shape {input_type.shape} is "
                    f"incompatible with output shape {output_type.shape} on "
                    f"axis {input_axis}"
                )
    expected = list(output_type.shape)
    expected[axis] = sum(input_type.shape[axis] for input_type in input_types)
    if tuple(expected) != output_type.shape:
        raise ValueError(f"cat output shape {output_type.shape} != {tuple(expected)}")


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


def _check_cast(graph: Graph, op: Operation) -> None:
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    if input_shape != output_shape:
        raise ValueError(f"cast shape {input_shape} != {output_shape}")


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


def _index_output_type(graph: Graph, op: Operation) -> TensorType:
    output_type = graph.value(op.outputs[0]).type
    if output_type.dtype != DType.I64:
        raise ValueError(
            f"{op.op} must return I64 indices, got {output_type.dtype.value}"
        )
    return output_type


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


# ---------------------------------------------------------------------------
# Hooks for operators whose operands constrain one another
# ---------------------------------------------------------------------------


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


def _check_embedding(graph: Graph, op: Operation) -> None:
    weight_type = graph.value(op.inputs[0]).type
    indices_type = graph.value(op.inputs[1]).type
    if len(weight_type.shape) != 2:
        raise ValueError(f"embedding weight must have rank 2, got {weight_type.shape}")
    if weight_type.dtype not in FLOATING_DTYPES:
        raise ValueError(
            f"embedding weight must be floating point, got {weight_type.dtype.value}"
        )
    _index_tensor(graph, op.inputs[1], "embedding indices")
    expected = TensorType(
        indices_type.shape + (weight_type.shape[1],), weight_type.dtype
    )
    _require_output(graph, op, expected, "output metadata")

    _only_attrs(op, {"padding_idx", "scale_grad_by_freq", "sparse"})
    padding_idx = op.attrs.get("padding_idx", -1)
    if padding_idx is not None:
        if not isinstance(padding_idx, int) or isinstance(padding_idx, bool):
            raise ValueError(
                f"embedding.padding_idx must be an integer, got {padding_idx!r}"
            )
        if not -weight_type.shape[0] <= padding_idx < weight_type.shape[0]:
            raise ValueError(
                f"embedding.padding_idx {padding_idx} is outside the weight table"
            )
    for name in ("scale_grad_by_freq", "sparse"):
        if not isinstance(op.attrs.get(name, False), bool):
            raise ValueError(f"embedding.{name} must be a boolean")


def _check_index_select(graph: Graph, op: Operation) -> None:
    source_type = graph.value(op.inputs[0]).type
    dim, _ = _positive_dim(graph, op)
    indices_type = _index_tensor(graph, op.inputs[1], "index_select indices", rank=1)
    expected = TensorType(
        source_type.shape[:dim]
        + (indices_type.shape[0],)
        + source_type.shape[dim + 1 :],
        source_type.dtype,
    )
    _require_output(graph, op, expected, "output metadata")
    _unsupported_attrs(op, {"dim"})


def _check_index(graph: Graph, op: Operation) -> None:
    source_type = graph.value(op.inputs[0]).type
    axes_value = op.attrs["axes"]
    if len(op.inputs) != len(axes_value) + 1:
        raise ValueError(
            f"index has {len(op.inputs) - 1} indices for axes {axes_value}"
        )
    first_axis = axes_value[0]
    last_axis = axes_value[-1]
    if axes_value != tuple(range(first_axis, last_axis + 1)):
        raise NotImplementedError(
            "non-contiguous advanced indexing is not supported: "
            f"indexed axes {axes_value}"
        )
    if first_axis < 0 or last_axis >= len(source_type.shape):
        raise ValueError(
            f"index axes {axes_value} exceed source rank {len(source_type.shape)}"
        )
    index_shapes = tuple(
        _index_tensor(graph, value_id, "index indices").shape
        for value_id in op.inputs[1:]
    )
    expected = TensorType(
        source_type.shape[:first_axis]
        + broadcast_shapes(index_shapes)
        + source_type.shape[last_axis + 1 :],
        source_type.dtype,
    )
    _require_output(graph, op, expected, "output metadata")
    _unsupported_attrs(op, {"axes"})


def _check_gather(graph: Graph, op: Operation) -> None:
    source_type = graph.value(op.inputs[0]).type
    dim, rank = _positive_dim(graph, op)
    indices_type = _index_tensor(graph, op.inputs[1], "gather indices", rank=rank)
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != dim and index_dim > source_dim:
            raise ValueError(
                f"gather indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-gather axis {index_axis}"
            )
    _require_output(
        graph,
        op,
        TensorType(indices_type.shape, source_type.dtype),
        "output metadata",
    )
    _unsupported_attrs(op, {"dim"})


def _check_scatter(graph: Graph, op: Operation) -> None:
    source_type = graph.value(op.inputs[0]).type
    dim, rank = _positive_dim(graph, op)
    indices_type = _index_tensor(graph, op.inputs[1], "scatter indices", rank=rank)
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != dim and index_dim > source_dim:
            raise ValueError(
                f"scatter indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-scatter axis {index_axis}"
            )
    _require_output(graph, op, source_type, "output metadata")

    if len(op.inputs) == 2:
        _only_attrs(op, {"dim", "value"})
        if not isinstance(op.attrs["value"], (bool, int, float)):
            raise ValueError(
                f"scatter.value must be a scalar number, got {op.attrs['value']!r}"
            )
        return

    # The update operand constrains the write regardless of how it reduces.
    update_type = graph.value(op.inputs[2]).type
    if update_type.dtype != source_type.dtype:
        raise ValueError(
            "scatter source dtype must match the input dtype, "
            f"got {update_type.dtype.value} and {source_type.dtype.value}"
        )
    if len(update_type.shape) != rank:
        raise ValueError(
            f"scatter source must have rank {rank}, got {update_type.shape}"
        )
    for update_axis, (update_dim, index_dim) in enumerate(
        zip(update_type.shape, indices_type.shape, strict=True)
    ):
        if update_dim < index_dim:
            raise ValueError(
                f"scatter source shape {update_type.shape} is smaller than "
                f"indices shape {indices_type.shape} on axis {update_axis}"
            )

    reduce = op.attrs.get("reduce")
    allowed = {"dim"} if reduce is None else {"dim", "reduce", "include_self"}
    _unsupported_attrs(op, allowed)
    if reduce is None:
        return
    if reduce not in SCATTER_REDUCES:
        raise ValueError(
            f"scatter.reduce {reduce!r} is not supported; expected one of "
            f"{sorted(SCATTER_REDUCES)}"
        )
    include_self = op.attrs.get("include_self", True)
    if not isinstance(include_self, bool):
        raise ValueError(
            f"scatter.include_self must be a boolean, got {include_self!r}"
        )
    if reduce == "mean" and source_type.dtype not in FLOATING_DTYPES:
        raise NotImplementedError(
            "scatter.reduce=mean needs a floating-point input, got "
            f"{source_type.dtype.value}"
        )
    if not include_self and reduce in {"amax", "amin"}:
        if source_type.dtype not in FLOATING_DTYPES:
            raise NotImplementedError(
                f"scatter.reduce={reduce} with include_self=False needs a "
                "floating-point input for its identity value, got "
                f"{source_type.dtype.value}"
            )


def _check_index_add(graph: Graph, op: Operation) -> None:
    source_type = graph.value(op.inputs[0]).type
    dim, _ = _positive_dim(graph, op)
    _unsupported_attrs(op, {"dim", "alpha"})
    alpha = op.attrs.get("alpha", 1.0)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"index_add.alpha must be a scalar number, got {alpha!r}")
    indices_type = graph.value(op.inputs[1]).type
    if len(indices_type.shape) != 1:
        raise NotImplementedError(
            f"index_add requires a 1-D index tensor, got {indices_type.shape}"
        )
    _index_tensor(graph, op.inputs[1], "index_add indices")
    update_type = graph.value(op.inputs[2]).type
    if update_type.dtype != source_type.dtype:
        raise ValueError(
            "index_add source dtype must match the input dtype, got "
            f"{update_type.dtype.value} and {source_type.dtype.value}"
        )
    expected_update = (
        source_type.shape[:dim]
        + (indices_type.shape[0],)
        + source_type.shape[dim + 1 :]
    )
    if update_type.shape != expected_update:
        raise ValueError(
            f"index_add source shape {update_type.shape} does not match "
            f"{expected_update}"
        )
    _require_output(graph, op, source_type, "output metadata")


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


# ---------------------------------------------------------------------------
# Layout rules
# ---------------------------------------------------------------------------

LayoutRule = Callable[[Operation, tuple[int, ...]], tuple[int, ...]]


def _reshape_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    target = op.attrs["shape"]
    if not target and (prod(shape) != 1):
        raise ValueError(
            f"a rank-0 reshape needs exactly one input element, got {shape}"
        )
    if any(dim == 0 or dim < -1 for dim in target):
        raise ValueError(f"reshape.shape contains an invalid dimension: {target}")
    if target.count(-1) > 1:
        raise ValueError(f"reshape.shape has multiple inferred dimensions: {target}")
    input_count = prod(shape) or 1
    known_count = prod(dim for dim in target if dim != -1) or 1
    if -1 not in target:
        return target
    if input_count % known_count:
        raise ValueError(
            f"reshape shape {target} cannot infer a dimension for "
            f"{input_count} elements"
        )
    inferred = input_count // known_count
    return tuple(inferred if dim == -1 else dim for dim in target)


def _transpose_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    first, second = op.attrs["dim0"], op.attrs["dim1"]
    normalized = tuple(
        normalize_axis(dim, len(shape), "transpose dimension")
        for dim in (first, second)
    )
    expected = list(shape)
    expected[normalized[0]], expected[normalized[1]] = (
        expected[normalized[1]],
        expected[normalized[0]],
    )
    return tuple(expected)


def _permute_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = op.attrs["dims"]
    if len(dims) != len(shape):
        raise ValueError(f"permute.dims must contain {len(shape)} integer dimensions")
    normalized = [normalize_axis(dim, len(shape), "permute dimensions") for dim in dims]
    if sorted(normalized) != list(range(len(shape))):
        raise ValueError(
            f"permute dimensions {dims} are not a permutation of rank {len(shape)}"
        )
    return tuple(shape[dim] for dim in normalized)


def _expand_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    target = op.attrs["shape"]
    if not all(dim == -1 or dim > 0 for dim in target):
        raise ValueError(f"expand.shape contains invalid dimensions: {target}")
    aligned = (1,) * (len(target) - len(shape)) + shape
    resolved = tuple(
        aligned[axis] if dim == -1 else dim for axis, dim in enumerate(target)
    )
    if not can_broadcast(aligned, resolved):
        raise ValueError(f"cannot expand shape {shape} to {resolved}")
    return resolved


def _slice_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    axis = normalize_axis(op.attrs["dim"], len(shape), "slice.dim")
    start = op.attrs["start"]
    end = op.attrs["end"]
    step = op.attrs["step"]
    sliced_length = len(range(*slice(start, end, step).indices(shape[axis])))
    return shape[:axis] + (sliced_length,) + shape[axis + 1 :]


def _identity_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape


def _flip_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    _axes(op, "dims", shape, "flip.dims")
    return shape


def _roll_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    shifts = op.attrs["shifts"]
    dims = op.attrs.get("dims")
    if dims is None:
        if len(shifts) != 1:
            raise ValueError(
                f"roll without dims requires exactly one shift, got {len(shifts)}"
            )
        return shape
    if len(dims) != len(shifts):
        raise ValueError(
            "roll shifts and dims must have the same length, got "
            f"{len(shifts)} and {len(dims)}"
        )
    _axes(op, "dims", shape, "roll.dims")
    return shape


def _axes(
    op: Operation, attribute: str, shape: tuple[int, ...], label: str
) -> tuple[int, ...]:
    values = op.attrs[attribute]
    if not values:
        raise ValueError(f"{label} must not be empty")
    normalized = tuple(normalize_axis(dim, len(shape), label) for dim in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate axes: {values}")
    return normalized


def _unsqueeze_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    output_rank = len(shape) + 1
    axis = normalize_axis(op.attrs["dim"], output_rank, "unsqueeze.dim")
    return shape[:axis] + (1,) + shape[axis:]


def _squeeze_shape(op: Operation, shape: tuple[int, ...]) -> tuple[int, ...]:
    axis = normalize_axis(op.attrs["dim"], len(shape), "squeeze.dim")
    if shape[axis] != 1:
        return shape
    return shape[:axis] + shape[axis + 1 :]


LAYOUT_RULES: dict[str, LayoutRule] = {
    "reshape": _reshape_shape,
    "transpose": _transpose_shape,
    "permute": _permute_shape,
    "expand": _expand_shape,
    "slice": _slice_shape,
    "flip": _flip_shape,
    "roll": _roll_shape,
    "unsqueeze": _unsqueeze_shape,
    "squeeze": _squeeze_shape,
    "contiguous": _identity_shape,
    "alias": _identity_shape,
}


def _check_layout(graph: Graph, op: Operation) -> None:
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    expected = LAYOUT_RULES[op.op](op, input_shape)
    if expected != output_shape:
        raise ValueError(f"{op.op} output shape {output_shape} != {expected}")


_ARANGE_ATTRS = (
    AttrSpec("start", AttrKind.FINITE_NUMBER),
    AttrSpec("end", AttrKind.FINITE_NUMBER),
    AttrSpec("step", AttrKind.FINITE_NUMBER),
)


OP_CONTRACTS: dict[str, OpContract] = {
    # -- reductions -------------------------------------------------------
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
    # -- scans ------------------------------------------------------------
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
    # -- orderings --------------------------------------------------------
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
    # -- elementwise views ------------------------------------------------
    "triu": OpContract(
        arity=(1, 1),
        rank=(2, 8),
        attrs=(AttrSpec("diagonal", AttrKind.INT),),
    ),
    "tril": OpContract(
        arity=(1, 1),
        rank=(2, 8),
        attrs=(AttrSpec("diagonal", AttrKind.INT),),
    ),
    "cast": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        extra=_check_cast,
    ),
    # -- structure --------------------------------------------------------
    "cat": OpContract(
        arity=(1, 64),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_cat,
    ),
    # -- kernels with hand-written source --------------------------------
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
    "pad": OpContract(arity=(1, 1), dtype=DtypeRule.ANY, shape=ShapeRule.FREE),
    # -- pointwise -------------------------------------------------------
    # -- indexing ---------------------------------------------------------
    "embedding": OpContract(
        arity=(2, 2),
        rank=(2, 2),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        extra=_check_embedding,
    ),
    "index_select": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_index_select,
    ),
    "index": OpContract(
        arity=(2, 9),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("axes", AttrKind.AXES),),
        extra=_check_index,
    ),
    "gather": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_gather,
    ),
    "scatter": OpContract(
        arity=(2, 3),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_scatter,
    ),
    "index_add": OpContract(
        arity=(3, 3),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_index_add,
    ),
    # -- fused operators --------------------------------------------------
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
    # -- views ------------------------------------------------------------
    "reshape": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shape", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "transpose": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim0", AttrKind.INT),
            AttrSpec("dim1", AttrKind.INT),
        ),
        extra=_check_layout,
    ),
    "permute": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_layout,
    ),
    "expand": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shape", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "slice": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("dim", AttrKind.INT),
            AttrSpec("start", AttrKind.INT_OR_NONE),
            AttrSpec("end", AttrKind.INT_OR_NONE),
            AttrSpec("step", AttrKind.INT),
        ),
        extra=_check_layout,
    ),
    "flip": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dims", AttrKind.AXES),),
        extra=_check_layout,
    ),
    "roll": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("shifts", AttrKind.INT_TUPLE),),
        extra=_check_layout,
    ),
    "unsqueeze": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_layout,
    ),
    "squeeze": OpContract(
        arity=(1, 1),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_layout,
    ),
    "contiguous": OpContract(arity=(1, 1), shape=ShapeRule.FREE, extra=_check_layout),
    "alias": OpContract(arity=(1, 1), shape=ShapeRule.FREE, extra=_check_layout),
    # -- constants --------------------------------------------------------
    "arange": OpContract(
        arity=(0, 0),
        shape=ShapeRule.FREE,
        attrs=_ARANGE_ATTRS,
        extra=_check_arange,
    ),
}

# Every pointwise operator shares one shape contract: an elementwise pass over
# broadcast operands. The expression-level checks live with the emitter that
# has to turn the body into Metal, because they are about expressibility rather
# than about the operation's own metadata.
for _pointwise_op in _POINTWISE_OPS:
    OP_CONTRACTS.setdefault(
        _pointwise_op,
        OpContract(
            arity=(1, 3),
            rank=(0, 8),
            dtype=DtypeRule.ANY,
            shape=ShapeRule.FREE,
            native=True,
        ),
    )

for _constant_op in _CONSTANT_OPS:
    OP_CONTRACTS[_constant_op] = OpContract(
        arity=(0, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("value", AttrKind.LITERAL),),
        native=True,
    )


def contract_for(op_name: str) -> OpContract:
    """The contract for ``op_name``, or ``KeyError`` naming the operator."""

    try:
        return OP_CONTRACTS[op_name]
    except KeyError as error:
        raise KeyError(f"no contract is registered for IR op {op_name!r}") from error
