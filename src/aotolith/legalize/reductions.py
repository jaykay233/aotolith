"""Reductions whose ATen form is not one core operator: ``nansum``, logsumexp,
``count_nonzero``, ``all``/``any``, the extremum dim variants, and the sort
family that has to keep both values and indices.

They share a shape: reduce over axes, then combine. Keeping the arguments
helpers next to the builders means the (dims, keepdim) parsing is written once
per family rather than inline in the dispatch."""

from __future__ import annotations

import operator
from math import isfinite, prod
from collections.abc import Callable
from typing import Any
from ..ir import DType, Operation, TensorType, Value
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _AND_OR_TARGETS,
    _COUNT_NONZERO_TARGETS,
    _EXTREMUM_DIM_TARGETS,
    _LOGSUMEXP_TARGET,
    _MEDIAN_TARGETS,
    _NANSUM_TARGET,
    _SORT_TARGETS,
    _TORCH_DTYPES,
    _VARIANCE_TARGETS,
    _VECTOR_NORM_TARGET,
)


def _reduced_extent(
    reduced_shape: tuple[int, ...],
    dims: tuple[int, ...],
    keepdim: bool,
) -> tuple[int, ...]:
    """Return the output shape of a reduction that keeps singleton axes."""

    if keepdim:
        return reduced_shape
    return tuple(
        size for axis, size in enumerate(reduced_shape) if axis not in set(dims)
    )


def _boolean_reduction_arguments(
    node: object,
    op: str,
) -> tuple[object, tuple[int, ...], bool, bool]:
    """Normalize the ``all``/``any``/``count_nonzero`` dims and keepdim."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{op} expects an input tensor")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if not isinstance(keepdim, bool):
        raise ValueError(f"{op}.keepdim must be a boolean, got {keepdim!r}")

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif (
        isinstance(dims, (tuple, list))
        and dims
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(f"{op}.dim must contain integers, got {dims!r}")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"{op}.dim repeats an axis: {dims!r}")

    reduced_shape = tuple(
        1 if axis in normalized_dims else size
        for axis, size in enumerate(input_type.shape)
    )
    return input_node, normalized_dims, keepdim, reduced_shape


def _logsumexp_arguments(
    node: object,
) -> tuple[object, tuple[int, ...], bool]:
    """Normalize the ``logsumexp`` dims and keepdim arguments."""

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("logsumexp expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError("logsumexp requires a ranked tensor")

    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if not isinstance(keepdim, bool):
        raise ValueError(f"logsumexp.keepdim must be a boolean, got {keepdim!r}")
    if isinstance(dims, int) and not isinstance(dims, bool):
        normalized = (dims % rank,)
    elif (
        isinstance(dims, (tuple, list))
        and dims
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
    ):
        normalized = tuple(dim % rank for dim in dims)
    elif isinstance(dims, (tuple, list)) and not dims:
        raise ValueError(
            "logsumexp needs at least one reduction axis; an empty dim tuple "
            "has no ATen meaning"
        )
    else:
        raise ValueError(f"logsumexp.dim must contain integers, got {dims!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"logsumexp.dim repeats an axis: {dims!r}")
    return input_node, normalized, keepdim


def _logsumexp_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.logsumexp`` into amax/masked_fill/exp/sum/log/add.

    ATen shifts by the reduced maximum for stability and replaces an infinite
    maximum with zero before the subtraction, so a +inf row sums to +inf and an
    all -inf row keeps -inf instead of producing ``inf - inf``.
    """

    input_node, dims, keepdim = _logsumexp_arguments(node)
    input_type = _node_type(input_node)
    shape = input_type.shape
    reduced_shape = tuple(
        1 if axis in dims else size for axis, size in enumerate(shape)
    )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
        dtype: DType | None = None,
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, dtype or input_type.dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    max_id = emit(
        "amax",
        (current,),
        {"dims": dims, "keepdim": True},
        reduced_shape,
    )
    infinite_id = emit(
        "isinf",
        (max_id,),
        {},
        reduced_shape,
        DType.BOOL,
    )
    shift_id = emit(
        "masked_fill",
        (max_id, infinite_id),
        {"value": 0.0},
        reduced_shape,
    )
    shifted_id = emit("sub", (current, shift_id), {}, shape)
    exp_id = emit("exp", (shifted_id,), {}, shape)
    sum_id = emit("sum", (exp_id,), {"dims": dims, "keepdim": True}, reduced_shape)
    log_id = emit("log", (sum_id,), {}, reduced_shape)
    total_id = emit("add", (log_id, shift_id), {}, reduced_shape)

    expected = _node_type(node).shape
    if not keepdim:
        total_id = emit("reshape", (total_id,), {"shape": expected}, expected)
    elif expected != reduced_shape:
        raise ValueError(
            f"logsumexp output shape {expected} does not match the reduced "
            f"{reduced_shape}"
        )
    return tuple(operations), total_id


def _nansum_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.nansum`` as ``masked_fill(isnan(x), 0) -> sum``.

    That is ATen's own definition, so an all-NaN slice sums to zero and an
    integer input skips the mask entirely. A requested ``dtype`` casts the
    operand first, which also matches ATen's accumulation width.
    """

    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("nansum expects an input tensor")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError("nansum requires a ranked tensor")

    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    dtype = args[3] if len(args) > 3 else kwargs.get("dtype")
    if not isinstance(keepdim, bool):
        raise ValueError(f"nansum.keepdim must be a boolean, got {keepdim!r}")

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif (
        isinstance(dims, (tuple, list))
        and dims
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(f"nansum.dim must contain integers, got {dims!r}")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"nansum.dim repeats an axis: {dims!r}")

    output_type = _node_type(node)
    reduced_shape = tuple(
        1 if axis in normalized_dims else size
        for axis, size in enumerate(input_type.shape)
    )
    expected_shape = (
        reduced_shape
        if keepdim
        else tuple(
            size
            for axis, size in enumerate(input_type.shape)
            if axis not in normalized_dims
        )
    )
    if output_type.shape != expected_shape:
        raise ValueError(
            f"nansum output shape {output_type.shape} does not match the "
            f"reduced {expected_shape}"
        )

    target_dtype = input_type.dtype
    if dtype is not None:
        mapped = _TORCH_DTYPES.get(str(dtype))
        if mapped is None:
            raise ValueError(f"unsupported nansum dtype: {dtype}")
        target_dtype = mapped
    if output_type.dtype != target_dtype:
        raise ValueError(
            f"nansum output dtype {output_type.dtype} does not match the "
            f"requested {target_dtype}"
        )

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
        value_dtype: DType,
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, value_dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    if target_dtype != input_type.dtype:
        current = emit(
            "cast",
            (current,),
            {"dtype": target_dtype},
            input_type.shape,
            target_dtype,
        )
    if target_dtype in {DType.F16, DType.BF16, DType.F32}:
        nan_mask = emit(
            "isnan",
            (current,),
            {},
            input_type.shape,
            DType.BOOL,
        )
        current = emit(
            "masked_fill",
            (current, nan_mask),
            {"value": 0.0},
            input_type.shape,
            target_dtype,
        )
    total_id = emit(
        "sum",
        (current,),
        {"dims": normalized_dims, "keepdim": keepdim},
        expected_shape,
        target_dtype,
    )
    return tuple(operations), total_id


def _count_nonzero_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``aten.count_nonzero`` as ``sum(x != 0)`` in I64."""

    input_node, dims, keepdim, reduced_shape = _boolean_reduction_arguments(
        node, "count_nonzero"
    )
    input_type = _node_type(input_node)
    output_type = _node_type(node)
    expected_shape = _reduced_extent(reduced_shape, dims, keepdim)
    if output_type.shape != expected_shape:
        raise ValueError(
            f"count_nonzero output shape {output_type.shape} does not match "
            f"the reduced {expected_shape}"
        )
    if output_type.dtype != DType.I64:
        raise ValueError(f"count_nonzero output dtype {output_type.dtype} must be I64")

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
        value_dtype: DType,
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, value_dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    if input_type.dtype != DType.BOOL:
        current = emit(
            "ne",
            (current,),
            {"scalar": 0, "scalar_position": "rhs"},
            input_type.shape,
            DType.BOOL,
        )
    total_id = emit(
        "sum",
        (current,),
        {"dims": dims, "keepdim": keepdim},
        expected_shape,
        DType.I64,
    )
    return tuple(operations), total_id


def _all_any_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
    *,
    conjunction: bool,
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``all``/``any`` as a counted bool reduction.

    A conjunction is ``sum(x != 0) == reduced_elements`` and a disjunction is
    ``sum(x != 0) != 0``; both match ATen, including a NaN input counting as
    true and an all-NaN slice being NaN-free by construction.
    """

    op = "all" if conjunction else "any"
    input_node, dims, keepdim, reduced_shape = _boolean_reduction_arguments(node, op)
    input_type = _node_type(input_node)
    output_type = _node_type(node)
    expected_shape = _reduced_extent(reduced_shape, dims, keepdim)
    if output_type.shape != expected_shape:
        raise ValueError(
            f"{op} output shape {output_type.shape} does not match the "
            f"reduced {expected_shape}"
        )
    if output_type.dtype != DType.BOOL:
        raise ValueError(f"{op} output dtype {output_type.dtype} must be BOOL")

    operations: list[tuple[Operation, Value]] = []

    def emit(
        op: str,
        inputs: tuple[str, ...],
        attrs: dict[str, object],
        output_shape: tuple[int, ...],
        value_dtype: DType,
    ) -> str:
        value_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(op, inputs, (value_id,), attrs),
                Value(value_id, TensorType(output_shape, value_dtype)),
            )
        )
        return value_id

    current = _argument_ids(input_node, node_ids)
    if input_type.dtype != DType.BOOL:
        current = emit(
            "ne",
            (current,),
            {"scalar": 0, "scalar_position": "rhs"},
            input_type.shape,
            DType.BOOL,
        )
    counted_id = emit(
        "sum",
        (current,),
        {"dims": dims, "keepdim": keepdim},
        expected_shape,
        DType.I64,
    )
    total = prod(input_type.shape[axis] for axis in dims)
    scalar = total if conjunction else 0
    result_id = emit(
        "eq" if conjunction else "ne",
        (counted_id,),
        {"scalar": scalar, "scalar_position": "rhs"},
        expected_shape,
        DType.BOOL,
    )
    return tuple(operations), result_id


def _vector_norm_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError("linalg_vector_norm expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError("linalg_vector_norm requires a ranked tensor")

    ord_value = args[1] if len(args) > 1 else kwargs.get("ord", 2)
    if isinstance(ord_value, bool) or not isinstance(ord_value, (int, float)):
        raise NotImplementedError(
            f"linalg_vector_norm.ord must be a real number, got {ord_value!r}"
        )
    dims = args[2] if len(args) > 2 else kwargs.get("dim")
    keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
    if not isinstance(keepdim, bool):
        raise ValueError(
            f"linalg_vector_norm.keepdim must be a boolean, got {keepdim!r}"
        )

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif (
        isinstance(dims, (tuple, list))
        and dims
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(f"linalg_vector_norm.dim must contain integers, got {dims!r}")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"linalg_vector_norm.dim repeats an axis: {dims!r}")

    return (
        (_argument_ids(input_node, node_ids),),
        {
            "ord": float(ord_value),
            "dims": tuple(sorted(normalized_dims)),
            "keepdim": keepdim,
        },
        _node_type(node),
    )


def _variance_operation(
    node: object,
    node_ids: dict[object, str],
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    """Normalize ``var``/``std`` spellings onto ``dims``/``keepdim``/``correction``."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects an input tensor")
    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    if target.endswith(".correction"):
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        correction = kwargs.get("correction", 1)
        if correction is None:
            correction = 1
        keepdim = kwargs.get("keepdim", False)
    elif target.endswith(".dim"):
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        unbiased = args[2] if len(args) > 2 else kwargs.get("unbiased", True)
        correction = 1 if unbiased else 0
        keepdim = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
    else:
        unbiased = args[1] if len(args) > 1 else kwargs.get("unbiased", True)
        correction = 1 if unbiased else 0
        dims = None
        keepdim = False

    if not isinstance(keepdim, bool):
        raise ValueError(f"{target}.keepdim must be a boolean, got {keepdim!r}")
    if isinstance(correction, bool) or not isinstance(correction, (int, float)):
        raise NotImplementedError(
            f"{target}.correction must be a number, got {correction!r}"
        )
    if isinstance(correction, float) and not isfinite(correction):
        raise ValueError(f"{target}.correction must be finite, got {correction!r}")
    if isinstance(correction, float) and correction.is_integer():
        correction = int(correction)

    if dims is None:
        normalized_dims = tuple(range(rank))
    elif isinstance(dims, int) and not isinstance(dims, bool):
        normalized_dims = (dims % rank,)
    elif (
        isinstance(dims, (tuple, list))
        and dims
        and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims)
    ):
        normalized_dims = tuple(dim % rank for dim in dims)
    else:
        raise ValueError(f"{target}.dim must contain integers, got {dims!r}")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError(f"{target}.dim repeats an axis: {dims!r}")

    return (
        (_argument_ids(input_node, node_ids),),
        {
            "dims": normalized_dims,
            "keepdim": keepdim,
            "correction": correction,
        },
        _node_type(node),
    )


def _extremum_dim_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], dict[int, str]]:
    """Lower ``max(dim)``/``min(dim)`` selections into single-output operations.

    The values are the same reduction ``amax``/``amin`` performs, and the
    indices are the matching ``argmax``/``argmin``; ATen's tie rule (the first
    maximal element) is the one the selection path already follows.
    """

    target = str(getattr(node, "target", ""))
    value_op, index_op = _EXTREMUM_DIM_TARGETS[target]
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects a tensor input")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not -rank <= dim < rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
    if not isinstance(keepdim, bool):
        raise ValueError(f"{target}.keepdim must be a boolean, got {keepdim!r}")
    axis = dim % rank

    wanted: dict[int, object] = {}
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{target} outputs must be selected with getitem")
        user_args = getattr(user, "args", ())
        if len(user_args) != 2 or not isinstance(user_args[1], int):
            raise ValueError(f"malformed {target} getitem: {user!r}")
        index = user_args[1]
        if index not in {0, 1}:
            raise ValueError(f"{target} output index must be 0 or 1, got {index}")
        if user.users:
            wanted[index] = user

    output_shape = (
        input_type.shape[:axis] + (1,) + input_type.shape[axis + 1 :]
        if keepdim
        else input_type.shape[:axis] + input_type.shape[axis + 1 :]
    )
    operations: list[tuple[Operation, Value]] = []
    produced: dict[int, str] = {}

    def emit(name: str, attrs: dict[str, object], index: int, dtype: DType) -> None:
        output_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(
                    name,
                    (_argument_ids(input_node, node_ids),),
                    (output_id,),
                    attrs,
                ),
                Value(output_id, TensorType(output_shape, dtype)),
            )
        )
        produced[index] = output_id

    if 0 in wanted:
        emit(value_op, {"dims": (axis,), "keepdim": keepdim}, 0, input_type.dtype)
    if 1 in wanted:
        emit(index_op, {"dim": axis, "keepdim": keepdim}, 1, DType.I64)
    return tuple(operations), produced


def _sort_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], dict[int, str]]:
    """Lower ``sort``/``topk`` selections into single-output IR operations."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects a tensor input")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    if target == "aten.sort.default":
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        descending = args[2] if len(args) > 2 else kwargs.get("descending", False)
        stable = args[3] if len(args) > 3 else kwargs.get("stable", False)
        if not isinstance(descending, bool):
            raise ValueError(
                f"{target}.descending must be a boolean, got {descending!r}"
            )
        if not isinstance(stable, bool):
            raise ValueError(f"{target}.stable must be a boolean, got {stable!r}")
    else:
        if len(args) < 2:
            raise ValueError(f"{target} expects an input and k")
        dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
        largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
        ordered = args[4] if len(args) > 4 else kwargs.get("sorted", True)
        if not isinstance(largest, bool):
            raise ValueError(f"{target}.largest must be a boolean, got {largest!r}")
        if not isinstance(ordered, bool):
            raise ValueError(f"{target}.sorted must be a boolean, got {ordered!r}")
        k = args[1]
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError(f"{target}.k must be a non-negative integer, got {k!r}")

    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not -rank <= dim < rank:
        raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
    axis = dim % rank
    if target == "aten.topk.default" and k > input_type.shape[axis]:
        raise ValueError(
            f"{target}.k {k} exceeds the size {input_type.shape[axis]} of "
            f"dimension {axis}"
        )

    wanted: dict[int, object] = {}
    for user in getattr(node, "users", {}):
        if getattr(user, "target", None) is not operator.getitem:
            raise ValueError(f"{target} outputs must be selected with getitem")
        user_args = getattr(user, "args", ())
        if len(user_args) != 2 or not isinstance(user_args[1], int):
            raise ValueError(f"malformed {target} getitem: {user!r}")
        index = user_args[1]
        if index not in {0, 1}:
            raise ValueError(f"{target} output index must be 0 or 1, got {index}")
        if user.users:
            wanted[index] = user

    operations: list[tuple[Operation, Value]] = []
    produced: dict[int, str] = {}

    def emit(
        name: str,
        attrs: dict[str, object],
        index: int,
        output_type: TensorType,
    ) -> None:
        output_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(
                    name, (_argument_ids(input_node, node_ids),), (output_id,), attrs
                ),
                Value(output_id, output_type),
            )
        )
        produced[index] = output_id

    if target == "aten.sort.default":
        attrs: dict[str, object] = {"dim": axis, "descending": descending}
        if 0 in wanted:
            emit("sort", dict(attrs), 0, input_type)
        if 1 in wanted:
            emit(
                "argsort",
                dict(attrs),
                1,
                TensorType(input_type.shape, DType.I64),
            )
        return tuple(operations), produced

    k = int(args[1])
    output_shape = input_type.shape[:axis] + (k,) + input_type.shape[axis + 1 :]
    attrs = {"dim": axis, "k": k, "largest": largest}
    if 0 in wanted:
        emit("topk", dict(attrs), 0, TensorType(output_shape, input_type.dtype))
    if 1 in wanted:
        emit(
            "topk_indices",
            dict(attrs),
            1,
            TensorType(output_shape, DType.I64),
        )
    return tuple(operations), produced


def _median_operations(
    node: object,
    node_ids: dict[object, str],
    first_op_index: int,
) -> tuple[tuple[tuple[Operation, Value], ...], dict[int, str]]:
    """Lower the median family into ``median`` and ``median_indices`` ops.

    ``median.dim`` returns ``(values, indices)`` while ``median.default`` reduces
    every axis and returns only the value; both are one operator with the
    reduced axes named, and a selection nothing reads is left out.
    """

    target = str(getattr(node, "target", ""))
    ignore_nan = _MEDIAN_TARGETS[target]
    args = getattr(node, "args", ())
    kwargs = getattr(node, "kwargs", {})
    if not args:
        raise ValueError(f"{target} expects a tensor input")

    input_node = args[0]
    input_type = _node_type(input_node)
    rank = len(input_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a ranked tensor")

    dims_provided = target.endswith(".dim")
    if dims_provided:
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
        if not -rank <= dim < rank:
            raise ValueError(f"{target}.dim {dim} exceeds rank {rank}")
        if not isinstance(keepdim, bool):
            raise ValueError(f"{target}.keepdim must be a boolean, got {keepdim!r}")
        dims = (dim % rank,)
    else:
        # ``median(x)`` reduces everything, and a scalar result cannot keep dims.
        dims = tuple(range(rank))
        keepdim = False

    wanted: dict[int, object] = {}
    if dims_provided:
        for user in getattr(node, "users", {}):
            if getattr(user, "target", None) is not operator.getitem:
                raise ValueError(f"{target} outputs must be selected with getitem")
            user_args = getattr(user, "args", ())
            if len(user_args) != 2 or not isinstance(user_args[1], int):
                raise ValueError(f"malformed {target} getitem: {user!r}")
            index = user_args[1]
            if index not in {0, 1}:
                raise ValueError(f"{target} output index must be 0 or 1, got {index}")
            if user.users:
                wanted[index] = user
    if not wanted:
        # A full reduction has a single result, so its only reader is the node's
        # own consumer and the value output is exactly what it wants.
        wanted[0] = node

    output_shape = tuple(
        size for axis, size in enumerate(input_type.shape) if axis not in dims
    )
    if keepdim:
        output_shape = tuple(
            1 if axis in dims else size for axis, size in enumerate(input_type.shape)
        )

    operations: list[tuple[Operation, Value]] = []
    produced: dict[int, str] = {}
    attrs: dict[str, object] = {
        "dims": dims,
        "keepdim": keepdim,
        "ignore_nan": ignore_nan,
    }

    def emit(name: str, index: int, dtype: DType) -> None:
        output_id = f"v{first_op_index + len(operations)}"
        operations.append(
            (
                Operation(
                    name, (_argument_ids(input_node, node_ids),), (output_id,), attrs
                ),
                Value(output_id, TensorType(output_shape, dtype)),
            )
        )
        produced[index] = output_id

    if 0 in wanted:
        emit("median", 0, input_type.dtype)
    if 1 in wanted:
        emit("median_indices", 1, DType.I64)
    return tuple(operations), produced


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _lower_bundle(
    lowering: Lowering, node: Any, builder: Callable[..., Any], **kwargs: Any
) -> bool:
    """Emit a builder's operations and bind the node to its single output.

    Every reduction family here has the same contract - "lower this node into
    whatever operations it takes, and tell me the id of the result" - so the
    glue is written once.
    """

    produced, output_id = builder(
        node, lowering.node_ids, lowering.next_index(), **kwargs
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


def _bind_selected_users(
    lowering: Lowering, node: Any, produced: dict[int, str]
) -> None:
    """Bind each ``getitem`` consumer of a multi-result node to its result.

    ``sort``, ``topk`` and the extremum dim form return values together with
    indices; the graph reads them through ``getitem`` nodes. A selection nothing
    reads needs no value at all, so it is marked dead rather than emitted.
    """

    for user in node.users:
        index = user.args[1]
        if index in produced:
            lowering.bind(user, produced[index])
        else:
            lowering.dead_nodes.add(user)


def _lower_extremum_dim(lowering: Lowering, node: Any, target: str) -> bool:
    produced, ids = _extremum_dim_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    _bind_selected_users(lowering, node, ids)
    return True


def _lower_sort(lowering: Lowering, node: Any, target: str) -> bool:
    produced, ids = _sort_operations(node, lowering.node_ids, lowering.next_index())
    lowering.emit_all(produced)
    _bind_selected_users(lowering, node, ids)
    return True


def _lower_median(lowering: Lowering, node: Any, target: str) -> bool:
    produced, ids = _median_operations(node, lowering.node_ids, lowering.next_index())
    lowering.emit_all(produced)
    if str(getattr(node, "target", "")).endswith(".default"):
        # The full reduction has one result and no ``getitem`` in between.
        lowering.bind(node, ids[0])
        return True
    _bind_selected_users(lowering, node, ids)
    return True


def _lower_all_any(lowering: Lowering, node: Any, target: str) -> bool:
    return _lower_bundle(
        lowering, node, _all_any_operations, conjunction=target.startswith("aten.all.")
    )


def _lower_vector_norm(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _vector_norm_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("vector_norm", inputs, attrs, output_type))
    return True


def _lower_variance(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _variance_operation(node, lowering.node_ids)
    op_name = _VARIANCE_TARGETS[target]
    lowering.bind(node, lowering.emit(op_name, inputs, attrs, output_type))
    return True


def _lower_logsumexp(lowering: Lowering, node: Any, target: str) -> bool:
    return _lower_bundle(lowering, node, _logsumexp_operations)


def _lower_nansum(lowering: Lowering, node: Any, target: str) -> bool:
    return _lower_bundle(lowering, node, _nansum_operations)


def _lower_count_nonzero(lowering: Lowering, node: Any, target: str) -> bool:
    return _lower_bundle(lowering, node, _count_nonzero_operations)


HANDLERS: dict[str, tuple[Handler, ...]] = {
    **{target: (_lower_extremum_dim,) for target in _EXTREMUM_DIM_TARGETS},
    **{target: (_lower_sort,) for target in _SORT_TARGETS},
    **{target: (_lower_median,) for target in _MEDIAN_TARGETS},
    **{target: (_lower_all_any,) for target in _AND_OR_TARGETS},
    **{target: (_lower_variance,) for target in _VARIANCE_TARGETS},
    **{target: (_lower_count_nonzero,) for target in _COUNT_NONZERO_TARGETS},
    _LOGSUMEXP_TARGET: (_lower_logsumexp,),
    _NANSUM_TARGET: (_lower_nansum,),
    _VECTOR_NORM_TARGET: (_lower_vector_norm,),
}
