"""Gather, scatter and embedding: the operators whose lowering has to read an
index tensor.

``scatter`` has the most variants - plain, ``add``, and the ``reduce`` form with
its reduction name - so the canonicalisation that maps an in-place target onto
its out-of-place equivalent lives here beside the builder that consumes it."""

from __future__ import annotations

from math import isfinite
from typing import Any
from ..ir import DType, Operation, TensorType, Value
from ..shapes import broadcast_shapes, can_broadcast
from .fx import _node_type
from .state import Lowering
from .state import Handler
from .refs import _argument_ids
from .targets import (
    _HISTC_TARGET,
    _INDEX_ADD_TARGET,
    _INDEX_PUT_TARGETS,
    _SEARCHSORTED_TARGETS,
    _INPLACE_SCATTER_TARGETS,
    _SCATTER_ADD_TARGET,
    _SCATTER_REDUCE_NAMES,
    _SCATTER_REDUCE_TARGET,
    _SCATTER_TARGETS,
)


def _canonical_scatter_target(target: str) -> str:
    """Map an in-place scatter spelling onto its functional ATen target."""

    return _INPLACE_SCATTER_TARGETS.get(target, target)


def _validate_inplace_legalization(node: object) -> None:
    args = getattr(node, "args", ())
    if not args:
        raise ValueError("in-place operation has no input")
    source = args[0]
    users = set(getattr(source, "users", ()))
    if users != {node}:
        raise NotImplementedError(
            "in-place operation has aliasing observers and cannot be lowered "
            f"safely: {getattr(node, 'target', '<unknown>')}"
        )


def _embedding_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    """Validate an embedding gather while ignoring inference-only gradient flags."""

    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    parameter_names = (
        "weight",
        "indices",
        "padding_idx",
        "scale_grad_by_freq",
        "sparse",
    )
    if len(args) < 2 or len(args) > len(parameter_names):
        raise ValueError(f"embedding expects two to five arguments, got {len(args)}")
    for position, name in enumerate(parameter_names[: len(args)]):
        if name in kwargs:
            raise ValueError(f"embedding received duplicate {name} argument")
        kwargs[name] = args[position]

    unknown = set(kwargs) - set(parameter_names)
    if unknown:
        raise ValueError(
            f"embedding received unsupported keyword arguments: {sorted(unknown)}"
        )
    missing = set(parameter_names[:2]) - set(kwargs)
    if missing:
        raise ValueError(f"embedding is missing arguments: {sorted(missing)}")

    padding_idx = kwargs.get("padding_idx", -1)
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq", False)
    sparse = kwargs.get("sparse", False)
    if padding_idx is not None and (
        not isinstance(padding_idx, int) or isinstance(padding_idx, bool)
    ):
        raise ValueError(
            f"embedding.padding_idx must be an integer, got {padding_idx!r}"
        )
    if not isinstance(scale_grad_by_freq, bool):
        raise ValueError(
            "embedding.scale_grad_by_freq must be a boolean, "
            f"got {scale_grad_by_freq!r}"
        )
    if not isinstance(sparse, bool):
        raise ValueError(f"embedding.sparse must be a boolean, got {sparse!r}")

    weight_id = _argument_ids(kwargs["weight"], node_ids)
    indices_id = _argument_ids(kwargs["indices"], node_ids)
    weight_type = _node_type(kwargs["weight"])
    indices_type = _node_type(kwargs["indices"])
    if len(weight_type.shape) != 2:
        raise NotImplementedError(
            f"embedding weight must have rank 2, got {weight_type.shape}"
        )
    if weight_type.dtype not in {DType.F16, DType.BF16, DType.F32}:
        raise ValueError(
            f"embedding weight must be floating point, got {weight_type.dtype.value}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            f"embedding indices must be int32 or int64, got {indices_type.dtype.value}"
        )
    if padding_idx is not None and not (
        -weight_type.shape[0] <= padding_idx < weight_type.shape[0]
    ):
        raise ValueError(
            f"embedding.padding_idx {padding_idx} is outside the weight table"
        )

    output_type = _node_type(node)
    expected_shape = indices_type.shape + (weight_type.shape[1],)
    if output_type.shape != expected_shape or output_type.dtype != weight_type.dtype:
        raise ValueError(
            f"embedding output metadata {(output_type.shape, output_type.dtype)} does "
            f"not match expected {(expected_shape, weight_type.dtype)}"
        )
    attrs = {
        "padding_idx": padding_idx,
        "scale_grad_by_freq": scale_grad_by_freq,
        "sparse": sparse,
    }
    return (weight_id, indices_id), attrs, output_type


def _index_select_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            f"index_select received unsupported keyword arguments: {sorted(kwargs)}"
        )
    if len(args) != 3:
        raise ValueError(f"index_select expects three arguments, got {len(args)}")

    source, dim, indices = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_select.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("index_select requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"index_select.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != 1:
        raise ValueError(
            f"index_select indices must have rank 1, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            "index_select indices must be int32 or int64, "
            f"got {indices_type.dtype.value}"
        )

    output_type = _node_type(node)
    expected_shape = (
        source_type.shape[:axis]
        + (indices_type.shape[0],)
        + source_type.shape[axis + 1 :]
    )
    if output_type != TensorType(expected_shape, source_type.dtype):
        raise ValueError(
            f"index_select output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(expected_shape, source_type.dtype)}"
        )
    return (
        (_argument_ids(source, node_ids), _argument_ids(indices, node_ids)),
        {"dim": axis},
        output_type,
    )


def _gather_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, str], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    sparse_grad = kwargs.pop("sparse_grad", False)
    if kwargs:
        raise ValueError(
            f"gather received unsupported keyword arguments: {sorted(kwargs)}"
        )
    if not isinstance(sparse_grad, bool):
        raise ValueError(f"gather.sparse_grad must be a boolean, got {sparse_grad!r}")
    if len(args) != 3:
        raise ValueError(f"gather expects three arguments, got {len(args)}")

    source, dim, indices = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"gather.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("gather requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"gather.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"gather indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            f"gather indices must be int32 or int64, got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != axis and index_dim > source_dim:
            raise ValueError(
                f"gather indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-gather axis {index_axis}"
            )

    output_type = _node_type(node)
    if output_type != TensorType(indices_type.shape, source_type.dtype):
        raise ValueError(
            f"gather output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match expected {(indices_type.shape, source_type.dtype)}"
        )
    return (
        (_argument_ids(source, node_ids), _argument_ids(indices, node_ids)),
        {"dim": axis},
        output_type,
    )


def _scatter_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    target = _canonical_scatter_target(str(getattr(node, "target", "")))
    reduce_name: object = None
    if target == _SCATTER_REDUCE_TARGET:
        # `scatter_reduce` carries its reduction as a positional argument.
        if len(args) != 5:
            raise ValueError(f"scatter_reduce expects five arguments, got {len(args)}")
        source, dim, indices, update, reduce_name = args
    else:
        if len(args) != 4:
            raise ValueError(f"scatter expects four arguments, got {len(args)}")
        source, dim, indices, update = args
    reduce, include_self = _scatter_reduce_attrs(target, reduce_name, kwargs)
    if reduce is None and kwargs:
        raise ValueError(
            f"scatter received unsupported keyword arguments: {sorted(kwargs)}"
        )

    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"scatter.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("scatter requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"scatter.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != rank:
        raise ValueError(
            f"scatter indices must have rank {rank}, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            f"scatter indices must be int32 or int64, got {indices_type.dtype.value}"
        )
    for index_axis, (index_dim, source_dim) in enumerate(
        zip(indices_type.shape, source_type.shape, strict=True)
    ):
        if index_axis != axis and index_dim > source_dim:
            raise ValueError(
                f"scatter indices shape {indices_type.shape} exceeds source shape "
                f"{source_type.shape} on non-scatter axis {index_axis}"
            )

    attrs: dict[str, object] = {"dim": axis}
    if target == "aten.scatter.value":
        if not isinstance(update, (bool, int, float)):
            raise ValueError(f"scatter.value must be a scalar number, got {update!r}")
        attrs["value"] = update
        inputs = (_argument_ids(source, node_ids), _argument_ids(indices, node_ids))
    else:
        update_type = _node_type(update)
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
        inputs = (
            _argument_ids(source, node_ids),
            _argument_ids(indices, node_ids),
            _argument_ids(update, node_ids),
        )
        if reduce is not None:
            attrs["reduce"] = reduce
            attrs["include_self"] = include_self

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"scatter output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return inputs, attrs, output_type


def _scatter_reduce_attrs(
    target: str, reduce_name: object, kwargs: dict[str, object]
) -> tuple[str | None, bool]:
    """Read the reduction of ``scatter_reduce``/``scatter_add``.

    Returns ``(reduce, include_self)``, with ``reduce`` ``None`` for the plain
    overwrite forms.
    """

    if target == "aten.scatter_add.default":
        if kwargs:
            raise ValueError(
                f"scatter_add received unsupported keyword arguments: {sorted(kwargs)}"
            )
        return "add", True
    if target != _SCATTER_REDUCE_TARGET:
        return None, True
    unknown = sorted(set(kwargs) - {"reduce", "include_self"})
    if unknown:
        raise ValueError(
            f"scatter_reduce received unsupported keyword arguments: {unknown}"
        )
    if reduce_name is None:
        reduce_name = kwargs.get("reduce", "sum")
    if reduce_name not in _SCATTER_REDUCE_NAMES:
        raise ValueError(
            f"scatter_reduce.reduce {reduce_name!r} is not supported; expected "
            f"one of {sorted(_SCATTER_REDUCE_NAMES)}"
        )
    include_self = kwargs.get("include_self", True)
    if not isinstance(include_self, bool):
        raise ValueError(
            f"scatter_reduce.include_self must be a boolean, got {include_self!r}"
        )
    return _SCATTER_REDUCE_NAMES[str(reduce_name)], include_self


def _index_add_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    unknown = sorted(set(kwargs) - {"alpha"})
    if unknown:
        raise ValueError(f"index_add received unsupported keyword arguments: {unknown}")
    if len(args) != 4:
        raise ValueError(f"index_add expects four arguments, got {len(args)}")

    source, dim, indices, update = args
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"index_add.dim must be an integer, got {dim!r}")
    source_type = _node_type(source)
    indices_type = _node_type(indices)
    update_type = _node_type(update)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError("index_add requires a tensor with rank >= 1")
    if dim < -rank or dim >= rank:
        raise ValueError(f"index_add.dim {dim} is outside rank {rank}")
    axis = dim + rank if dim < 0 else dim
    if len(indices_type.shape) != 1:
        raise NotImplementedError(
            f"index_add requires a 1-D index tensor, got {indices_type.shape}"
        )
    if indices_type.dtype not in {DType.I32, DType.I64}:
        raise ValueError(
            f"index_add indices must be int32 or int64, got {indices_type.dtype.value}"
        )
    if update_type.dtype != source_type.dtype:
        raise ValueError(
            "index_add source dtype must match the input dtype, got "
            f"{update_type.dtype.value} and {source_type.dtype.value}"
        )
    expected_update = (
        source_type.shape[:axis]
        + (indices_type.shape[0],)
        + source_type.shape[axis + 1 :]
    )
    if update_type.shape != expected_update:
        raise ValueError(
            f"index_add source shape {update_type.shape} does not match "
            f"{expected_update}"
        )
    alpha = kwargs.get("alpha", 1)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"index_add.alpha must be a scalar number, got {alpha!r}")
    if not isfinite(float(alpha)):
        raise ValueError(f"index_add.alpha must be finite, got {alpha!r}")

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"index_add output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return (
        (
            _argument_ids(source, node_ids),
            _argument_ids(indices, node_ids),
            _argument_ids(update, node_ids),
        ),
        {"dim": axis, "alpha": float(alpha)},
        output_type,
    )


def _index_put_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    """Lower the write-through-index family into one ``index_put`` operation."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if target.startswith("aten.index_put"):
        unknown = sorted(set(kwargs) - {"accumulate"})
        if unknown:
            raise ValueError(
                f"index_put received unsupported keyword arguments: {unknown}"
            )
        if len(args) not in {3, 4}:
            raise ValueError(
                f"index_put expects three or four arguments, got {len(args)}"
            )
        source, index_list, values = args[:3]
        accumulate = args[3] if len(args) == 4 else kwargs.get("accumulate", False)
        if not isinstance(index_list, (tuple, list)) or not index_list:
            raise ValueError("index_put expects a non-empty index list")
        axes = tuple(
            position for position, index in enumerate(index_list) if index is not None
        )
        indices = [index for index in index_list if index is not None]
        filled = False
    elif target.startswith("aten.index_copy"):
        if kwargs:
            raise ValueError(
                f"index_copy received unsupported keyword arguments: {sorted(kwargs)}"
            )
        if len(args) != 4:
            raise ValueError(f"index_copy expects four arguments, got {len(args)}")
        source, dim, indices, values = args
        axes = (_positive_axis(dim, _node_type(source), target),)
        indices = [indices]
        filled = False
        accumulate = False
    else:
        if kwargs:
            raise ValueError(
                f"index_fill received unsupported keyword arguments: {sorted(kwargs)}"
            )
        if len(args) != 4:
            raise ValueError(f"index_fill expects four arguments, got {len(args)}")
        source, dim, indices, value = args
        axes = (_positive_axis(dim, _node_type(source), target),)
        indices = [indices]
        filled = True
        accumulate = False
        if not isinstance(value, (bool, int, float)):
            raise NotImplementedError(
                f"{target} expects a scalar fill value, got {value!r}"
            )

    if not isinstance(accumulate, bool):
        raise ValueError(f"{target} accumulate must be a boolean, got {accumulate!r}")

    source_type = _node_type(source)
    rank = len(source_type.shape)
    if rank == 0:
        raise NotImplementedError(f"{target} requires a tensor with rank >= 1")
    first_axis = axes[0]
    last_axis = axes[-1]
    if axes != tuple(range(first_axis, last_axis + 1)):
        raise NotImplementedError(
            f"non-contiguous {target} is not supported: indexed axes {axes}"
        )
    index_types = [_node_type(index) for index in indices]
    for index_type in index_types:
        if index_type.dtype not in {DType.I32, DType.I64}:
            raise ValueError(
                f"{target} indices must be int32 or int64, got {index_type.dtype.value}"
            )

    selection = (
        source_type.shape[:first_axis]
        + broadcast_shapes(tuple(index.shape for index in index_types))
        + source_type.shape[last_axis + 1 :]
    )
    inputs = [_argument_ids(source, node_ids)]
    inputs.extend(_argument_ids(index, node_ids) for index in indices)
    attrs: dict[str, object] = {"axes": axes}
    if filled:
        attrs["value"] = value
    else:
        values_type = _node_type(values)
        if values_type.dtype != source_type.dtype:
            raise ValueError(
                f"{target} values dtype must match the input dtype, got "
                f"{values_type.dtype.value} and {source_type.dtype.value}"
            )
        if not can_broadcast(values_type.shape, selection):
            raise ValueError(
                f"{target} values shape {values_type.shape} does not broadcast "
                f"to the selection {selection}"
            )
        if accumulate:
            attrs["accumulate"] = True
        inputs.append(_argument_ids(values, node_ids))

    output_type = _node_type(node)
    if output_type != source_type:
        raise ValueError(
            f"{target} output metadata {(output_type.shape, output_type.dtype)} "
            f"does not match input {(source_type.shape, source_type.dtype)}"
        )
    return tuple(inputs), attrs, output_type


def _positive_axis(dim: object, source_type: TensorType, target: str) -> int:
    """The checked ``dim`` of a single-axis write."""

    rank = len(source_type.shape)
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise ValueError(f"{target}.dim must be an integer, got {dim!r}")
    if not -rank <= dim < rank:
        raise ValueError(f"{target}.dim {dim} is outside rank {rank}")
    return dim % rank


def _searchsorted_operations(
    node: object, node_ids: dict[object, str], first_op_index: int
) -> tuple[tuple[tuple[Operation, Value], ...], str]:
    """Lower ``searchsorted``/``bucketize`` into one placement search.

    ``bucketize`` puts the values first and the boundaries second, so the
    operand order is the only difference between the two spellings. A scalar
    value becomes a constant so the search has a single shape rule.
    """

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    unknown = sorted(set(kwargs) - {"right", "side"})
    if unknown:
        raise ValueError(f"{target} received unsupported keyword arguments: {unknown}")
    if len(args) != 2:
        raise ValueError(f"{target} expects two arguments, got {len(args)}")
    right = kwargs.get("right")
    if "side" in kwargs:
        side = kwargs["side"]
        if side not in {"left", "right"}:
            raise ValueError(f"{target}.side must be left or right, got {side!r}")
        right = side == "right"
    if right is None:
        right = False
    if not isinstance(right, bool):
        raise ValueError(f"{target}.right must be a boolean, got {right!r}")

    if target.startswith("aten.bucketize"):
        values, boundaries = args
    else:
        boundaries, values = args

    operations: list[tuple[Operation, Value]] = []
    boundary_type = _node_type(boundaries)
    if isinstance(values, (bool, int, float)):
        scalar_id = f"v{first_op_index}"
        operations.append(
            (
                Operation("scalar_tensor", (), (scalar_id,), {"value": values}),
                Value(scalar_id, TensorType((), boundary_type.dtype)),
            )
        )
        values_id = scalar_id
        values_shape: tuple[int, ...] = ()
    else:
        values_type = _node_type(values)
        if values_type.dtype != boundary_type.dtype:
            raise ValueError(
                f"{target} values dtype must match the boundaries, got "
                f"{values_type.dtype.value} and {boundary_type.dtype.value}"
            )
        values_id = _argument_ids(values, node_ids)
        values_shape = values_type.shape

    output_id = f"v{first_op_index + len(operations)}"
    operations.append(
        (
            Operation(
                "searchsorted",
                (_argument_ids(boundaries, node_ids), values_id),
                (output_id,),
                {"right": right},
            ),
            Value(output_id, TensorType(values_shape, DType.I64)),
        )
    )
    return tuple(operations), output_id


def _histc_operation(
    node: object, node_ids: dict[object, str]
) -> tuple[tuple[str, ...], dict[str, object], TensorType]:
    """Read ``histc``'s bin count and range, which are attributes on ATen."""

    target = str(getattr(node, "target", ""))
    args = getattr(node, "args", ())
    kwargs = dict(getattr(node, "kwargs", {}))
    if kwargs:
        raise ValueError(
            f"{target} received unsupported keyword arguments: {sorted(kwargs)}"
        )
    if not 1 <= len(args) <= 4:
        raise ValueError(f"{target} expects one to four arguments, got {len(args)}")
    values = args[0]
    bins = args[1] if len(args) > 1 else 100
    low = args[2] if len(args) > 2 else 0
    high = args[3] if len(args) > 3 else 0
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError(f"{target}.bins must be an integer, got {bins!r}")
    for name, value in (("min", low), ("max", high)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{target}.{name} must be a scalar number, got {value!r}")
        if not isfinite(float(value)):
            raise ValueError(f"{target}.{name} must be finite, got {value!r}")
    if bins <= 0:
        raise ValueError(f"{target}.bins must be positive, got {bins}")
    if low > high:
        raise ValueError(f"{target}.min {low} exceeds {target}.max {high}")

    values_type = _node_type(values)
    output_type = TensorType((bins,), values_type.dtype)
    return (
        (_argument_ids(values, node_ids),),
        {"bins": bins, "min": float(low), "max": float(high)},
        output_type,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _lower_embedding(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _embedding_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("embedding", inputs, attrs, output_type))
    return True


def _lower_index_select(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _index_select_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("index_select", inputs, attrs, output_type))
    return True


def _lower_gather(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _gather_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("gather", inputs, attrs, output_type))
    return True


def _lower_scatter(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _scatter_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("scatter", inputs, attrs, output_type))
    return True


def _lower_index_add(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _index_add_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("index_add", inputs, attrs, output_type))
    return True


def _lower_index_put(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _index_put_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("index_put", inputs, attrs, output_type))
    return True


def _lower_searchsorted(lowering: Lowering, node: Any, target: str) -> bool:
    produced, output_id = _searchsorted_operations(
        node, lowering.node_ids, lowering.next_index()
    )
    lowering.emit_all(produced)
    lowering.bind(node, output_id)
    return True


def _lower_histc(lowering: Lowering, node: Any, target: str) -> bool:
    inputs, attrs, output_type = _histc_operation(node, lowering.node_ids)
    lowering.bind(node, lowering.emit("histc", inputs, attrs, output_type))
    return True


def _checked_in_place(handler: Handler) -> Handler:
    """Guard an in-place target with the mutation rule.

    The compiler models values, not mutation, so an in-place write is only
    legalizable when its destination has no other reader. That check belongs to
    the spelling, not to the lowering: the functional target the node becomes is
    the same either way.
    """

    def guarded(lowering: Lowering, node: Any, target: str) -> bool:
        _validate_inplace_legalization(node)
        return handler(lowering, node, target)

    return guarded


_BY_CANONICAL: dict[str, tuple[Handler, ...]] = {
    **{target: (_lower_scatter,) for target in _SCATTER_TARGETS},
    _SCATTER_ADD_TARGET: (_lower_scatter,),
    _SCATTER_REDUCE_TARGET: (_lower_scatter,),
    _INDEX_ADD_TARGET: (_lower_index_add,),
    **{
        target: (_lower_index_put,)
        for target, in_place in _INDEX_PUT_TARGETS.items()
        if not in_place
    },
    **{target: (_lower_searchsorted,) for target in _SEARCHSORTED_TARGETS},
    _HISTC_TARGET: (_lower_histc,),
}

HANDLERS: dict[str, tuple[Handler, ...]] = {
    "aten.embedding.default": (_lower_embedding,),
    "aten.index_select.default": (_lower_index_select,),
    "aten.gather.default": (_lower_gather,),
    **_BY_CANONICAL,
    # An in-place spelling dispatches to the same lowering as the functional
    # target it canonicalises to, behind the mutation check.
    **{
        in_place: tuple(
            _checked_in_place(handler) for handler in _BY_CANONICAL[canonical]
        )
        for in_place, canonical in _INPLACE_SCATTER_TARGETS.items()
    },
    **{
        target: (_checked_in_place(_lower_index_put),)
        for target, in_place in _INDEX_PUT_TARGETS.items()
        if in_place
    },
}
