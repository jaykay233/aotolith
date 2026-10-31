"""Indexing: gather, scatter and the layout-free selections.

Every operator here reads an integer index tensor and therefore has to verify
the same three things - that the index dtype is integral, that its values fit
inside the axis it addresses, and that the index and the values agree on the
shape they write. Those checks are inter-operator, so they live in hooks.
"""

from __future__ import annotations

from .. import DType, Graph, Operation, TensorType
from .rules import (
    AttrKind,
    AttrSpec,
    DtypeRule,
    FLOATING_DTYPES,
    OpContract,
    SCATTER_REDUCES,
    ShapeRule,
)
from .shared import (
    _index_tensor,
    _only_attrs,
    _positive_dim,
    _require_output,
    _unsupported_attrs,
)
from ...shapes import broadcast_shapes, can_broadcast


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


def _check_index_put(graph: Graph, op: Operation) -> None:
    """``index_put`` writes through the same index list ``index`` reads."""

    source_type = graph.value(op.inputs[0]).type
    axes_value = op.attrs["axes"]
    filled = "value" in op.attrs
    if filled:
        value = op.attrs["value"]
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"index_put.value must be a scalar number, got {value!r}")
    if len(op.inputs) != len(axes_value) + (1 if filled else 2):
        raise ValueError(
            f"index_put has {len(op.inputs) - 1} index operands for axes "
            f"{axes_value} with {'a' if filled else 'no'} fill value"
        )
    first_axis = axes_value[0]
    last_axis = axes_value[-1]
    if axes_value != tuple(range(first_axis, last_axis + 1)):
        raise NotImplementedError(
            f"non-contiguous index_put is not supported: indexed axes {axes_value}"
        )
    if first_axis < 0 or last_axis >= len(source_type.shape):
        raise ValueError(
            f"index_put axes {axes_value} exceed source rank {len(source_type.shape)}"
        )
    index_values = op.inputs[1:-1] if not filled else op.inputs[1:]
    index_shapes = tuple(
        _index_tensor(graph, value_id, "index_put indices").shape
        for value_id in index_values
    )
    selection = (
        source_type.shape[:first_axis]
        + broadcast_shapes(index_shapes)
        + source_type.shape[last_axis + 1 :]
    )
    if not filled:
        # The update can be a scalar or omit the leading axes, so it only has
        # to broadcast to the region being written.
        values_type = graph.value(op.inputs[-1]).type
        if values_type.dtype != source_type.dtype:
            raise ValueError(
                "index_put values dtype must match the input dtype, got "
                f"{values_type.dtype.value} and {source_type.dtype.value}"
            )
        if not can_broadcast(values_type.shape, selection):
            raise ValueError(
                f"index_put values shape {values_type.shape} does not broadcast "
                f"to the selection {selection}"
            )
    accumulate = op.attrs.get("accumulate", False)
    if not isinstance(accumulate, bool):
        raise ValueError(f"index_put.accumulate must be a boolean, got {accumulate!r}")
    _require_output(graph, op, source_type, "output metadata")
    _unsupported_attrs(op, {"axes", "accumulate", "value"})


def _check_searchsorted(graph: Graph, op: Operation) -> None:
    """``searchsorted`` reads a sorted boundary list and places each value in it."""

    sorted_type = graph.value(op.inputs[0]).type
    values_type = graph.value(op.inputs[1]).type
    if len(sorted_type.shape) != 1:
        # ATen grows the boundaries to a leading batch, which needs the batch
        # dims of the values to line up; the 1-D list is the common case.
        raise NotImplementedError(
            f"searchsorted needs a 1-D boundary list, got {sorted_type.shape}"
        )
    if sorted_type.dtype != values_type.dtype:
        raise ValueError(
            "searchsorted boundaries and values must share a dtype, got "
            f"{sorted_type.dtype.value} and {values_type.dtype.value}"
        )
    _require_output(
        graph, op, TensorType(values_type.shape, DType.I64), "output metadata"
    )
    _unsupported_attrs(op, {"right"})


def _check_histc(graph: Graph, op: Operation) -> None:
    """``histc`` counts into a fixed number of bins over ``[min, max]``."""

    input_type = graph.value(op.inputs[0]).type
    bins = op.attrs["bins"]
    if bins <= 0:
        raise ValueError(f"histc.bins must be positive, got {bins}")
    low = op.attrs["min"]
    high = op.attrs["max"]
    if low > high:
        raise ValueError(f"histc.min {low} exceeds histc.max {high}")
    _require_output(graph, op, TensorType((bins,), input_type.dtype), "output metadata")
    _unsupported_attrs(op, {"bins", "min", "max"})


CONTRACTS: dict[str, OpContract] = {
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
    "index_put": OpContract(
        arity=(2, 10),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("axes", AttrKind.AXES),),
        extra=_check_index_put,
    ),
    "searchsorted": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("right", AttrKind.BOOL),),
        extra=_check_searchsorted,
    ),
    "histc": OpContract(
        arity=(1, 1),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("bins", AttrKind.NONNEGATIVE_INT),
            AttrSpec("min", AttrKind.NUMBER),
            AttrSpec("max", AttrKind.NUMBER),
        ),
        extra=_check_histc,
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
}
