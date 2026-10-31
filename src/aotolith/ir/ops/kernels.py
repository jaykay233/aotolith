"""Operators the backend executes as one kernel.

Matmul, the fused normalizations, attention and the spatial kernels are the
operations the backend does not express as elementwise expressions: matmul and
the normalizations own hand-written Metal source, while convolution and the
pooling windows are handed to MLX's own kernels. Their contracts are therefore
mostly attribute declarations: the shape and dtype promises are what the
implementation relies on, and the hooks check the constraints it assumes (a head
dimension width, a rope pair layout, an attention mask rank, a window that fits).
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


def _spatial_integers(
    op: Operation, name: str, spatial: int, *, floor: int
) -> tuple[int, ...]:
    """One per-axis attribute of a spatial operator, checked for range."""

    value = op.attrs.get(name)
    if (
        not isinstance(value, tuple)
        or len(value) != spatial
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    ):
        raise ValueError(f"{op.op}.{name} must be {spatial} integers, got {value!r}")
    if any(item < floor for item in value):
        raise ValueError(f"{op.op}.{name} must be at least {floor}, got {value!r}")
    return value


def _pool_mode(op: Operation) -> str:
    mode = op.attrs.get("mode")
    if mode not in {"max", "avg"}:
        raise ValueError(f"{op.op}.mode must be 'max' or 'avg', got {mode!r}")
    return mode


def _check_pool_outputs(graph: Graph, op: Operation, shape: tuple[int, ...]) -> None:
    """The values, and the flattened indices a maximum may also report."""

    for position, value_id in enumerate(op.outputs):
        value_type = graph.value(value_id).type
        if value_type.shape != shape:
            raise ValueError(
                f"{op.op} output {position} shape {value_type.shape} != {shape}"
            )
        expected_dtype = DType.I64 if position else graph.value(op.inputs[0]).type.dtype
        if value_type.dtype != expected_dtype:
            raise ValueError(
                f"{op.op} output {position} must be {expected_dtype.value}, got "
                f"{value_type.dtype.value}"
            )


def _check_resize(graph: Graph, op: Operation) -> None:
    input_type = graph.value(op.inputs[0]).type
    spatial = len(input_type.shape) - 2
    mode = op.attrs.get("mode")
    if mode not in {"nearest", "linear", "bicubic"}:
        raise ValueError(
            f"{op.op}.mode must be nearest, linear or bicubic, got {mode!r}"
        )
    if mode == "bicubic" and spatial != 2:
        raise ValueError("resize.bicubic is only defined over two spatial axes")
    output_size = _spatial_integers(op, "output_size", spatial, floor=1)
    scales = op.attrs.get("source_scale")
    if (
        not isinstance(scales, tuple)
        or len(scales) != spatial
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and float(item) > 0
            for item in scales
        )
    ):
        raise ValueError(
            f"{op.op}.source_scale must be {spatial} positive numbers, got {scales!r}"
        )
    if not isinstance(op.attrs.get("align_corners", False), bool):
        raise ValueError(f"{op.op}.align_corners must be a boolean")
    expected = (*input_type.shape[:2], *output_size)
    actual = graph.value(op.outputs[0]).type.shape
    if actual != expected:
        raise ValueError(f"{op.op} output shape {actual} != {expected}")


def _check_pool(graph: Graph, op: Operation) -> None:
    input_type = graph.value(op.inputs[0]).type
    spatial = len(input_type.shape) - 2
    mode = _pool_mode(op)
    kernel = _spatial_integers(op, "kernel", spatial, floor=1)
    stride = _spatial_integers(op, "stride", spatial, floor=1)
    padding = _spatial_integers(op, "padding", spatial, floor=0)
    dilation = _spatial_integers(op, "dilation", spatial, floor=1)
    ceil_mode = op.attrs.get("ceil_mode", False)
    if not isinstance(ceil_mode, bool):
        raise ValueError(f"{op.op}.ceil_mode must be a boolean, got {ceil_mode!r}")
    if mode == "avg":
        if not isinstance(op.attrs.get("count_include_pad", True), bool):
            raise ValueError(f"{op.op}.count_include_pad must be a boolean")
        override = op.attrs.get("divisor_override")
        if override is not None and (
            not isinstance(override, int) or isinstance(override, bool) or override <= 0
        ):
            raise ValueError(
                f"{op.op}.divisor_override must be a positive integer or None, "
                f"got {override!r}"
            )
    shape = list(input_type.shape[:2])
    for size, step, window, spread, pad in zip(
        input_type.shape[2:], stride, kernel, dilation, padding
    ):
        extent = size + 2 * pad - spread * (window - 1) - 1
        if extent < 0:
            raise ValueError(
                f"{op.op}.kernel {kernel} does not fit the padded input "
                f"{input_type.shape[2:]}"
            )
        if ceil_mode:
            out = -(-extent // step) + 1
            if (out - 1) * step >= size + pad:
                out -= 1
        else:
            out = extent // step + 1
        shape.append(out)
    if len(op.outputs) == 2 and mode != "max":
        raise ValueError(f"{op.op} only a maximum reports indices")
    _check_pool_outputs(graph, op, tuple(shape))


def _check_adaptive_pool(graph: Graph, op: Operation) -> None:
    input_type = graph.value(op.inputs[0]).type
    spatial = len(input_type.shape) - 2
    _pool_mode(op)
    output_size = _spatial_integers(op, "output_size", spatial, floor=1)
    if len(op.outputs) == 2 and op.attrs.get("mode") != "max":
        raise ValueError(f"{op.op} only a maximum reports indices")
    _check_pool_outputs(graph, op, (*input_type.shape[:2], *output_size))


def _conv_geometry(
    op: Operation, graph: Graph
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """The window geometry of one convolution, checked against its operands."""

    input_shape = graph.value(op.inputs[0]).type.shape
    spatial = len(input_shape) - 2
    geometry: list[tuple[int, ...]] = []
    for name in ("stride", "padding", "dilation"):
        value = op.attrs.get(name)
        if (
            not isinstance(value, tuple)
            or len(value) != spatial
            or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            )
        ):
            raise ValueError(
                f"{op.op}.{name} must be {spatial} integers, got {value!r}"
            )
        # Padding may be zero; a zero stride or dilation would not terminate.
        floor = 0 if name == "padding" else 1
        if any(item < floor for item in value):
            raise ValueError(f"{op.op}.{name} must be at least {floor}, got {value!r}")
        geometry.append(value)
    groups = op.attrs.get("groups")
    if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
        raise ValueError(f"{op.op}.groups must be a positive integer, got {groups!r}")
    return geometry[0], geometry[1], geometry[2], groups


def _check_conv(graph: Graph, op: Operation) -> None:
    input_type = graph.value(op.inputs[0]).type
    weight_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    if len(input_type.shape) != len(weight_type.shape):
        raise ValueError(
            f"conv operand ranks differ: {input_type.shape} and {weight_type.shape}"
        )
    if len(input_type.shape) < 3:
        raise ValueError(f"conv requires rank 3 or more, got {input_type.shape}")
    if not (input_type.dtype == weight_type.dtype == output_type.dtype):
        raise NotImplementedError("conv requires identical input/weight/output dtypes")
    stride, padding, dilation, groups = _conv_geometry(op, graph)
    transposed = op.attrs.get("transposed", False)
    if not isinstance(transposed, bool):
        raise ValueError(f"conv.transposed must be a boolean, got {transposed!r}")
    output_padding = op.attrs.get("output_padding", ())
    if (
        not isinstance(output_padding, tuple)
        or len(output_padding) != len(input_type.shape) - 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in output_padding
        )
    ):
        raise ValueError(
            f"conv.output_padding must be {len(input_type.shape) - 2} integers, "
            f"got {output_padding!r}"
        )
    batch, channels = input_type.shape[:2]
    if transposed:
        # A transposed convolution keeps the weight in the input-major layout
        # Torch uses for it: one row per input channel, one column block per
        # output channel of that group.
        weight_channels, block = weight_type.shape[:2]
        if weight_channels != channels:
            raise ValueError(
                f"conv weight has {weight_channels} rows, which does not match "
                f"{channels} inputs"
            )
        if channels % groups:
            raise ValueError(
                f"conv groups {groups} must divide the {channels} input channels"
            )
        out_channels = block * groups
        expected = [batch, out_channels]
        for size, kernel, step, pad, spread, extra in zip(
            input_type.shape[2:],
            weight_type.shape[2:],
            stride,
            padding,
            dilation,
            output_padding,
        ):
            if extra < 0 or extra >= max(step, spread):
                raise ValueError(
                    f"conv.output_padding {extra} must be smaller than the "
                    f"stride {step} or the dilation {spread}"
                )
            expected.append(
                (size - 1) * step - 2 * pad + spread * (kernel - 1) + 1 + extra
            )
        if any(size < 1 for size in expected[2:]):
            raise ValueError(
                f"conv output shape {tuple(expected)} has an empty spatial axis"
            )
    else:
        if any(output_padding):
            raise ValueError(
                "conv.output_padding is only nonzero for a transposed convolution"
            )
        out_channels, weight_channels = weight_type.shape[:2]
        if channels % groups or out_channels % groups:
            raise ValueError(
                f"conv groups {groups} must divide the {channels} input and "
                f"{out_channels} output channels"
            )
        if weight_channels != channels // groups:
            raise ValueError(
                f"conv weight has {weight_channels} channels per group, which does "
                f"not match {channels} inputs in {groups} groups"
            )
        expected = [batch, out_channels]
        for size, kernel, step, pad, spread in zip(
            input_type.shape[2:], weight_type.shape[2:], stride, padding, dilation
        ):
            extent = size + 2 * pad - spread * (kernel - 1) - 1
            if extent < 0:
                raise ValueError(
                    f"conv kernel {weight_type.shape[2:]} does not fit the padded "
                    f"input {input_type.shape[2:]}"
                )
            expected.append(extent // step + 1)
    if output_type.shape != tuple(expected):
        raise ValueError(f"conv output shape {output_type.shape} != {tuple(expected)}")
    if len(op.inputs) == 3:
        bias_type = graph.value(op.inputs[2]).type
        if bias_type.shape != (out_channels,):
            raise ValueError(f"conv bias shape {bias_type.shape} != {(out_channels,)}")
        if bias_type.dtype != input_type.dtype:
            raise NotImplementedError("conv requires a bias of the input dtype")


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


def _check_dequantize_i4(graph: Graph, op: Operation) -> None:
    """The three facts a 4-bit unpack relies on, checked together.

    The packed operand is a row of int32 words, the scales carry one value per
    group along the same row, and the result is the row expanded eightfold. Each
    of those is a relation *between* operands, which is why none of them fits the
    shape and dtype fields.
    """

    packed_type = graph.value(op.inputs[0]).type
    scales_type = graph.value(op.inputs[1]).type
    output_type = graph.value(op.outputs[0]).type
    if packed_type.dtype is not DType.I32:
        raise ValueError(
            f"dequantize_i4 packed weights must be int32 words, got {packed_type.dtype}"
        )
    if len(packed_type.shape) != 2:
        raise ValueError(
            f"dequantize_i4 packed weights must be [out, in/8], got {packed_type.shape}"
        )
    if len(scales_type.shape) != 2:
        raise ValueError(
            f"dequantize_i4 scales must be [out, in/group_size], got {scales_type.shape}"
        )
    group_size = op.attrs["group_size"]
    assert isinstance(group_size, int)
    rows, words = packed_type.shape
    if scales_type.shape[0] != rows:
        raise ValueError(
            f"dequantize_i4 scales has {scales_type.shape[0]} rows but the packed "
            f"weights have {rows}"
        )
    inner = words * 8
    if inner % group_size != 0 or scales_type.shape[1] != inner // group_size:
        raise ValueError(
            f"dequantize_i4 scales shape {scales_type.shape} does not cover "
            f"{inner} weights in groups of {group_size}"
        )
    if scales_type.dtype != output_type.dtype:
        raise ValueError(
            f"dequantize_i4 output dtype {output_type.dtype} must match the scales' "
            f"{scales_type.dtype}"
        )
    if output_type.shape != (rows, inner):
        raise ValueError(
            f"dequantize_i4 output shape {output_type.shape} != {(rows, inner)}"
        )


def _matmul_weight_group_size(op: Operation) -> int | None:
    """Return the int4 group size when the right operand is a packed weight.

    A folded quantized matmul keeps the same operator and the same names: the
    weight operand is the packed word tensor, the slot right after it holds the
    group scales, and the transpose flag is gone (the packed layout *is* the
    ``[out, in]`` orientation, so there is nothing left to transpose). Every
    consumer therefore asks this function rather than looking for a marker in a
    different place each time, and it lives with the contract table rather than
    with the codegen, because the pass that creates the pair and the kernels
    that read it sit on opposite sides of the pipeline.
    """

    value = op.attrs.get("weight_group_size")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{op.op}.weight_group_size must be a positive integer, got {value!r}"
        )
    return value


CONTRACTS: dict[str, OpContract] = {
    # Their compilers check the operands against the kernel they emit, so the
    # table only pins the arity and the number of results.
    "matmul": OpContract(
        arity=(2, 6), dtype=DtypeRule.ANY, shape=ShapeRule.FREE, native=True
    ),
    "addmm": OpContract(
        arity=(3, 6), dtype=DtypeRule.ANY, shape=ShapeRule.FREE, native=True
    ),
    # The unpack of a 4-bit weight. It has no kernel of its own: the pass that
    # folds it into the matmul that reads it removes every instance a compiled
    # model contains, and the MLX fallback only sees the ones a graph built by
    # hand can hold.
    "dequantize_i4": OpContract(
        arity=(2, 2),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("group_size", AttrKind.INT),),
        extra=_check_dequantize_i4,
    ),
    # A fused region may also hold no graph inputs at all: a chain that starts
    # from a constant producer (``x.zero_()`` fused into its consumer) is that
    # constant evaluated elementwise, and the runtime materializes it from the
    # shape it carries.
    "fused_elementwise": OpContract(
        arity=(0, 8),
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
    # Convolution has no Metal source: MLX's own kernel is the implementation,
    # which is why the contract is `free` over the shapes and checked by hand.
    "conv": OpContract(
        arity=(2, 3),
        rank=(3, 5),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("stride", AttrKind.INT_TUPLE),
            AttrSpec("padding", AttrKind.INT_TUPLE),
            AttrSpec("dilation", AttrKind.INT_TUPLE),
            AttrSpec("groups", AttrKind.INT),
            AttrSpec("transposed", AttrKind.BOOL),
            AttrSpec("output_padding", AttrKind.INT_TUPLE),
        ),
        extra=_check_conv,
    ),
    # A pooling window is read once and reduced; the two modes differ only in
    # the padding value and the reduction, so one contract covers both.
    "pool": OpContract(
        arity=(1, 1),
        rank=(3, 5),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        outputs=(1, 2),
        attrs=(
            AttrSpec("kernel", AttrKind.INT_TUPLE),
            AttrSpec("stride", AttrKind.INT_TUPLE),
            AttrSpec("padding", AttrKind.INT_TUPLE),
            AttrSpec("dilation", AttrKind.INT_TUPLE),
        ),
        extra=_check_pool,
    ),
    # A resize is a strided read plus per-axis interpolation weights, which is
    # why its geometry is checked by hand rather than from the output shape.
    "resize": OpContract(
        arity=(1, 1),
        rank=(3, 5),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        attrs=(
            AttrSpec("output_size", AttrKind.INT_TUPLE),
            AttrSpec("source_scale", AttrKind.NUMBER_TUPLE),
        ),
        extra=_check_resize,
    ),
    "adaptive_pool": OpContract(
        arity=(1, 1),
        rank=(3, 5),
        dtype=DtypeRule.FLOATING_IN,
        shape=ShapeRule.FREE,
        outputs=(1, 2),
        attrs=(AttrSpec("output_size", AttrKind.INT_TUPLE),),
        extra=_check_adaptive_pool,
    ),
}
