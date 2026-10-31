"""The ATen target names the frontend recognises, and what each one maps to.

Every ``str(node.target)`` string the legalizer compares against lives here, so
that "does this build support this operator" is one table to read rather than a
scatter of literals through the lowering code. ``_SUPPORTED_OPS`` is also the
one place where a target is given the core op name it becomes; the rest of the
tables classify those targets by how they have to be lowered.

These are ATen's own names, including the ``.default`` / ``.dim`` / ``.Tensor``
overload suffix, because that is what ``make_fx`` produces.
"""

from __future__ import annotations


from ..ir import DType

_SUPPORTED_OPS = {
    "aten.mm.default": "matmul",
    "aten.bmm.default": "matmul",
    "aten.addmm.default": "addmm",
    "aten.baddbmm.default": "addmm",
    "aten.view.default": "reshape",
    "aten._unsafe_view.default": "reshape",
    "aten.reshape.default": "reshape",
    "aten.transpose.int": "transpose",
    "aten.permute.default": "permute",
    "aten.expand.default": "expand",
    "aten.slice.Tensor": "slice",
    "aten.flip.default": "flip",
    "aten.roll.default": "roll",
    "aten.constant_pad_nd.default": "pad",
    "aten.triu.default": "triu",
    "aten.tril.default": "tril",
    "aten.unsqueeze.default": "unsqueeze",
    "aten.squeeze.dim": "squeeze",
    "aten.clone.default": "contiguous",
    "aten.alias.default": "alias",
    # ``detach`` only severs autograd edges, which the IR does not model, so it
    # is a pure alias. Torch's own composites emit it (``nanmean`` detaches the
    # non-NaN count), and an empty placeholder is still skipped above.
    "aten.detach.default": "alias",
    # A tensor literal built inside ``forward`` reaches the graph as a constant
    # plus this fresh-copy marker. The copy carries no arithmetic, and every
    # in-place op is legalized before it reaches the IR, so rebinding the value
    # is enough and avoids materializing a second buffer.
    "aten.lift_fresh_copy.default": "alias",
    "aten.lift_fresh.default": "alias",
    "aten._to_copy.default": "cast",
    "aten._softmax.default": "softmax",
    "aten._safe_softmax.default": "safe_softmax",
    "aten._log_softmax.default": "log_softmax",
    "aten.sum.default": "sum",
    "aten.sum.dim_IntList": "sum",
    "aten.prod.dim_int": "prod",
    "aten.prod.default": "prod",
    "aten.amax.default": "amax",
    "aten.amin.default": "amin",
    # The full-reduction spellings: ``max(x)``/``min(x)`` reduce every axis,
    # which is what an ``amax``/``amin`` with no dims means.
    "aten.max.default": "amax",
    "aten.min.default": "amin",
    "aten.cumsum.default": "cumsum",
    "aten.cumprod.default": "cumprod",
    "aten.logcumsumexp.default": "logcumsumexp",
    "aten.argmax.default": "argmax",
    "aten.argmin.default": "argmin",
    "aten.mean.default": "mean",
    "aten.mean.dim": "mean",
    "aten.silu.default": "silu",
    "aten.gelu.default": "gelu",
    "aten.softplus.default": "softplus",
    "aten.mish.default": "mish",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.mul_.Tensor": "mul",
    "aten.mul_.Scalar": "mul",
    "aten.add.Tensor": "add",
    "aten.add.Scalar": "add",
    "aten.add_.Tensor": "add",
    "aten.add_.Scalar": "add",
    "aten.sub.Tensor": "sub",
    "aten.sub.Scalar": "sub",
    "aten.sub_.Tensor": "sub",
    "aten.sub_.Scalar": "sub",
    "aten.rsub.Scalar": "rsub",
    "aten.div.Tensor": "div",
    "aten.div.Scalar": "div",
    "aten.div_.Tensor": "div",
    "aten.div_.Scalar": "div",
    "aten.bitwise_and_.Tensor": "bitwise_and",
    "aten.bitwise_and_.Scalar": "bitwise_and",
    "aten.bitwise_or_.Tensor": "bitwise_or",
    "aten.bitwise_or_.Scalar": "bitwise_or",
    "aten.bitwise_xor_.Tensor": "bitwise_xor",
    "aten.bitwise_xor_.Scalar": "bitwise_xor",
    "aten.bitwise_not_.default": "bitwise_not",
    "aten.logical_not_.default": "logical_not",
    "aten.floor_divide.default": "floor_divide",
    "aten.floor_divide.Tensor": "floor_divide",
    "aten.floor_divide.Scalar": "floor_divide",
    "aten.remainder.Tensor": "remainder",
    "aten.remainder.Scalar": "remainder",
    # ``scalar % tensor``: the dividend is the scalar, so the result still
    # follows the divisor's sign.
    "aten.remainder.Scalar_Tensor": "remainder",
    "aten.remainder_.Tensor": "remainder",
    "aten.remainder_.Scalar": "remainder",
    "aten.fmod.Tensor": "fmod",
    "aten.fmod.Scalar": "fmod",
    "aten.lerp.Tensor": "lerp",
    "aten.lerp.Scalar": "lerp",
    "aten.addcmul.default": "addcmul",
    "aten.addcdiv.default": "addcdiv",
    "aten.pow.Tensor_Tensor": "pow",
    "aten.pow.Tensor_Scalar": "pow",
    "aten.logit.default": "logit",
    "aten.erfc.default": "erfc",
    "aten.frac.default": "frac",
    "aten.sinc.default": "sinc",
    "aten.hypot.default": "hypot",
    "aten.heaviside.default": "heaviside",
    "aten.copysign.Tensor": "copysign",
    "aten.copysign.Scalar": "copysign",
    "aten.copysign.Scalar_Scalar": "copysign",
    "aten.xlogy.Tensor": "xlogy",
    "aten.xlogy.Scalar_Self": "xlogy",
    "aten.xlogy.Scalar_Other": "xlogy",
    "aten.nan_to_num.default": "nan_to_num",
    "aten.pow.Scalar": "pow",
    "aten.neg.default": "neg",
    "aten.abs.default": "abs",
    "aten.sigmoid.default": "sigmoid",
    "aten.tanh.default": "tanh",
    "aten.relu.default": "relu",
    "aten.exp.default": "exp",
    "aten.expm1.default": "expm1",
    "aten.exp2.default": "exp2",
    "aten.log.default": "log",
    "aten.log2.default": "log2",
    "aten.log10.default": "log10",
    "aten.log1p.default": "log1p",
    "aten.erf.default": "erf",
    # Special functions. ``lgamma`` and ``i0`` become pointwise Metal
    # expressions; ``erfinv`` stays an MLX call because PyTorch refines it with
    # Newton steps against an accurate ``erf``, which MSL does not provide.
    "aten.lgamma.default": "lgamma",
    # ``digamma`` has no MSL equivalent, so like ``lgamma`` it is a polynomial
    # helper in ``_MSL_HEADER``; the MLX path mirrors it.
    "aten.digamma.default": "digamma",
    "aten.i0.default": "i0",
    "aten.erfinv.default": "erfinv",
    # A bit test, so it is exact for both floats and integers: ``signbit`` is
    # true for -0.0, which a comparison against zero would miss.
    "aten.signbit.default": "signbit",
    "aten.sin.default": "sin",
    "aten.cos.default": "cos",
    # The rest of the trigonometric and hyperbolic family. Torch treats these
    # as core ATen primitives, so no decomposition rewrites them and the
    # frontend has to lower them itself; Metal's standard library has all nine.
    "aten.tan.default": "tan",
    "aten.asin.default": "asin",
    "aten.acos.default": "acos",
    "aten.atan.default": "atan",
    "aten.sinh.default": "sinh",
    "aten.cosh.default": "cosh",
    "aten.asinh.default": "asinh",
    "aten.acosh.default": "acosh",
    "aten.atanh.default": "atanh",
    "aten.floor.default": "floor",
    "aten.ceil.default": "ceil",
    "aten.round.default": "round",
    "aten.round.decimals": "round",
    "aten.trunc.default": "trunc",
    "aten.sign.default": "sign",
    "aten.square.default": "square",
    "aten.sqrt.default": "sqrt",
    "aten.rsqrt.default": "rsqrt",
    "aten.reciprocal.default": "reciprocal",
    "aten.clamp.default": "clamp",
    "aten.clamp.Tensor": "clamp",
    "aten.clamp_min.default": "clamp_min",
    "aten.clamp_min.Tensor": "clamp_min",
    "aten.clamp_max.default": "clamp_max",
    "aten.clamp_max.Tensor": "clamp_max",
    "aten.clamp_.default": "clamp",
    "aten.clamp_.Tensor": "clamp",
    "aten.clamp_.Scalar": "clamp",
    "aten.clamp_min_.default": "clamp_min",
    "aten.clamp_min_.Tensor": "clamp_min",
    "aten.clamp_max_.default": "clamp_max",
    "aten.clamp_max_.Tensor": "clamp_max",
    "aten.hardtanh.default": "clamp",
    "aten.leaky_relu.default": "leaky_relu",
    "aten.elu.default": "elu",
    "aten.hardswish.default": "hardswish",
    "aten.hardsigmoid.default": "hardsigmoid",
    "aten.threshold.default": "threshold",
    "aten.eq.Tensor": "eq",
    "aten.eq.Scalar": "eq",
    "aten.ne.Tensor": "ne",
    "aten.ne.Scalar": "ne",
    "aten.lt.Tensor": "lt",
    "aten.lt.Scalar": "lt",
    "aten.le.Tensor": "le",
    "aten.le.Scalar": "le",
    "aten.gt.Tensor": "gt",
    "aten.gt.Scalar": "gt",
    "aten.ge.Tensor": "ge",
    "aten.ge.Scalar": "ge",
    "aten.where.self": "where",
    "aten.logical_and.default": "logical_and",
    "aten.logical_or.default": "logical_or",
    "aten.logical_xor.default": "logical_xor",
    "aten.logical_not.default": "logical_not",
    "aten.bitwise_and.Tensor": "bitwise_and",
    "aten.bitwise_and.Scalar": "bitwise_and",
    "aten.bitwise_or.Tensor": "bitwise_or",
    "aten.bitwise_or.Scalar": "bitwise_or",
    "aten.bitwise_xor.Tensor": "bitwise_xor",
    "aten.bitwise_xor.Scalar": "bitwise_xor",
    "aten.bitwise_not.default": "bitwise_not",
    # Shifts have three spellings each: two tensors, a tensor and a scalar, and
    # a scalar and a tensor.
    "aten.bitwise_left_shift.Tensor": "bitwise_left_shift",
    "aten.bitwise_left_shift.Tensor_Scalar": "bitwise_left_shift",
    "aten.bitwise_left_shift.Scalar_Tensor": "bitwise_left_shift",
    "aten.bitwise_right_shift.Tensor": "bitwise_right_shift",
    "aten.bitwise_right_shift.Tensor_Scalar": "bitwise_right_shift",
    "aten.bitwise_right_shift.Scalar_Tensor": "bitwise_right_shift",
    "aten.fmax.default": "fmax",
    "aten.fmin.default": "fmin",
    "aten.gcd.default": "gcd",
    "aten.lcm.default": "lcm",
    "aten.nextafter.default": "nextafter",
    "aten.isnan.default": "isnan",
    "aten.isinf.default": "isinf",
    "aten.masked_fill.Scalar": "masked_fill",
    "aten.masked_fill.Tensor": "where",
    "aten.maximum.default": "maximum",
    "aten.minimum.default": "minimum",
    "aten.logaddexp.default": "logaddexp",
    "aten.atan2.default": "atan2",
    "aten.scalar_tensor.default": "scalar_tensor",
    "aten.arange.default": "arange",
    "aten.arange.start": "arange",
    "aten.arange.start_step": "arange",
    "aten.zeros.default": "zeros",
    "aten.ones.default": "ones",
    "aten.full.default": "full",
    "aten.new_zeros.default": "new_zeros",
    "aten.new_ones.default": "new_ones",
    "aten.new_full.default": "new_full",
    "aten.zeros_like.default": "zeros_like",
    "aten.ones_like.default": "ones_like",
    "aten.full_like.default": "full_like",
    "aten.cat.default": "cat",
    "aten.embedding.default": "embedding",
    "aten.index_select.default": "index_select",
    "aten.gather.default": "gather",
    "aten.scatter.src": "scatter",
    "aten.scatter.value": "scatter",
    "aten.scatter_add.default": "scatter",
    "aten.index_add.default": "index_add",
}


_EMPTY_COPY_TARGETS = frozenset(
    {
        "aten.lift_fresh_copy.default",
        "aten.clone.default",
        "aten.detach.default",
        "aten.alias.default",
        "aten._to_copy.default",
        "aten.contiguous.default",
        "aten.lift_fresh.default",
    }
)


_TORCH_DTYPES = {
    "torch.float16": DType.F16,
    "torch.bfloat16": DType.BF16,
    "torch.float32": DType.F32,
    "torch.bool": DType.BOOL,
    "torch.int32": DType.I32,
    "torch.int64": DType.I64,
}


_NUMERIC_ATTRS = {
    "addmm": {"alpha": 1.0, "beta": 1.0},
    "add": {"alpha": 1.0},
    "sub": {"alpha": 1.0},
    "rsub": {"alpha": 1.0},
    "leaky_relu": {"negative_slope": 0.01},
    "elu": {"alpha": 1.0, "scale": 1.0, "input_scale": 1.0},
    "softplus": {"beta": 1.0, "threshold": 20.0},
    "threshold": {"threshold": 0.0, "value": 0.0},
    "addcmul": {"value": 1.0},
    "addcdiv": {"value": 1.0},
}


_COMPARISON_OPS = {"eq", "ne", "lt", "le", "gt", "ge"}


_DIVIDE_ROUNDING_TARGETS = frozenset(
    {
        "aten.div.Tensor_mode",
        "aten.div.Scalar_mode",
        "aten.div_.Tensor_mode",
        "aten.div_.Scalar_mode",
    }
)
"""``div`` spellings that carry a ``rounding_mode`` keyword.

They are not a straight target-to-op mapping: ``rounding_mode='floor'`` is the
existing ``floor_divide``, ``'trunc'`` is ``trunc_divide``, and ``None`` is true
division, so ``lower_supported`` reads the keyword before choosing the IR op.
"""


_FLOAT_ONLY_POINTWISE_OPS = {
    "logit",
    "erfc",
    "frac",
    "sinc",
    "lgamma",
    "digamma",
    "i0",
    "erfinv",
    "hypot",
    "copysign",
    "xlogy",
    "lerp",
    "addcmul",
    "addcdiv",
    "nan_to_num",
    # ATen promotes integer operands to float for these and computes in the
    # default float dtype; Metal only has the floating-point overloads.
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    # ``nextafter`` is float-only in ATen; ``gcd``/``lcm`` and the shifts are
    # the mirror image and only accept integral operands.
    "nextafter",
}

# Operators whose operands must be integral. ATen raises for a float argument,
# so a float pair reaching these means the lowering is wrong.
_INTEGRAL_ONLY_POINTWISE_OPS = {
    "gcd",
    "lcm",
    "bitwise_left_shift",
    "bitwise_right_shift",
}


_LOGICAL_BINARY_OPS = {"logical_and", "logical_or", "logical_xor"}


_BITWISE_BINARY_OPS = {"bitwise_and", "bitwise_or", "bitwise_xor"}


_BINARY_OPS = _COMPARISON_OPS | {
    "mul",
    "add",
    "sub",
    "rsub",
    "div",
    "floor_divide",
    "trunc_divide",
    "remainder",
    "fmod",
    "hypot",
    "copysign",
    "xlogy",
    "pow",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "nextafter",
    "gcd",
    "lcm",
    "logaddexp",
    "atan2",
    "heaviside",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
}


_REDUCTION_OPS = {"sum", "mean", "prod", "amax", "amin"}


_INTEGRAL_DTYPES = {DType.I32, DType.I64}


_FLOATING_DTYPES = {DType.F16, DType.BF16, DType.F32}


_PROMOTABLE_POINTWISE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "floor_divide",
    "trunc_divide",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "nextafter",
    "gcd",
    "lcm",
    "logaddexp",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
} | _COMPARISON_OPS


_DIRECT_ATTENTION_TARGETS = {
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten.scaled_dot_product_attention.default",
}


_TUPLE_ATTENTION_TARGETS = {
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
}


_SEQUENCE_VIEW_TARGETS = {
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
}


_SELECT_TARGET = "aten.select.int"


_INDEX_TARGET = "aten.index.Tensor"


_MATRIX_TRANSPOSE_TARGET = "aten.t.default"


_STACK_TARGET = "aten.stack.default"


_SCATTER_TARGETS = {"aten.scatter.src", "aten.scatter.value"}


_SCATTER_ADD_TARGET = "aten.scatter_add.default"


_SCATTER_REDUCE_TARGET = "aten.scatter_reduce.two"


_INDEX_ADD_TARGET = "aten.index_add.default"


_SEARCHSORTED_TARGETS = {
    "aten.searchsorted.Tensor": "searchsorted",
    "aten.searchsorted.Scalar": "searchsorted",
    "aten.bucketize.Tensor": "bucketize",
    "aten.bucketize.Scalar": "bucketize",
}
"""The search spellings, mapped to the operand order each one takes.

``bucketize`` is ``searchsorted`` with the arguments the other way round: the
values being placed come first.
"""


_HISTC_TARGET = "aten.histc.default"


_INDEX_PUT_TARGETS = {
    "aten.index_put.default": False,
    "aten.index_put_.default": True,
    "aten.index_copy.default": False,
    "aten.index_copy_.default": True,
    "aten.index_fill.int_Scalar": False,
    "aten.index_fill.int_Tensor": False,
    "aten.index_fill_.int_Scalar": True,
    "aten.index_fill_.int_Tensor": True,
}
"""The write-through-index spellings, mapped to whether they mutate in place.

``index_put`` names every axis it writes, ``index_copy`` names one, and
``index_fill`` is ``index_copy`` with a scalar; all three lower to the same
``index_put`` IR operation once the selection is spelled out.
"""


_SCATTER_REDUCE_NAMES = {
    "sum": "add",
    "prod": "prod",
    "amax": "amax",
    "amin": "amin",
    "mean": "mean",
}


_LOGSIGMOID_TARGET = "aten.log_sigmoid_forward.default"


_PRELU_TARGET = "aten._prelu_kernel.default"


_NANSUM_TARGET = "aten.nansum.default"


_SORT_TARGETS = {"aten.sort.default", "aten.topk.default"}


_CUM_EXTREMUM_TARGETS = {
    "aten.cummax.default": ("cummax", "cummax_indices"),
    "aten.cummin.default": ("cummin", "cummin_indices"),
}
"""The running-extremum scans, mapped to their value and index operators.

Both return ``(values, indices)`` and are consumed through ``getitem``, so the
lowering reads the selection exactly like ``sort`` does.
"""


_MEDIAN_TARGETS = {
    "aten.median.default": False,
    "aten.median.dim": False,
    "aten.nanmedian.default": True,
    "aten.nanmedian.dim": True,
}
"""The median spellings, mapped to whether they ignore NaN.

``median.dim``/``nanmedian.dim`` return ``(values, indices)`` and are consumed
through ``getitem``; the ``default`` forms reduce every axis and return a
scalar, but they take the same lowering with all axes named.
"""


_EXTREMUM_DIM_TARGETS = {
    "aten.max.dim": ("amax", "argmax"),
    "aten.min.dim": ("amin", "argmin"),
}


_VARIANCE_TARGETS = {
    "aten.var.correction": "var",
    "aten.var.dim": "var",
    "aten.var.default": "var",
    "aten.std.correction": "std",
    "aten.std.dim": "std",
    "aten.std.default": "std",
}


_REPEAT_TARGET = "aten.repeat.default"


_DIAGONAL_TARGET = "aten.diagonal.default"


_DIAGONAL_COPY_TARGET = "aten.diagonal_copy.default"


_VECTOR_NORM_TARGET = "aten.linalg_vector_norm.default"


_PAD_MODE_TARGETS = {
    "aten.reflection_pad1d.default": ("reflect", 1),
    "aten.reflection_pad2d.default": ("reflect", 2),
    "aten.replication_pad1d.default": ("replicate", 1),
    "aten.replication_pad2d.default": ("replicate", 2),
}


_INPLACE_TARGETS = {
    "aten.add_.Tensor",
    "aten.add_.Scalar",
    "aten.sub_.Tensor",
    "aten.sub_.Scalar",
    "aten.mul_.Tensor",
    "aten.mul_.Scalar",
    "aten.div_.Tensor",
    "aten.div_.Scalar",
    "aten.remainder_.Tensor",
    "aten.remainder_.Scalar",
    "aten.bitwise_and_.Tensor",
    "aten.bitwise_and_.Scalar",
    "aten.bitwise_or_.Tensor",
    "aten.bitwise_or_.Scalar",
    "aten.bitwise_xor_.Tensor",
    "aten.bitwise_xor_.Scalar",
    "aten.bitwise_not_.default",
    "aten.logical_not_.default",
    "aten.clamp_.default",
    "aten.clamp_.Tensor",
    "aten.clamp_.Scalar",
    "aten.clamp_min_.default",
    "aten.clamp_min_.Tensor",
    "aten.clamp_max_.default",
    "aten.clamp_max_.Tensor",
}


_CONSTANT_PAD_TARGET = "aten.constant_pad_nd.default"


_MV_TARGET = "aten.mv.default"


_TRACE_TARGET = "aten.trace.default"


_LOGSUMEXP_TARGET = "aten.logsumexp.default"


_INPLACE_SCATTER_TARGETS = {
    "aten.scatter_.src": "aten.scatter.src",
    "aten.scatter_.value": "aten.scatter.value",
    "aten.scatter_add_.default": "aten.scatter_add.default",
    "aten.scatter_reduce_.two": "aten.scatter_reduce.two",
    "aten.index_add_.default": "aten.index_add.default",
}


_AND_OR_TARGETS = frozenset(
    {
        "aten.all.default",
        "aten.all.dim",
        "aten.all.dims",
        "aten.any.default",
        "aten.any.dim",
        "aten.any.dims",
    }
)


_BERNOULLI_TARGET = "aten.bernoulli_.float"


_LAYER_NORM_TARGET = "aten.native_layer_norm.default"


_SQUEEZE_ALL_TARGET = "aten.squeeze.default"


_COUNT_NONZERO_TARGETS = frozenset(
    {"aten.count_nonzero.default", "aten.count_nonzero.dim_IntList"}
)


NATIVE_TARGETS: frozenset[str] = frozenset(
    set(_SUPPORTED_OPS)
    | set(_EMPTY_COPY_TARGETS)
    | _DIRECT_ATTENTION_TARGETS
    | _SEQUENCE_VIEW_TARGETS
    | _SORT_TARGETS
    | set(_EXTREMUM_DIM_TARGETS)
    | set(_VARIANCE_TARGETS)
    | set(_PAD_MODE_TARGETS)
    | _INPLACE_TARGETS
    | _DIVIDE_ROUNDING_TARGETS
    | set(_INPLACE_SCATTER_TARGETS)
    | _AND_OR_TARGETS
    | _SCATTER_TARGETS
    | _COUNT_NONZERO_TARGETS
    | set(_MEDIAN_TARGETS)
    | set(_CUM_EXTREMUM_TARGETS)
    | set(_INDEX_PUT_TARGETS)
    | set(_SEARCHSORTED_TARGETS)
    | {_HISTC_TARGET}
    | {
        _SELECT_TARGET,
        _INDEX_TARGET,
        _MATRIX_TRANSPOSE_TARGET,
        _STACK_TARGET,
        _SCATTER_ADD_TARGET,
        _SCATTER_REDUCE_TARGET,
        _INDEX_ADD_TARGET,
        _LOGSIGMOID_TARGET,
        _PRELU_TARGET,
        _NANSUM_TARGET,
        _REPEAT_TARGET,
        _DIAGONAL_TARGET,
        _DIAGONAL_COPY_TARGET,
        _VECTOR_NORM_TARGET,
        _CONSTANT_PAD_TARGET,
        _MV_TARGET,
        _TRACE_TARGET,
        _LOGSUMEXP_TARGET,
        _BERNOULLI_TARGET,
        _LAYER_NORM_TARGET,
        _SQUEEZE_ALL_TARGET,
    }
)
