"""Pointwise evaluation through MLX.

Most operators are one MLX call, but a few are not: ``lgamma`` and ``i0`` have no
MLX binding, so they are approximated here with the same polynomials the MSL
kernel uses, and ``signbit`` is a bit test rather than a comparison so that
``-0.0`` reports negative.
"""

from __future__ import annotations

import mlx.core as mx
from math import log, pi
from ...ir import DType
from ...ir import Operation
from ...ir import TensorType
from ...ir.ops import FLOATING_DTYPES as _FLOATING_DTYPES
from ...ir.ops import INTEGRAL_DTYPES as _INTEGRAL_DTYPES
from ...codegen.dtypes import _MLX_DTYPES, _MLX_DTYPE_TO_DTYPE
from ...codegen.expr import _FINITE_LIMITS, _wrap_integer_scalar

_LANCZOS_COEFFICIENTS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-06,
    1.5056327351493116e-07,
)


def _evaluate_pointwise_mlx(
    op: Operation, env: dict[str, mx.array], output_type: TensorType
) -> mx.array:
    arguments = [env[value_id] for value_id in op.inputs]
    if op.op == "scalar_tensor":
        value = op.attrs.get("value")
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"scalar_tensor.value must be scalar, got {value!r}")
        return mx.array(value, dtype=_MLX_DTYPES[output_type.dtype])
    if op.op in {
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
    }:
        value = op.attrs.get("value")
        if not isinstance(value, (bool, int, float)):
            raise ValueError(f"{op.op}.value must be scalar, got {value!r}")
        fill = mx.array(value, dtype=_MLX_DTYPES[output_type.dtype])
        return mx.full(output_type.shape, fill)
    if op.op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "eq":
            return left == right
        if op.op == "ne":
            return left != right
        if op.op == "lt":
            return left < right
        if op.op == "le":
            return left <= right
        if op.op == "gt":
            return left > right
        return left >= right
    if op.op == "logical_and":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_and(left, right)
    if op.op == "logical_or":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_or(left, right)
    if op.op == "logical_xor":
        left, right = _binary_mx_operands(op, arguments)
        return mx.logical_xor(left, right)
    if op.op == "logical_not":
        return mx.logical_not(arguments[0])
    if op.op == "bitwise_and":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_and(left, right)
    if op.op == "bitwise_or":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_or(left, right)
    if op.op == "bitwise_xor":
        left, right = _binary_mx_operands(op, arguments)
        return mx.bitwise_xor(left, right)
    if op.op == "bitwise_not":
        return ~arguments[0]
    if op.op in {"bitwise_left_shift", "bitwise_right_shift"}:
        left, right = _binary_mx_operands(op, arguments)
        return _mlx_shift(op.op, left, right)
    if op.op == "isnan":
        return mx.isnan(arguments[0])
    if op.op == "isinf":
        return mx.isinf(arguments[0])
    if op.op == "where":
        return mx.where(arguments[0], arguments[1], arguments[2])
    if op.op == "masked_fill":
        fill = mx.array(op.attrs["value"], dtype=arguments[0].dtype)
        return mx.where(arguments[1], fill, arguments[0])
    if op.op == "silu":
        value = arguments[0]
        return value / (1.0 + mx.exp(-value))
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
            return 0.5 * value * (1.0 + mx.tanh(inner))
        return 0.5 * value * (1.0 + mx.erf(value * 0.7071067811865476))
    if op.op == "softplus":
        value = arguments[0]
        beta = float(op.attrs["beta"])
        threshold = float(op.attrs["threshold"])
        scaled = beta * value
        return mx.where(
            scaled > threshold,
            value,
            mx.log1p(mx.exp(scaled)) / beta,
        )
    if op.op == "mish":
        value = arguments[0]
        softplus = mx.where(
            value > 20.0,
            value,
            mx.log1p(mx.exp(value)),
        )
        return value * mx.tanh(softplus)
    if op.op == "logsigmoid":
        value = arguments[0]
        body = mx.minimum(value, 0.0) - mx.log1p(mx.exp(-mx.abs(value)))
        return mx.where(mx.isnan(value), value, body)
    if op.op in {
        "mul",
        "div",
        "floor_divide",
        "remainder",
        "add",
        "sub",
        "rsub",
        "pow",
    }:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return left * right
        if op.op == "div":
            return left / right
        if op.op == "floor_divide":
            return mx.floor_divide(left, right)
        if op.op == "remainder":
            return mx.remainder(left, right)
        if op.op == "pow":
            return mx.power(left, right)
        alpha = float(op.attrs.get("alpha", 1.0))
        if op.op == "add":
            return left + alpha * right
        return left - alpha * right
    if op.op in {"maximum", "minimum"}:
        left, right = _binary_mx_operands(op, arguments)
        comparison = left < right if op.op == "maximum" else left > right
        selected = mx.where(comparison, right, left)
        return mx.where(mx.isnan(left) | mx.isnan(right), float("nan"), selected)
    if op.op in {"fmax", "fmin"}:
        left, right = _binary_mx_operands(op, arguments)
        # C99 fmax/fmin, which ignore a NaN operand; MLX's maximum/minimum
        # propagate NaN the way the torch maximum/minimum kernels do. There is
        # no NaN to ignore on an integer pair, so those go straight through.
        if _MLX_DTYPE_TO_DTYPE.get(left.dtype) not in _FLOATING_DTYPES:
            return (
                mx.maximum(left, right) if op.op == "fmax" else mx.minimum(left, right)
            )
        if op.op == "fmax":
            return _mlx_fmax(left, right)
        return _mlx_fmin(left, right)
    if op.op == "nextafter":
        left, right = _binary_mx_operands(op, arguments)
        return _mlx_nextafter(left, right)
    if op.op in {"trunc_divide", "gcd", "lcm"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "trunc_divide":
            return _mlx_trunc_divide(left, right)
        if op.op == "gcd":
            return _mlx_gcd(left, right)
        return _mlx_lcm(left, right)
    if op.op == "logaddexp":
        left, right = _binary_mx_operands(op, arguments)
        return _mlx_logaddexp(left, right)
    if op.op == "atan2":
        left, right = _binary_mx_operands(op, arguments)
        return mx.arctan2(left, right)
    if op.op == "fmod":
        left, right = _binary_mx_operands(op, arguments)
        # MLX has no fmod, and its remainder floored-adjusts the result, so
        # rebuild the truncated form: fmod(a, b) = a - trunc(a / b) * b. On an
        # integer dtype the quotient has to truncate as well, or a negative
        # dividend would land on the floor-multiple instead.
        if output_type.dtype in _INTEGRAL_DTYPES:
            return left - _mlx_trunc_divide(left, right) * right
        return left - mx.trunc(left / right) * right
    if op.op == "hypot":
        left, right = _binary_mx_operands(op, arguments)
        return mx.sqrt(left * left + right * right)
    if op.op == "copysign":
        left, right = _binary_mx_operands(op, arguments)
        negative = mx.less(right, 0.0) | (
            (right == 0.0) & mx.less(mx.divide(1.0, right), 0.0)
        )
        return mx.where(negative, -mx.abs(left), mx.abs(left))
    if op.op == "xlogy":
        left, right = _binary_mx_operands(op, arguments)
        return mx.where(
            (left == 0.0) & ~mx.isnan(right),
            mx.zeros_like(left),
            left * mx.log(right),
        )
    if output_type.dtype == DType.BOOL and op.op in {"mul", "add"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return mx.logical_and(left, right)
        return mx.logical_or(left, right)
    if op.op == "heaviside":
        value, boundary = _binary_mx_operands(op, arguments)
        dtype = (value if isinstance(value, mx.array) else boundary).dtype
        zero = mx.array(0, dtype=dtype)
        one = mx.array(1, dtype=dtype)
        return mx.where(
            mx.equal(value, zero),
            boundary,
            mx.where(mx.greater(value, zero), one, zero),
        )
    if op.op == "lerp":
        start, end = arguments[0], arguments[1]
        weight = (
            arguments[2] if len(arguments) == 3 else float(op.attrs.get("weight", 1.0))
        )
        return mx.where(
            mx.abs(weight) < 0.5,
            start + weight * (end - start),
            end - (end - start) * (1.0 - weight),
        )
    if op.op in {"addcmul", "addcdiv"}:
        accumulator, left, right = arguments
        value = float(op.attrs.get("value", 1.0))
        scaled = value * left
        return accumulator + (scaled * right if op.op == "addcmul" else scaled / right)
    if op.op == "logit":
        value = arguments[0]
        if "eps" in op.attrs:
            eps = float(op.attrs["eps"])
            value = mx.minimum(mx.maximum(value, eps), 1.0 - eps)
        return mx.log(value / (1.0 - value))
    if op.op == "erfc":
        return 1.0 - mx.erf(arguments[0])
    if op.op == "frac":
        return arguments[0] - mx.trunc(arguments[0])
    if op.op == "sinc":
        value = arguments[0]
        scaled = pi * value
        return mx.where(
            value == 0.0,
            mx.ones_like(value),
            mx.sin(scaled) / scaled,
        )
    if op.op == "nan_to_num":
        value = arguments[0]
        dtype = _MLX_DTYPE_TO_DTYPE.get(value.dtype)
        if dtype not in _FLOATING_DTYPES:
            return value
        limit = _FINITE_LIMITS[dtype]
        nan_value = float(op.attrs.get("nan", 0.0))
        posinf_value = float(op.attrs.get("posinf", limit))
        neginf_value = float(op.attrs.get("neginf", -limit))
        return mx.where(
            mx.isnan(value),
            nan_value,
            mx.where(
                mx.isinf(value),
                mx.where(value > 0.0, posinf_value, neginf_value),
                value,
            ),
        )
    if op.op == "neg":
        return -arguments[0]
    if op.op == "abs":
        return mx.abs(arguments[0])
    if op.op == "sigmoid":
        return 1.0 / (1.0 + mx.exp(-arguments[0]))
    if op.op == "tanh":
        return mx.tanh(arguments[0])
    if op.op == "relu":
        return mx.maximum(arguments[0], 0.0)
    if op.op == "leaky_relu":
        value = arguments[0]
        slope = float(op.attrs.get("negative_slope", 0.01))
        return mx.where(value >= 0.0, value, slope * value)
    if op.op == "elu":
        value = arguments[0]
        alpha = float(op.attrs.get("alpha", 1.0))
        return mx.where(value >= 0.0, value, alpha * (mx.exp(value) - 1.0))
    if op.op == "hardswish":
        value = arguments[0]
        body = value * mx.minimum(mx.maximum(value + 3.0, 0.0), 6.0) / 6.0
        return mx.where(mx.isnan(value), float("nan"), body)
    if op.op == "hardsigmoid":
        value = arguments[0]
        body = mx.minimum(mx.maximum(value + 3.0, 0.0), 6.0) / 6.0
        return mx.where(mx.isnan(value), float("nan"), body)
    if op.op == "threshold":
        value = arguments[0]
        threshold = float(op.attrs["threshold"])
        fill_value = float(op.attrs["value"])
        return mx.where(
            mx.isnan(value),
            value,
            mx.where(value > threshold, value, fill_value),
        )
    if op.op == "exp":
        return mx.exp(arguments[0])
    if op.op == "expm1":
        return mx.expm1(arguments[0])
    if op.op == "exp2":
        return mx.exp2(arguments[0])
    if op.op == "log":
        return mx.log(arguments[0])
    if op.op == "log2":
        return mx.log2(arguments[0])
    if op.op == "log10":
        return mx.log10(arguments[0])
    if op.op == "log1p":
        return mx.log1p(arguments[0])
    if op.op == "erf":
        return mx.erf(arguments[0])
    if op.op == "erfinv":
        value = arguments[0]
        if value.dtype not in (mx.float32, mx.float64):
            value = value.astype(mx.float32)
        # MLX's own implementation, which is what keeps this within an ulp of
        # ATen: refining Pavlis' rational approximation needs an accurate
        # ``erf`` that the Metal standard library does not provide.
        return mx.erfinv(value).astype(_MLX_DTYPES[output_type.dtype])
    if op.op == "lgamma":
        return _mlx_lgamma(arguments[0]).astype(_MLX_DTYPES[output_type.dtype])
    if op.op == "digamma":
        return _mlx_digamma(arguments[0]).astype(_MLX_DTYPES[output_type.dtype])
    if op.op == "i0":
        return _mlx_i0(arguments[0]).astype(_MLX_DTYPES[output_type.dtype])
    if op.op == "signbit":
        return _mlx_signbit(arguments[0])
    if op.op == "sin":
        return mx.sin(arguments[0])
    if op.op == "cos":
        return mx.cos(arguments[0])
    if op.op == "tan":
        return mx.tan(arguments[0])
    if op.op == "asin":
        return mx.arcsin(arguments[0])
    if op.op == "acos":
        return mx.arccos(arguments[0])
    if op.op == "atan":
        return mx.arctan(arguments[0])
    if op.op == "sinh":
        return mx.sinh(arguments[0])
    if op.op == "cosh":
        return mx.cosh(arguments[0])
    if op.op == "asinh":
        return mx.arcsinh(arguments[0])
    if op.op == "acosh":
        return mx.arccosh(arguments[0])
    if op.op == "atanh":
        return mx.arctanh(arguments[0])
    if op.op == "floor":
        return mx.floor(arguments[0])
    if op.op == "ceil":
        return mx.ceil(arguments[0])
    if op.op == "round":
        decimals = int(op.attrs.get("decimals", 0))
        if not decimals:
            return mx.round(arguments[0])
        scale = 10.0**decimals
        return mx.round(arguments[0] * scale) / scale
    if op.op == "trunc":
        return mx.trunc(arguments[0])
    if op.op == "sign":
        return mx.sign(arguments[0])
    if op.op == "square":
        return mx.square(arguments[0])
    if op.op == "sqrt":
        return mx.sqrt(arguments[0])
    if op.op == "rsqrt":
        return 1.0 / mx.sqrt(arguments[0])
    if op.op == "reciprocal":
        return 1.0 / arguments[0]
    if op.op in {"clamp", "clamp_min", "clamp_max"}:
        value = arguments[0]
        integral = value.dtype in {mx.int32, mx.int64}
        integer_min = -(1 << (31 if value.dtype == mx.int32 else 63))
        integer_max = (1 << (31 if value.dtype == mx.int32 else 63)) - 1
        lower: mx.array | float = (
            mx.array(integer_min, dtype=value.dtype) if integral else -float("inf")
        )
        upper: mx.array | float = (
            mx.array(integer_max, dtype=value.dtype) if integral else float("inf")
        )
        if "min_input" in op.attrs:
            lower = arguments[int(op.attrs["min_input"])]
        elif "min" in op.attrs:
            lower = (
                mx.array(op.attrs["min"], dtype=value.dtype)
                if integral
                else float(op.attrs["min"])
            )
        if "max_input" in op.attrs:
            upper = arguments[int(op.attrs["max_input"])]
        elif "max" in op.attrs:
            upper = (
                mx.array(op.attrs["max"], dtype=value.dtype)
                if integral
                else float(op.attrs["max"])
            )
        clipped = mx.minimum(mx.maximum(value, lower), upper)
        if integral:
            return clipped
        invalid = mx.isnan(value) | mx.isnan(lower) | mx.isnan(upper)
        return mx.where(invalid, float("nan"), clipped)
    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _binary_mx_operands(
    op: Operation, arguments: list[mx.array]
) -> tuple[mx.array | float, mx.array | float]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar_value = op.attrs["scalar"]
    if isinstance(scalar_value, bool):
        scalar = mx.array(scalar_value)
    elif isinstance(scalar_value, int) and arguments[0].dtype in {
        mx.int32,
        mx.int64,
    }:
        dtype = DType.I32 if arguments[0].dtype == mx.int32 else DType.I64
        scalar = mx.array(
            _wrap_integer_scalar(scalar_value, dtype), dtype=_MLX_DTYPES[dtype]
        )
    else:
        scalar = float(scalar_value)
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _mlx_lgamma(value: mx.array) -> mx.array:
    """log|Gamma(x)|, mirroring the Metal helper ``tsc_lgamma``."""

    compute = (
        value if value.dtype in (mx.float32, mx.float64) else value.astype(mx.float32)
    )

    def series(arguments_value: mx.array) -> mx.array:
        z = arguments_value - 1.0
        total = _LANCZOS_COEFFICIENTS[0]
        for index, coefficient in enumerate(_LANCZOS_COEFFICIENTS[1:], start=1):
            total = total + coefficient / (z + float(index))
        return (
            0.9189385332046727 + (z + 0.5) * mx.log(z + 7.5) - (z + 7.5) + mx.log(total)
        )

    reflected = log(pi) - mx.log(mx.abs(mx.sin(pi * compute))) - series(1.0 - compute)
    # ATen reports +inf at the poles and for both infinities, and NaN only for
    # a NaN argument.
    at_pole = (compute <= 0.0) & (compute == mx.round(compute))
    body = mx.where(
        at_pole, float("inf"), mx.where(compute < 0.5, reflected, series(compute))
    )
    return mx.where(mx.isinf(compute), float("inf"), body)


_PSI_10 = 2.25175258906672110764


def _mlx_digamma(value: mx.array) -> mx.array:
    """Digamma, mirroring the Metal helper ``tsc_digamma``."""

    compute = (
        value if value.dtype in (mx.float32, mx.float64) else value.astype(mx.float32)
    )
    pi = 3.14159265358979323846

    def series(argument: mx.array) -> mx.array:
        # Nine steps is the worst case: the walk starts at x = 1 and stops at 10.
        accumulated = mx.zeros_like(argument)
        current = argument
        for _ in range(9):
            active = current < 10.0
            accumulated = accumulated - mx.where(active, 1.0 / current, 0.0)
            current = mx.where(active, current + 1.0, current)
        inverse_square = 1.0 / (current * current)
        tail = inverse_square * (
            8.33333333333333333333e-2
            + inverse_square
            * (
                -2.10927960927960927961e-2
                + inverse_square
                * (
                    7.57575757575757575758e-3
                    + inverse_square
                    * (
                        -4.16666666666666666667e-3
                        + inverse_square
                        * (
                            3.96825396825396825397e-3
                            + inverse_square
                            * (
                                -8.33333333333333333333e-3
                                + inverse_square * 8.33333333333333333333e-2
                            )
                        )
                    )
                )
            )
        )
        asymptotic = mx.log(current) - 0.5 / current - tail
        return accumulated + mx.where(current == 10.0, _PSI_10, asymptotic)

    # The same reduction the Metal helper does: subtract the nearest integer so
    # the angle is in [-0.5, 0.5] (exact by Sterbenz' lemma), and use cos/sin
    # instead of 1/tan so the error stays relative at both ends.
    reduced = compute - mx.round(compute)
    reflected = series(1.0 - compute) - pi * mx.cos(pi * reduced) / mx.sin(pi * reduced)
    negative = compute < 0.0
    # ATen reports NaN at the negative integers, where the reflection terms
    # disagree. It is evaluated eagerly here and selected away, so the tan pole
    # in an unused branch never reaches the output.
    at_pole = negative & (compute == mx.trunc(compute))
    body = mx.where(
        negative, mx.where(at_pole, float("nan"), reflected), series(compute)
    )
    # At ±0 ATen returns the infinity whose sign is the opposite of the zero's,
    # so +0 maps to -inf and -0 to +inf; the reciprocal is what tells the two
    # zeros apart.
    at_zero = compute == 0.0
    signed_infinity = mx.where(1.0 / compute < 0.0, float("inf"), float("-inf"))
    return mx.where(
        at_zero, signed_infinity, mx.where(mx.isnan(compute), compute, body)
    )


def _mlx_i0(value: mx.array) -> mx.array:
    """Modified Bessel I0, mirroring the Metal helper ``tsc_i0``."""

    compute = (
        value if value.dtype in (mx.float32, mx.float64) else value.astype(mx.float32)
    )
    magnitude = mx.abs(compute)
    ratio = magnitude / 3.75
    square = ratio * ratio
    small = 1.0 + square * (
        3.5156229
        + square
        * (
            3.0899424
            + square
            * (
                1.2067492
                + square * (0.2659732 + square * (0.0360768 + square * 0.0045813))
            )
        )
    )
    scaled = 3.75 / magnitude
    polynomial = 0.39894228 + scaled * (
        0.01328592
        + scaled
        * (
            0.00225319
            + scaled
            * (
                -0.00157565
                + scaled
                * (
                    0.00916281
                    + scaled
                    * (
                        -0.02057706
                        + scaled
                        * (0.02635537 + scaled * (-0.01647633 + scaled * 0.00392377))
                    )
                )
            )
        )
    )
    large = mx.exp(magnitude) / mx.sqrt(magnitude) * polynomial
    return mx.where(magnitude < 3.75, small, large)


def _mlx_signbit(value: mx.array) -> mx.array:
    """The sign bit of each element, true for -0.0 as well as for negatives."""

    if value.dtype in (mx.int32, mx.int64, mx.uint32, mx.uint64):
        return value < 0
    negative_zero = (value == 0.0) & (mx.sign(1.0 / value) < 0.0)
    return (value < 0.0) | negative_zero


def _mlx_logaddexp(left: mx.array, right: mx.array) -> mx.array:
    """``logaddexp`` on MLX, the shift ATen's log-domain kernels share."""

    maximum = mx.where(left < right, right, left)
    body = maximum + mx.log1p(mx.exp(-mx.abs(left - right)))
    body = mx.where(mx.isinf(maximum), maximum, body)
    return mx.where(mx.isnan(left) | mx.isnan(right), float("nan"), body)


def _mlx_fmax(left: mx.array, right: mx.array) -> mx.array:
    """C99 ``fmax`` on MLX, mirroring the Metal helper ``tsc_fmax``.

    MLX's ``maximum`` propagates NaN, so the NaN cases are peeled off first and
    a pair of zeros is summed, which is exact and lands on the sign ATen
    reports: +0 unless both operands are -0.
    """

    selected = mx.where(left > right, left, right)
    zeros = (left == 0.0) & (right == 0.0)
    selected = mx.where(zeros, left + right, selected)
    return mx.where(mx.isnan(left), right, mx.where(mx.isnan(right), left, selected))


def _mlx_fmin(left: mx.array, right: mx.array) -> mx.array:
    """C99 ``fmin`` on MLX, mirroring the Metal helper ``tsc_fmin``."""

    selected = mx.where(left < right, left, right)
    zeros = (left == 0.0) & (right == 0.0)
    selected = mx.where(zeros, -((-left) + (-right)), selected)
    return mx.where(mx.isnan(left), right, mx.where(mx.isnan(right), left, selected))


# The integer view that shares a width with each floating dtype, so the bits of
# a value can be stepped by one to get its neighbour.
_NEXTAFTER_BITS = {
    mx.float32: mx.int32,
    mx.float16: mx.int16,
    mx.bfloat16: mx.int16,
}


def _mlx_nextafter(value: mx.array, target: mx.array) -> mx.array:
    """``nextafter`` on MLX, mirroring the Metal helper ``tsc_nextafter``.

    MLX has no binding for it, so the IEEE-754 bit pattern is stepped by one in
    the direction of the target: the magnitude grows when ``|target| > |value|``
    and shrinks otherwise, which covers both signs with one rule.
    """

    bits_dtype = _NEXTAFTER_BITS.get(value.dtype)
    if bits_dtype is None:
        raise ValueError(f"nextafter is not implemented for {value.dtype}")
    magnitude = mx.abs(value)
    # The smallest subnormal of this dtype, as the value of bit pattern 1.
    smallest = mx.view(mx.array([1], dtype=bits_dtype), value.dtype).reshape(())
    step = mx.where(mx.abs(target) > magnitude, 1, -1)
    neighbour = mx.view(mx.view(magnitude, bits_dtype) + step, value.dtype)
    signed = mx.where(_mlx_signbit(value), -neighbour, neighbour)
    from_zero = mx.where(_mlx_signbit(target), -smallest, smallest)
    result = mx.where(value == 0.0, from_zero, signed)
    return mx.where(value == target, target, result)


def _mlx_trunc_divide(left: mx.array, right: mx.array) -> mx.array:
    """Division truncated towards zero, mirroring ``tsc_trunc_div_*``."""

    if _MLX_DTYPE_TO_DTYPE.get(left.dtype) in _FLOATING_DTYPES:
        return mx.trunc(left / right)
    # Integer division in C truncates, but MLX's floor_divide rounds down, so a
    # negative quotient with a non-zero remainder has to be stepped back up.
    quotient = mx.floor_divide(left, right)
    remainder = mx.remainder(left, right)
    inexact = (remainder != 0) & ((left < 0) != (right < 0))
    return quotient + mx.where(inexact, 1, 0).astype(quotient.dtype)


def _mlx_gcd(left: mx.array, right: mx.array) -> mx.array:
    """Greatest common divisor of the magnitudes, mirroring ``tsc_gcd_*``."""

    _, signed = _UNSIGNED_VIEWS[left.dtype]
    divisor, _, _ = _euclid(left, right)
    return mx.view(divisor, signed)


def _mlx_lcm(left: mx.array, right: mx.array) -> mx.array:
    """Least common multiple of the magnitudes, mirroring ``tsc_lcm_*``.

    ``|a| / gcd * b`` is formed in the operand's own width and then made
    positive, which is the order ATen uses: the product wraps first, so an
    overflowing overflow pair lands on the same value as the reference kernel.
    """

    _, signed = _UNSIGNED_VIEWS[left.dtype]
    divisor, left_bits, right_bits = _euclid(left, right)
    zero = divisor == 0
    safe = mx.where(zero, mx.ones_like(divisor), divisor)
    quotient = mx.view(mx.floor_divide(left_bits, safe), signed)
    product = quotient * right
    return mx.where(
        zero, mx.zeros_like(product), mx.view(_magnitude_bits(product), signed)
    )


# The unsigned view of each integer dtype: Euclid runs on bit patterns so that
# the most negative value, whose magnitude does not fit the signed type, is
# still handled the way the two's-complement MSL helpers handle it.
_UNSIGNED_VIEWS = {
    mx.int8: (mx.uint8, mx.int8),
    mx.uint8: (mx.uint8, mx.uint8),
    mx.int16: (mx.uint16, mx.int16),
    mx.uint16: (mx.uint16, mx.uint16),
    mx.int32: (mx.uint32, mx.int32),
    mx.uint32: (mx.uint32, mx.uint32),
    mx.int64: (mx.uint64, mx.int64),
    mx.uint64: (mx.uint64, mx.uint64),
}


def _magnitude_bits(value: mx.array) -> mx.array:
    """``|value|`` as the unsigned view of the same width."""

    unsigned, _ = _UNSIGNED_VIEWS[value.dtype]
    bits = mx.view(value, unsigned)
    zero = mx.zeros_like(bits)
    return mx.where(value < 0, zero - bits, bits)


def _euclid(left: mx.array, right: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Euclid over unsigned magnitudes: ``(gcd, |left|, |right|)``."""

    a = _magnitude_bits(left)
    b = _magnitude_bits(right)
    left_bits, right_bits = a, b
    while bool(mx.any(b != 0).item()):
        nonzero = b != 0
        safe = mx.where(nonzero, b, mx.ones_like(b))
        remainder = mx.where(nonzero, mx.remainder(a, safe), mx.zeros_like(b))
        # A lane whose divisor is already zero keeps its dividend: that value is
        # the answer, and swapping unconditionally would drop it.
        a, b = mx.where(nonzero, b, a), remainder
    return a, left_bits, right_bits


_SHIFT_WIDTHS = {
    mx.int8: 8,
    mx.uint8: 8,
    mx.int16: 16,
    mx.uint16: 16,
    mx.int32: 32,
    mx.uint32: 32,
    mx.int64: 64,
    mx.uint64: 64,
}


def _mlx_shift(operation: str, left: mx.array, right: mx.array) -> mx.array:
    """``bitwise_left_shift``/``bitwise_right_shift``, mirroring ``tsc_shl_*``.

    ATen defines an out-of-range count to shift every bit out, so the left shift
    yields zero and the right shift yields the sign fill. MLX inherits the
    undefined behaviour of the hardware shift, so the count is clamped before
    the shift and the out-of-range lanes are overwritten afterwards.
    """

    width = _SHIFT_WIDTHS.get(left.dtype)
    if width is None:
        raise ValueError(f"bitwise shift is not implemented for {left.dtype}")
    zero = mx.zeros_like(right)
    in_range = (right >= zero) & (right < mx.full(right.shape, width, right.dtype))
    clamped = mx.minimum(
        mx.maximum(right, zero), mx.full(right.shape, width - 1, right.dtype)
    )
    if operation == "bitwise_left_shift":
        shifted = mx.left_shift(left, clamped)
        return mx.where(in_range, shifted, mx.zeros_like(left))
    shifted = mx.right_shift(left, clamped)
    fill = mx.where(
        left < mx.zeros_like(left), -mx.ones_like(left), mx.zeros_like(left)
    )
    return mx.where(in_range, shifted, fill)
