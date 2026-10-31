"""The Metal source every kernel starts from, and the compiled artifact.

A kernel is a preamble plus a body: the preamble declares the dtype aliases,
the checked index arithmetic and the small vector helpers that every family
would otherwise repeat, and ``_metal_kernel`` compiles the concatenation once
per key and caches it. ``KernelArtifact`` is what survives compilation - the
source, the argument order, and the ids the kernel writes - so a plan can be
serialised without the compiler in the picture.
"""

from __future__ import annotations

import mlx.core as mx
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from ..ir import TensorType

_MSL_HEADER = """
inline float tsc_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = metal::abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * metal::exp(-x * x));
}

inline float tsc_erfc(float value) {
    // Abramowitz-Stegun 7.1.26, the same approximation tsc_erf uses. MSL has
    // no erfc in the metal namespace, and `1 - tsc_erf(x)` would cancel away
    // most of the digits in the tail.
    float x = metal::abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    float tail = polynomial * metal::exp(-x * x);
    return value < 0.0f ? 2.0f - tail : tail;
}

inline float tsc_expm1(float value) {
    if (metal::abs(value) <= 0.5f) {
        return value * (1.0f + value * (0.5f + value * (
            0.16666666666666666f + value * (0.041666666666666664f + value * (
            0.008333333333333333f + value * (
            0.001388888888888889f + value * (
            0.0001984126984126984f + value * 0.0000248015873015873f)))))));
    }
    return metal::exp(value) - 1.0f;
}

inline float tsc_softplus(float value, float beta, float threshold) {
    float scaled = beta * value;
    return scaled > threshold
        ? value
        : log1p(metal::exp(scaled)) / beta;
}

inline float tsc_logsigmoid(float value) {
    if (value != value) {
        return value;
    }
    return metal::min(value, 0.0f)
        - log1p(metal::exp(-metal::abs(value)));
}

inline float tsc_logaddexp(float left, float right) {
    if (metal::isnan(left) || metal::isnan(right)) {
        return NAN;
    }
    float maximum = left < right ? right : left;
    if (metal::isinf(maximum)) {
        return maximum;
    }
    return maximum + log1p(metal::exp(-metal::abs(left - right)));
}

inline float tsc_atan2(float y, float x) {
    constexpr float pi = 3.14159265358979323846f;
    if (metal::isnan(y) || metal::isnan(x)) {
        return NAN;
    }
    bool x_negative = (as_type<uint>(x) & 0x80000000u) != 0u;
    if (metal::isinf(y)) {
        if (metal::isinf(x)) {
            float magnitude = x_negative ? 0.75f * pi : 0.25f * pi;
            return metal::copysign(magnitude, y);
        }
        return metal::copysign(0.5f * pi, y);
    }
    if (metal::isinf(x)) {
        return x_negative
            ? metal::copysign(pi, y)
            : metal::copysign(0.0f, y);
    }
    if (x == 0.0f) {
        if (y == 0.0f) {
            return x_negative
                ? metal::copysign(pi, y)
                : metal::copysign(0.0f, y);
        }
        return metal::copysign(0.5f * pi, y);
    }
    return atan2(y, x);
}

inline float tsc_mish(float value) {
    return value > 20.0f
        ? value
        : value * metal::tanh(tsc_softplus(value, 1.0f, 20.0f));
}

inline float tsc_floor_div_f32(float dividend, float divisor) {
    return metal::floor(dividend / divisor);
}

inline float tsc_remainder_f32(float dividend, float divisor) {
    float mod = metal::fmod(dividend, divisor);
    if (mod != 0.0f && ((divisor < 0.0f) != (mod < 0.0f))) {
        mod += divisor;
    }
    return mod;
}

inline float tsc_lerp_f32(float start, float end, float weight) {
    // ATen's lerp picks the branch with the smaller cancellation error.
    return metal::abs(weight) < 0.5f
        ? start + weight * (end - start)
        : end - (end - start) * (1.0f - weight);
}

inline float tsc_sinc(float value) {
    float scaled = 3.14159265358979323846f * value;
    if (metal::abs(value) < 1.0e-4f) {
        // Metal's sin can flush an argument this small to zero, which would
        // turn sinc(1e-8) into 0 instead of the 1 ATen returns. The series is
        // exact to well below one ulp over this window.
        float square = scaled * scaled;
        return 1.0f - square / 6.0f + square * square / 120.0f;
    }
    return metal::sin(scaled) / scaled;
}

inline float tsc_i0(float value) {
    // Abramowitz-Stegun 9.8.1 and 9.8.2, split at 3.75. MSL has no Bessel
    // function, and both branches are plain polynomials, so the peak relative
    // error is 1.1e-6 - about ten float ulps over the whole range.
    float x = metal::abs(value);
    if (x < 3.75f) {
        float ratio = x / 3.75f;
        float square = ratio * ratio;
        return 1.0f + square * (3.5156229f + square * (3.0899424f
            + square * (1.2067492f + square * (0.2659732f
            + square * (0.0360768f + square * 0.0045813f)))));
    }
    // The scaled form keeps the leading exp out of the polynomial and cannot
    // overflow before ``exp`` itself does.
    float ratio = 3.75f / x;
    float polynomial = 0.39894228f + ratio * (0.01328592f + ratio * (0.00225319f
        + ratio * (-0.00157565f + ratio * (0.00916281f + ratio * (-0.02057706f
        + ratio * (0.02635537f + ratio * (-0.01647633f
        + ratio * 0.00392377f)))))));
    return metal::exp(x) / metal::sqrt(x) * polynomial;
}

inline float tsc_lgamma_series(float value) {
    // Lanczos g=7, N=9 (the usual published coefficient set). Metal has no
    // tgamma/lgamma, and MSL forbids recursion, so the reflection branch below
    // calls this helper instead of calling tsc_lgamma again.
    float z = value - 1.0f;
    float shifted = z + 7.5f;
    float series = 0.99999999999980993f
        + 676.5203681218851f / (z + 1.0f)
        + -1259.1392167224028f / (z + 2.0f)
        + 771.32342877765313f / (z + 3.0f)
        + -176.61502916214059f / (z + 4.0f)
        + 12.507343278686905f / (z + 5.0f)
        + -0.13857109526572012f / (z + 6.0f)
        + 9.9843695780195716e-6f / (z + 7.0f)
        + 1.5056327351493116e-7f / (z + 8.0f);
    return 0.9189385332046727f + (z + 0.5f) * metal::log(shifted)
        - shifted + metal::log(series);
}

inline float tsc_lgamma(float value) {
    // ``lgamma`` is log|Gamma|, so negative arguments go through the
    // reflection formula rather than a separate series.
    const float pi = 3.14159265358979323846f;
    if (metal::isnan(value)) {
        return value;
    }
    if (metal::isinf(value)) {
        // ATen reports +inf for both infinities: |Gamma| grows without bound
        // towards either end of the real line.
        return INFINITY;
    }
    if (value <= 0.0f && value == metal::rint(value)) {
        // The poles at zero and the negative integers. sin is exactly zero
        // there, so the reflection branch below would divide by log(0).
        return INFINITY;
    }
    if (value < 0.5f) {
        return metal::log(pi) - metal::log(metal::abs(metal::sin(pi * value)))
            - tsc_lgamma_series(1.0f - value);
    }
    return tsc_lgamma_series(value);
}

inline float tsc_digamma_series(float value) {
    // ATen's calc_digamma: walk the argument up to 10 accumulating -1/x, then
    // either use the precomputed PSI(10) or the asymptotic expansion. MSL
    // forbids recursion, so the reflection branch calls this instead of
    // calling tsc_digamma again.
    float accumulated = 0.0f;
    float x = value;
    while (x < 10.0f) {
        accumulated -= 1.0f / x;
        x += 1.0f;
    }
    if (x == 10.0f) {
        return accumulated + 2.25175258906672110764f;
    }
    float inverse_square = 1.0f / (x * x);
    float series = inverse_square * (8.33333333333333333333E-2f
        + inverse_square * (-2.10927960927960927961E-2f
        + inverse_square * (7.57575757575757575758E-3f
        + inverse_square * (-4.16666666666666666667E-3f
        + inverse_square * (3.96825396825396825397E-3f
        + inverse_square * (-8.33333333333333333333E-3f
        + inverse_square * 8.33333333333333333333E-2f))))));
    return accumulated + metal::log(x) - 0.5f / x - series;
}

inline float tsc_digamma(float value) {
    const float pi = 3.14159265358979323846f;
    if (metal::isnan(value)) {
        return value;
    }
    if (value == 0.0f) {
        // ATen returns the infinity whose sign is the opposite of the zero's,
        // so +0 maps to -inf and -0 to +inf.
        return metal::copysign(INFINITY, -value);
    }
    if (value < 0.0f) {
        if (value == metal::trunc(value)) {
            // The poles at the negative integers; ATen reports NaN there
            // rather than an infinity, because the reflection terms disagree.
            return NAN;
        }
        // Reflection: psi(x) = psi(1 - x) - pi * cot(pi x). The fractional part
        // is reduced to [-0.5, 0.5] by subtracting the nearest integer, which
        // is exact by Sterbenz' lemma; evaluating tan(pi * r) at a fractional
        // part close to +-1 is not. cot is written as cos/sin rather than
        // 1/tan so the error stays relative at both ends of the reduced range.
        float reduced = value - metal::rint(value);
        return tsc_digamma_series(1.0f - value)
            - pi * metal::cos(pi * reduced) / metal::sin(pi * reduced);
    }
    return tsc_digamma_series(value);
}

inline float tsc_xlogy_f32(float value, float other) {    // ATen only short-circuits a zero left operand, and only when the right
    // operand is not NaN, so 0 * log(-1) is 0 while 0 * log(NaN) stays NaN.
    if (value == 0.0f && !metal::isnan(other)) {
        return 0.0f;
    }
    return value * metal::log(other);
}

inline int tsc_neg_i32(int value) {
    return (int)(0u - (uint)value);
}

inline long tsc_neg_i64(long value) {
    return (long)((unsigned long)0 - (unsigned long)value);
}

inline int tsc_add_i32(int left, int right) {
    return (int)((uint)left + (uint)right);
}

inline long tsc_add_i64(long left, long right) {
    return (long)((unsigned long)left + (unsigned long)right);
}

inline int tsc_sub_i32(int left, int right) {
    return (int)((uint)left - (uint)right);
}

inline long tsc_sub_i64(long left, long right) {
    return (long)((unsigned long)left - (unsigned long)right);
}

inline int tsc_mul_i32(int left, int right) {
    return (int)((uint)left * (uint)right);
}

inline long tsc_mul_i64(long left, long right) {
    return (long)((unsigned long)left * (unsigned long)right);
}

inline int tsc_floor_div_i32(int dividend, int divisor) {
    if (divisor == -1) {
        return tsc_neg_i32(dividend);
    }
    int quotient = dividend / divisor;
    int remainder = dividend % divisor;
    bool round_down = remainder != 0 && ((remainder < 0) != (divisor < 0));
    return quotient - (round_down ? 1 : 0);
}

inline long tsc_floor_div_i64(long dividend, long divisor) {
    if (divisor == -1L) {
        return tsc_neg_i64(dividend);
    }
    long quotient = dividend / divisor;
    long remainder = dividend % divisor;
    bool round_down = remainder != 0 && ((remainder < 0) != (divisor < 0));
    return quotient - (round_down ? 1L : 0L);
}

inline int tsc_remainder_i32(int dividend, int divisor) {
    if (divisor == -1) {
        return 0;
    }
    int remainder = dividend % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        remainder += divisor;
    }
    return remainder;
}

inline long tsc_remainder_i64(long dividend, long divisor) {
    if (divisor == -1L) {
        return 0L;
    }
    long remainder = dividend % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        remainder += divisor;
    }
    return remainder;
}

inline int tsc_pow_i32(int base, int exponent) {
    if (exponent < 0) {
        return 0;
    }
    int result = 1;
    int factor = base;
    unsigned remaining = (unsigned)exponent;
    while (remaining != 0u) {
        if ((remaining & 1u) != 0u) {
            result = tsc_mul_i32(result, factor);
        }
        factor = tsc_mul_i32(factor, factor);
        remaining >>= 1u;
    }
    return result;
}

inline long tsc_pow_i64(long base, long exponent) {
    if (exponent < 0L) {
        return 0L;
    }
    long result = 1L;
    long factor = base;
    unsigned long remaining = (unsigned long)exponent;
    while (remaining != 0UL) {
        if ((remaining & 1UL) != 0UL) {
            result = tsc_mul_i64(result, factor);
        }
        factor = tsc_mul_i64(factor, factor);
        remaining >>= 1UL;
    }
    return result;
}
"""


_MPP_HEADER = (
    _MSL_HEADER
    + "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
)


_METAL_KERNEL_CACHE_SIZE = 256


@lru_cache(maxsize=_METAL_KERNEL_CACHE_SIZE)
def _metal_kernel(
    name: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    source: str,
    header: str,
) -> Any:
    """Reuse an mx.fast.metal_kernel for an identical static kernel spec."""

    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        source=source,
        header=header,
    )


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    """One generated kernel and the IR values it writes.

    Most kernels write a single tensor. A kernel that already computes several
    results writes them together and lists every output here, which is what
    lets a multi-output IR operation stay one kernel instead of being split
    into a producer plus a re-derivation of its siblings.
    """

    name: str
    source: str
    header: str
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    tensor_types: tuple[TensorType, ...]
    kind: str

    def __post_init__(self) -> None:
        if not self.output_ids:
            raise ValueError(f"kernel {self.name} must write at least one value")
        if len(self.output_ids) != len(self.tensor_types):
            raise ValueError(
                f"kernel {self.name} lists {len(self.output_ids)} outputs but "
                f"{len(self.tensor_types)} types"
            )

    @property
    def output_id(self) -> str:
        """The only output of a single-output kernel."""

        if len(self.output_ids) != 1:
            raise ValueError(f"kernel {self.name} writes {len(self.output_ids)} values")
        return self.output_ids[0]

    @property
    def tensor_type(self) -> TensorType:
        """The type of a single-output kernel's only output."""

        if len(self.tensor_types) != 1:
            raise ValueError(
                f"kernel {self.name} writes {len(self.tensor_types)} values"
            )
        return self.tensor_types[0]
