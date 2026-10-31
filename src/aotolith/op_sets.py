"""How each IR operator executes, independent of who is asking.

An operator's execution class is a property of the operator itself, not of the
layer that first needed to know it: ``silu`` is elementwise whether a fusion
pass is deciding whether it may join a region, an IR contract is choosing a
default dtype rule, or the backend is picking a kernel family. Keeping these
sets inside one of those consumers forces every other consumer to import it,
which is how the contract table ended up importing the graph rewriter purely to
read a set of names.

So the classification lives here, at the bottom of the stack, next to nothing
else. The three users are:

* ``ir.ops`` - every pointwise operator shares one contract shape, and the
  constant producers share one attribute rule.
* ``passes`` - fusion may only absorb an operator that runs elementwise, and
  layout folding only moves pure views.
* ``backends`` - the codegen and the MLX fallback both dispatch on the same
  three classes.

Membership rules worth stating, because they are not obvious from the names:

* ``where`` and ``masked_fill`` are pointwise: they are elementwise selects
  that fuse into the surrounding region like any arithmetic.
* ``scalar_tensor`` and the ``zeros``/``ones``/``full`` family are pointwise
  as well, because a broadcast constant is exactly an elementwise operand.
  They are also collected separately in ``CONSTANT_OPS``: an operator can be
  pointwise in *fusion* terms and still need a different contract and a
  different kernel, and the codegen keys off that second set.
* ``pad`` is a layout operation in spirit but not in this set: it allocates
  rather than viewing, so layout *folding* must not move it. The backend adds
  it to its own runtime set where a copy is acceptable.
"""

from __future__ import annotations

# Elementwise operators: one output element per input element, no axis, no
# cross-lane communication. Fusion may absorb them into a neighbouring region.
POINTWISE_OPS = frozenset(
    {
        # Activations and their fused-loss counterparts.
        "silu",
        "gelu",
        "softplus",
        "mish",
        "logsigmoid",
        "relu",
        "leaky_relu",
        "elu",
        "hardswish",
        "hardsigmoid",
        "threshold",
        # Arithmetic and rounding.
        "mul",
        "add",
        "sub",
        "rsub",
        "div",
        "floor_divide",
        "trunc_divide",
        "remainder",
        "fmod",
        "lerp",
        "addcmul",
        "addcdiv",
        "pow",
        "neg",
        "abs",
        "square",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "floor",
        "ceil",
        "round",
        "trunc",
        "sign",
        "clamp",
        "clamp_min",
        "clamp_max",
        "maximum",
        "minimum",
        "fmax",
        "fmin",
        # Shift counts make these two's-complement integer operations, but they
        # are otherwise ordinary binary pointwise calls.
        "bitwise_left_shift",
        "bitwise_right_shift",
        # Transcendental functions, including the helpers the backend has to
        # implement itself because MSL has no equivalent.
        "sigmoid",
        "tanh",
        "exp",
        "expm1",
        "exp2",
        "log",
        "log2",
        "log10",
        "log1p",
        "erf",
        "erfc",
        "lgamma",
        "digamma",
        "i0",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "logit",
        "frac",
        "sinc",
        "hypot",
        "copysign",
        "xlogy",
        "logaddexp",
        "atan2",
        "heaviside",
        "nan_to_num",
        "nextafter",
        "gcd",
        "lcm",
        # Comparisons and masks. A comparison result is a boolean tensor that
        # feeds the region like any other intermediate.
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "isnan",
        "isinf",
        "signbit",
        "logical_and",
        "logical_or",
        "logical_xor",
        "logical_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "where",
        "masked_fill",
        # Broadcast constants.
        "scalar_tensor",
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
    }
)

# Pure views: the output aliases the input's storage under a changed shape or
# stride order. Folding may reorder or drop them, so they must never carry
# arithmetic and must never allocate.
LAYOUT_OPS = frozenset(
    {
        "reshape",
        "transpose",
        "permute",
        "expand",
        "slice",
        "flip",
        "roll",
        "unsqueeze",
        "squeeze",
        "unfold",
        "contiguous",
        "alias",
    }
)

# Operators that produce a tensor from attributes alone, reading no graph
# values. They need a contract that permits zero operands and a kernel that
# writes a buffer without reading one.
CONSTANT_OPS = frozenset(
    {
        "scalar_tensor",
        "zeros",
        "ones",
        "full",
        "new_zeros",
        "new_ones",
        "new_full",
        "zeros_like",
        "ones_like",
        "full_like",
    }
)
