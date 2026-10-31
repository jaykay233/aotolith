"""ATen decompositions that widen what the frontend can lower.

``capture_fx`` traces with ``make_fx``, which accepts a decomposition table:
a mapping from an ATen operator to a function that rewrites it into smaller
operators *during* tracing. Torch ships 941 such rules in
``core_aten_decompositions``, which is most of the long tail the frontend
would otherwise have to reject one operator at a time - ``lgamma``,
``digamma``, ``i0``, ``aminmax``, ``celu``, ``erfc``, ``nanmean``, and the
``out=`` and gradient spellings of operators that already have a forward
lowering.

The FFT family is the exception worth knowing about: it does decompose, but
into complex primitives, and the IR has no complex dtype, so ``torch.fft``
still refuses - at the dtype rather than at the operator.

Two reasons to use it rather than hand-writing every lowering:

* The rules are maintained upstream, so an operator that PyTorch itself
  redefines keeps working here without a change.
* A decomposition is expressed in operators the frontend already lowers, so
  the result goes through the same layout simplification, pointwise fusion
  and dtype promotion. There is no second code path to keep correct.

Which operators keep their own lowering is the *caller's* decision, passed in
as ``keep``. This module deliberately does not know the frontend's operator
table: the table is a fact about legalization, and reaching back for it would
make tracing and legalization import each other. The legalizer already owns the
list (``legalize.targets.NATIVE_TARGETS``), so the caller passes it in and the
dependency points one way.

For the operators that are kept, a decomposition would be a regression rather
than a shortcut: it replaces a dedicated Metal kernel (``attention``,
``layer_norm``, ``softmax``), an exact emulation (``arange`` rebuilds its float
sequence from the fma residual and a two-sum so it rounds once, the way
PyTorch's double expression does), or a lowering whose attribute normalisation
the rest of the compiler relies on (``sum``/``var`` spell their axes and
``keepdim`` differently in each schema). A test asserts the two sets stay
disjoint.

The table is Torch's on top of a handful of rules of our own: the normalizations
have no upstream rule because every backend implements them as one kernel, and
expressing them in operators this frontend already lowers costs nothing (see
below).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A decomposition function takes an FX node's arguments and emits operations
# while tracing, so its signature is only meaningful to ``make_fx``.
Decomposition = Callable[..., Any]


# ---------------------------------------------------------------------------
# The normalizations Torch leaves to whoever runs them
# ---------------------------------------------------------------------------


# ``native_batch_norm`` and ``native_group_norm`` ship without a rule upstream:
# every backend implements them as one fused kernel, so ``core_aten_decompositions``
# leaves them alone. Here there is no kernel to protect - the moments, the
# reciprocal standard deviation and the affine fix-up are all operators the
# frontend already lowers - so they are written out below and the surrounding
# pointwise fusion absorbs them.


def _moments(value: Any, dimensions: tuple[int, ...]) -> tuple[Any, Any]:
    """The batch moments ATen's normalization kernels compute, unbiased."""

    return (
        value.mean(dim=dimensions, keepdim=True),
        value.var(dim=dimensions, unbiased=False, keepdim=True),
    )


def _channel_parameter(parameter: Any, channels: int, rank: int) -> Any:
    """A per-channel parameter reshaped to broadcast over the channel axis."""

    return parameter.reshape([1, channels, *([1] * (rank - 2))])


def _compute_dtype(dtype: Any) -> Any:
    """The dtype ATen accumulates the moments of ``dtype`` in.

    Half precision reduces in float32, which is why a decomposition that keeps
    the input dtype would lose accuracy the operator does not: ``batch_norm`` on
    a ``float16`` tensor is not a ``float16`` reduction.
    """

    import torch

    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _empty_statistics(moments: Any, dtype: Any) -> Any:
    """The zero-length statistic an inference-mode ``batch_norm`` reports.

    ATen only computes the training moments; in inference it returns empty
    tensors in their place. Slicing the moment the caller already has gives a
    real value of that shape, and a graph that never reads it costs nothing
    once the unreachable operations are dropped.
    """

    return moments.reshape(-1)[:0].to(dtype)


def _batch_norm(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    training: bool,
    momentum: Any,
    eps: float,
) -> tuple[Any, Any, Any]:
    """``native_batch_norm`` over the channel axis, without its running update."""

    import torch

    channels = input.shape[1]
    rank = input.dim()
    dtype = _compute_dtype(input.dtype)
    value = input.to(dtype)
    shape = [1, channels, *([1] * (rank - 2))]
    if training:
        if running_mean is not None or running_var is not None:
            raise NotImplementedError(
                "batch_norm in training mode updates the running moments, which "
                "are bound as immutable inputs here. Compile the module in eval "
                "mode, or construct it with track_running_stats=False, so the "
                "normalization has no state to write"
            )
        mean, variance = _moments(value, (0, *range(2, rank)))
    elif running_mean is None or running_var is None:
        mean, variance = _moments(value, (0, *range(2, rank)))
    else:
        # Both statistics are already per-channel, so both broadcast against
        # the input exactly like the affine parameters do.
        mean = running_mean.to(dtype).reshape(shape)
        variance = running_var.to(dtype).reshape(shape)
    inv_std = torch.rsqrt(variance + eps)
    normalized = (value - mean) * inv_std
    if weight is not None:
        normalized = normalized * _channel_parameter(weight.to(dtype), channels, rank)
    if bias is not None:
        normalized = normalized + _channel_parameter(bias.to(dtype), channels, rank)
    output = normalized.to(input.dtype)
    statistics = (channels,) if training else None
    if statistics is None:
        empty = _empty_statistics(mean, input.dtype)
        return output, empty, empty
    return (
        output,
        mean.reshape(statistics).to(input.dtype),
        inv_std.reshape(statistics).to(input.dtype),
    )


def _batch_norm_without_statistics(
    input: Any, weight: Any, bias: Any, training: bool, momentum: Any, eps: float
) -> tuple[Any, Any, Any]:
    """The ``no_stats`` spelling, which has no running moments to write even
    when it is normalizing: it only ever normalizes with the batch moments."""

    if not training:
        raise NotImplementedError(
            "batch_norm without running statistics needs training=True to have "
            "moments to normalize with"
        )
    return _batch_norm(input, weight, bias, None, None, True, momentum, eps)


def _batch_norm_no_training(
    input: Any,
    weight: Any,
    bias: Any,
    running_mean: Any,
    running_var: Any,
    momentum: Any,
    eps: float,
) -> tuple[Any, Any, Any]:
    """The ``no_training`` spelling: the running moments, never updated."""

    return _batch_norm(
        input, weight, bias, running_mean, running_var, False, momentum, eps
    )


def _group_norm(
    input: Any,
    weight: Any,
    bias: Any,
    batch: Any,
    channels: Any,
    spatial: Any,
    groups: int,
    eps: float,
) -> tuple[Any, Any, Any]:
    """``native_group_norm``: the channel axis split into ``groups`` rows.

    Splitting the channels into a group axis and flattening what is left of each
    group turns the normalization into the same row reduction layer norm is, and
    the size arguments ATen passes are ignored in favour of the traced shape,
    which is the one the rest of the graph is typed against.
    """

    import torch

    channels = input.shape[1]
    rank = input.dim()
    batch = input.shape[0]
    rows = channels // groups
    for value, name in ((groups, "groups"), (rows, "channels")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"group_norm.{name} must be a positive integer")
    if channels % groups:
        raise ValueError(
            f"group_norm groups {groups} does not divide {channels} channels"
        )
    spatial = 1
    for dimension in input.shape[2:]:
        spatial *= dimension

    dtype = _compute_dtype(input.dtype)
    value = input.to(dtype)
    grouped = value.reshape([batch, groups, rows * spatial])
    mean, variance = _moments(grouped, (2,))
    inv_std = torch.rsqrt(variance + eps)
    normalized = ((grouped - mean) * inv_std).reshape(input.shape)
    if weight is not None:
        normalized = normalized * _channel_parameter(weight.to(dtype), channels, rank)
    if bias is not None:
        normalized = normalized + _channel_parameter(bias.to(dtype), channels, rank)
    output = normalized.to(input.dtype)
    statistics = (batch, groups)
    return (
        output,
        mean.reshape(statistics).to(input.dtype),
        inv_std.reshape(statistics).to(input.dtype),
    )


def _backend_normalizations() -> dict[object, Decomposition]:
    """The rules this module adds to Torch's table.

    Three spellings of the same batch normalization share one implementation
    because they differ only in which state they receive: ``native_batch_norm``
    is the functional entry point, ``_native_batch_norm_legit`` the one whose
    running moments are declared mutable, and ``_native_batch_norm_legit_no_training``
    the inference-only one. A backend picks between them by device.
    """

    import torch

    aten = torch.ops.aten
    return {
        aten.native_batch_norm.default: _batch_norm,
        aten._native_batch_norm_legit.default: _batch_norm,
        aten._native_batch_norm_legit.no_stats: _batch_norm_without_statistics,
        aten._native_batch_norm_legit_no_training.default: _batch_norm_no_training,
        aten.native_group_norm.default: _group_norm,
    }


# Keyed by the keep set, since the table is a function of it. Building the
# table walks Torch's registries and ``asdict``-copies each rule, which is far
# more expensive than the compile that follows it.
_CACHE: dict[frozenset[str], dict[object, Decomposition]] = {}


def aten_decompositions(
    keep: frozenset[str] = frozenset(),
) -> dict[object, Decomposition]:
    """Torch's core decomposition table minus the targets in ``keep``.

    ``keep`` holds the ATen target names, as strings, whose own lowering is
    better than the decomposition: a fused kernel, an exact emulation, or a
    normalization the rest of the compiler depends on.
    """

    cached = _CACHE.get(keep)
    if cached is not None:
        return cached

    from torch._decomp import core_aten_decompositions

    table: dict[object, Decomposition] = {
        target: rule
        for target, rule in core_aten_decompositions().items()
        if str(target) not in keep
    }
    table.update(
        {
            target: rule
            for target, rule in _backend_normalizations().items()
            if str(target) not in keep
        }
    )
    _CACHE[keep] = table
    return table
