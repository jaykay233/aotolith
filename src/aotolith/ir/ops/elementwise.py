"""Elementwise operators: one output element per input element.

``CAST``, ``TRIU``/``TRIL`` and ``CAT`` are shape-simple but need a hook, so they
live here with the pointwise defaults they are shaped like. Everything in
``POINTWISE_OPS`` shares one contract - broadcast operands, free shape, any
dtype - and the per-operator dtype restrictions live with the emitter that has
to express them in Metal, because those are about expressibility rather than
about the operation's own metadata.
"""

from __future__ import annotations

from .. import Graph, Operation
from .rules import (
    AttrKind,
    AttrSpec,
    DtypeRule,
    OpContract,
    ShapeRule,
    normalize_axis,
)


def _check_cast(graph: Graph, op: Operation) -> None:
    input_shape = graph.value(op.inputs[0]).type.shape
    output_shape = graph.value(op.outputs[0]).type.shape
    if input_shape != output_shape:
        raise ValueError(f"cast shape {input_shape} != {output_shape}")


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


CONTRACTS: dict[str, OpContract] = {
    # ``erfinv`` is elementwise but deliberately not in ``POINTWISE_OPS``: that
    # set means "emittable as an MSL expression", and this one stays an MLX
    # call because PyTorch refines it with Newton steps against an accurate
    # ``erf`` that Metal's standard library does not have. It still needs a
    # contract, otherwise the backend's validation loop skips it by name and a
    # malformed ``erfinv`` reaches the runtime unchecked.
    "erfinv": OpContract(
        arity=(1, 1),
        rank=(0, 8),
        dtype=DtypeRule.ANY,
        shape=ShapeRule.FREE,
        native=False,
    ),
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
    "cat": OpContract(
        arity=(1, 64),
        shape=ShapeRule.FREE,
        attrs=(AttrSpec("dim", AttrKind.INT),),
        extra=_check_cat,
    ),
}
