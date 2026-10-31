"""Reading a traced FX graph: what a node produces, and what it baked in.

Nothing here emits an operation. These helpers answer the questions the lowering
asks about the graph ``make_fx`` handed it - the tensor type of a node, the value
of a tensor it captured as an attribute, which nodes are zero-length constants,
and which graphs are rejectable outright.

They live beside the legalizer rather than beside the tracer because that is who
reads them: the tracer's job ends when it returns a graph, and the one-way
dependency ``capture -> legalize`` is what keeps the two from importing each
other.

The zero-length handling is worth calling out: ``use_cache=True`` models open
their KV cache from empty tensors, which tracing records as constants plus
identity copies. They carry no arithmetic and are legal only as a concatenation
operand, where a zero-length piece is a no-op, so they are collected here and
folded away by the legalizer when it reaches the concatenation.
"""

from __future__ import annotations


from ..ir import TensorType
from .refs import _argument_ids
from .targets import _TORCH_DTYPES


def _node_type(node: object) -> TensorType:
    meta = getattr(node, "meta", {})
    tensor = meta.get("val")
    if tensor is None:
        raise ValueError(f"missing concrete shape/dtype metadata for FX node: {node}")
    return _tensor_type(tensor, f"FX node {node}")


def _tensor_type(tensor: object, context: str) -> TensorType:
    if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
        raise ValueError(f"missing concrete shape/dtype metadata for {context}")

    dtype = _TORCH_DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(
            f"unsupported dtype: {tensor.dtype} in {context}. The backend speaks "
            "float16, bfloat16, float32, bool and the integer types. A float64 "
            "tensor usually comes from a decomposition rather than the model - "
            "quantile, nanquantile and matrix_rank build float64 constants out "
            "of a float32 input - so those operators refuse here even though no "
            "input asked for float64"
        )
    if any(dim <= 0 for dim in tensor.shape):
        raise ValueError(
            f"dynamic or empty shapes are not supported yet: {tuple(tensor.shape)}"
        )
    return TensorType(tuple(int(dim) for dim in tensor.shape), dtype)


def _constant_value(traced: object, node: object) -> object:
    return getattr(traced, getattr(node, "target", ""), None)


def _constant_tensor_message(traced: object, node: object) -> str:
    target = getattr(node, "target", "?")
    value = _constant_value(traced, node)
    shape = getattr(value, "shape", None)
    rendered = "?" if shape is None else tuple(shape)
    return (
        f"captured graph embeds constant tensor {target!r} of shape {rendered}; "
        "constant tensors are not plan inputs. A HuggingFace model traced with "
        "use_cache=True is the usual source: pass "
        "example_kwargs={'use_cache': False} to keep the cache out of the graph. "
        "compile_module binds these constants as private plan inputs on its own, "
        "so this only affects a bare capture_fx call; pass the tensor through "
        "example_inputs instead to make it an explicit graph input"
    )


def _reject_unsupported_dropout(graph: object) -> None:
    """Refuse any graph that draws a dropout mask.

    ``aten.bernoulli_.float`` is the one random operator dropout leaves in a
    traced graph, whether it comes from ``nn.Dropout``, ``F.dropout``, or the
    dropout hand-rolled inside attention. The mask is sampled per call, and a
    static plan has no RNG state to reproduce it, so nothing downstream could
    be faithful. The advice differs by caller, so the message names the
    operator and the two fixed points (eval mode, ``p=0``) rather than assuming
    which spelling produced it.
    """

    for node in getattr(graph, "nodes", ()):
        if str(getattr(node, "target", "")) == "aten.bernoulli_.float":
            raise NotImplementedError(
                "dropout is not supported: the traced graph draws a random "
                "bernoulli mask (aten.bernoulli_.float), and a static plan has "
                "no RNG state to reproduce it. Trace with dropout off - "
                "model.eval() or p=0.0 - so dropout is the identity"
            )


# Operator whose metadata produced a symbolic length, and the name a user knows
# it by. ``unique`` reports itself through the tuple-returning ``_unique2`` and a
# ``getitem``, so the trace records a builtin rather than the operator the model
# called; climbing to the producer is what makes the refusal readable.
_DATA_DEPENDENT_NAMES = {
    "aten.nonzero.default": "nonzero",
    "aten.unique.default": "unique",
    "aten._unique2.default": "unique",
    "aten.unique_dim.default": "unique",
    "aten.unique_consecutive.default": "unique_consecutive",
    "aten.masked_select.default": "masked_select",
    "aten.bincount.default": "bincount",
    "aten.index.Tensor": "boolean-mask indexing",
}


def _producing_target(node: object) -> str:
    """The operator whose metadata gave this node its tensor."""

    target = str(getattr(node, "target", "?"))
    args = getattr(node, "args", ())
    while target.startswith("<built-in function") and args:
        source = next((arg for arg in args if hasattr(arg, "target")), None)
        if source is None:
            break
        target = str(source.target)
        args = getattr(source, "args", ())
    return target


def _reject_data_dependent_shapes(graph: object) -> None:
    """Refuse a graph whose tensor lengths are computed from the values.

    ``nonzero``, ``unique``, ``masked_select``, ``bincount`` and boolean-mask
    indexing all return a tensor whose size depends on what is *in* the tensor.
    The IR requires every dimension to be a positive constant, so there is
    nothing to lower, and Torch's own tracing will not guard on the symbol
    either. The node carries that symbol in its metadata before any lowerer
    reads it, which is the one place where the offending operator still has a
    name - so the refusal is written here, next to the dropout refusal, rather
    than surfacing as a guard exception from a shape comparison.
    """

    for node in getattr(graph, "nodes", ()):
        shape = getattr(getattr(node, "meta", {}).get("val"), "shape", None)
        if shape is None or not any(hasattr(dim, "node") for dim in shape):
            continue
        target = _producing_target(node)
        named = _DATA_DEPENDENT_NAMES.get(target)
        subject = f"{named} ({target})" if named else target
        raise ValueError(
            f"{subject} returns a length computed from its values, and a plan "
            "can only hold static shapes: the size is not known until the data "
            "is, so Torch refuses to guard on it too. Pass the extent in from "
            "the input shape instead - a dense mask of the input length, or an "
            "explicit length argument - or run this operator outside the "
            "compiled region"
        )


def _empty_value_nodes(traced: object) -> set[object]:
    """Collect FX nodes whose value is an empty tensor.

    A model with ``use_cache=True`` starts its KV cache as zero-length tensors.
    ``make_fx`` lifts those into graph constants, and an empty tensor cannot be a
    plan input because the IR requires positive static dimensions. They are only
    ever concatenated, where a zero-length operand contributes nothing, so they
    are dropped at the concatenation instead of being given a type.
    """

    found: set[object] = set()
    for node in getattr(getattr(traced, "graph", None), "nodes", ()):
        tensor = getattr(node, "meta", {}).get("val")
        shape = getattr(tensor, "shape", None)
        if shape is not None and 0 in tuple(shape):
            found.add(node)
    return found


def _fold_empty_cat_operands(
    node: object,
    empty_nodes: set[object],
    node_ids: dict[object, str],
) -> tuple[object, ...] | None:
    """Drop zero-length operands from a ``cat``.

    Returns the arguments to lower, or ``None`` when a single operand survives
    and the concatenation is therefore the identity.
    """

    args = getattr(node, "args", ())
    if not args:
        return args
    operands = args[0]
    if not isinstance(operands, (tuple, list)):
        return args
    kept = tuple(operand for operand in operands if operand not in empty_nodes)
    if len(kept) == len(operands):
        return args
    if not kept:
        raise ValueError(
            "cat only concatenates zero-length constants and has no result"
        )
    if len(kept) == 1:
        node_ids[node] = _argument_ids(kept[0], node_ids)
        return None
    return (kept, *args[1:])
