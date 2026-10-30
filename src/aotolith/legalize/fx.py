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
        raise ValueError(f"unsupported dtype: {tensor.dtype}")
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


def _reject_unsupported_attention_dropout(graph: object) -> None:
    for node in getattr(graph, "nodes", ()):
        if str(getattr(node, "target", "")) == "aten.bernoulli_.float":
            raise NotImplementedError(
                "scaled_dot_product_attention dropout is not supported"
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
