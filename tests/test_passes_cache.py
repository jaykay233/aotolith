"""The KV-cache roll rewrite: what it folds, what it refuses, and what it computes.

The roll is one statement written as a dozen operators, and the pass that folds it
is worth testing on three levels. The rewrite has to fire on the spelling the
frontend actually emits; it has to *not* fire when the chain only looks similar
(a window that drops more than one slot, a pair read by a third consumer, a value
that is not the projection's); and the operator it produces has to compute the
same window, in the same dtype, as the concatenate it replaced - through the
generated kernel and through the MLX fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import aotolith as aoto
from aotolith.backends import AppleMetalBackend
from aotolith.capture import capture_fx
from aotolith.ir import DType, Graph, Operation, TensorType, Value
from aotolith.ir.ops.table import check_operation
from aotolith.passes.cache import fuse_kv_cache_update

LAYERS = 3
CAPACITY = 12
HEADS = 2
HEAD_DIM = 8
PACKED = HEADS * HEAD_DIM * 3


def _mx(tensor: torch.Tensor):
    import mlx.core as mx

    return mx.array(tensor.float().numpy()).astype(mx.bfloat16)


class Roll(torch.nn.Module):
    """The decode adapter's spelling: roll a window, read the new pair back."""

    def forward(self, cache, projected, key):
        pairs = []
        key_windows = []
        value_windows = []
        for layer in range(LAYERS):
            window = cache[layer][:, :, :, 1:, :]
            value = (
                projected[:, :, 2 * HEADS * HEAD_DIM :]
                .view(1, 1, HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            new_pair = torch.cat(
                (window, torch.stack((key[layer], value), dim=0)),
                dim=3,
            )
            pairs.append(new_pair)
            key_windows.append(new_pair[0])
            value_windows.append(new_pair[1])
        return (
            torch.stack(pairs, dim=0),
            torch.stack(key_windows, dim=0),
            torch.stack(value_windows, dim=0),
        )


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(
            (LAYERS, 2, 1, HEADS, CAPACITY, HEAD_DIM), generator=generator
        ).to(torch.bfloat16),
        torch.randn((1, 1, PACKED), generator=generator).to(torch.bfloat16),
        torch.randn((LAYERS, 1, HEADS, 1, HEAD_DIM), generator=generator).to(
            torch.bfloat16
        ),
    )


def _lowered():
    graph = capture_fx(Roll().eval(), _inputs())
    return graph, fuse_kv_cache_update(graph)


def test_the_roll_becomes_one_operator_per_layer() -> None:
    graph, rewritten = _lowered()

    appends = [op for op in rewritten.ops if op.op == "kv_cache_append"]
    assert len(appends) == LAYERS
    assert [op.attrs["layer"] for op in appends] == list(range(LAYERS))
    assert all(op.attrs["value_start"] == 2 * HEADS * HEAD_DIM for op in appends)
    # The chain is a dozen operators per layer; the window slice, the pair
    # slice, both squeezes, the concatenate and the value's slice/reshape/
    # transpose all go away with it.
    assert len(rewritten.ops) < len(graph.ops)
    assert len(graph.ops) - len(rewritten.ops) >= LAYERS * 6
    rewritten.validate()


def test_the_rewritten_graph_computes_the_same_window() -> None:
    cache, projected, key = _inputs()
    reference = Roll()(cache, projected, key)

    compiled = aoto.compile_module(Roll().eval(), (cache, projected, key))
    actual = compiled(_mx(cache), _mx(projected), _mx(key))
    expected = [np.asarray(t.float().numpy()) for t in reference]

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert tuple(got.shape) == want.shape
        np.testing.assert_array_equal(np.asarray(got.astype(_float32())), want)


def test_the_fallback_and_the_kernel_agree() -> None:
    cache, projected, key = _inputs(seed=7)
    lowered = fuse_kv_cache_update(capture_fx(Roll().eval(), (cache, projected, key)))

    native = AppleMetalBackend().compile(lowered)
    fallback = AppleMetalBackend(
        selection_overrides={
            selection.output_id: "mlx" for selection in native.selections
        }
    ).compile(lowered)

    assert any(selection.is_native for selection in native.selections)
    assert not any(selection.is_native for selection in fallback.selections)

    inputs = (_mx(cache), _mx(projected), _mx(key))
    want = native(*inputs)
    got = fallback(*inputs)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a.astype(_float32())), np.asarray(b.astype(_float32()))
        )


def test_the_multi_output_operator_keeps_the_unrolled_runner() -> None:
    """One three-output operator must not push the plan back to the interpreter.

    The unrolled runner used to require every operator to return a single
    value, so a graph holding one ``kv_cache_append`` lost the fast path for all
    of its other operators too. It now binds the tuple, and both paths have to
    agree bit for bit.
    """

    cache, projected, key = _inputs(seed=3)
    lowered = fuse_kv_cache_update(capture_fx(Roll().eval(), (cache, projected, key)))
    plan = AppleMetalBackend().compile(lowered)
    inputs = (_mx(cache), _mx(projected), _mx(key))

    unrolled = plan(*inputs)
    assert plan._runner is not None
    interpreted, _ = plan.call_with_stats(*inputs)
    for a, b in zip(unrolled, interpreted, strict=True):
        np.testing.assert_array_equal(
            np.asarray(a.astype(_float32())), np.asarray(b.astype(_float32()))
        )


def test_a_window_that_drops_two_slots_is_left_alone() -> None:
    class Narrower(torch.nn.Module):
        def forward(self, cache, projected, key):
            pairs = []
            for layer in range(LAYERS):
                window = cache[layer][:, :, :, 2:, :]
                value = (
                    projected[:, :, 2 * HEADS * HEAD_DIM :]
                    .view(1, 1, HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                pairs.append(
                    torch.cat((window, torch.stack((key[layer], value), 0)), 3)
                )
            return torch.stack(pairs)

    graph = capture_fx(Narrower().eval(), _inputs())
    assert not any(op.op == "kv_cache_append" for op in fuse_kv_cache_update(graph).ops)


def test_a_value_that_is_not_the_projection_is_left_alone() -> None:
    class Shaped(torch.nn.Module):
        def forward(self, cache, projected, key):
            pairs = []
            for layer in range(LAYERS):
                window = cache[layer][:, :, :, 1:, :]
                value = torch.zeros(
                    (1, HEADS, 1, HEAD_DIM), dtype=torch.bfloat16
                )
                pairs.append(
                    torch.cat((window, torch.stack((key[layer], value), 0)), 3)
                )
            return torch.stack(pairs)

    graph = capture_fx(Shaped().eval(), _inputs())
    assert not any(op.op == "kv_cache_append" for op in fuse_kv_cache_update(graph).ops)


def test_a_pair_shared_with_another_reader_is_left_alone() -> None:
    """Deleting the layer slice would strand whoever else reads the pair."""

    class Shared(torch.nn.Module):
        def forward(self, cache, projected, key):
            pairs = []
            layers = []
            for layer in range(LAYERS):
                pair = cache[layer]
                layers.append(pair)
                value = (
                    projected[:, :, 2 * HEADS * HEAD_DIM :]
                    .view(1, 1, HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                pairs.append(
                    torch.cat(
                        (pair[:, :, :, 1:, :], torch.stack((key[layer], value), 0)),
                        3,
                    )
                )
            return torch.stack(pairs), torch.stack(layers)

    graph = capture_fx(Shared().eval(), _inputs())
    assert not any(op.op == "kv_cache_append" for op in fuse_kv_cache_update(graph).ops)


def test_a_pair_that_is_a_graph_output_is_left_alone() -> None:
    """The consumer table cannot see graph outputs, so the pass has to check.

    Only the exported layer keeps its chain; the others still fold, which is
    what tells this guard apart from a matcher that stopped firing at all.
    """

    class Exported(torch.nn.Module):
        def forward(self, cache, projected, key):
            pairs = []
            key_windows = []
            value_windows = []
            exported = None
            for layer in range(LAYERS):
                pair = cache[layer]
                if layer == 0:
                    exported = pair
                value = (
                    projected[:, :, 2 * HEADS * HEAD_DIM :]
                    .view(1, 1, HEADS, HEAD_DIM)
                    .transpose(1, 2)
                )
                new_pair = torch.cat(
                    (pair[:, :, :, 1:, :], torch.stack((key[layer], value), 0)),
                    dim=3,
                )
                pairs.append(new_pair)
                key_windows.append(new_pair[0])
                value_windows.append(new_pair[1])
            return (
                torch.stack(pairs),
                torch.stack(key_windows),
                torch.stack(value_windows),
                exported,
            )

    graph = capture_fx(Exported().eval(), _inputs())
    appends = [
        op for op in fuse_kv_cache_update(graph).ops if op.op == "kv_cache_append"
    ]
    assert [op.attrs["layer"] for op in appends] == list(range(1, LAYERS))


def test_the_contract_rejects_a_window_of_the_wrong_length() -> None:
    """A hand-built graph cannot claim a roll the shapes do not support."""

    def value(value_id: str, shape: tuple[int, ...]) -> Value:
        return Value(value_id, TensorType(shape, DType.BF16))

    graph = Graph(
        inputs=("cache", "key", "source"),
        values={
            "cache": value("cache", (LAYERS, 2, 1, HEADS, CAPACITY, HEAD_DIM)),
            "key": value("key", (1, HEADS, 1, HEAD_DIM)),
            "source": value("source", (1, 1, PACKED)),
            "pair": value("pair", (2, 1, HEADS, CAPACITY, HEAD_DIM)),
            # One slot short of what the pair says the window must be.
            "key_window": value("key_window", (1, HEADS, CAPACITY - 2, HEAD_DIM)),
            "value_window": value("value_window", (1, HEADS, CAPACITY, HEAD_DIM)),
        },
        ops=(
            Operation(
                "kv_cache_append",
                ("cache", "key", "source"),
                ("pair", "key_window", "value_window"),
                {"layer": 0, "value_start": 2 * HEADS * HEAD_DIM},
            ),
        ),
        outputs=("pair", "key_window", "value_window"),
    )
    with pytest.raises(ValueError, match="key window"):
        check_operation(graph, graph.ops[0])


def test_the_pipeline_drops_the_roll_from_a_decoded_model() -> None:
    compiled = aoto.compile_module(Roll().eval(), _inputs())
    ops = [op for op in compiled.lowered_graph.ops if op.op == "kv_cache_append"]
    assert len(ops) == LAYERS
    # Nothing of the roll survives as general operators.
    assert not any(
        op.op == "cat" and int(op.attrs.get("dim", -1)) % 4 == 3
        for op in compiled.lowered_graph.ops
    )


def _float32():
    import mlx.core as mx

    return mx.float32
