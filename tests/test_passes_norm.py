"""The normalization scale fold: five operators in, one operator out.

PyTorch writes RMSNorm as an fp32 upcast, the normalization, a scaled multiply
and a narrowing cast back. The pass under test replaces that chain with one
normalization that takes the weight as an operand, so the tests check the shape
of the rewrite, that the result is unchanged to the bit, and that the pass backs
off when the chain is not what it looks like.
"""

from __future__ import annotations

import mlx.core as mx
import torch

from aotolith.capture import capture_fx
from aotolith.compiler import compile_graph
from aotolith.passes import fuse_norm_scale, rewrite

EPS = 1e-6


def _rms_norm_scale(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The spelling Qwen and Llama use, which is what the pass exists for."""

    upcast = value.to(torch.float32)
    variance = upcast.pow(2).mean(-1, keepdim=True)
    normalized = upcast * torch.rsqrt(variance + EPS)
    # ``weight`` is bf16, so Torch promotes the product to fp32 and the model
    # narrows it once at the end. That is the chain the pass folds.
    return (weight * normalized).to(value.dtype)


def _graph(*values: torch.Tensor):
    return capture_fx(_rms_norm_scale, values)


def _counts(graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for op in graph.ops:
        counts[op.op] = counts.get(op.op, 0) + 1
    return counts


def _case(hidden: int, *, seed: int = 0, rows: tuple[int, ...] = (1, 1)):
    torch.manual_seed(seed)
    value = torch.randn(*rows, hidden, dtype=torch.bfloat16)
    weight = torch.randn(hidden, dtype=torch.bfloat16)
    return value, weight


def test_the_captured_normalization_is_five_operators() -> None:
    # The premise of the pass: two casts on the row, a cast on the weight, the
    # normalization, and the scale multiply.
    value, weight = _case(64)

    counts = _counts(_graph(value, weight))

    assert counts.get("rms_norm") == 1
    assert counts.get("mul") == 1
    assert counts.get("cast") == 3


def test_the_fold_leaves_one_weighted_normalization() -> None:
    value, weight = _case(64)

    graph = rewrite(_graph(value, weight))
    counts = _counts(graph)

    assert counts.get("rms_norm") == 1
    assert "mul" not in counts
    assert "cast" not in counts
    norm = next(op for op in graph.ops if op.op == "rms_norm")
    assert len(norm.inputs) == 2
    assert norm.attrs["has_weight"] is True


def test_the_folded_normalization_is_bit_identical() -> None:
    # The chain upcasts losslessly, multiplies in fp32 and rounds once, which is
    # exactly what the fused kernel does. Comparing the two compiled graphs
    # rather than Torch isolates the rewrite: both call the same normalization
    # reduction, so only the fold could move a bit.
    for seed in range(4):
        value, weight = _case(128, seed=seed, rows=(2, 3))
        graph = _graph(value, weight)
        folded = fuse_norm_scale(graph)
        assert len(folded.ops) < len(graph.ops)

        inputs = [
            mx.array(value.float().numpy()).astype(mx.bfloat16),
            mx.array(weight.float().numpy()).astype(mx.bfloat16),
        ]
        expected = compile_graph(graph).plan(*inputs)[0]
        actual = compile_graph(folded).plan(*inputs)[0]
        mx.eval(expected, actual)

        assert mx.array_equal(actual, expected), seed
        with torch.no_grad():
            torch_value = _rms_norm_scale(value, weight)
        assert (
            float(
                mx.abs(
                    actual.astype(mx.float32) - mx.array(torch_value.float().numpy())
                ).max()
            )
            < 1e-2
        )


def test_the_fold_keeps_the_normalized_row_when_it_has_two_readers() -> None:
    # A graph that also consumes the normalized row needs it materialized, so
    # folding would have to keep the original chain anyway.
    def two_readers(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        upcast = value.to(torch.float32)
        variance = upcast.pow(2).mean(-1, keepdim=True)
        normalized = upcast * torch.rsqrt(variance + EPS)
        scaled = weight * normalized
        return (scaled + normalized).to(value.dtype)

    value, weight = _case(64)

    graph = rewrite(capture_fx(two_readers, (value, weight)))
    counts = _counts(graph)

    norms = [op for op in graph.ops if op.op == "rms_norm"]
    assert len(norms) == 1
    # The normalization kept its standalone form, so the scale was not moved in.
    assert len(norms[0].inputs) == 1
    assert counts.get("cast") == 3


def test_the_fold_runs_on_the_graph_it_is_handed() -> None:
    # The pass is exported and standalone, so it has to be callable on its own
    # rather than only through the pipeline.
    value, weight = _case(32)
    graph = _graph(value, weight)

    folded = fuse_norm_scale(graph)

    assert len(folded.ops) < len(graph.ops)
    assert all(op.op != "cast" for op in folded.ops)
