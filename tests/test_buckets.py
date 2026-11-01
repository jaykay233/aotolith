"""Bucket dispatch: what a request of any width costs, and what it may change.

The point of a bucket set is that a request the compiler never saw still runs on
a plan that was compiled ahead of time. Two things have to hold for that to be
worth anything: the answer must match the answer a plan compiled at exactly that
width gives, and the memory and padding costs must be known before the request
arrives. Both are measured here rather than argued.
"""

from __future__ import annotations

import json
import threading

import mlx.core as mx
import pytest
import torch

import aotolith as aoto
from aotolith.buckets import BucketedPlans, compile_bucketed, load_bucketed


class _Rowwise(torch.nn.Module):
    """A model whose rows are independent, which is the batch-axis contract."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value @ self.weight).sum(dim=-1, keepdim=True)


class _Elementwise(torch.nn.Module):
    """No matmul, so its arithmetic is bit-exact across widths."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value * 2.0 + 1.0).relu()


def _compile(module: torch.nn.Module, sizes, axis: int = 0) -> BucketedPlans:
    return compile_bucketed(module, (torch.randn(1, 16),), axis=axis, sizes=sizes)


def test_dispatch_matches_a_plan_compiled_at_the_requested_width() -> None:
    module = _Elementwise().eval()
    plans = _compile(module, (1, 4, 8))

    direct = aoto.compile_module(module, (torch.randn(3, 16),))
    value = mx.array(torch.randn(3, 16).numpy())
    expected = direct(value)
    actual = plans(value)
    mx.eval(*expected, *actual)

    assert actual[0].shape == (3, 16)
    assert mx.array_equal(actual[0], expected[0])


def test_a_widened_bucket_agrees_with_the_backend_within_its_own_noise() -> None:
    """Padding must not change the arithmetic beyond what MLX already does.

    MLX is not row-exact at float32: the same row in a one-row matmul and in a
    four-row matmul of the same expression differed by 6.8e-3 here, because the
    tile shape decides the accumulation order. The bound asserted below is that
    noise, measured on this exact expression at both widths in the same process,
    so a padding bug cannot hide inside a tolerance picked by hand.
    """

    module = _Rowwise().eval()
    plans = _compile(module, (1, 4))
    value = torch.randn(3, 16)
    weights = mx.array(module.weight.detach().numpy())

    def expression(rows: torch.Tensor) -> mx.array:
        out = mx.tanh(mx.matmul(mx.array(rows.numpy()), weights))
        return out.sum(axis=-1, keepdims=True)

    requested = expression(value)
    padded = expression(value.repeat(4, 1))[:3]
    noise = mx.max(mx.abs(requested - padded))

    actual = plans(mx.array(value.numpy()))[0]
    mx.eval(actual, requested, noise)
    difference = mx.max(mx.abs(actual - requested))
    mx.eval(difference)

    assert float(difference) <= float(noise) + 1e-5


def test_padding_tiles_real_rows_and_never_a_constant() -> None:
    """The padded rows repeat the request, so a variance cannot become NaN."""

    class _Capture(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            # ``var`` over a constant fill of a single row is 0/0; over tiled
            # rows it is the row's own variance.
            return value + value.var(dim=-1, keepdim=True)

    plans = compile_bucketed(
        _Capture().eval(), (torch.randn(1, 4),), axis=0, sizes=(1, 4)
    )
    value = mx.array([[1.0, 2.0, 3.0, 4.0]])
    output = plans(value)[0]
    mx.eval(output)

    assert output.shape == (1, 4)
    assert not bool(mx.any(mx.isnan(output)))


def test_outputs_are_trimmed_to_the_requested_width() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4, 8))

    for requested in (1, 2, 3, 4, 5, 8):
        output = plans(mx.array(torch.randn(requested, 16).numpy()))[0]
        mx.eval(output)
        assert output.shape[0] == requested, requested


def test_a_request_wider_than_every_bucket_is_refused() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))

    with pytest.raises(aoto.BucketRangeError, match="exceeds the largest bucket"):
        plans(mx.zeros((5, 16), dtype=mx.float32))
    # The code is the part a serving layer branches on.
    assert aoto.BucketRangeError.code == "AOTO-2005"


def test_bucket_widths_must_be_positive_sorted_and_unique() -> None:
    module = _Elementwise().eval()
    with pytest.raises(ValueError, match="strictly increasing"):
        _compile(module, (4, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        _compile(module, (4, 4))
    with pytest.raises(ValueError, match="positive"):
        _compile(module, (0, 4))
    with pytest.raises(ValueError, match="at least one"):
        _compile(module, ())


def test_the_example_width_may_be_a_bucket() -> None:
    """Tracing at width 2 and bucketing at (1, 2, 4) is the normal case."""

    plans = compile_bucketed(
        _Elementwise().eval(), (torch.randn(2, 16),), axis=0, sizes=(1, 2, 4)
    )
    assert plans.widths == (1, 2, 4)
    assert plans(mx.zeros((3, 16), dtype=mx.float32))[0].shape == (3, 16)


def test_the_weights_are_one_copy_for_the_whole_set() -> None:
    plans = _compile(_Rowwise().eval(), (1, 4, 8))
    names = plans.buckets[0].plan.state_names
    assert "weight" in names
    references = [
        next(
            value
            for name, value in zip(bucket.plan.state_names, bucket.plan._state_values)
            if name == "weight"
        )
        for bucket in plans.buckets
    ]
    assert all(value is references[0] for value in references)

    weight_bytes = references[0].nbytes
    footprint = plans.footprint()
    assert footprint.state_bytes == weight_bytes
    assert footprint.bucket_count == 3


def test_the_reported_bound_is_the_worst_bucket() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4, 8))
    footprint = plans.footprint()
    widest = max(bucket.peak_bound_bytes for bucket in plans.buckets)
    assert footprint.peak_bound_bytes == widest
    # The report is the widest bucket's, because that is the one a request of
    # the largest supported size runs on.
    assert plans.memory_report().peak_bound_bytes == widest


def test_a_bucket_set_refuses_a_budget_it_cannot_fit() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    needed = plans.footprint().state_bytes + plans.memory_report().peak_bound_bytes

    with pytest.raises(aoto.MemoryBudgetError, match="bucket set needs"):
        plans.check_memory_budget(needed - 1)
    assert plans.check_memory_budget(needed).bucket_count == 2


def test_a_request_without_the_bucket_axis_is_refused() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    with pytest.raises(aoto.PlanInputError, match="has no axis"):
        plans(mx.array(1.0))


def test_inputs_that_disagree_about_the_width_are_refused() -> None:
    class _Pair(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    plans = compile_bucketed(
        _Pair().eval(),
        (torch.randn(1, 4), torch.randn(1, 4)),
        axis=0,
        sizes=(1, 4),
    )
    assert plans.widening == (0, 1)
    with pytest.raises(aoto.PlanInputError, match="disagree"):
        plans(
            mx.zeros((2, 4), dtype=mx.float32),
            mx.zeros((3, 4), dtype=mx.float32),
        )


def test_a_fixed_extent_on_the_same_axis_is_not_widened() -> None:
    """A table of a different width is not a request dimension."""

    class _Project(torch.nn.Module):
        def forward(self, value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            return value @ table

    plans = compile_bucketed(
        _Project().eval(),
        (torch.randn(1, 8), torch.randn(8, 8)),
        axis=0,
        sizes=(1, 4),
    )
    assert plans.widening == (0,)
    output = plans(
        mx.zeros((3, 8), dtype=mx.float32),
        mx.zeros((8, 8), dtype=mx.float32),
    )[0]
    mx.eval(output)
    assert output.shape == (3, 8)


def test_custom_hooks_own_the_padding_of_an_axis_that_needs_one() -> None:
    """Left padding is the sequence case: the mask belongs to the caller.

    A causal model cannot be right-padded, because the pads would be attended
    to. The hooks let the caller left-pad and pass the matching mask, and the
    dispatcher stays out of the semantics it cannot know.
    """

    def pad(inputs: tuple, width: int) -> tuple:
        value, mask = inputs
        missing = width - int(value.shape[1])
        if missing <= 0:
            return inputs
        padded = mx.concatenate(
            [mx.zeros((int(value.shape[0]), missing), dtype=value.dtype), value],
            axis=1,
        )
        masked = mx.concatenate(
            [mx.zeros((1, missing), dtype=mask.dtype), mask], axis=1
        )
        return (padded, masked)

    def unpad(outputs: tuple, requested: int) -> tuple:
        (output,) = outputs
        return (output[:, -requested:],)

    class _Length(torch.nn.Module):
        def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (value * mask).sum(dim=1, keepdim=True)

    plans = compile_bucketed(
        _Length().eval(),
        (torch.randn(1, 8), torch.ones(1, 8)),
        axis=1,
        sizes=(8, 16),
        pad=pad,
        unpad=unpad,
    )
    assert plans.pad_mode == "custom"
    value = mx.array([[1.0, 2.0, 3.0]])
    mask = mx.ones((1, 3))
    output = plans(value, mask)[0]
    mx.eval(output)
    assert output.shape == (1, 1)
    assert float(output[0, 0]) == pytest.approx(6.0)


def test_save_and_load_round_trip_a_set(tmp_path) -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    written = plans.save(tmp_path / "model.aoto")
    index = tmp_path / "model.buckets.json"
    assert index in written
    payload = json.loads(index.read_text())
    assert payload["pad"] == "tile"
    assert payload["widths"] == [1, 4]
    assert [entry["artifact"] for entry in payload["buckets"]] == [
        "model.1.aoto",
        "model.4.aoto",
    ]

    loaded = load_bucketed(index)
    assert loaded.widths == (1, 4)
    value = mx.array(torch.randn(3, 16).numpy())
    mx.eval(*plans(value), *loaded(value))
    assert mx.array_equal(plans(value)[0], loaded(value)[0])
    # The loaded plans share the weights, exactly like the compiled ones.
    assert loaded.footprint().state_bytes == plans.footprint().state_bytes


def test_the_index_names_the_backend_the_set_was_built_for(tmp_path) -> None:
    """The archives each record a target; the index says what the set is.

    Without the index entry, a reference bundle and a Metal bundle of the same
    shapes are indistinguishable until the loader reaches the first archive and
    fails on that file, which reads as "one artifact is wrong" rather than "this
    whole set came from another backend".
    """

    plans = _compile(_Elementwise().eval(), (1, 4))
    plans.save(tmp_path / "model.aoto")
    index = tmp_path / "model.buckets.json"
    payload = json.loads(index.read_text())
    assert payload["target"] == "apple-metal"

    # An index that claims the other backend must be refused before anything is
    # loaded, with the set-level message rather than a per-artifact one.
    payload["target"] = "mlx-reference"
    index.write_text(json.dumps(payload))
    with pytest.raises(aoto.ArtifactCompatibilityError, match="the index records"):
        load_bucketed(index)


def test_the_index_without_a_target_reads_as_the_metal_one(tmp_path) -> None:
    """A set recorded before the target existed loads as what it was."""

    plans = _compile(_Elementwise().eval(), (1, 4))
    plans.save(tmp_path / "model.aoto")
    index = tmp_path / "model.buckets.json"
    payload = json.loads(index.read_text())
    del payload["target"]
    index.write_text(json.dumps(payload))
    loaded = load_bucketed(index)
    assert loaded.widths == (1, 4)


def test_a_set_with_custom_hooks_refuses_to_be_saved(tmp_path) -> None:
    plans = compile_bucketed(
        _Elementwise().eval(),
        (torch.randn(1, 16),),
        axis=0,
        sizes=(1, 4),
        pad=lambda inputs, width: inputs,
    )
    with pytest.raises(aoto.CompilationError, match="cannot be saved"):
        plans.save(tmp_path / "model.aoto")


def test_stats_record_the_bucket_and_the_padding_it_wasted() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4, 8))
    for requested in (1, 3, 3, 8):
        mx.eval(*plans(mx.zeros((requested, 16), dtype=mx.float32)))

    stats = plans.stats.snapshot()
    assert stats["counts"] == {1: 1, 4: 2, 8: 1}
    # 3 needs a 4 bucket twice and 8 needs nothing.
    assert stats["slack_tokens"] == 2


def test_prepare_unrolls_every_bucket() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    reports = plans.prepare()
    assert len(reports) == 2
    assert all(report.policy == aoto.DEFAULT_MEMORY for report in reports)


def test_dispatch_is_safe_from_several_threads() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    errors: list[BaseException] = []

    def worker(width: int) -> None:
        try:
            for _ in range(5):
                mx.eval(*plans(mx.zeros((width, 16), dtype=mx.float32)))
        except BaseException as exc:  # pragma: no cover - the failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(width,)) for width in (1, 2, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sum(plans.stats.snapshot()["counts"].values()) == 15


def test_describe_lists_every_bucket_and_its_bound() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    summary = plans.describe()
    assert summary["buckets"] == 2
    assert summary["widths"] == [1, 4]
    assert summary["pad"] == "tile"
    assert [entry["width"] for entry in summary["per_bucket"]] == [1, 4]


def test_a_loaded_set_refuses_an_artifact_the_index_does_not_describe(
    tmp_path,
) -> None:
    """The index is the artifact a caller names, so it has to match its files.

    ``planned_peak_bytes`` is a fingerprint of the graph's shapes: an entry and a
    file that disagree mean the index would dispatch a width to a plan compiled
    for a different one. Swapping the two files reproduces exactly that.
    """

    plans = _compile(_Elementwise().eval(), (1, 4))
    index = tmp_path / "model.buckets.json"
    plans.save(tmp_path / "model.aoto")
    payload = json.loads(index.read_text())
    narrow, wide = (tmp_path / entry["artifact"] for entry in payload["buckets"])
    narrow_bytes, wide_bytes = narrow.read_bytes(), wide.read_bytes()
    narrow.write_bytes(wide_bytes)
    wide.write_bytes(narrow_bytes)

    with pytest.raises(aoto.ArtifactCompatibilityError, match="index recorded"):
        load_bucketed(index)


def test_a_request_shaped_keyword_is_widened_with_the_request() -> None:
    """A mask that follows the batch is part of the request, not a constant.

    The keyword is re-widened at compile time as well as at dispatch: a plan
    traced at width 4 whose keyword stayed 1 wide would reject the call.
    """

    class _Masked(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4))

        def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (value @ self.weight) * mask

    plans = compile_bucketed(
        _Masked().eval(),
        (torch.randn(1, 4),),
        axis=0,
        sizes=(1, 4),
        example_kwargs={"mask": torch.ones(1, 1)},
    )
    assert plans.widening_kwargs == ("mask",)
    assert plans.describe()["widening_kwargs"] == ["mask"]

    value = mx.array(torch.arange(12, dtype=torch.float32).reshape(3, 4).numpy())
    mask = mx.array([[1.0], [0.0], [2.0]])
    output = plans(value, mask=mask)[0]
    mx.eval(output)
    assert output.shape == (3, 4)
    expected = (value * mask).__array__()
    assert (output.__array__() == expected).all()


def test_a_fixed_keyword_is_not_widened() -> None:
    """A table whose extent did not follow the request stays exactly as traced."""

    class _Fixed(torch.nn.Module):
        def forward(self, value: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
            return value @ table

    plans = compile_bucketed(
        _Fixed().eval(),
        (torch.randn(1, 4),),
        axis=0,
        sizes=(1, 4),
        example_kwargs={"table": torch.eye(4)},
    )
    assert plans.widening_kwargs == ()
    output = plans(mx.zeros((3, 4)), table=mx.eye(4))[0]
    mx.eval(output)
    assert output.shape == (3, 4)


def test_an_output_that_is_not_bucket_wide_is_left_alone() -> None:
    """Only an axis that *is* the bucket width is treated as the request.

    An output whose extent on the axis happens to equal the bucket width is
    indistinguishable from a request-shaped one, so the default trims it; that
    is the documented rule, and ``unpad`` is the way to override it.
    """

    class _Const(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.ones(2, dtype=value.dtype)

    plans = compile_bucketed(
        _Const().eval(), (torch.randn(1, 8),), axis=0, sizes=(1, 4)
    )
    output = plans(mx.zeros((3, 8), dtype=mx.float32))[0]
    mx.eval(output)
    assert output.shape == (2,)


def test_the_set_reports_its_widths_and_a_readable_footprint() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4, 8))
    assert plans.min_width == 1
    assert plans.max_width == 8
    text = plans.footprint().describe()
    assert "3 buckets" in text
    assert "peak bound" in text


def test_a_zero_width_request_and_a_short_argument_list_are_refused() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    with pytest.raises(aoto.PlanInputError, match="cannot serve a request"):
        plans.bucket_for(0)
    with pytest.raises(aoto.PlanInputError, match="expected 1 inputs"):
        plans()  # ty: ignore[missing-argument]
    with pytest.raises(aoto.PlanInputError, match="expected 1 inputs"):
        plans(mx.zeros((1, 16)), mx.zeros((1, 16)))  # ty: ignore[too-many-positional-arguments]


def test_an_unknown_memory_policy_is_refused() -> None:
    plans = _compile(_Elementwise().eval(), (1, 4))
    with pytest.raises(aoto.PlanInputError, match="unknown memory policy"):
        plans.memory_report("turbo")


def test_a_set_orders_its_buckets_and_refuses_duplicates() -> None:
    from aotolith.buckets import BucketedPlans

    plans = _compile(_Elementwise().eval(), (1, 4))
    unsorted = BucketedPlans(
        list(reversed(plans.buckets)), axis=0, widening=[0], input_count=1
    )
    # A set is a set: the order it was handed is not part of its meaning, but a
    # duplicated width is, because it would leave two plans for one boundary.
    assert unsorted.widths == (1, 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        BucketedPlans(
            [plans.buckets[0], plans.buckets[0]], axis=0, widening=[0], input_count=1
        )
    with pytest.raises(ValueError, match="at least one bucket"):
        BucketedPlans([], axis=0, widening=[0], input_count=1)


def test_an_empty_axis_is_refused_rather_than_tiled() -> None:
    """A zero-row request has no smallest bucket, so dispatch refuses it."""

    plans = _compile(_Elementwise().eval(), (1, 4))
    with pytest.raises(aoto.PlanInputError, match="cannot serve a request"):
        plans(mx.zeros((0, 16), dtype=mx.float32))
