"""The memory contract: what a call allocates, and what bounds it.

MLX is lazy, so a runner that only builds the graph holds every intermediate
until the caller evaluates the outputs, and the evaluator then allocates an
output for every node. That is why the peak of a naive unrolled plan is the sum
of its intermediates rather than the largest live set, and why the memory policy
exists: it synchronizes at storage-plan release points so the buffers can
actually go back.

These tests measure that with ``mx.get_peak_memory`` rather than trusting the
plan: a policy that is supposed to lower the peak and does not is worse than no
policy at all, because it costs latency for nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest
import torch

import aotolith as aoto
from aotolith import bench
from aotolith.runtime import plan as plan_module


def _doubling_chain(rounds: int = 8) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    """A chain whose intermediates double in size, which no pass can fuse."""

    def fn(value: torch.Tensor) -> torch.Tensor:
        current = value
        for step in range(1, rounds + 1):
            current = torch.cat((current, current * float(step + 1)), dim=-1)
        return current.sum(dim=-1, keepdim=True)

    return fn, (torch.randn(64, 1024),)


def _peak_extra(call) -> tuple[int, object]:
    """Run ``call`` and return the bytes it added on top of what was resident."""

    call()  # warm the kernels so the measurement is not Metal's first compile
    mx.eval(mx.array(0.0))
    mx.synchronize()
    mx.reset_peak_memory()
    resident = mx.get_active_memory()
    outputs = call()
    mx.eval(*outputs)
    mx.synchronize()
    return max(0, mx.get_peak_memory() - resident), outputs


def test_the_minimal_policy_lowers_the_measured_peak() -> None:
    fn, inputs = _doubling_chain()
    compiled = aoto.compile_callable(fn, inputs)
    arrays = tuple(mx.array(value.numpy()) for value in inputs)

    lazy, lazy_outputs = _peak_extra(lambda: compiled.plan(*arrays, memory="latency"))
    bounded, bounded_outputs = _peak_extra(lambda: compiled.plan(*arrays, memory="minimal"))

    # The shipped workloads are all below the balanced floor, so this is the
    # test that exercises the mechanism at all.
    assert bounded < lazy, (
        f"expected the bounded policy to hold less than {lazy} bytes, held "
        f"{bounded}"
    )
    assert mx.allclose(lazy_outputs[0], bounded_outputs[0], rtol=1e-5, atol=1e-5)


def test_every_policy_produces_the_same_answer() -> None:
    fn, inputs = _doubling_chain()
    compiled = aoto.compile_callable(fn, inputs)
    arrays = tuple(mx.array(value.numpy()) for value in inputs)

    reference = compiled.plan(*arrays, memory="latency")
    mx.eval(*reference)
    for policy in aoto.MEMORY_POLICIES:
        outputs = compiled.plan(*arrays, memory=policy)
        mx.eval(*outputs)
        assert mx.allclose(outputs[0], reference[0], rtol=1e-6, atol=1e-3), policy


def test_the_balanced_peak_stays_inside_its_reported_bound() -> None:
    fn, inputs = _doubling_chain()
    compiled = aoto.compile_callable(fn, inputs)
    arrays = tuple(mx.array(value.numpy()) for value in inputs)

    bounded, bounded_outputs = _peak_extra(lambda: compiled.plan(*arrays, memory="balanced"))
    lazy, lazy_outputs = _peak_extra(lambda: compiled.plan(*arrays, memory="latency"))

    assert bounded <= lazy
    assert mx.allclose(bounded_outputs[0], lazy_outputs[0], rtol=1e-5, atol=1e-5)


def test_a_clean_process_holds_the_peak_inside_the_bound() -> None:
    """The bound is a claim about a fresh process, so it is measured in one.

    In the test process the allocator already holds whatever the earlier tests
    left behind, and MLX counts those bytes in ``get_peak_memory``; a fresh
    interpreter is the state a server starts in, and the only one where the
    numbers mean what the report says.
    """

    script = textwrap.dedent(
        """
        import torch, mlx.core as mx, aotolith as aoto

        def fn(value):
            current = value
            for step in range(1, 9):
                current = torch.cat((current, current * float(step + 1)), dim=-1)
            return current.sum(dim=-1, keepdim=True)

        compiled = aoto.compile_callable(fn, (torch.randn(64, 1024),))
        plan = compiled.plan
        arrays = (mx.zeros((64, 1024), dtype=mx.float32),)
        for policy in ("latency", "minimal"):
            mx.eval(*plan(*arrays, memory=policy))
        mx.synchronize()

        def peak(policy):
            mx.eval(*plan(*arrays, memory=policy))
            mx.synchronize()
            mx.reset_peak_memory()
            resident = mx.get_active_memory()
            mx.eval(*plan(*arrays, memory=policy))
            mx.synchronize()
            return mx.get_peak_memory() - resident

        report = plan.memory_report("minimal")
        print(peak("minimal"), report.peak_bound_bytes, peak("latency"))
        """
    )
    # More than once: before the runner waited for the device at a release
    # point, the same plan came in under the bound in some processes and over it
    # in others - by a whole release group when the allocator got unlucky, which
    # on the doubling chain was 32 MB.
    for _ in range(4):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        bounded, bound, lazy = (int(value) for value in result.stdout.split())
        assert bounded <= bound, (bounded, bound)
        assert bounded < lazy, (bounded, lazy)


def test_balanced_does_not_synchronize_a_plan_below_the_floor() -> None:
    """The shipped shapes are small enough that the floor makes it a no-op."""

    def fn(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, weight).relu()

    compiled = aoto.compile_callable(fn, (torch.randn(8, 1024), torch.randn(4096, 1024)))
    report = compiled.plan.memory_report("balanced")
    assert report.sync_count == 0
    assert report.threshold_bytes == plan_module.MIN_SYNC_BYTES
    assert not _emits_a_sync(compiled.plan, "balanced")


def _emits_a_sync(plan, policy: str) -> bool:
    """Whether the generated body of this runner synchronizes at all.

    The unrolled function refers to ``mx`` only through the ``mx.eval`` the
    policy emits, so the name in its code object is the signal: a body that
    never synchronizes never mentions the module.
    """

    runner, _ = plan._runner_for(policy)
    for cell in getattr(runner, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # pragma: no cover - empty cell in a refcycle
            continue
        if callable(value) and getattr(value, "__name__", None) == "run":
            return "mx" in value.__code__.co_names
    return False


def test_a_synchronization_point_waits_for_the_device() -> None:
    """Evaluating schedules work; releasing the buffer needs the wait.

    ``mx.eval`` returns once the graph is scheduled, and MLX cannot return a
    buffer to the allocator while the GPU may still be writing it. A release
    point therefore has to name both, and the generated body is the evidence:
    the names it looks up tell a synchronizing runner from a lazy one without
    re-measuring anything.
    """

    compiled = aoto.compile_callable(_doubling_chain()[0], _doubling_chain()[1])
    runner, _ = compiled.plan._runner_for("minimal")
    names: set[str] = set()
    for cell in getattr(runner, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # pragma: no cover - empty cell in a refcycle
            continue
        if callable(value) and getattr(value, "__name__", None) == "run":
            names = set(value.__code__.co_names)
    assert "eval" in names
    assert "synchronize" in names

    lazy, _ = compiled.plan._runner_for("latency")
    assert not _emits_a_sync(compiled.plan, "latency")


def test_a_plan_that_cannot_unroll_still_reports_its_policy() -> None:
    compiled = aoto.compile_callable(lambda v: v * 2.0, (torch.randn(4),))
    assert compiled.plan.memory_report("latency").sync_count == 0


def test_an_unknown_policy_is_refused_with_the_choices() -> None:
    compiled = aoto.compile_callable(lambda v: v * 2.0, (torch.randn(4),))
    with pytest.raises(aoto.PlanInputError, match="unknown memory policy"):
        compiled.plan(mx.zeros((4,), dtype=mx.float32), memory="cheap")


def test_runners_are_cached_per_policy(monkeypatch) -> None:
    compiled = aoto.compile_callable(_doubling_chain(2)[0], _doubling_chain(2)[1])
    plan = compiled.plan
    builds: list[str] = []
    original = plan_module._compile_plan_runner

    def counted(plan_to_build, memory):
        builds.append(memory)
        return original(plan_to_build, memory)

    monkeypatch.setattr(plan_module, "_compile_plan_runner", counted)
    arrays = tuple(mx.array(value.numpy()) for value in _doubling_chain(2)[1])

    for _ in range(3):
        mx.eval(*plan(*arrays, memory="latency"))
        mx.eval(*plan(*arrays, memory="minimal"))

    assert sorted(builds) == ["latency", "minimal"], builds


def test_prepare_builds_the_runner_before_the_first_call() -> None:
    compiled = aoto.compile_callable(_doubling_chain(2)[0], _doubling_chain(2)[1])
    assert compiled.plan._runner is None
    report = compiled.prepare()
    assert compiled.plan._runner is not None
    assert report.policy == aoto.DEFAULT_MEMORY
    assert compiled.plan.memory_report().peak_bound_bytes > 0


def test_the_footprint_splits_what_the_caller_keeps_from_what_the_call_allocates() -> None:
    def fn(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.linear(value, weight)
        return hidden + 1.0

    compiled = aoto.compile_callable(fn, (torch.randn(8, 64), torch.randn(32, 64)))
    footprint = compiled.plan.footprint
    storage = compiled.plan.storage_plan

    assert footprint.input_bytes == 8 * 64 * 4 + 32 * 64 * 4
    assert footprint.activation_bytes > 0
    assert footprint.peak_live_bytes == storage.peak_live_bytes
    assert footprint.arena_bytes == storage.arena_size_bytes
    # The peak of the whole plan is at least the inputs it must hold while it
    # runs, so the split can never add up to less than the plan's own figure.
    assert footprint.peak_live_bytes <= footprint.input_bytes + footprint.activation_bytes
    assert "inputs" in footprint.describe()


def test_the_report_names_a_bound_for_every_policy() -> None:
    fn, inputs = _doubling_chain()
    compiled = aoto.compile_callable(fn, inputs)
    latency = compiled.plan.memory_report("latency")
    balanced = compiled.plan.memory_report("balanced")
    minimal = compiled.plan.memory_report("minimal")

    assert latency.sync_count == 0
    assert latency.peak_bound_bytes == (
        latency.planned_peak_bytes + latency.transient_bytes
    )
    assert minimal.threshold_bytes == 0
    assert minimal.peak_bound_bytes == minimal.planned_peak_bytes
    # The transient here is well past the balanced floor, so the policy really
    # does promise less than holding everything.
    assert balanced.threshold_bytes == max(
        plan_module.MIN_SYNC_BYTES, balanced.transient_bytes // plan_module.SYNC_BUDGET
    )
    assert balanced.peak_bound_bytes < latency.peak_bound_bytes
    assert balanced.sync_count > 0


def test_a_plan_below_the_floor_reports_no_worse_bound_than_lazy() -> None:
    compiled = aoto.compile_callable(_doubling_chain(2)[0], _doubling_chain(2)[1])
    latency = compiled.plan.memory_report("latency")
    balanced = compiled.plan.memory_report("balanced")

    assert balanced.sync_count == 0
    # The floor is larger than the whole transient, so the tightest true bound
    # is the same one the lazy policy has; a policy that promised more than the
    # traffic would be a lie.
    assert balanced.peak_bound_bytes == latency.peak_bound_bytes


def test_compare_fails_a_plan_that_starts_holding_more_memory() -> None:
    def report(planned: int, peak: int) -> dict[str, object]:
        return {
            "schema": bench.BENCH_SCHEMA,
            "device": {"name": "Apple M5 Pro"},
            "measurement": {"warmups": 3, "repeats": 11, "memory": "balanced"},
            "workloads": {
                "tiny": {
                    "input_ops": 1,
                    "lowered_ops": 1,
                    "kernels": 1,
                    "min_ms": 1.0,
                    "planned_peak_bytes": planned,
                    "peak_bound_bytes": planned,
                    "peak_extra_bytes": peak,
                    "first_call_ms": 1.0,
                    "compile_ms": 1.0,
                }
            },
        }

    comparison = bench.compare(report(1024, 512), report(1024, 512))
    assert comparison.ok

    # The planned figure is deterministic, so any growth is a regression with
    # no tolerance at all.
    assert not bench.compare(report(2048, 512), report(1024, 512)).ok
    # The measured peak is only gated once it is large enough to mean something.
    assert bench.compare(report(1024, 4096), report(1024, 512)).ok
    assert not bench.compare(report(1024, 3 << 20), report(1024, 1 << 20)).ok
