from __future__ import annotations

import pytest

from aotolith import bench
from aotolith.bench import Workload
from aotolith.cli import main


def _report(
    *,
    device: str = "Apple M5 Pro",
    workloads: dict[str, dict[str, float]] | None = None,
    reference: float = 2.0,
) -> dict[str, object]:
    return {
        "schema": bench.BENCH_SCHEMA,
        "device": {"name": device},
        "measurement": {"warmups": 0, "repeats": 1},
        "reference_ms": reference,
        "workloads": workloads or {},
        "skipped": {},
    }


def _entry(
    *,
    median_ms: float = 1.0,
    input_ops: int = 2,
    lowered_ops: int = 1,
    kernels: int = 1,
) -> dict[str, float]:
    return {
        "compile_ms": 10.0,
        "median_ms": median_ms,
        "min_ms": median_ms,
        "max_ms": median_ms,
        "ratio": median_ms / 2.0,
        "input_ops": input_ops,
        "lowered_ops": lowered_ops,
        "kernels": kernels,
    }


def test_compare_flags_a_growth_in_op_counts() -> None:
    """Op counts are deterministic, so any increase is a regression."""

    baseline = _report(workloads={"mlp": _entry(input_ops=8, lowered_ops=3, kernels=2)})
    current = _report(workloads={"mlp": _entry(input_ops=9, lowered_ops=4, kernels=3)})

    comparison = bench.compare(current, baseline)

    assert not comparison.ok
    assert comparison.timings_compared
    metrics = sorted(item.metric for item in comparison.regressions)
    assert metrics == ["input_ops", "kernels", "lowered_ops"]
    assert all(item.workload == "mlp" for item in comparison.regressions)


def test_compare_allows_a_fewer_op_lowering() -> None:
    """A better lowering shrinks the counts; that must not look like a failure."""

    baseline = _report(workloads={"mlp": _entry(input_ops=8, lowered_ops=3, kernels=2)})
    current = _report(workloads={"mlp": _entry(input_ops=8, lowered_ops=2, kernels=1)})

    comparison = bench.compare(current, baseline)

    assert comparison.ok
    assert comparison.regressions == ()


def test_compare_gates_timings_only_on_the_baseline_device() -> None:
    """A millisecond on one Apple GPU is not a millisecond on another."""

    baseline = _report(device="Apple M5 Pro", workloads={"op": _entry(median_ms=1.0)})
    slower = _report(device="Apple M5 Pro", workloads={"op": _entry(median_ms=1.5)})
    elsewhere = _report(device="Apple Paravirtual", workloads={"op": _entry(median_ms=1.5)})

    same = bench.compare(slower, baseline, tolerance=0.15)
    assert same.timings_compared
    assert [item.metric for item in same.regressions] == ["median_ms"]

    other = bench.compare(elsewhere, baseline, tolerance=0.15)
    assert not other.timings_compared
    assert other.ok
    assert any("kernel counts and timings not compared" in note for note in other.notes)


def test_compare_keeps_the_pipeline_gate_on_another_device() -> None:
    """Only the selection-dependent counts drop out; the pipeline counts do not."""

    baseline = _report(device="Apple M5 Pro", workloads={"op": _entry(input_ops=4)})
    elsewhere = _report(device="Apple Paravirtual", workloads={"op": _entry(input_ops=5)})

    comparison = bench.compare(elsewhere, baseline)

    assert not comparison.timings_compared
    assert [item.metric for item in comparison.regressions] == ["input_ops"]


def test_compare_tolerance_covers_noise_below_the_threshold() -> None:
    baseline = _report(workloads={"op": _entry(median_ms=1.0)})
    slightly_slower = _report(workloads={"op": _entry(median_ms=1.1)})

    assert bench.compare(slightly_slower, baseline, tolerance=0.15).ok
    assert not bench.compare(slightly_slower, baseline, tolerance=0.05).ok


def test_compare_rejects_a_schema_mismatch() -> None:
    baseline = _report()
    baseline["schema"] = bench.BENCH_SCHEMA + 1

    with pytest.raises(ValueError, match="re-record the baseline"):
        bench.compare(_report(), baseline)


def test_compare_reports_workloads_it_cannot_match() -> None:
    baseline = _report(workloads={"old": _entry()})
    current = _report(workloads={"new": _entry()})

    comparison = bench.compare(current, baseline)

    assert comparison.ok
    assert any("new: no baseline entry" in note for note in comparison.notes)
    assert any("old: in the baseline" in note for note in comparison.notes)


def _tiny_workload() -> tuple[object, tuple[object, ...]]:
    import torch

    return torch.nn.Linear(8, 8).eval(), (torch.randn(4, 8),)


def test_run_produces_a_report_that_matches_itself() -> None:
    report = bench.run(
        warmups=0,
        repeats=2,
        workloads=[Workload("tiny", _tiny_workload)],
    )

    assert report["schema"] == bench.BENCH_SCHEMA
    assert report["measurement"] == {"warmups": 0, "repeats": 2}
    assert report["reference_ms"] > 0
    entry = report["workloads"]["tiny"]
    assert entry["median_ms"] > 0
    assert entry["ratio"] == pytest.approx(entry["median_ms"] / report["reference_ms"])
    assert entry["input_ops"] >= 1
    assert entry["kernels"] >= 0
    assert report["skipped"] == {}

    comparison = bench.compare(report, report)
    assert comparison.ok


def test_run_skips_an_optional_workload_without_its_dependency() -> None:
    """A missing optional dependency is a skip, not a failure."""

    def _needs_nothing() -> tuple[object, tuple[object, ...]]:
        raise ImportError("no module named 'absent'")

    report = bench.run(
        warmups=0,
        repeats=1,
        workloads=[
            Workload("tiny", _tiny_workload),
            Workload("absent", _needs_nothing, optional=True),
        ],
    )

    assert report["skipped"] == {"absent": "no module named 'absent'"}
    assert set(report["workloads"]) == {"tiny"}


def test_run_refuses_to_skip_a_required_workload() -> None:
    def _needs_nothing() -> tuple[object, tuple[object, ...]]:
        raise ImportError("no module named 'absent'")

    with pytest.raises(ImportError, match="absent"):
        bench.run(warmups=0, repeats=1, workloads=[Workload("absent", _needs_nothing)])


def test_render_table_lists_every_workload_and_its_skips() -> None:
    report = bench.run(
        warmups=0,
        repeats=1,
        workloads=[Workload("tiny", _tiny_workload)],
    )
    report["skipped"] = {"absent": "no module named 'absent'"}

    text = bench.render_table(report)

    assert "tiny" in text
    assert "reference matmul" in text
    assert "skipped absent: no module named 'absent'" in text


def test_bench_command_rejects_an_unknown_workload_name(capsys) -> None:
    assert main(["bench", "--workload", "not-a-workload"]) == 1
    assert "unknown workload" in capsys.readouterr().err
