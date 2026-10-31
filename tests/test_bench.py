from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aotolith import bench
from aotolith.bench import Workload
from aotolith.cli import main


def _report(
    *,
    device: str = "Apple M5 Pro",
    workloads: dict[str, dict[str, float]] | None = None,
    reference: float = 2.0,
    repeats: int = 11,
) -> dict[str, object]:
    return {
        "schema": bench.BENCH_SCHEMA,
        "device": {"name": device},
        "measurement": {"warmups": 0, "repeats": repeats},
        "reference_ms": reference,
        "workloads": workloads or {},
        "skipped": {},
    }


def _entry(
    *,
    median_ms: float = 1.0,
    min_ms: float | None = None,
    max_ms: float | None = None,
    input_ops: int = 2,
    lowered_ops: int = 1,
    kernels: int = 1,
) -> dict[str, float]:
    best = median_ms if min_ms is None else min_ms
    worst = median_ms if max_ms is None else max_ms
    return {
        "compile_ms": 10.0,
        "median_ms": median_ms,
        "min_ms": best,
        "max_ms": worst,
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
    assert [item.metric for item in same.regressions] == ["min_ms"]

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


def test_compare_gates_timings_on_the_best_sample_not_the_median() -> None:
    """Noise inflates the median and leaves the floor alone.

    Measured on this machine, an unchanged 0.35 ms kernel produced medians of
    1.14, 0.38 and 0.36 ms across three runs while its minimum stayed at
    0.38, 0.35 and 0.34. Gating the median would report a 3x compiler
    regression; gating the minimum reports none.
    """

    baseline = _report(workloads={"op": _entry(median_ms=0.35, min_ms=0.34)})
    noisy = _report(
        workloads={"op": _entry(median_ms=1.14, min_ms=0.38, max_ms=1.5)}
    )

    assert bench.compare(noisy, baseline, tolerance=0.15).ok

    # A real slowdown raises the floor too, and that is what fails.
    slower_floor = _report(workloads={"op": _entry(median_ms=0.40, min_ms=0.50)})
    assert [item.metric for item in bench.compare(slower_floor, baseline).regressions] == [
        "min_ms"
    ]


def test_compare_skips_timings_a_single_sample_cannot_support() -> None:
    """`repeats=1` has no median to compare, so only the counts are gated.

    This is not a technicality: two single-shot runs of an unchanged kernel on
    a shared runner differed by 1.6x, which would have been reported as a
    compiler regression.
    """

    baseline = _report(workloads={"op": _entry(median_ms=1.0)}, repeats=1)
    current = _report(workloads={"op": _entry(median_ms=3.0)}, repeats=1)

    comparison = bench.compare(current, baseline)

    assert comparison.ok
    assert not comparison.timings_compared
    assert any("at least 3 repeats" in note for note in comparison.notes)

    # Kernel counts depend on the GPU but not on the sample count, so a device
    # match still gates them.
    regressed = _report(workloads={"op": _entry(median_ms=3.0, kernels=4)}, repeats=1)
    assert [item.metric for item in bench.compare(regressed, baseline).regressions] == [
        "kernels"
    ]


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


# The cheapest real workload, used to exercise the CLI's exit codes end to end.
_CLI_WORKLOAD = "rms_norm_32x4096"
# Enough repeats that the timing half of the gate is allowed to run; a
# single sample would be skipped for having no best case to compare.
_CLI_REPEATS = "3"


def _bench_cli(*args: str):
    """Run `python -m aotolith bench` the way a user or CI does.

    A subprocess, not `main()` in this process, so each run starts with a cold
    MLX context like the real command. Calling `run()` repeatedly inside one
    process measured the same 0.15 ms kernel at up to 6x its floor, which is
    not a property of the compiler and not something these tests should be
    sensitive to.
    """

    return subprocess.run(
        [sys.executable, "-m", "aotolith", "bench", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _recorded_baseline(tmp_path, *, shrink_ops: int = 0):
    """Record a baseline through the CLI, optionally tightened by hand.

    Recording through the CLI rather than building the JSON here keeps the test
    honest about the file the gate actually reads back.
    """

    path = tmp_path / "baseline.json"
    recorded = _bench_cli(
        "--workload", _CLI_WORKLOAD, "--repeats", _CLI_REPEATS, "--warmups", "1",
        "--update-baseline", str(path),
    )
    assert recorded.returncode == 0, recorded.stderr
    if shrink_ops:
        payload = json.loads(path.read_text())
        entry = payload["workloads"][_CLI_WORKLOAD]
        entry["input_ops"] -= shrink_ops
        entry["lowered_ops"] -= shrink_ops
        path.write_text(json.dumps(payload))
    return path


def test_bench_command_passes_against_its_own_baseline(tmp_path) -> None:
    """Rerunning an unchanged compiler still exits 0.

    The tolerance is wide on purpose. This asserts the end-to-end path -- record
    a baseline, read it back, compare equal counts, exit 0 -- and the machine
    running a test suite is not a quiet benchmark host. The timing half of the
    gate is pinned by the synthetic reports above, where the numbers are exact
    and the only variable is what the gate does with them.
    """

    baseline = _recorded_baseline(tmp_path)

    compared = _bench_cli(
        "--workload", _CLI_WORKLOAD, "--repeats", _CLI_REPEATS, "--warmups", "1",
        "--tolerance", "100", "--baseline", str(baseline),
    )

    assert compared.returncode == 0, compared.stdout + compared.stderr
    assert "gate: ok" in compared.stdout


def test_bench_command_fails_the_gate_on_an_op_count_regression(tmp_path) -> None:
    """The threshold check is a process exit code, not just a printed note.

    This is what makes the CI job a gate: a pass that drops a fusion grows the
    op count, and the step has to fail rather than log a warning. The assertion
    names the two regressions rather than counting them, because whether a
    timing regression also fires depends on the machine running the test.
    """

    baseline = _recorded_baseline(tmp_path, shrink_ops=1)

    compared = _bench_cli(
        "--workload", _CLI_WORKLOAD, "--repeats", _CLI_REPEATS, "--warmups", "1",
        "--tolerance", "100", "--baseline", str(baseline),
    )

    assert compared.returncode == 1
    assert f"regression: {_CLI_WORKLOAD}: input_ops went from" in compared.stdout
    assert f"regression: {_CLI_WORKLOAD}: lowered_ops went from" in compared.stdout
    assert "gate: ok" not in compared.stdout
