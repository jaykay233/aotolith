from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aotolith import BaselineError, bench
from aotolith.bench import Workload
from aotolith.cli import main


def _report(
    *,
    device: str = "Apple M5 Pro",
    workloads: dict[str, dict[str, float]] | None = None,
    reference: float = 2.0,
    repeats: int = 11,
) -> dict[str, object]:
    reference_floor = reference / 2.0
    payload = workloads or {}
    for entry in payload.values():
        entry["min_ratio"] = entry["min_ms"] / reference_floor
    return {
        "schema": bench.BENCH_SCHEMA,
        "device": {"name": device},
        "measurement": {"warmups": 0, "repeats": repeats},
        "reference_ms": reference,
        "reference_min_ms": reference_floor,
        "workloads": payload,
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
    assert [item.metric for item in same.regressions] == ["min_ratio"]

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
        "min_ratio"
    ]


def test_compare_blames_the_machine_when_everything_is_slower() -> None:
    """A busy host must not be reported as a compiler regression.

    Measured here: two runs of unchanged code gave the same 0.34 ms kernel a
    floor of 0.344 and 0.431 ms while the reference matmul went from 0.89 to
    1.56 ms. Comparing milliseconds called that a regression; comparing the
    ratio against the reference taken at the same moment calls it noise.
    """

    baseline = _report(device="Apple M5 Pro", workloads={"op": _entry(median_ms=1.0)})
    loaded = _report(
        device="Apple M5 Pro",
        reference=4.0,
        workloads={"op": _entry(median_ms=1.6, min_ms=1.55)},
    )

    comparison = bench.compare(loaded, baseline, tolerance=0.15)

    assert comparison.ok
    assert any("is inside the tolerance" in note for note in comparison.notes)


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
    assert report["measurement"] == {
        "warmups": 0,
        "repeats": 2,
        "rounds": 1,
        "memory": "balanced",
    }
    assert report["reference_ms"] > 0
    assert report["reference_min_ms"] <= report["reference_ms"]
    entry = report["workloads"]["tiny"]
    assert entry["median_ms"] > 0
    assert entry["first_call_ms"] > 0
    assert entry["peak_extra_bytes"] > 0
    assert entry["planned_peak_bytes"] > 0
    assert entry["peak_bound_bytes"] >= entry["planned_peak_bytes"]
    # One workload and one round is one reference measurement, so the ratio is
    # exactly the median over that reference; with more slots each ratio is
    # measured against a reference taken at the same moment instead.
    assert entry["ratio"] == pytest.approx(entry["median_ms"] / report["reference_ms"])
    assert entry["min_ratio"] == pytest.approx(
        entry["min_ms"] / report["reference_min_ms"]
    )
    assert entry["input_ops"] >= 1
    assert entry["kernels"] >= 0
    assert report["skipped"] == {}

    comparison = bench.compare(report, report)
    assert comparison.ok


def test_run_collects_the_samples_of_every_round() -> None:
    """A baseline floor is a best case, so extra rounds are what stabilize it."""

    report = bench.run(
        warmups=0,
        repeats=2,
        rounds=3,
        workloads=[Workload("tiny", _tiny_workload)],
    )

    assert report["measurement"]["rounds"] == 3
    entry = report["workloads"]["tiny"]
    assert entry["min_ms"] <= entry["median_ms"] <= entry["max_ms"]
    # Nine samples make a wider spread than two, which is the point.
    assert entry["max_ms"] >= entry["min_ms"]


def test_the_reference_is_sampled_between_the_workload_samples() -> None:
    """The two numbers a floor divides must share a moment.

    When the reference was timed once per round, after the whole batch of
    workload samples, one round measured the same 0.2 ms attention kernel
    against a 1.15 ms reference and the next against 0.78 ms. The first round's
    ``min_ratio`` was a floor no later run could reach, so the gate failed on
    unchanged code. Interleaving is what removes that skew, and the order of
    calls is the whole mechanism, so it is asserted directly.
    """

    calls: list[str] = []

    def reference() -> float:
        calls.append("reference")
        return 1.0

    # ``mx.eval`` on a tuple of MLX scalars is what the harness always evaluates,
    # so the plan has to return one.
    import mlx.core as mx

    class RealPlan:
        def __call__(self, *inputs: object) -> tuple[object, ...]:
            calls.append("workload")
            return (mx.array(1.0),)

    samples, references = bench._timed(RealPlan(), (object(),), 3, reference)
    assert calls == ["workload", "reference"] * 3
    assert len(samples) == len(references) == 3
    # Without a reference callable the workload is timed alone, which is what
    # the first-call measurement does.
    calls.clear()
    samples, references = bench._timed(RealPlan(), (object(),), 2)
    assert calls == ["workload", "workload"]
    assert references == []


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


def test_a_bucketed_workload_sums_its_counts_over_the_whole_set() -> None:
    """The gate has to see every bucket, not just the one a sample ran on.

    An extra kernel in the widest bucket is exactly as much a regression as one
    in a single plan, so the entry reports the sum, and the memory figures are
    the widest bucket's, which is what an admission decision clears.
    """

    report = bench.run(
        warmups=0,
        repeats=2,
        workloads=[Workload("bucketed", _tiny_workload, buckets=(2, 4))],
    )

    entry = report["workloads"]["bucketed"]
    assert entry["buckets"] == 2
    assert entry["widths"] == [2, 4]
    # Two plans' worth of ops and kernels, even though the example input is one
    # row and the timing runs on one bucket.
    assert entry["input_ops"] == 2 * _tiny_ops()
    assert entry["kernels"] == 2 * _tiny_kernels()
    assert entry["peak_bound_bytes"] >= 1


def _tiny_ops() -> int:
    """Ops in the tiny workload's input graph, from a plain compile."""

    from aotolith import compile_module

    module, inputs = _tiny_workload()
    return len(compile_module(module, inputs).input_graph.ops)


def _tiny_kernels() -> int:
    from aotolith import compile_module

    module, inputs = _tiny_workload()
    return len(compile_module(module, inputs).plan.kernels)


# -- the chip matrix -------------------------------------------------------------
#
# A baseline directory is the gap matrix's answer to "M1 through M4": one file
# per chip, named by the chip, chosen by the chip. None of this needs a GPU, so
# all of it is testable on any machine - which is the point, because the
# numbers themselves can only be recorded where the hardware is.


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Apple M5 Pro", "apple-m5-pro"),
        ("Apple M2", "apple-m2"),
        ("Apple M4 Max", "apple-m4-max"),
        ("Apple M1 Ultra", "apple-m1-ultra"),
        ("  Apple  M3   Pro ", "apple-m3-pro"),
        ("Apple M2 (GPU)", "apple-m2-gpu"),
    ],
)
def test_a_chip_name_becomes_a_file_name(name: str, slug: str) -> None:
    assert bench.device_slug(name) == slug


def test_an_unnamed_device_cannot_name_a_file() -> None:
    # Better to refuse than to write ".json" and have every unknown machine
    # share one file.
    with pytest.raises(BaselineError) as caught:
        bench.device_slug(None)
    assert "unnamed device" in str(caught.value)
    with pytest.raises(BaselineError):
        bench.device_slug("   ")


def _matrix(directory, *devices: str):
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for device in devices:
        (directory / f"{bench.device_slug(device)}.json").write_text(
            json.dumps(
                _report(
                    device=device,
                    workloads={
                        "projection": _entry(input_ops=4, lowered_ops=4, kernels=1)
                    },
                )
            )
        )
    return directory


def test_the_matrix_picks_the_file_for_this_chip(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro", "Apple M3 Max")

    chosen = bench.resolve_baseline(directory, device="Apple M3 Max")

    assert chosen.path.name == "apple-m3-max.json"
    assert chosen.same_device is True
    assert chosen.note is None


def test_a_chip_without_a_file_still_gates_the_deterministic_half(tmp_path) -> None:
    # "We have not recorded an M4 yet" must not mean "anything goes on an M4".
    # Op counts and the planned peak are properties of the graph, so they are
    # gated from another chip's file; the milliseconds are not.
    directory = _matrix(tmp_path, "Apple M5 Pro")

    chosen = bench.resolve_baseline(directory, device="Apple M4 Pro")

    assert chosen.path.name == "apple-m5-pro.json"
    assert chosen.same_device is False
    assert "Apple M4 Pro" in (chosen.note or "")
    assert "apple-m5-pro.json" in (chosen.note or "")

    current = _report(
        device="Apple M4 Pro",
        workloads={"projection": _entry(input_ops=5, lowered_ops=4, kernels=1)},
    )
    comparison = bench.compare(current, chosen.report)

    assert comparison.timings_compared is False
    assert [(item.workload, item.metric) for item in comparison.regressions] == [
        ("projection", "input_ops")
    ]


def test_a_file_named_for_one_chip_but_recording_another_is_not_trusted(tmp_path) -> None:
    # The failure this guard exists for: someone copies a baseline and renames
    # it. The name says M4, the numbers came from an M5, and gating M4
    # milliseconds against M5 milliseconds would fail the wrong machine.
    directory = _matrix(tmp_path, "Apple M5 Pro")
    (directory / "apple-m4-pro.json").write_text(
        (directory / "apple-m5-pro.json").read_text()
    )

    chosen = bench.resolve_baseline(directory, device="Apple M4 Pro")

    assert chosen.same_device is False
    assert "records 'Apple M5 Pro'" in (chosen.note or "")


def test_an_explicit_file_is_used_as_given(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")
    target = directory / "apple-m5-pro.json"

    chosen = bench.resolve_baseline(target, device="Apple M3 Max")

    assert chosen.path == target
    assert chosen.same_device is True
    # An explicit path is an explicit decision; ``compare`` still refuses to
    # gate timings across devices on its own.
    assert bench.compare(
        _report(device="Apple M3 Max", workloads={"projection": _entry()}),
        chosen.report,
    ).timings_compared is False


def test_an_empty_matrix_is_an_error_not_a_pass(tmp_path) -> None:
    # The one failure mode a gate cannot have: comparing nothing and reporting
    # success.
    with pytest.raises(BaselineError) as caught:
        bench.resolve_baseline(tmp_path, device="Apple M5 Pro")

    assert "no baseline files" in str(caught.value)


def test_a_valid_matrix_reports_one_slug_per_chip(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro", "Apple M3 Max", "Apple M2")

    assert bench.validate_baselines(directory) == (
        "apple-m2",
        "apple-m3-max",
        "apple-m5-pro",
    )


def test_the_shipped_matrix_validates() -> None:
    from pathlib import Path

    shipped = Path(__file__).resolve().parent.parent / "benchmarks" / "baselines"

    assert bench.validate_baselines(shipped)


def test_validation_names_the_file_and_the_problem(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")
    report = json.loads((directory / "apple-m5-pro.json").read_text())

    stale = json.loads(json.dumps(report))
    stale["schema"] = bench.BENCH_SCHEMA - 1
    (directory / "apple-m1.json").write_text(json.dumps(stale))

    with pytest.raises(BaselineError) as caught:
        bench.validate_baselines(directory)

    message = str(caught.value)
    assert "apple-m1.json" in message
    assert f"schema {bench.BENCH_SCHEMA - 1} is not {bench.BENCH_SCHEMA}" in message


def test_validation_rejects_a_chip_recorded_twice(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")
    (directory / "apple-m5-pro-copy.json").write_text(
        (directory / "apple-m5-pro.json").read_text()
    )

    with pytest.raises(BaselineError) as caught:
        bench.validate_baselines(directory)

    assert "is also recorded in" in str(caught.value)


def test_validation_rejects_a_report_with_no_repeats(tmp_path) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")
    report = json.loads((directory / "apple-m5-pro.json").read_text())
    report["measurement"]["repeats"] = 0
    (directory / "apple-m5-pro.json").write_text(json.dumps(report))

    with pytest.raises(BaselineError) as caught:
        bench.validate_baselines(directory)

    assert "cannot gate" in str(caught.value)


def test_the_cli_checks_a_matrix_without_measuring(tmp_path, capsys) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")

    assert main(["bench", "--check-baselines", str(directory)]) == 0
    assert "apple-m5-pro" in capsys.readouterr().out


def test_the_cli_reports_a_bad_matrix_as_a_coded_error(tmp_path, capsys) -> None:
    directory = _matrix(tmp_path, "Apple M5 Pro")
    (directory / "apple-m4-pro.json").write_text(
        (directory / "apple-m5-pro.json").read_text()
    )

    assert main(["bench", "--check-baselines", str(directory)]) == 1
    assert "AOTO-6001" in capsys.readouterr().err


def test_recording_into_a_matrix_names_the_file_after_the_chip(tmp_path) -> None:
    directory = tmp_path / "baselines"
    directory.mkdir()

    recorded = _bench_cli(
        "--workload", _CLI_WORKLOAD, "--repeats", _CLI_REPEATS, "--warmups", "1",
        "--update-baseline", str(directory),
    )

    assert recorded.returncode == 0, recorded.stderr
    written = sorted(entry.name for entry in directory.iterdir())
    report = json.loads((directory / written[0]).read_text())
    # The file is the chip's identity, so the name has to be the slug of the
    # device the report recorded - not a name a flag guessed.
    assert written == [f"{bench.device_slug(report['device']['name'])}.json"]


def test_recording_into_a_matrix_refuses_to_replace_another_chip(tmp_path) -> None:
    # A copied baseline plus a recording on a different machine is how a matrix
    # silently loses a chip's numbers. The guard is that the file names its
    # device and the run names its own: this machine's file exists, and it says
    # it came from another chip.
    current = bench._device().get("name")
    if not current:
        pytest.skip("no device to name a baseline after")

    directory = tmp_path / "baselines"
    directory.mkdir()
    taken = directory / f"{bench.device_slug(current)}.json"
    taken.write_text(
        json.dumps(
            _report(
                device="Apple M1 Max",
                workloads={"projection": _entry(input_ops=4, lowered_ops=4)},
            )
        )
    )
    before = taken.read_text()

    recorded = _bench_cli(
        "--workload", _CLI_WORKLOAD, "--repeats", _CLI_REPEATS, "--warmups", "1",
        "--update-baseline", str(directory),
    )

    assert recorded.returncode == 1
    assert "AOTO-6001" in recorded.stderr
    assert taken.read_text() == before
    assert sorted(entry.name for entry in directory.iterdir()) == [taken.name]
