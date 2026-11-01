"""What a deployment can observe: the logger, the artifact log, and threads.

None of this changes a number the compiler computes. It exists because the first
question after "it is slow" or "it failed on the server" is "what did it decide",
and the answer used to live only in an object the operator did not have.
"""

from __future__ import annotations

import json
import logging
import platform as host_platform
import sys
import threading
import time
import zipfile
from pathlib import Path

import mlx.core as mx
import pytest
import torch

import aotolith
import aotolith as aoto
from aotolith import _log
from aotolith import artifact as artifact_module
from aotolith.errors import (
    ArtifactCompatibilityError,
    PlatformError,
)
from aotolith.runtime import plan as plan_module


def test_the_package_logger_is_quiet_and_namespaced() -> None:
    logger = _log.get_logger()
    assert logger.name == "aotolith"
    assert _log.get_logger("backends.apple").name == "aotolith.backends.apple"
    assert any(isinstance(item, logging.NullHandler) for item in logger.handlers)


def test_basic_config_adds_one_handler_and_sets_the_level() -> None:
    logger = _log.get_logger()
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
    _log.basic_config("INFO")
    _log.basic_config("INFO")
    streams = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.NullHandler)
    ]
    assert len(streams) == 1
    assert logger.level == logging.INFO
    # Leave the logger as the suite found it: a handler left open would print
    # for every later test.
    logger.removeHandler(streams[0])
    logger.setLevel(logging.WARNING)


def test_the_level_can_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("AOTOLITH_LOG", "debug")
    assert _log.level_from_environment() == logging.DEBUG
    monkeypatch.setenv("AOTOLITH_LOG", "25")
    assert _log.level_from_environment() == 25
    # A typo must not stop a process from starting.
    monkeypatch.setenv("AOTOLITH_LOG", "loud")
    assert _log.level_from_environment() is None
    monkeypatch.delenv("AOTOLITH_LOG")


def test_a_compile_logs_what_it_selected(caplog) -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value) + 1.0

    caplog.set_level(logging.INFO, logger="aotolith")
    aoto.compile_callable(fn, (torch.randn(4, 8),))
    messages = [record.getMessage() for record in caplog.records]
    summary = next(
        (message for message in messages if message.startswith("compiled ")), None
    )
    assert summary is not None, messages
    assert " selections on " in summary


def test_saving_and_loading_an_artifact_is_logged(tmp_path: Path, caplog) -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return value * 2.0

    compiled = aoto.compile_callable(fn, (torch.randn(4),))
    caplog.set_level(logging.INFO, logger="aotolith")
    path = compiled.save(tmp_path / "double.aoto")
    aoto.load_compiled_module(path)
    events = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("aotolith")
    ]
    assert any("artifact written" in event for event in events), events
    assert any("artifact read" in event for event in events), events


def test_a_format_four_artifact_still_loads(tmp_path: Path, caplog) -> None:
    """The reader claims format-4 support, so a format-4 file has to load.

    Format 4 wrote one output per kernel under singular keys. Rewriting a fresh
    artifact that way is the only way to produce one without keeping an old
    compiler around, and it tests the same code path a stored artifact takes.
    """

    def fn(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value)

    compiled = aoto.compile_callable(fn, (torch.randn(4),))
    saved = compiled.save(tmp_path / "current.aoto")

    legacy = tmp_path / "legacy.aoto"
    with zipfile.ZipFile(saved) as source:
        entries = {name: source.read(name) for name in source.namelist()}

    kernels = json.loads(entries["kernels.json"])
    entries["kernels.json"] = artifact_module._canonical_json(
        [
            {
                **{
                    key: item
                    for key, item in kernel.items()
                    if key not in {"output_ids", "tensor_types"}
                },
                "output_id": kernel["output_ids"][0],
                "tensor_type": kernel["tensor_types"][0],
            }
            for kernel in kernels
        ]
    )
    manifest = json.loads(entries["manifest.json"])
    manifest["format_version"] = 4
    # The payload changed shape, so its digest has to follow: otherwise the
    # reader is right to call the archive damaged, and the test would be
    # asserting on the wrong failure.
    manifest["payload_digests"]["kernels"] = artifact_module._digest(
        entries["kernels.json"]
    )
    entries["manifest.json"] = json.dumps(manifest).encode()

    with zipfile.ZipFile(legacy, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, payload in entries.items():
            target.writestr(name, payload)

    caplog.set_level(logging.WARNING, logger="aotolith")
    loaded = aoto.load_compiled_module(legacy)
    assert any(
        "older format" in record.getMessage() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]

    value = mx.array([1.0, -1.0, 2.0, 0.0], dtype=mx.float32)
    actual = loaded(value)[0]
    expected = torch.nn.functional.silu(torch.tensor([1.0, -1.0, 2.0, 0.0]))
    mx.eval(actual)
    assert mx.allclose(actual, mx.array(expected.numpy()), atol=1e-6)


def test_a_format_five_artifact_is_not_flagged_as_older(tmp_path: Path, caplog) -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return value + 1.0

    compiled = aoto.compile_callable(fn, (torch.randn(4),))
    saved = compiled.save(tmp_path / "current.aoto")
    caplog.set_level(logging.WARNING, logger="aotolith")
    aoto.load_compiled_module(saved)
    assert not any("older format" in record.getMessage() for record in caplog.records)


def test_a_future_format_is_refused(tmp_path: Path) -> None:
    def fn(value: torch.Tensor) -> torch.Tensor:
        return value + 1.0

    compiled = aoto.compile_callable(fn, (torch.randn(4),))
    saved = compiled.save(tmp_path / "current.aoto")
    future = tmp_path / "future.aoto"
    with zipfile.ZipFile(saved) as source, zipfile.ZipFile(
        future, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "manifest.json":
                manifest = json.loads(payload)
                manifest["format_version"] = 99
                payload = json.dumps(manifest).encode()
            target.writestr(name, payload)

    with pytest.raises(ArtifactCompatibilityError, match="format version"):
        aoto.load_compiled_module(future)


def test_concurrent_first_calls_build_the_runner_once(monkeypatch) -> None:
    """Two threads arriving on a fresh plan must not unroll it twice.

    Unrolling compiles nothing, but it does ``exec`` a thousand-line function per
    thread, and the duplicate is pure waste on the cold path a server hits at
    startup. The counter is what makes the assertion about the lock rather than
    about the answer.
    """

    def fn(value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value + 1.0) * 2.0

    compiled = aoto.compile_callable(fn, (torch.randn(8),))
    plan = compiled.plan
    assert plan._runner is None

    builds: list[int] = []
    original = plan_module._compile_plan_runner

    def counted(plan_to_build, memory):
        builds.append(1)
        # Widen the window between "checked _runner" and "assigned it", which is
        # exactly where the old code let a second thread in.
        time.sleep(0.05)
        return original(plan_to_build, memory)

    monkeypatch.setattr(plan_module, "_compile_plan_runner", counted)

    source = mx.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 1.5], dtype=mx.float32)
    results: list[list[float]] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def call() -> None:
        try:
            barrier.wait()
            outputs = plan(source, verify_index_bounds=False)
            mx.eval(outputs[0])
            results.append(outputs[0].tolist())
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(builds) == 1, f"the runner was unrolled {len(builds)} times"
    assert len(results) == 4
    assert all(result == results[0] for result in results)


def test_a_graph_that_cannot_unroll_still_runs(monkeypatch) -> None:
    """The interpreter remains the correctness path, not just the slow path."""

    def fn(value: torch.Tensor) -> torch.Tensor:
        return value * 2.0

    compiled = aoto.compile_callable(fn, (torch.randn(4),))
    monkeypatch.setattr(
        plan_module, "_compile_plan_runner", lambda plan, memory: None
    )
    outputs = compiled.plan(mx.zeros((4,), dtype=mx.float32))
    mx.eval(outputs[0])
    assert outputs[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert isinstance(compiled.plan._runner, plan_module._NoRunner)


def test_a_plan_can_be_called_from_several_threads_at_once() -> None:
    def fn(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(value, weight)

    table = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    indices = torch.tensor([0, 11, 3], dtype=torch.int64)
    compiled = aoto.compile_callable(fn, (indices, table))
    source = mx.array(table.numpy())

    def call(seed: int) -> list[float]:
        chosen = mx.array([seed % 12, (seed + 1) % 12, (seed + 2) % 12], dtype=mx.int64)
        output = compiled.plan(chosen, source)[0]
        mx.eval(output)
        return output.tolist()

    expected = [call(seed) for seed in range(8)]
    observed: dict[int, list[float]] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(seed: int) -> None:
        try:
            value = call(seed)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
            return
        with lock:
            observed[seed] = value

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert [observed[seed] for seed in range(8)] == expected


def test_a_non_apple_host_is_a_platform_error(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(host_platform, "machine", lambda: "x86_64")
    with pytest.raises(PlatformError, match="macOS on Apple Silicon") as caught:
        aotolith._require_supported_host()
    assert caught.value.code == PlatformError.code
    assert "linux/x86_64" in str(caught.value)


def test_an_intel_mac_is_a_platform_error(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(host_platform, "machine", lambda: "x86_64")
    with pytest.raises(PlatformError, match="darwin/x86_64"):
        aotolith._require_supported_host()


def test_this_host_passes_the_guard() -> None:
    # The suite only runs where the backend can run, so this is the check that
    # the guard is not simply refusing everything.
    aotolith._require_supported_host()
