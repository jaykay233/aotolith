"""Fixed-workload benchmarks and the regression gate built on them.

The compiler's own speed is not the number a user cares about, but it is the
number that notices a bad change first: a pass that starts materializing a
second copy of every buffer, a kernel selection that silently loses the
tensor-core path, or an epilogue fold that stops firing. This module measures
the same handful of shapes on every run and can compare the result against a
recorded baseline.

Two properties make the gate usable, and both matter:

* **Front-end and mid-end op counts are deterministic.** ``input_ops`` and
  ``lowered_ops`` do not depend on the machine, the load, or the phase of the
  moon, so the gate compares them exactly, everywhere. A pipeline regression
  that adds a copy, or removes a fusion, fails here with no timing noise at
  all. ``kernels`` counts the selected native kernels instead, which follows
  what the GPU can actually load, so it is gated only on the baseline's device.
* **Timings are only comparable on the same GPU.** Absolute milliseconds are
  recorded and gated, but only when the baseline names the same device. On a
  different device the gate falls back to reporting ``ratio`` -- the workload's
  median against one raw MLX matmul measured in the same process -- which
  cancels most of the difference between machines without pretending to be
  precise.

Nothing here needs network access or downloaded weights: every workload is a
small torch module with seeded inputs, so the gate runs in the same environment
as the test suite.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from .compiler import compile_module

BENCH_SCHEMA = 1

# One raw matmul dominates the reference measurement so that its own overhead
# is negligible, and it is short enough to repeat twenty times in a second.
_REFERENCE_SHAPE = (2048, 2048, 2048)
_REFERENCE_DTYPE = mx.bfloat16


@dataclass(frozen=True, slots=True)
class Workload:
    """One fixed shape, and the compiler path it is meant to exercise."""

    name: str
    build: Callable[[], tuple[Any, tuple[Any, ...]]]
    codegen_matmul: bool = False
    codegen_attention: bool = False
    #: A workload whose builder needs an optional dependency. When the builder
    #: raises ``ImportError`` the workload is reported as skipped instead of
    #: failing the run.
    optional: bool = False


def _linear() -> tuple[Any, tuple[Any, ...]]:
    """One large projection: the plain matmul path, on the generated kernel."""

    import torch

    torch.manual_seed(0)
    module = torch.nn.Linear(1024, 1024).eval()
    return module, (torch.randn(256, 1024),)


def _gated_mlp() -> tuple[Any, tuple[Any, ...]]:
    """A gated MLP: exercises the paired-silu epilogue fold.

    Both projections consume the same normalized input, so the pipeline folds
    them into one launch. If that fold stops matching, ``kernels`` grows and the
    gate fails on the count rather than on a timing sample.
    """

    import torch
    import torch.nn.functional as F

    torch.manual_seed(1)

    class Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(1024, 4096, bias=False)
            self.up = torch.nn.Linear(1024, 4096, bias=False)
            self.down = torch.nn.Linear(4096, 1024, bias=False)

        def forward(self, hidden: Any) -> Any:
            return self.down(F.silu(self.gate(hidden)) * self.up(hidden))

    return Mlp().eval(), (torch.randn(8, 1024),)


def _rms_norm() -> tuple[Any, tuple[Any, ...]]:
    """A bandwidth-bound reduction: the shape every layer boundary pays for."""

    import torch

    torch.manual_seed(2)

    class RmsNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4096))

        def forward(self, hidden: Any) -> Any:
            variance = hidden.pow(2).mean(-1, keepdim=True)
            return hidden * torch.rsqrt(variance + 1e-6) * self.weight

    return RmsNorm().eval(), (torch.randn(32, 4096),)


def _causal_attention() -> tuple[Any, tuple[Any, ...]]:
    """Grouped-query causal attention, on the generated attention kernel."""

    import torch
    import torch.nn.functional as F

    class Attention(torch.nn.Module):
        def forward(self, query: Any, key: Any, value: Any) -> Any:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                enable_gqa=True,
            )

    return Attention().eval(), (
        torch.randn(1, 8, 128, 64),
        torch.randn(1, 2, 128, 64),
        torch.randn(1, 2, 128, 64),
    )


def _qwen2_prefill() -> tuple[Any, tuple[Any, ...]]:
    """The whole Qwen2 adapter at one bucket width, end to end.

    A tiny random model keeps this dependency-light; what it gates is the
    adapter's lowering, not its accuracy.
    """

    import torch
    import transformers

    from .models.qwen2 import Qwen2PrefillModule, prefill_mask

    config = transformers.Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        use_cache=True,
    )
    torch.manual_seed(3)
    model = transformers.Qwen2ForCausalLM(config).eval()
    module = Qwen2PrefillModule(model, cache_capacity=128).eval()
    mask = torch.from_numpy(prefill_mask(128, 128))
    return module, (
        torch.zeros((1, 128), dtype=torch.int64),
        torch.arange(128, dtype=torch.int64)[None, :],
        mask,
    )


WORKLOADS: tuple[Workload, ...] = (
    Workload("linear_1024x1024", _linear, codegen_matmul=True),
    Workload("gated_mlp_1024_4096", _gated_mlp),
    Workload("rms_norm_32x4096", _rms_norm),
    Workload("attention_causal_128", _causal_attention, codegen_attention=True),
    Workload("qwen2_prefill_128", _qwen2_prefill, optional=True),
)


def _reference_ms(warmups: int, repeats: int) -> float:
    """Time one raw MLX matmul, the per-process speed unit.

    Everything else in the report is also a ratio against this, so a busy or
    slow machine moves the whole report together instead of looking like a
    compiler regression.
    """

    rng = np.random.default_rng(0)
    batch, inner, out = _REFERENCE_SHAPE
    left = mx.array(rng.standard_normal((batch, inner), dtype=np.float32)).astype(
        _REFERENCE_DTYPE
    )
    right = mx.array(rng.standard_normal((inner, out), dtype=np.float32)).astype(
        _REFERENCE_DTYPE
    )
    for _ in range(warmups):
        mx.eval(mx.matmul(left, right))
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(mx.matmul(left, right))
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def measure(
    workload: Workload,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    """Compile and time one workload."""

    module, torch_inputs = workload.build()
    compile_start = time.perf_counter()
    compiled = compile_module(
        module,
        torch_inputs,
        codegen_matmul=workload.codegen_matmul,
        codegen_attention=workload.codegen_attention,
    )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    inputs = tuple(mx.array(value.detach().numpy()) for value in torch_inputs)
    for _ in range(warmups):
        mx.eval(*compiled(*inputs))
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = compiled(*inputs)
        mx.eval(*outputs)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "compile_ms": compile_ms,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "input_ops": len(compiled.input_graph.ops),
        "lowered_ops": len(compiled.lowered_graph.ops),
        "kernels": len(compiled.plan.kernels),
    }


def _device() -> dict[str, Any]:
    try:
        info = mx.device_info()
    except (RuntimeError, ValueError):
        return {}
    return {
        "name": info.get("device_name"),
        "architecture": info.get("architecture"),
        "memory_bytes": info.get("memory_size"),
    }


def run(
    *,
    warmups: int = 3,
    repeats: int = 11,
    workloads: Iterable[Workload] = WORKLOADS,
) -> dict[str, Any]:
    """Measure every workload and return one report.

    The report is the unit the gate compares, so it holds the measurement
    settings alongside the numbers: a baseline recorded with one repeat is not
    a fair reference for a run with twenty.
    """

    if warmups < 0 or repeats < 1:
        raise ValueError(
            f"need warmups >= 0 and repeats >= 1, got {warmups}/{repeats}"
        )
    reference = _reference_ms(warmups, repeats)
    measured: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for workload in workloads:
        try:
            entry = measure(workload, warmups=warmups, repeats=repeats)
        except ImportError as exc:
            if not workload.optional:
                raise
            skipped[workload.name] = str(exc)
            continue
        entry["ratio"] = entry["median_ms"] / reference
        measured[workload.name] = entry
    return {
        "schema": BENCH_SCHEMA,
        "device": _device(),
        "measurement": {"warmups": warmups, "repeats": repeats},
        "reference_ms": reference,
        "workloads": measured,
        "skipped": skipped,
    }


@dataclass(frozen=True, slots=True)
class Regression:
    """One way the current run is worse than the baseline."""

    workload: str
    metric: str
    baseline: float
    current: float

    def describe(self) -> str:
        return (
            f"{self.workload}: {self.metric} went from {self.baseline:.4g} "
            f"to {self.current:.4g}"
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    regressions: tuple[Regression, ...] = ()
    notes: tuple[str, ...] = ()
    #: True when the baseline was recorded on this same device, so absolute
    #: timings were compared rather than just the deterministic counts.
    timings_compared: bool = False

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerance: float = 0.15,
) -> Comparison:
    """Compare one report against a baseline.

    Op counts may only go down: ``input_ops`` and ``lowered_ops`` are
    deterministic, so any increase is a pipeline regression and no tolerance
    applies. ``kernels`` and timings are gated only when both reports name the
    same device, because kernel selection follows what the GPU can load and a
    millisecond is not portable; a tolerance is the fraction of slowdown that
    is allowed.
    """

    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if current.get("schema") != baseline.get("schema"):
        raise ValueError(
            f"baseline schema {baseline.get('schema')} does not match "
            f"report schema {current.get('schema')}; re-record the baseline"
        )

    notes: list[str] = []
    regressions: list[Regression] = []
    current_device = (current.get("device") or {}).get("name")
    baseline_device = (baseline.get("device") or {}).get("name")
    timings_compared = bool(current_device) and current_device == baseline_device
    if not timings_compared:
        notes.append(
            f"kernel counts and timings not compared: baseline device "
            f"{baseline_device!r} is not {current_device!r}; the ratio column is "
            f"still reported"
        )
    # The front end and the mid end do not depend on the device, so these two
    # counts are the part of the gate that holds everywhere.
    deterministic = ("input_ops", "lowered_ops")
    device_dependent = ("kernels",)

    current_workloads = current.get("workloads") or {}
    baseline_workloads = baseline.get("workloads") or {}
    for name in sorted(current_workloads):
        if name not in baseline_workloads:
            notes.append(f"{name}: no baseline entry, not compared")
            continue
        now = current_workloads[name]
        then = baseline_workloads[name]
        metrics = deterministic + (device_dependent if timings_compared else ())
        for metric in metrics:
            if now[metric] > then[metric]:
                regressions.append(
                    Regression(name, metric, then[metric], now[metric])
                )
        if timings_compared:
            allowed = then["median_ms"] * (1.0 + tolerance)
            if now["median_ms"] > allowed:
                regressions.append(
                    Regression(name, "median_ms", then["median_ms"], now["median_ms"])
                )
    for name in sorted(set(baseline_workloads) - set(current_workloads)):
        notes.append(f"{name}: in the baseline but not measured now")
    return Comparison(
        regressions=tuple(regressions),
        notes=tuple(notes),
        timings_compared=timings_compared,
    )


_TABLE_HEADER = (
    "workload",
    "median ms",
    "ratio",
    "compile ms",
    "input ops",
    "lowered ops",
    "kernels",
)


def render_table(report: Mapping[str, Any]) -> str:
    """Render a report as fixed-width text for a terminal or a CI log."""

    rows = [_TABLE_HEADER]
    for name, entry in (report.get("workloads") or {}).items():
        rows.append(
            (
                name,
                f"{entry['median_ms']:.3f}",
                f"{entry['ratio']:.3f}",
                f"{entry['compile_ms']:.0f}",
                str(entry["input_ops"]),
                str(entry["lowered_ops"]),
                str(entry["kernels"]),
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = [
        "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        for row in rows
    ]
    reference = report.get("reference_ms")
    if reference is not None:
        lines.append(f"reference matmul: {reference:.3f} ms (ratio 1.0)")
    for name, reason in (report.get("skipped") or {}).items():
        lines.append(f"skipped {name}: {reason}")
    return "\n".join(lines)


def summarize(comparison: Comparison) -> str:
    """Render a comparison as the block the CLI prints after the table."""

    lines: list[str] = []
    lines.extend(f"note: {note}" for note in comparison.notes)
    if comparison.ok:
        lines.append(
            "gate: ok"
            + (" (op counts and timings)" if comparison.timings_compared else " (op counts)")
        )
    else:
        lines.extend(f"regression: {item.describe()}" for item in comparison.regressions)
        lines.append(f"gate: {len(comparison.regressions)} regression(s)")
    return "\n".join(lines)
