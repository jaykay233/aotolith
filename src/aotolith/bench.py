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
  recorded and gated, but only when the baseline names the same device and both
  reports have more than one sample. On a different device the gate falls back
  to reporting ``ratio`` -- the workload's median against one raw MLX matmul
  measured in the same process -- which cancels most of the difference between
  machines without pretending to be precise.

Nothing here needs network access or downloaded weights: every workload is a
small torch module with seeded inputs, so the gate runs in the same environment
as the test suite.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .backends import REFERENCE_TARGET
from .errors import BaselineError
from .buckets import BucketedPlans, compile_bucketed
from .compiler import CompiledModulePlan, compile_module
from .runtime.plan import DEFAULT_MEMORY

BENCH_SCHEMA = 2

#: A median needs a few samples to mean anything. Comparing single-shot runs on
#: a shared machine measured a 1.6x swing on an unchanged kernel, so reports
#: below this count skip the timing half of the gate instead of failing it.
MIN_TIMING_REPEATS = 3

#: How much slower a workload's device-normalized floor may be than the
#: baseline's before the gate fails. The exact half of the gate (op counts, the
#: planned peak, the memory bound) has no tolerance at all, so this only covers
#: the measured milliseconds - and on a machine with a desktop, a language model
#: and a browser on it, a best-of-thirty-three floor still moved by 15% between
#: two runs of unchanged code (1.7 ms packed GEMV at 0.147x and 0.170x the same
#: run's reference matmul). Twenty-five percent is above that noise and still
#: below any regression worth calling one.
DEFAULT_TOLERANCE = 0.25

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
    #: Compile the builder's module as a bucket set at these widths on
    #: ``bucket_axis`` instead of as a single plan. The measured call is the
    #: builder's example input, dispatched through the set.
    buckets: tuple[int, ...] = ()
    bucket_axis: int = 0
    #: Which backend to compile for. ``"mlx-reference"`` generates no kernels,
    #: so its workloads gate the interpreter path instead of the Metal one.
    target: str = "apple-metal"


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


def _qwen2_decode() -> tuple[Any, tuple[Any, ...]]:
    """One Qwen2 decode step: the whole adapter, including the KV-cache roll.

    The roll is the part worth gating here. Spelled out it is a slice of the
    layer's pair, a stack of the two new rows, a concatenate, and a slice plus a
    squeeze per half; ``fuse_kv_cache_update`` folds all of that into one
    ``kv_cache_append`` per layer. When that match stops firing, ``input_ops``
    and ``lowered_ops`` jump and the gate fails on the count, with no timing
    noise involved.
    """

    import torch
    import transformers

    from .models.qwen2 import Qwen2DecodeModule

    capacity = 128
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
    torch.manual_seed(5)
    model = transformers.Qwen2ForCausalLM(config).eval()
    module = Qwen2DecodeModule(model, cache_capacity=capacity).eval()
    cache = torch.zeros(
        (2, 2, 1, 2, capacity, module.head_dim), dtype=torch.float32
    )
    return module, (
        torch.zeros((1, 1), dtype=torch.int64),
        torch.tensor([[7]], dtype=torch.int64),
        torch.ones((1, 1, 1, capacity), dtype=torch.bool),
        cache,
    )


def _llama_prefill() -> tuple[Any, tuple[Any, ...]]:
    """The Llama adapter at one bucket width, end to end.

    The second model family, gated the same way as the first: same builder
    shape, bias-free attention, the same 128-token prefill. What this catches is
    a family-specific regression - the packed q/k/v projection losing its
    epilogue fold, a norm dropping out of the graph - rather than a new one.
    """

    import torch
    import transformers

    from .models.llama import LlamaPrefillModule, prefill_mask

    config = transformers.LlamaConfig(
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
    model = transformers.LlamaForCausalLM(config).eval()
    module = LlamaPrefillModule(model, cache_capacity=128).eval()
    mask = torch.from_numpy(prefill_mask(128, 128))
    return module, (
        torch.zeros((1, 128), dtype=torch.int64),
        torch.arange(128, dtype=torch.int64)[None, :],
        mask,
    )


def _llama_decode() -> tuple[Any, tuple[Any, ...]]:
    """One Llama decode step, including the fused KV-cache append per layer."""

    import torch
    import transformers

    from .models.llama import LlamaDecodeModule

    capacity = 128
    config = transformers.LlamaConfig(
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
    torch.manual_seed(5)
    model = transformers.LlamaForCausalLM(config).eval()
    module = LlamaDecodeModule(model, cache_capacity=capacity).eval()
    cache = torch.zeros(
        (2, 2, 1, 2, capacity, module.head_dim), dtype=torch.float32
    )
    return module, (
        torch.zeros((1, 1), dtype=torch.int64),
        torch.tensor([[7]], dtype=torch.int64),
        torch.ones((1, 1, 1, capacity), dtype=torch.bool),
        cache,
    )


def _packed_gemv() -> tuple[Any, tuple[Any, ...]]:
    """A decode-shaped matmul against an int4 weight, on the streaming kernel.

    This is the shape the fold exists for: one row of activations multiplied by
    a packed weight, so the weight is never materialized. It gates both the
    rewrite and the GEMV prologue, and the weights are real random values rather
    than zeros so the driver cannot skip the read.
    """

    import torch

    from .models.quantized import dequantize_i4
    from .quantize import quantize_int4

    group_size = 64
    weight = torch.randn(2048, 2048)
    packed, scales = quantize_int4(weight.numpy(), group_size)

    class Packed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("words", torch.from_numpy(packed))
            self.register_buffer("scales", torch.from_numpy(scales))

        def forward(self, x: Any) -> Any:
            weight = dequantize_i4(self.words, self.scales, group_size)
            return torch.matmul(x, weight.t())

    return Packed().eval(), (torch.randn(1, 2048),)


def _bucket_dispatch() -> tuple[Any, tuple[Any, ...]]:
    """A projection whose request width is not a compiled width.

    The example input is 96 rows and the buckets are 128, so every call pads a
    quarter of its rows and slices them off again. That makes the workload gate
    the whole dispatch path - the tile, the trim, and the per-bucket lowering -
    rather than only the matmul inside one plan.
    """

    import torch

    torch.manual_seed(3)
    module = torch.nn.Linear(1024, 1024).eval()
    return module, (torch.randn(96, 1024),)


WORKLOADS: tuple[Workload, ...] = (
    Workload("linear_1024x1024", _linear, codegen_matmul=True),
    Workload(
        "bucket_dispatch_96_into_128",
        _bucket_dispatch,
        codegen_matmul=True,
        buckets=(128,),
    ),
    Workload("gated_mlp_1024_4096", _gated_mlp),
    Workload("rms_norm_32x4096", _rms_norm),
    Workload("attention_causal_128", _causal_attention, codegen_attention=True),
    Workload("packed_gemv_2048", _packed_gemv),
    Workload("qwen2_prefill_128", _qwen2_prefill, optional=True),
    Workload("qwen2_decode_128", _qwen2_decode, optional=True),
    Workload("llama_prefill_128", _llama_prefill, optional=True),
    Workload("llama_decode_128", _llama_decode, optional=True),
    # The second backend. No kernels, no codegen: this is the MLX interpreter
    # running the same projection the first workload generates a kernel for.
    Workload("reference_linear_1024x1024", _linear, target=REFERENCE_TARGET),
)


def _reference_operands() -> tuple[mx.array, mx.array]:
    """The raw MLX matmul that defines one unit of this machine's speed."""

    rng = np.random.default_rng(0)
    batch, inner, out = _REFERENCE_SHAPE
    left = mx.array(rng.standard_normal((batch, inner), dtype=np.float32)).astype(
        _REFERENCE_DTYPE
    )
    right = mx.array(rng.standard_normal((inner, out), dtype=np.float32)).astype(
        _REFERENCE_DTYPE
    )
    return left, right


def _reference_sample(operands: tuple[mx.array, mx.array]) -> Callable[[], float]:
    """Return a callable that times one reference matmul in milliseconds.

    It is called between a workload's samples rather than around its batch, so
    the two numbers a floor divides are measured under the same conditions - see
    :func:`_timed` for the measurement that made this necessary.
    """

    left, right = operands

    def take() -> float:
        start = time.perf_counter()
        mx.eval(mx.matmul(left, right))
        return (time.perf_counter() - start) * 1000.0

    return take


def _compile(
    workload: Workload,
) -> tuple[CompiledModulePlan | BucketedPlans, tuple[Any, ...], float]:
    """Compile one workload and return its plan, its MLX inputs, and the cost."""

    module, torch_inputs = workload.build()
    compile_start = time.perf_counter()
    if workload.buckets:
        compiled: CompiledModulePlan | BucketedPlans = compile_bucketed(
            module,
            torch_inputs,
            axis=workload.bucket_axis,
            sizes=workload.buckets,
            codegen_matmul=workload.codegen_matmul,
            codegen_attention=workload.codegen_attention,
            target=workload.target,
        )
    else:
        compiled = compile_module(
            module,
            torch_inputs,
            codegen_matmul=workload.codegen_matmul,
            codegen_attention=workload.codegen_attention,
            target=workload.target,
        )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    inputs = tuple(mx.array(value.detach().numpy()) for value in torch_inputs)
    return compiled, inputs, compile_ms


def _timed(
    compiled: CompiledModulePlan | BucketedPlans,
    inputs: tuple[Any, ...],
    repeats: int,
    reference: Callable[[], float] | None = None,
) -> tuple[list[float], list[float]]:
    """Time ``repeats`` calls, sampling the reference matmul between them.

    The reference has to be sampled *inside* the same window as the workload,
    not once before or after it. Two runs of unchanged code put the same 0.2 ms
    attention kernel at 0.203 ms and the reference matmul at 0.78 ms in one
    round and 1.15 ms in another, and dividing the first number by the second
    made a floor of 0.177 that no later run could match - a gate failure with no
    regression behind it. Interleaving costs one extra matmul per sample and
    gives the numerator and the denominator the same clock, the same thermal
    state and the same neighbour on the queue.

    Returns the workload samples and the reference samples, one pair per repeat,
    so a caller can divide minima measured moments apart.
    """

    samples: list[float] = []
    references: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = compiled(*inputs)
        mx.eval(*outputs)
        samples.append((time.perf_counter() - start) * 1000.0)
        if reference is not None:
            references.append(reference())
    return samples, references


def _first_call_ms(
    compiled: CompiledModulePlan | BucketedPlans, inputs: tuple[Any, ...]
) -> float:
    """Time the first call, which unrolls the plan and compiles its kernels.

    This is the number a server pays once per artifact at startup, and it is
    invisible to every other measurement here because they all run after a
    warmup. It is measured with the peak counter reset so the call's own
    allocation is what the counter reports.
    """

    start = time.perf_counter()
    outputs = compiled(*inputs)
    mx.eval(*outputs)
    first_call = (time.perf_counter() - start) * 1000.0
    mx.synchronize()
    return first_call


def _peak_bytes(
    compiled: CompiledModulePlan | BucketedPlans, inputs: tuple[Any, ...]
) -> int:
    """Peak device bytes one call allocates above what is already resident.

    ``mx.get_peak_memory`` counts every allocation the device has made since the
    counter was reset, including tensors other workloads left behind, so an
    absolute reading drifts upwards as a benchmark run proceeds. Subtracting the
    resident figure gives the number that belongs to this call - its
    intermediates together with its outputs - which is the number the memory
    policy controls.
    """

    mx.synchronize()
    mx.reset_peak_memory()
    resident = mx.get_active_memory()
    outputs = compiled(*inputs)
    mx.eval(*outputs)
    mx.synchronize()
    peak = mx.get_peak_memory()
    return max(0, peak - resident)


def _entry(
    compiled: CompiledModulePlan | BucketedPlans,
    samples: list[float],
    compile_ms: float,
    first_call_ms: float,
    peak_bytes: int,
) -> dict[str, Any]:
    """The gateable numbers for one workload.

    A bucket set reports the *sum* over its buckets: the gate exists to catch an
    extra kernel or an extra op, and an extra one in the 128 bucket is exactly as
    much a regression as one in a single plan. The memory figures are the widest
    bucket's, because that is the number an admission decision has to clear.
    """

    if isinstance(compiled, BucketedPlans):
        report = compiled.memory_report(DEFAULT_MEMORY)
        return {
            "compile_ms": compile_ms,
            "first_call_ms": first_call_ms,
            "peak_extra_bytes": peak_bytes,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "input_ops": sum(
                len(bucket.plan.input_graph.ops) for bucket in compiled.buckets
            ),
            "lowered_ops": sum(
                len(bucket.plan.lowered_graph.ops) for bucket in compiled.buckets
            ),
            "kernels": sum(
                len(bucket.plan.plan.kernels) for bucket in compiled.buckets
            ),
            "buckets": len(compiled.buckets),
            "widths": list(compiled.widths),
            "planned_peak_bytes": report.planned_peak_bytes,
            "peak_bound_bytes": report.peak_bound_bytes,
            "syncs": report.sync_count,
        }
    report = compiled.plan.memory_report(DEFAULT_MEMORY)
    return {
        "compile_ms": compile_ms,
        "first_call_ms": first_call_ms,
        "peak_extra_bytes": peak_bytes,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "input_ops": len(compiled.input_graph.ops),
        "lowered_ops": len(compiled.lowered_graph.ops),
        "kernels": len(compiled.plan.kernels),
        # Deterministic: shapes decide these, so the gate compares them exactly.
        "planned_peak_bytes": report.planned_peak_bytes,
        "peak_bound_bytes": report.peak_bound_bytes,
        "syncs": report.sync_count,
    }


def measure(
    workload: Workload,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    """Compile and time one workload on its own."""

    compiled, inputs, compile_ms = _compile(workload)
    first_call = _first_call_ms(compiled, inputs)
    peak = _peak_bytes(compiled, inputs)
    for _ in range(warmups):
        mx.eval(*compiled(*inputs))
    samples = _timed(compiled, inputs, repeats)
    return _entry(compiled, samples, compile_ms, first_call, peak)


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


def device_slug(name: str | None) -> str:
    """The file name a chip's baseline gets in a baseline directory.

    ``"Apple M5 Pro"`` becomes ``"apple-m5-pro"``. A slug is what lets
    ``benchmarks/baselines/`` be a matrix instead of one file that gets
    overwritten whenever somebody records on a different machine: the file *is*
    the chip's identity, so a mismatched pair of name and file is detectable
    rather than being trusted.
    """

    if not name:
        raise BaselineError("cannot name a baseline file after an unnamed device")
    slug = "".join(
        character if character.isalnum() else "-" for character in name.lower()
    )
    collapsed = "-".join(part for part in slug.split("-") if part)
    if not collapsed:
        raise BaselineError(f"device name {name!r} has no alphanumeric characters")
    return collapsed


@dataclass(frozen=True, slots=True)
class ResolvedBaseline:
    """The baseline chosen for this run, and why it is that one."""

    path: Path
    report: Mapping[str, Any]
    #: True when the file was recorded on the current device. False means the
    #: deterministic half of the gate runs and the timings do not.
    same_device: bool
    note: str | None = None


def resolve_baseline(path: str | Path, *, device: str | None = None) -> ResolvedBaseline:
    """Pick the baseline to compare against, from a file or a matrix directory.

    A file is used as given - an explicit path is an explicit decision. A
    directory is a chip matrix: its files are named by :func:`device_slug`, so
    the run picks the file for its own device, and falls back to another chip's
    file when there is none. The fallback is deliberate and not a compromise:
    ``input_ops``, ``lowered_ops``, ``planned_peak_bytes`` and
    ``peak_bound_bytes`` are decided by the graph, not the GPU, so a machine
    nobody has recorded on still gates every pipeline regression. What it does
    not gate is the millisecond half, and ``compare`` already says so with a
    note naming both devices.

    An empty directory is an error. A gate pointed at it would otherwise pass
    by comparing nothing, which is the one failure mode a gate must not have.
    """

    resolved = Path(path)
    if resolved.is_dir():
        device = (device or (_device().get("name")) or "")
        candidates = sorted(
            (entry for entry in resolved.glob("*.json") if entry.is_file()),
            key=lambda entry: entry.name,
        )
        if not candidates:
            raise BaselineError(
                f"{resolved} holds no baseline files; record one with "
                f"`aoto bench --update-baseline {resolved}`"
            )
        if device:
            wanted = resolved / f"{device_slug(device)}.json"
            if wanted.is_file():
                report = _read_baseline(wanted)
                recorded = (report.get("device") or {}).get("name")
                if recorded == device:
                    return ResolvedBaseline(wanted, report, same_device=True)
                # The file is named for this chip but records another one. Trust
                # the file, not the name: a renamed file is exactly the mistake
                # ``validate_baselines`` exists to catch, and comparing timings
                # across it would gate the wrong machine's milliseconds.
                return ResolvedBaseline(
                    wanted,
                    report,
                    same_device=False,
                    note=(
                        f"{wanted.name} is named for {device!r} but records "
                        f"{recorded!r}; only the device-independent metrics are "
                        f"gated"
                    ),
                )
        # No file for this chip. Prefer one that at least declares a device, so
        # the note can name it; otherwise the first file sorted by name keeps
        # the choice deterministic.
        chosen = candidates[0]
        report = _read_baseline(chosen)
        return ResolvedBaseline(
            chosen,
            report,
            same_device=False,
            note=(
                f"no baseline for {device or 'this device'} in {resolved}; "
                f"using {chosen.name}, so only the device-independent metrics "
                f"are gated"
            ),
        )
    return ResolvedBaseline(
        resolved, _read_baseline(resolved), same_device=True
    )


def _read_baseline(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_baselines(directory: str | Path) -> tuple[str, ...]:
    """Check every baseline in a matrix directory, and return their slugs.

    These files are committed, reviewed and edited by hand, so the things worth
    checking are the things a hand edit breaks: the schema version, a device
    name that no longer matches the file it sits in (which would make the loader
    pick the wrong chip's numbers), a report with no workloads, and a chip that
    appears twice.
    """

    resolved = Path(directory)
    if not resolved.is_dir():
        raise BaselineError(f"{resolved} is not a directory")
    problems: list[str] = []
    slugs: list[str] = []
    seen_devices: dict[str, str] = {}
    files = sorted(entry for entry in resolved.glob("*.json") if entry.is_file())
    if not files:
        problems.append(f"{resolved} holds no baseline files")
    for entry in files:
        try:
            report = _read_baseline(entry)
        except json.JSONDecodeError as exc:
            problems.append(f"{entry.name}: not JSON ({exc})")
            continue
        if not isinstance(report, Mapping):
            problems.append(f"{entry.name}: not a report object")
            continue
        schema = report.get("schema")
        if schema != BENCH_SCHEMA:
            problems.append(
                f"{entry.name}: schema {schema} is not {BENCH_SCHEMA}; re-record it"
            )
        name = (report.get("device") or {}).get("name")
        if not name:
            problems.append(f"{entry.name}: no device name recorded")
            continue
        try:
            slug = device_slug(name)
        except ValueError as exc:
            problems.append(f"{entry.name}: {exc}")
            continue
        slugs.append(slug)
        if slug != entry.stem:
            problems.append(
                f"{entry.name}: records device {name!r}, which belongs in "
                f"{slug}.json; the loader would never select it"
            )
        previous = seen_devices.get(slug)
        if previous is not None:
            problems.append(
                f"{entry.name}: device {name!r} is also recorded in {previous}"
            )
        else:
            seen_devices[slug] = entry.name
        workloads = report.get("workloads")
        if not workloads:
            problems.append(f"{entry.name}: records no workloads")
        elif not isinstance(workloads, Mapping):
            problems.append(f"{entry.name}: workloads is not a mapping")
        measurement = report.get("measurement") or {}
        if not measurement.get("repeats"):
            problems.append(
                f"{entry.name}: records no repeats, so its timings cannot gate"
            )
    if problems:
        raise BaselineError(
            "baseline directory problems:\n  " + "\n  ".join(problems)
        )
    return tuple(slugs)


def run(
    *,
    warmups: int = 3,
    repeats: int = 11,
    rounds: int = 1,
    workloads: Iterable[Workload] = WORKLOADS,
) -> dict[str, Any]:
    """Measure every workload and return one report.

    The report is the unit the gate compares, so it holds the measurement
    settings alongside the numbers: a baseline recorded with one repeat is not
    a fair reference for a run with twenty.

    ``rounds`` repeats the timed half and keeps every sample. The gate compares
    floors, and a floor is a best case, so the way to record a stable baseline
    is to sample more of them: one round of eleven repeats put the same 0.6 ms
    decode plan anywhere between 0.58 and 0.75 ms on this machine. Compilation,
    the first call, and the peak are one-shot by construction and are measured
    once.

    Every workload is compiled and executed once before any timing starts. The
    first plan a process runs pays for the Metal kernel compilation of every
    kernel it uses, and that cost leaked into the first workload's samples even
    through its warmups: the same 0.35 ms projection measured a 0.34 ms floor
    when something else had run first and 0.82 ms when nothing had. Warmup runs
    after the first plan are already fair, so this pass is what makes the
    workload order irrelevant.
    """

    if warmups < 0 or repeats < 1 or rounds < 1:
        raise ValueError(
            f"need warmups >= 0, repeats >= 1 and rounds >= 1, got "
            f"{warmups}/{repeats}/{rounds}"
        )
    prepared: list[
        tuple[
            Workload,
            CompiledModulePlan | BucketedPlans,
            tuple[Any, ...],
            float,
            float,
            int,
        ]
    ] = []
    skipped: dict[str, str] = {}
    for workload in workloads:
        try:
            compiled, inputs, compile_ms = _compile(workload)
        except ImportError as exc:
            if not workload.optional:
                raise
            skipped[workload.name] = str(exc)
            continue
        # The first call is the one that builds and caches the Metal kernels
        # this plan uses, so it is measured rather than discarded. Everything
        # after it is warm.
        first_call = _first_call_ms(compiled, inputs)
        peak = _peak_bytes(compiled, inputs)
        prepared.append((workload, compiled, inputs, compile_ms, first_call, peak))

    sample_inputs = _reference_operands()
    for _ in range(warmups):
        mx.eval(mx.matmul(*sample_inputs))

    samples: dict[str, list[float]] = {item[0].name: [] for item in prepared}
    #: Per-workload ratios against the reference measured at the same moment.
    #: Measuring the reference once per run and dividing everything by it is
    #: wrong on a machine whose speed drifts: a run whose matmul happened to be
    #: fast and whose projection happened to be slow reported a 60% projection
    #: "regression" that no code caused.
    ratios: dict[str, list[float]] = {item[0].name: [] for item in prepared}
    floors: dict[str, list[float]] = {item[0].name: [] for item in prepared}
    reference_medians: list[float] = []
    reference_floors: list[float] = []
    for _ in range(rounds):
        for workload, compiled, inputs, _compile_ms, _first_call, _peak in prepared:
            for _ in range(warmups):
                mx.eval(*compiled(*inputs))
            taken, references = _timed(
                compiled, inputs, repeats, _reference_sample(sample_inputs)
            )
            samples[workload.name].extend(taken)
            median = statistics.median(references)
            floor = min(references)
            reference_medians.append(median)
            reference_floors.append(floor)
            ratios[workload.name].append(statistics.median(taken) / median)
            floors[workload.name].append(min(taken) / floor)

    reference = statistics.median(reference_medians)
    reference_floor = min(reference_floors)
    measured: dict[str, Any] = {}
    for workload, compiled, _inputs, compile_ms, first_call, peak in prepared:
        name = workload.name
        entry = _entry(compiled, samples[name], compile_ms, first_call, peak)
        entry["ratio"] = statistics.median(ratios[name])
        entry["min_ratio"] = min(floors[name])
        measured[name] = entry
    return {
        "schema": BENCH_SCHEMA,
        "device": _device(),
        "measurement": {
            "warmups": warmups,
            "repeats": repeats,
            "rounds": rounds,
            "memory": DEFAULT_MEMORY,
        },
        "reference_ms": reference,
        "reference_min_ms": reference_floor,
        "workloads": measured,
        "skipped": skipped,
    }


#: Metrics the compilation itself decides, compared exactly and on any device.
_PLAN_METRICS = ("planned_peak_bytes", "peak_bound_bytes")
#: Metrics the run measures, as ``(name, floor, allowed growth)``. The floor is
#: the value below which a relative comparison says nothing (a call that
#: allocates 40 KB and then 80 KB is not a regression worth failing a build
#: over), and the growth is what is allowed on top of the baseline.
#:
#: Peak extra memory is device work, so the timing tolerance applies. The first
#: call and the compilation are one-shot host measurements dominated by Metal's
#: shader compiler: two runs of unchanged code here put the first decode call at
#: 5.1 ms and, on a loaded machine, at 14.2 ms, while every timing in the same
#: run moved by less than 2% apiece. A single sample cannot support a tight
#: band, so these get a factor of three - they catch the order-of-magnitude
#: regression, and the deterministic plan metrics stay strict.
_MEASURED_METRICS = (
    ("peak_extra_bytes", 1 << 20, 0.15),
    ("first_call_ms", 1.0, 2.0),
    ("compile_ms", 5.0, 2.0),
)


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
    tolerance: float = DEFAULT_TOLERANCE,
) -> Comparison:
    """Compare one report against a baseline.

    Op counts may only go down: ``input_ops`` and ``lowered_ops`` are
    deterministic, so any increase is a pipeline regression and no tolerance
    applies. ``kernels`` and timings are gated only when both reports name the
    same device, because kernel selection follows what the GPU can load and a
    millisecond is not portable; a tolerance is the fraction of slowdown that
    is allowed.

    Timings are compared on the best sample rather than ``median_ms``: the
    report's median is the number a reader wants, but for a gate the best
    sample is the right statistic, because scheduling noise only ever adds time.
    That best sample is then divided by the reference matmul the same run
    measured, so what the gate fails on is ``min_ratio`` - the workload's floor
    relative to the machine's speed at that moment - and not a millisecond that
    a loaded host can move by 1.8x without any code changing. A run that is
    slower in milliseconds but not in ratio produces a note instead.

    A timing comparison also needs more than one sample. A single-shot minimum
    is that sample: on a shared runner two single-shot runs of an unchanged
    kernel differed by 1.6x, which says nothing about the compiler. Reports with
    fewer than :data:`MIN_TIMING_REPEATS` repeats therefore skip the timing
    half and say so, while the op counts keep being checked.
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
    device_matched = bool(current_device) and current_device == baseline_device
    if not device_matched:
        notes.append(
            f"kernel counts and timings not compared: baseline device "
            f"{baseline_device!r} is not {current_device!r}; the ratio column is "
            f"still reported"
        )
    current_repeats = (current.get("measurement") or {}).get("repeats") or 0
    baseline_repeats = (baseline.get("measurement") or {}).get("repeats") or 0
    timings_compared = device_matched and min(current_repeats, baseline_repeats) >= (
        MIN_TIMING_REPEATS
    )
    if device_matched and not timings_compared:
        notes.append(
            f"timings not compared: need at least {MIN_TIMING_REPEATS} repeats for "
            f"a best-case estimate, got {current_repeats} now and "
            f"{baseline_repeats} in the baseline"
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
        metrics = deterministic + (device_dependent if device_matched else ())
        for metric in metrics:
            if now[metric] > then[metric]:
                regressions.append(
                    Regression(name, metric, then[metric], now[metric])
                )
        # The storage plan's figures come from the graph's shapes, so a plan
        # that starts holding more is a pipeline regression and is gated like an
        # op count: exactly, everywhere. ``peak_bound_bytes`` is what the memory
        # policy promises, and it is the one number that turns "we hope the
        # intermediate buffers are freed" into something a gate can fail on.
        for metric in _PLAN_METRICS:
            before = then.get(metric)
            if before is None:
                notes.append(f"{name}: baseline has no {metric}, not compared")
                continue
            if now[metric] > before:
                regressions.append(Regression(name, metric, before, now[metric]))
        if timings_compared:
            # Compare the best sample, not the median. Noise only ever adds
            # time, so it inflates the median and leaves the floor alone; three
            # back-to-back runs of an unchanged kernel put the median of a
            # 0.35 ms workload at 1.14, 0.38, and 0.36 ms while the minimum
            # stayed at 0.40, 0.35, 0.34. A real regression raises the floor.
            #
            # And compare it as a ratio against the reference matmul this run
            # measured, not in milliseconds: on a shared machine the absolute
            # floor moved by 1.8x between two runs of unchanged code (the
            # reference matmul went from 0.89 ms to 1.56 ms with it), while the
            # ratio stayed put. The absolute number is still reported, but a
            # run that is simply slower everywhere is a note and not a failure.
            before_ratio = then.get("min_ratio")
            if before_ratio is None:
                notes.append(
                    f"{name}: baseline has no min_ratio, timings not compared"
                )
            else:
                allowed = before_ratio * (1.0 + tolerance)
                if now["min_ratio"] > allowed:
                    regressions.append(
                        Regression(name, "min_ratio", before_ratio, now["min_ratio"])
                    )
                elif now["min_ms"] > then["min_ms"] * (1.0 + tolerance):
                    notes.append(
                        f"{name}: {now['min_ms']:.3f} ms against the baseline's "
                        f"{then['min_ms']:.3f} ms, but {now['min_ratio']:.4g}x "
                        f"against {before_ratio:.4g}x the same run's reference "
                        f"matmul is inside the tolerance"
                    )
            # Peak device memory is measured, so it moves with allocator state
            # the way a timing does; the same tolerance and the same same-device
            # rule apply. Cold start is measured too: it includes the unroll and
            # the Metal kernel compilation, both of which are real startup cost.
            for metric, floor, growth in _MEASURED_METRICS:
                before = then.get(metric)
                if before is None or before < floor:
                    continue
                allowed = before * (1.0 + growth)
                if now[metric] > allowed:
                    regressions.append(Regression(name, metric, before, now[metric]))
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
    "first ms",
    "extra MB",
    "bound MB",
    "syncs",
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
                f"{entry['first_call_ms']:.0f}",
                f"{entry['peak_extra_bytes'] / 1e6:.1f}",
                f"{entry['peak_bound_bytes'] / 1e6:.1f}",
                str(entry["syncs"]),
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
