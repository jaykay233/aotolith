"""The Apple GPU backend: choose a kernel per operation, then select a plan.

This module is policy, not emission. ``_lower_candidates`` asks each family's
lowering for a native candidate (or gets nothing, meaning MLX will run it), and
then either measures the candidates against the MLX reference or applies the
heuristic that prefers a native kernel where one is known to win.

The family lowerings are composed in as bases: each one lives beside the source
emitter it calls, so "how is a matmul compiled" has one answer in one file. What
stays here is what is true of every family - which operations get a candidate at
all, how a winner is picked, and how the result becomes a ``CompiledPlan``.
"""

from __future__ import annotations

import time
import mlx.core as mx
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod
from statistics import median
from typing import Any
from ..ir import DType
from ..ir import Graph
from ..selection import KernelBenchmark
from ..selection import KernelSelection
from ..op_sets import LAYOUT_OPS
from ..ir.ops import OP_CONTRACTS
from ..ir import Operation
from ..op_sets import POINTWISE_OPS
from ..ir import TensorType
from ..op_sets import CONSTANT_OPS as _CONSTANT_POINTWISE_OPS
from ..ir.ops import INTEGRAL_DTYPES as _INTEGRAL_DTYPES
from ..ir.ops import check_operation
from ..planning import plan_storage
from ..codegen.attention import AttentionLowering
from ..codegen.dtypes import _MLX_DTYPES
from ..codegen.index import IndexLowering
from ..codegen.kv_cache import KVCacheLowering
from ..codegen.matmul import (
    MatmulLowering,
    _matmul_batch_shape,
    _matmul_weight_group_size,
    _packed_matmul_supported,
)
from ..codegen.msl import KernelArtifact
from ..codegen.norm import NormLowering
from ..codegen.pointwise import PointwiseLowering
from ..codegen.quantize import QuantizationLowering
from ..codegen.reduction import ReductionLowering
from ..runtime.mlx_ops.__init__ import _evaluate_mlx_op
from ..runtime.plan import CompiledPlan, _as_arrays
from ..target import apple as target
from ..target.apple import (
    _block_tensor_cores,
    _device_profile,
    _mpp_supported,
)

_AUTOTUNE_WARMUP = 2


_AUTOTUNE_ITERATIONS = 7


@dataclass(frozen=True, slots=True)
class _NativeCandidate:
    artifact: KernelArtifact
    runtime: Any


def _tensor_core_candidates_load(
    graph: Graph, candidates: Mapping[str, _NativeCandidate]
) -> bool:
    """Try to load the emitted tensor-core kernels and evaluate them.

    `__HAVE_TENSOR__` answers whether the GPU has tensor ops; it does not answer
    whether the metallib will load. A paravirtualised GPU reports the macro and
    then rejects the kernel with "unsupported deferred-static-alloca-size
    function body", so this asks the driver by running the smallest thing that
    goes through the real kernels. Doing it here costs a launch per compile
    instead of an opaque `RuntimeError` on a user's first matmul.

    One probe per kernel kind and input dtype set: support is not guaranteed to
    be uniform, so a device that loads FP16 tensor ops and rejects FP32 ones is
    caught too. Within a group the smallest shapes are probed, because the
    dummy inputs are allocated at full size. Nothing is checked for
    correctness, only that the kernel loads and dispatches; if this wrongly
    fails, the plan still computes the right answer through the simdgroup path,
    and the tests that assert tensor-core source catch the throughput loss.
    """

    probes: dict[tuple[Any, ...], tuple[int, _NativeCandidate]] = {}
    for candidate in candidates.values():
        artifact = candidate.artifact
        if "tensorcore" not in artifact.name:
            continue
        types = [graph.value(value_id).type for value_id in artifact.input_ids]
        key = (artifact.kind, tuple(item.dtype for item in types))
        elements = sum(prod(item.shape) for item in types)
        smallest = probes.get(key)
        if smallest is None or elements < smallest[0]:
            probes[key] = (elements, candidate)

    for _, candidate in probes.values():
        try:
            inputs = [
                mx.zeros(
                    graph.value(value_id).type.shape,
                    dtype=_MLX_DTYPES[graph.value(value_id).type.dtype],
                )
                for value_id in candidate.artifact.input_ids
            ]
            # MLX is lazy, so `runtime(...)` alone never loads the kernel and
            # would report success for one the device rejects.
            mx.eval(*_as_arrays(candidate.runtime(*inputs)))
        except Exception as error:  # noqa: BLE001 - any failure means "no MPP".
            _block_tensor_cores(str(error))
            return False
    return True


_TENSOR_CORE_NOTE = "tensor cores rejected by this device"


def _annotate_tensor_core_fallback(
    selections: tuple[KernelSelection, ...],
) -> tuple[KernelSelection, ...]:
    """Say so when the plan is native but is not using tensor cores."""

    if target._TENSOR_CORE_BLOCK_ERROR is None:
        return selections
    return tuple(
        replace(selection, reason=f"{selection.reason} ({_TENSOR_CORE_NOTE})")
        if selection.is_native
        else selection
        for selection in selections
    )


def _synthetic_tune_input(tensor_type: TensorType) -> mx.array:
    dtype = _MLX_DTYPES[tensor_type.dtype]
    if tensor_type.dtype in {DType.F16, DType.BF16, DType.F32}:
        return mx.full(tensor_type.shape, 0.25, dtype=dtype)
    if tensor_type.dtype == DType.BOOL:
        return mx.ones(tensor_type.shape, dtype=dtype)
    return mx.zeros(tensor_type.shape, dtype=dtype)


def _coerce_tune_input(value: Any, tensor_type: TensorType) -> mx.array:
    if isinstance(value, mx.array):
        array = value
    else:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            host_value = value.detach().cpu()
            if str(host_value.dtype) == "torch.bfloat16":
                host_value = host_value.float()
            array = mx.array(host_value.numpy())
        else:
            array = mx.array(value)
    array = array.astype(_MLX_DTYPES[tensor_type.dtype])
    if tuple(array.shape) != tensor_type.shape:
        raise ValueError(
            f"autotune input has shape {tuple(array.shape)}, "
            f"expected {tensor_type.shape}"
        )
    return array


def _store_outputs(env: dict[str, Any], op: Operation, result: Any) -> None:
    """Publish every result of ``op`` so the next operation can read it."""

    if len(op.outputs) == 1:
        env[op.outputs[0]] = result
        return
    for value_id, array in zip(op.outputs, result, strict=True):
        env[value_id] = array


def _outputs_match(actual: Any, expected: Any, dtype: DType) -> bool:
    # An operator with several results is compared result by result: a kernel
    # that writes the pair and hands back two views of it has to agree with the
    # fallback on all three, not just on the one it wrote.
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or not isinstance(
            expected, (tuple, list)
        ):
            return False
        if len(actual) != len(expected):
            return False
        return all(
            _outputs_match(one, other, dtype)
            for one, other in zip(actual, expected, strict=True)
        )
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        return False
    if dtype in {DType.BOOL, DType.I32, DType.I64}:
        result = mx.array_equal(actual, expected)
    else:
        # MLX and native kernels can use different accumulation schedules.
        # Keep the gate aligned with the transformer parity tolerance rather
        # than rejecting a numerically valid candidate for a few ULPs.
        atol, rtol = (2e-2, 2e-2)
        result = mx.allclose(actual, expected, atol=atol, rtol=rtol)
    mx.eval(result)
    return bool(result.item())


def _time_runtime(
    runtime: Any,
    *,
    warmup_iterations: int,
    iterations: int,
) -> tuple[float, ...]:
    for _ in range(warmup_iterations):
        mx.eval(runtime())
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(runtime())
        samples.append(time.perf_counter() - start)
    return tuple(samples)


class AppleMetalBackend(
    PointwiseLowering,
    ReductionLowering,
    MatmulLowering,
    QuantizationLowering,
    AttentionLowering,
    NormLowering,
    IndexLowering,
    KVCacheLowering,
):
    def __init__(
        self,
        *,
        codegen_matmul: bool = False,
        codegen_attention: bool = False,
        autotune: bool = False,
        autotune_warmup: int = _AUTOTUNE_WARMUP,
        autotune_iterations: int = _AUTOTUNE_ITERATIONS,
        selection_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if autotune_warmup < 0:
            raise ValueError("autotune_warmup must be non-negative")
        if autotune_iterations <= 0:
            raise ValueError("autotune_iterations must be positive")
        self.codegen_matmul = codegen_matmul
        self.codegen_attention = codegen_attention
        self.autotune = autotune
        self.autotune_warmup = autotune_warmup
        self.autotune_iterations = autotune_iterations
        self.selection_overrides = dict(selection_overrides or {})

    def _lower_candidates(self, graph: Graph) -> dict[str, _NativeCandidate]:
        """Emit one native candidate per operation this backend can lower."""

        candidates: dict[str, _NativeCandidate] = {}
        for index, op in enumerate(graph.ops):
            if op.op in OP_CONTRACTS:
                check_operation(graph, op)
            compiled: tuple[KernelArtifact, Any] | None = None
            if op.op == "fused_elementwise":
                body = op.attrs.get("body")
                if not isinstance(body, tuple) or not all(
                    isinstance(item, Operation) for item in body
                ):
                    raise ValueError(
                        "fused_elementwise body must be a tuple of operations"
                    )
                compiled = self._compile_elementwise(
                    graph, op.inputs, body, op.outputs[0], index, "fused_elementwise"
                )
            elif op.op == "arange":
                compiled = self._compile_arange(graph, op, index)
            elif op.op == "dequantize_i4":
                compiled = self._compile_dequantize_i4(graph, op, index)
            elif (
                op.op in _CONSTANT_POINTWISE_OPS
                and graph.value(op.outputs[0]).type.dtype in _INTEGRAL_DTYPES
            ):
                compiled = self._compile_integer_constant(graph, op, index)
            elif op.op in POINTWISE_OPS:
                compiled = self._compile_elementwise(
                    graph, op.inputs, (op,), op.outputs[0], index, op.op
                )
            elif op.op in {"softmax", "safe_softmax"}:
                compiled = self._compile_softmax(graph, op, index)
            elif op.op in {"sum", "mean"}:
                compiled = self._compile_reduction(graph, op, index)
            elif op.op == "layer_norm":
                compiled = self._compile_layer_norm(graph, op, index)
            elif op.op == "rms_norm":
                compiled = self._compile_rms_norm(graph, op, index)
            elif op.op == "rope":
                # The vectorized kernel owns four neighbouring columns, so both
                # the head dimension and its half must be multiples of four.
                if graph.value(op.outputs[0]).type.shape[-1] % 8 == 0:
                    compiled = self._compile_rope(graph, op, index)
            elif op.op == "kv_cache_append":
                compiled = self._compile_kv_cache_append(graph, op, index)
            elif op.op == "attention":
                if self.codegen_attention or self.autotune or self._wants_native(op):
                    compiled = self._compile_attention(graph, op, index)
                else:
                    # A single-query attention needs one threadgroup per head
                    # and no online rescaling, so it beats the generic runtime
                    # kernel without the prefill-oriented codegen switch.
                    compiled = self._compile_decode_attention(graph, op, index)
            elif op.op in {"matmul", "addmm"} and _matmul_weight_group_size(op) is not None:
                # A packed weight streams natively whether or not dense matmul
                # codegen is on: the MLX fallback has to materialize the dense
                # weight first, which is the allocation quantizing removes. It
                # still has to be a shape the kernel handles, or it stays on MLX.
                if _packed_matmul_supported(graph, op):
                    compiled = self._compile_matmul(graph, op, index)
            elif (
                self.codegen_matmul or self.autotune or self._wants_native(op)
            ) and op.op in {"matmul", "addmm"}:
                left_position = 1 if op.op == "addmm" else 0
                right_position = 2 if op.op == "addmm" else 1
                left_type = graph.value(op.inputs[left_position]).type
                right_type = graph.value(op.inputs[right_position]).type
                output_type = graph.value(op.outputs[0]).type
                batch_shape = _matmul_batch_shape(left_type.shape, right_type.shape)
                if batch_shape is not None and output_type.shape[:-2] == batch_shape:
                    compiled = self._compile_matmul(graph, op, index)

            if compiled is not None:
                artifact, runtime = compiled
                # Candidates are keyed by the operation's first output, which
                # is also what a selection records; a multi-output kernel still
                # fills every output of that operation.
                candidates[artifact.output_ids[0]] = _NativeCandidate(artifact, runtime)
        return candidates

    def compile(
        self,
        graph: Graph,
        *,
        tune_inputs: Sequence[Any] | None = None,
    ) -> CompiledPlan:
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process")

        candidates = self._lower_candidates(graph)
        if _mpp_supported() and not _tensor_core_candidates_load(graph, candidates):
            # This device compiled the tensor-core kernels and refused to
            # load them, so re-lower once without them rather than leaving
            # a plan that raises on its first matmul.
            candidates = self._lower_candidates(graph)

        if self.autotune and not self.selection_overrides:
            kernels, selections, callables = self._autotune(
                graph, candidates, tune_inputs
            )
        else:
            kernels, selections, callables = self._heuristic_select(graph, candidates)

        return CompiledPlan(
            graph,
            kernels,
            _annotate_tensor_core_fallback(selections),
            callables,
            plan_storage(graph),
        )

    def _wants_native(self, op: Operation) -> bool:
        return self.selection_overrides.get(op.outputs[0]) == "metal_kernel"

    def _heuristic_select(
        self,
        graph: Graph,
        candidates: Mapping[str, _NativeCandidate],
    ) -> tuple[tuple[KernelArtifact, ...], tuple[KernelSelection, ...], dict[str, Any]]:
        kernels: list[KernelArtifact] = []
        selections: list[KernelSelection] = []
        callables: dict[str, Any] = {}
        device_name, device_architecture = _device_profile()
        for index, op in enumerate(graph.ops):
            output_id = op.outputs[0]
            candidate = candidates.get(output_id)
            override = self.selection_overrides.get(output_id)
            if override not in {None, "metal_kernel", "mlx"}:
                raise ValueError(
                    f"unknown kernel implementation override {override!r} "
                    f"for {output_id}"
                )
            if override == "metal_kernel" and candidate is None:
                raise ValueError(
                    f"artifact selected Metal for unsupported operation {output_id}"
                )
            use_native = override == "metal_kernel" or (
                override is None and candidate is not None
            )
            if use_native:
                assert candidate is not None
                kernels.append(candidate.artifact)
                callables[output_id] = candidate.runtime
                selection = KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation="metal_kernel",
                    kind=candidate.artifact.kind,
                    reason=(
                        "native Metal kernel selected"
                        if override is None
                        else "serialized artifact selected native Metal kernel"
                    ),
                    artifact_name=candidate.artifact.name,
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            else:
                selection = KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation="mlx",
                    kind=op.op,
                    reason=(
                        self._fallback_reason(graph, op)
                        if override is None
                        else "serialized artifact selected MLX fallback"
                    ),
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            selections.append(selection)
        return tuple(kernels), tuple(selections), callables

    def _autotune(
        self,
        graph: Graph,
        candidates: Mapping[str, _NativeCandidate],
        tune_inputs: Sequence[Any] | None,
    ) -> tuple[tuple[KernelArtifact, ...], tuple[KernelSelection, ...], dict[str, Any]]:
        if tune_inputs is None:
            env = {
                value_id: _synthetic_tune_input(graph.value(value_id).type)
                for value_id in graph.inputs
            }
        else:
            if len(tune_inputs) != len(graph.inputs):
                raise ValueError(
                    f"autotune expected {len(graph.inputs)} inputs, "
                    f"received {len(tune_inputs)}"
                )
            env = {
                value_id: _coerce_tune_input(value, graph.value(value_id).type)
                for value_id, value in zip(graph.inputs, tune_inputs, strict=True)
            }

        device_name, device_architecture = _device_profile()
        kernels: list[KernelArtifact] = []
        selections: list[KernelSelection] = []
        callables: dict[str, Any] = {}
        for index, op in enumerate(graph.ops):
            output_id = op.outputs[0]
            output_type = graph.value(output_id).type
            input_arrays = tuple(env[value_id] for value_id in op.inputs)

            def fallback_runtime() -> mx.array:
                return _evaluate_mlx_op(graph, op, env)

            candidate = candidates.get(output_id)
            try:
                fallback_result = fallback_runtime()
                mx.eval(fallback_result)
            except NotImplementedError:
                # Some ops only ever lower to a generated kernel: normalization,
                # rotary and attention have no MLX equivalent once they are
                # fused. There is nothing to compare against, so the kernel is
                # measured on its own and its own output feeds the next op.
                if candidate is None:
                    raise
                fallback_result = None

            if candidate is None:
                assert fallback_result is not None
                selections.append(
                    KernelSelection(
                        op_index=index,
                        op=op.op,
                        output_id=output_id,
                        implementation="mlx",
                        kind=op.op,
                        reason=self._fallback_reason(graph, op),
                        device_name=device_name,
                        device_architecture=device_architecture,
                    )
                )
                _store_outputs(env, op, fallback_result)
                continue

            if fallback_result is None:
                native_result = candidate.runtime(*input_arrays)
                mx.eval(native_result)
                native_benchmark = self._benchmark(
                    "metal_kernel",
                    candidate.artifact.name,
                    lambda: candidate.runtime(*input_arrays),
                )
                mlx_benchmark = None
                _store_outputs(env, op, native_result)
            else:
                mlx_benchmark = self._benchmark("mlx", None, fallback_runtime)
                try:
                    native_result = candidate.runtime(*input_arrays)
                    mx.eval(native_result)
                    if not _outputs_match(
                        native_result, fallback_result, output_type.dtype
                    ):
                        native_benchmark = KernelBenchmark(
                            candidate="metal_kernel",
                            artifact_name=candidate.artifact.name,
                            warmup_iterations=self.autotune_warmup,
                            iterations=self.autotune_iterations,
                            samples_seconds=(),
                            median_seconds=None,
                            min_seconds=None,
                            error="output differs from the MLX reference",
                        )
                    else:
                        native_benchmark = self._benchmark(
                            "metal_kernel",
                            candidate.artifact.name,
                            lambda: candidate.runtime(*input_arrays),
                        )
                except Exception as exc:
                    native_benchmark = KernelBenchmark(
                        candidate="metal_kernel",
                        artifact_name=candidate.artifact.name,
                        warmup_iterations=self.autotune_warmup,
                        iterations=self.autotune_iterations,
                        samples_seconds=(),
                        median_seconds=None,
                        min_seconds=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                _store_outputs(env, op, fallback_result)

            native_seconds = native_benchmark.median_seconds
            mlx_seconds = (
                None if mlx_benchmark is None else mlx_benchmark.median_seconds
            )
            use_native = native_seconds is not None and (
                mlx_seconds is None or native_seconds <= mlx_seconds
            )
            if use_native:
                kernels.append(candidate.artifact)
                callables[output_id] = candidate.runtime
                implementation = "metal_kernel"
                kind = candidate.artifact.kind
                artifact_name: str | None = candidate.artifact.name
                reason = self._measured_reason(
                    "native Metal kernel", native_seconds, mlx_seconds
                )
            else:
                implementation = "mlx"
                kind = op.op
                artifact_name = None
                reason = self._measured_reason(
                    "MLX fallback", mlx_seconds, native_seconds
                )
                _store_outputs(env, op, fallback_result)

            benchmarks = (native_benchmark,)
            if mlx_benchmark is not None:
                benchmarks = (native_benchmark, mlx_benchmark)
            selections.append(
                KernelSelection(
                    op_index=index,
                    op=op.op,
                    output_id=output_id,
                    implementation=implementation,
                    kind=kind,
                    reason=reason,
                    artifact_name=artifact_name,
                    benchmarks=benchmarks,
                    autotuned=True,
                    device_name=device_name,
                    device_architecture=device_architecture,
                )
            )

        return tuple(kernels), tuple(selections), callables

    def _benchmark(
        self,
        candidate: str,
        artifact_name: str | None,
        runtime: Any,
    ) -> KernelBenchmark:
        try:
            samples = _time_runtime(
                runtime,
                warmup_iterations=self.autotune_warmup,
                iterations=self.autotune_iterations,
            )
        except Exception as exc:
            return KernelBenchmark(
                candidate=candidate,
                artifact_name=artifact_name,
                warmup_iterations=self.autotune_warmup,
                iterations=self.autotune_iterations,
                samples_seconds=(),
                median_seconds=None,
                min_seconds=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        return KernelBenchmark(
            candidate=candidate,
            artifact_name=artifact_name,
            warmup_iterations=self.autotune_warmup,
            iterations=self.autotune_iterations,
            samples_seconds=samples,
            median_seconds=median(samples),
            min_seconds=min(samples),
        )

    @staticmethod
    def _measured_reason(
        winner: str,
        winner_seconds: float | None,
        alternative_seconds: float | None,
    ) -> str:
        if winner_seconds is None:
            return "autotune failed for every candidate; MLX fallback retained"
        if alternative_seconds is None:
            return f"autotuned: {winner} was the only valid candidate"
        return (
            f"autotuned: {winner} was faster "
            f"({winner_seconds * 1e3:.4f} ms vs "
            f"{alternative_seconds * 1e3:.4f} ms median)"
        )

    def _fallback_reason(self, graph: Graph, op: Operation) -> str:
        if op.op == "attention" and not self.codegen_attention:
            return "codegen_attention disabled; using MLX scaled dot-product attention"
        if op.op == "attention":
            return "attention shape or mask is outside native Metal constraints"
        if op.op in {"matmul", "addmm"} and not self.codegen_matmul:
            return "codegen_matmul disabled; using MLX matrix multiplication"
        if op.op in {"matmul", "addmm"}:
            return "matmul shape is outside native Metal constraints"
        if op.op == "cast":
            return "dtype conversion uses MLX astype"
        if op.op in LAYOUT_OPS:
            return "layout operation uses an MLX view or materialization"
        if op.op == "scatter":
            return "scatter and its reductions use the MLX runtime atomic path"
        if op.op == "index_add":
            return "index_add uses the MLX runtime scatter-add"
        if op.op in {"prod", "amax", "amin"}:
            return "reduction uses the MLX runtime reduction"
        if op.op == "log_softmax":
            return "log_softmax uses the MLX runtime log-sum-exp"
        return "operation uses the MLX runtime implementation"
