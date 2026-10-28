from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import mlx.core as mx

from ..ir import DType, Graph, Operation, TensorType

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
}


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    name: str
    source: str
    input_ids: tuple[str, ...]
    output_id: str
    tensor_type: TensorType


@dataclass(slots=True)
class CompiledPlan:
    graph: Graph
    kernels: tuple[KernelArtifact, ...]
    _callables: dict[str, Any]

    def __call__(self, *inputs: Any) -> tuple[mx.array, ...]:
        if len(inputs) != len(self.graph.inputs):
            raise ValueError(
                f"expected {len(self.graph.inputs)} inputs, received {len(inputs)}"
            )

        env: dict[str, mx.array] = {}
        for value_id, value in zip(self.graph.inputs, inputs, strict=True):
            array = value if isinstance(value, mx.array) else mx.array(value)
            expected = self.graph.value(value_id).type
            if tuple(array.shape) != expected.shape:
                raise ValueError(
                    f"input {value_id} has shape {tuple(array.shape)}, expected {expected.shape}"
                )
            if array.dtype != _MLX_DTYPES[expected.dtype]:
                raise TypeError(
                    f"input {value_id} has dtype {array.dtype}, expected {expected.dtype.value}"
                )
            env[value_id] = array

        for index, op in enumerate(self.graph.ops):
            if op.op == "matmul":
                env[op.outputs[0]] = mx.matmul(env[op.inputs[0]], env[op.inputs[1]])
            elif op.op == "addmm":
                bias, left, right = (env[value_id] for value_id in op.inputs)
                alpha = float(op.attrs.get("alpha", 1.0))
                beta = float(op.attrs.get("beta", 1.0))
                env[op.outputs[0]] = beta * bias + alpha * mx.matmul(left, right)
            elif op.op == "fused_elementwise":
                runtime = self._callables[op.outputs[0]]
                env[op.outputs[0]] = runtime(*(env[value_id] for value_id in op.inputs))
            elif op.op == "silu":
                value = env[op.inputs[0]]
                env[op.outputs[0]] = value / (1.0 + mx.exp(-value))
            elif op.op == "mul":
                env[op.outputs[0]] = env[op.inputs[0]] * env[op.inputs[1]]
            elif op.op == "add":
                alpha = float(op.attrs.get("alpha", 1.0))
                env[op.outputs[0]] = env[op.inputs[0]] + alpha * env[op.inputs[1]]
            else:
                raise NotImplementedError(f"no Apple runtime lowering for {op.op}")

        return tuple(env[value_id] for value_id in self.graph.outputs)


class AppleMetalBackend:
    """Lower supported IR regions to Metal kernels with an MLX runtime."""

    def compile(self, graph: Graph) -> CompiledPlan:
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process")

        kernels: list[KernelArtifact] = []
        callables: dict[str, Any] = {}
        for index, op in enumerate(graph.ops):
            if op.op != "fused_elementwise":
                continue
            artifact, runtime = self._compile_pointwise(graph, op, index)
            kernels.append(artifact)
            callables[artifact.output_id] = runtime

        return CompiledPlan(graph, tuple(kernels), callables)

    def _compile_pointwise(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        body = op.attrs.get("body")
        if not isinstance(body, tuple) or not all(isinstance(item, Operation) for item in body):
            raise ValueError("fused_elementwise body must be a tuple of operations")

        output_type = graph.value(op.outputs[0]).type
        for value_id in op.inputs:
            if graph.value(value_id).type != output_type:
                raise NotImplementedError(
                    "fused pointwise kernels currently require identical input/output types"
                )

        source = _emit_pointwise_source(op, body)
        name = f"fused_pointwise_{index}"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[f"in_{position}" for position in range(len(op.inputs))],
            output_names=["out"],
            source=source,
        )

        element_count = prod(output_type.shape)
        grid = (element_count, 1, 1)
        threadgroup = (min(256, element_count), 1, 1)
        dtype = _MLX_DTYPES[output_type.dtype]

        def runtime(*inputs: mx.array) -> mx.array:
            outputs = kernel(
                inputs=list(inputs),
                template=[("T", dtype)],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[output_type.shape],
                output_dtypes=[dtype],
            )
            return outputs[0]

        artifact = KernelArtifact(
            name=name,
            source=source,
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
        )
        return artifact, runtime


def _emit_pointwise_source(fused: Operation, body: tuple[Operation, ...]) -> str:
    lines = ["uint elem = thread_position_in_grid.x;"]
    env: dict[str, str] = {}

    for position, value_id in enumerate(fused.inputs):
        variable = f"input_{position}"
        lines.append(f"float {variable} = float(in_{position}[elem]);")
        env[value_id] = variable

    for index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        output = f"temp_{index}"
        if op.op == "silu":
            expression = f"{arguments[0]} / (1.0f + metal::exp(-{arguments[0]}))"
        elif op.op == "mul":
            expression = f"{arguments[0]} * {arguments[1]}"
        elif op.op == "add":
            alpha = float(op.attrs.get("alpha", 1.0))
            expression = f"{arguments[0]} + {alpha!r} * {arguments[1]}"
        else:
            raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")
        lines.append(f"float {output} = {expression};")
        env[op.outputs[0]] = output

    if fused.outputs[0] not in env:
        raise ValueError("fused region does not produce its declared output")
    lines.append(f"out[elem] = (T)({env[fused.outputs[0]]});")
    return "\n".join(f"        {line}" for line in lines)
