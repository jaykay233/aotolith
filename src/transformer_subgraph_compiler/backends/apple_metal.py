from __future__ import annotations

import textwrap
from dataclasses import dataclass
from math import ceil, prod
from typing import Any

import mlx.core as mx

from ..ir import DType, Graph, Operation, TensorType
from ..passes import POINTWISE_OPS

_MLX_DTYPES = {
    DType.F16: mx.float16,
    DType.BF16: mx.bfloat16,
    DType.F32: mx.float32,
}

_ELEMENTS_PER_THREAD = 4
_MATMUL_MICROTILE = (4, 4)
_MATMUL_LARGE_K_MICROTILE = (8, 4)
_MATMUL_LARGE_K_THRESHOLD = 1024

_MSL_HEADER = """
inline float tsc_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = metal::abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * metal::exp(-x * x));
}
"""


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    name: str
    source: str
    header: str
    input_ids: tuple[str, ...]
    output_id: str
    tensor_type: TensorType
    kind: str


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

        for op in self.graph.ops:
            runtime = self._callables.get(op.outputs[0])
            if runtime is not None:
                env[op.outputs[0]] = runtime(*(env[value_id] for value_id in op.inputs))
                continue

            if op.op == "matmul":
                env[op.outputs[0]] = mx.matmul(env[op.inputs[0]], env[op.inputs[1]])
            elif op.op == "addmm":
                bias, left, right = (env[value_id] for value_id in op.inputs)
                alpha = float(op.attrs.get("alpha", 1.0))
                beta = float(op.attrs.get("beta", 1.0))
                env[op.outputs[0]] = beta * bias + alpha * mx.matmul(left, right)
            else:
                env[op.outputs[0]] = _evaluate_pointwise_mlx(op, env)

        return tuple(env[value_id] for value_id in self.graph.outputs)


class AppleMetalBackend:
    """Lower supported IR regions to Metal kernels with an MLX runtime."""

    def __init__(self, *, codegen_matmul: bool = False) -> None:
        self.codegen_matmul = codegen_matmul

    def compile(self, graph: Graph) -> CompiledPlan:
        if not mx.metal.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process")

        kernels: list[KernelArtifact] = []
        callables: dict[str, Any] = {}
        for index, op in enumerate(graph.ops):
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
            elif op.op in POINTWISE_OPS:
                compiled = self._compile_elementwise(
                    graph, op.inputs, (op,), op.outputs[0], index, op.op
                )
            elif op.op == "softmax":
                compiled = self._compile_softmax(graph, op, index)
            elif op.op == "layer_norm":
                compiled = self._compile_layer_norm(graph, op, index)
            elif op.op == "rms_norm":
                compiled = self._compile_rms_norm(graph, op, index)
            elif self.codegen_matmul and op.op in {"matmul", "addmm"}:
                compiled = self._compile_matmul(graph, op, index)

            if compiled is not None:
                artifact, runtime = compiled
                kernels.append(artifact)
                callables[artifact.output_id] = runtime

        return CompiledPlan(graph, tuple(kernels), callables)

    def _compile_elementwise(
        self,
        graph: Graph,
        input_ids: tuple[str, ...],
        body: tuple[Operation, ...],
        output_id: str,
        index: int,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        output_type = graph.value(output_id).type
        for value_id in input_ids:
            input_type = graph.value(value_id).type
            if input_type.dtype != output_type.dtype:
                raise NotImplementedError(
                    "pointwise kernels require identical input/output dtypes"
                )
            if not _can_broadcast(input_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast pointwise input {input_type.shape} "
                    f"to {output_type.shape}"
                )

        source = _emit_pointwise_source(
            graph, input_ids, body, output_id, output_type.shape
        )
        name = f"{kind}_{index}"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[f"in_{position}" for position in range(len(input_ids))],
            output_names=["out"],
            source=source,
            header=_MSL_HEADER,
        )

        element_count = prod(output_type.shape)
        grid = (ceil(element_count / _ELEMENTS_PER_THREAD), 1, 1)
        threadgroup = (min(256, grid[0]), 1, 1)
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
            header=_MSL_HEADER,
            input_ids=input_ids,
            output_id=output_id,
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_softmax(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        if len(op.inputs) != 1:
            raise ValueError("softmax expects one input")

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                "Metal softmax requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"softmax output shape {output_type.shape} != {input_type.shape}"
            )
        if not input_type.shape:
            raise NotImplementedError("Metal softmax requires a ranked tensor")

        rank = len(input_type.shape)
        dim = op.attrs.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"softmax.dim must be an integer, got {dim!r}")
        axis = dim % rank
        if axis != rank - 1:
            raise NotImplementedError(
                "Metal softmax currently supports only the last dimension"
            )

        columns = input_type.shape[-1]
        rows = prod(input_type.shape[:-1]) or 1
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint offset = row * {columns}u;
        threadgroup float scratch[{threads}];

        float local_max = -INFINITY;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_max = metal::max(local_max, float(in_0[offset + col]));
        }}
        scratch[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] = metal::max(scratch[tid], scratch[tid + stride]);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += metal::exp(float(in_0[offset + col]) - max_value);
        }}
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float sum_value = scratch[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            out[offset + col] = (T)(
                metal::exp(float(in_0[offset + col]) - max_value) / sum_value
            );
        }}
        """
        name = f"metal_softmax_{threads}_{index}"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=["in_0"],
            output_names=["out"],
            source=source,
        )
        grid = (rows * threads, 1, 1)
        threadgroup = (threads, 1, 1)
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
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind="softmax",
        )
        return artifact, runtime

    def _compile_layer_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="layer_norm")

    def _compile_rms_norm(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        return self._compile_normalization(graph, op, index, kind="rms_norm")

    def _compile_normalization(
        self,
        graph: Graph,
        op: Operation,
        index: int,
        *,
        kind: str,
    ) -> tuple[KernelArtifact, Any]:
        normalized_shape = op.attrs.get("normalized_shape")
        if not isinstance(normalized_shape, tuple) or not normalized_shape:
            raise ValueError(f"{kind}.normalized_shape must be a non-empty tuple")
        if not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim > 0
            for dim in normalized_shape
        ):
            raise ValueError(
                f"{kind}.normalized_shape must contain positive integers"
            )

        has_weight = op.attrs.get("has_weight")
        if not isinstance(has_weight, bool):
            raise ValueError(f"{kind}.has_weight must be a boolean")
        has_bias = False
        if kind == "layer_norm":
            has_bias = op.attrs.get("has_bias")
            if not isinstance(has_bias, bool):
                raise ValueError("layer_norm.has_bias must be a boolean")
        expected_inputs = 1 + int(has_weight) + int(has_bias)
        if len(op.inputs) != expected_inputs:
            raise ValueError(
                f"{kind} expects {expected_inputs} inputs, got {len(op.inputs)}"
            )

        input_type = graph.value(op.inputs[0]).type
        output_type = graph.value(op.outputs[0]).type
        if input_type.dtype != output_type.dtype:
            raise NotImplementedError(
                f"Metal {kind} requires identical input/output dtypes"
            )
        if input_type.shape != output_type.shape:
            raise ValueError(
                f"{kind} output shape {output_type.shape} != {input_type.shape}"
            )
        normalized_rank = len(normalized_shape)
        if normalized_rank > len(input_type.shape):
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} exceeds input rank"
            )
        if input_type.shape[-normalized_rank:] != normalized_shape:
            raise ValueError(
                f"{kind} normalized_shape {normalized_shape} does not match "
                f"input shape {input_type.shape}"
            )

        input_position = 1
        weight_position: int | None = None
        bias_position: int | None = None
        if has_weight:
            weight_position = input_position
            input_position += 1
        if has_bias:
            bias_position = input_position
        for position in range(1, len(op.inputs)):
            parameter_type = graph.value(op.inputs[position]).type
            if parameter_type.dtype != input_type.dtype:
                raise NotImplementedError(
                    f"Metal {kind} requires identical parameter dtypes"
                )
            if parameter_type.shape != normalized_shape:
                raise ValueError(
                    f"{kind} parameter shape {parameter_type.shape} "
                    f"!= {normalized_shape}"
                )

        columns = prod(normalized_shape)
        rows = prod(input_type.shape[:-normalized_rank]) or 1
        threads = _reduction_thread_count(columns)
        reduction_stride = threads // 2
        eps = float(op.attrs.get("eps", 1e-5))

        if kind == "layer_norm":
            reduction_source = textwrap.dedent(f"""
        float local_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            local_sum += float(in_0[offset + col]);
        }}
        scratch[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean_value = scratch[0] / {columns}.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_squared_deviation = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float centered = float(in_0[offset + col]) - mean_value;
            local_squared_deviation += centered * centered;
        }}
        scratch[tid] = local_squared_deviation;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float variance = scratch[0] / {columns}.0f;
            """).strip()
            reduction_result = "variance"
            centered_expression = "current - mean_value"
        else:
            reduction_source = textwrap.dedent(f"""
        float local_square_sum = 0.0f;
        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            local_square_sum += current * current;
        }}
        scratch[tid] = local_square_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = {reduction_stride}u; stride > 0u; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] += scratch[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean_square = scratch[0] / {columns}.0f;
            """).strip()
            reduction_result = "mean_square"
            centered_expression = "current"

        scale_expression = (
            f"float(in_{weight_position}[col])"
            if weight_position is not None
            else "1.0f"
        )
        normalized_expression = (
            f"({centered_expression}) * inv_std * {scale_expression}"
        )
        if bias_position is not None:
            normalized_expression += f" + float(in_{bias_position}[col])"

        source = f"""
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;
        uint offset = row * {columns}u;
        threadgroup float scratch[{threads}];

{textwrap.indent(reduction_source, '        ')}
        float inv_std = metal::rsqrt({reduction_result} + {eps!r}f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint col = tid; col < {columns}u; col += {threads}u) {{
            float current = float(in_0[offset + col]);
            out[offset + col] = (T)({normalized_expression});
        }}
        """
        name = f"metal_{kind}_{threads}_{index}"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[
                f"in_{position}" for position in range(len(op.inputs))
            ],
            output_names=["out"],
            source=source,
        )
        grid = (rows * threads, 1, 1)
        threadgroup = (threads, 1, 1)
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
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=kind,
        )
        return artifact, runtime

    def _compile_matmul(
        self, graph: Graph, op: Operation, index: int
    ) -> tuple[KernelArtifact, Any]:
        expected_inputs = 3 if op.op == "addmm" else 2
        if len(op.inputs) != expected_inputs:
            raise ValueError(f"{op.op} expects {expected_inputs} inputs")

        if op.op == "addmm":
            bias_id, left_id, right_id = op.inputs
        else:
            bias_id = None
            left_id, right_id = op.inputs

        left_type = graph.value(left_id).type
        right_type = graph.value(right_id).type
        output_type = graph.value(op.outputs[0]).type
        if len(left_type.shape) != 2 or len(right_type.shape) != 2:
            raise NotImplementedError("Metal matmul currently requires rank-2 tensors")
        if len(output_type.shape) != 2:
            raise NotImplementedError("Metal matmul currently produces rank-2 tensors")

        rows, reduction = left_type.shape
        right_reduction, columns = right_type.shape
        if reduction != right_reduction:
            raise ValueError(
                f"matmul reduction dimensions differ: {reduction} and {right_reduction}"
            )
        if output_type.shape != (rows, columns):
            raise ValueError(
                f"matmul output shape {output_type.shape} != {(rows, columns)}"
            )

        dtypes = {left_type.dtype, right_type.dtype, output_type.dtype}
        if bias_id is not None:
            bias_type = graph.value(bias_id).type
            dtypes.add(bias_type.dtype)
            if not _can_broadcast(bias_type.shape, output_type.shape):
                raise NotImplementedError(
                    f"cannot broadcast addmm bias {bias_type.shape} "
                    f"to {output_type.shape}"
                )
        if len(dtypes) != 1:
            raise NotImplementedError("Metal matmul requires identical input/output dtypes")

        micro_rows, micro_columns = _select_matmul_microtile(reduction)
        source = _emit_register_matmul_source(
            op,
            rows,
            reduction,
            columns,
            micro_rows,
            micro_columns,
            graph,
        )
        name = f"metal_{op.op}_{micro_rows}x{micro_columns}_{index}"
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=[f"in_{position}" for position in range(len(op.inputs))],
            output_names=["out"],
            source=source,
        )
        grid = (
            ceil(columns / micro_columns),
            ceil(rows / micro_rows),
            1,
        )
        threadgroup = (min(32, grid[0]), min(8, grid[1]), 1)
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
            header="",
            input_ids=op.inputs,
            output_id=op.outputs[0],
            tensor_type=output_type,
            kind=op.op,
        )
        return artifact, runtime


def _can_broadcast(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    if len(input_shape) > len(output_shape):
        return False
    offset = len(output_shape) - len(input_shape)
    return all(
        input_dim in {1, output_shape[offset + axis]}
        for axis, input_dim in enumerate(input_shape)
    )


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected a positive integer, got {value}")
    return 1 << (value - 1).bit_length()


def _reduction_thread_count(columns: int) -> int:
    return min(256, _next_power_of_two(columns))


def _offset_expression(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    index_expression: str,
) -> str:
    if input_shape == output_shape:
        return index_expression

    offset = len(output_shape) - len(input_shape)
    terms: list[str] = []
    for output_axis, output_dim in enumerate(output_shape):
        input_axis = output_axis - offset
        if input_axis < 0 or input_shape[input_axis] == 1:
            continue
        output_stride = prod(output_shape[output_axis + 1 :])
        input_stride = prod(input_shape[input_axis + 1 :])
        coordinate = f"((({index_expression}) / {output_stride}) % {output_dim})"
        terms.append(f"{coordinate} * {input_stride}")
    return " + ".join(terms) if terms else "0"


def _emit_pointwise_source(
    graph: Graph,
    input_ids: tuple[str, ...],
    body: tuple[Operation, ...],
    output_id: str,
    output_shape: tuple[int, ...],
) -> str:
    element_count = prod(output_shape)
    lines = [
        "uint base = thread_position_in_grid.x * 4u;",
        "for (uint lane = 0u; lane < 4u; ++lane) {",
        "    uint elem = base + lane;",
        f"    if (elem >= {element_count}) {{",
        "        break;",
        "    }",
    ]
    env: dict[str, str] = {}

    for position, value_id in enumerate(input_ids):
        variable = f"input_{position}"
        offset = _offset_expression(
            graph.value(value_id).type.shape, output_shape, "elem"
        )
        lines.append(f"    float {variable} = float(in_{position}[{offset}]);")
        env[value_id] = variable

    for op_index, op in enumerate(body):
        arguments = [env[value_id] for value_id in op.inputs]
        output = f"temp_{op_index}"
        expression = _emit_pointwise_expression(op, arguments)
        lines.append(f"    float {output} = {expression};")
        env[op.outputs[0]] = output

    if output_id not in env:
        raise ValueError("pointwise region does not produce its declared output")
    lines.extend(
        [
            f"    out[elem] = (T)({env[output_id]});",
            "}",
        ]
    )
    return "\n".join(f"        {line}" for line in lines)


def _binary_operands(op: Operation, arguments: list[str]) -> tuple[str, str]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar = repr(float(op.attrs["scalar"]))
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar


def _emit_pointwise_expression(op: Operation, arguments: list[str]) -> str:
    if op.op == "silu":
        value = arguments[0]
        return f"{value} / (1.0f + metal::exp(-{value}))"
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            return (
                f"0.5f * {value} * (1.0f + metal::tanh("
                f"0.7978845608028654f * ({value} + "
                f"0.044715f * {value} * {value} * {value})))"
            )
        return (
            f"0.5f * {value} * (1.0f + tsc_erf("
            f"{value} * 0.7071067811865476f))"
        )
    if op.op in {"mul", "div"}:
        left, right = _binary_operands(op, arguments)
        operator = "*" if op.op == "mul" else "/"
        return f"{left} {operator} {right}"
    if op.op in {"add", "sub", "rsub"}:
        left, right = _binary_operands(op, arguments)
        operator = "+" if op.op == "add" else "-"
        alpha = float(op.attrs.get("alpha", 1.0))
        return f"{left} {operator} {alpha!r} * {right}"
    if op.op == "neg":
        return f"-{arguments[0]}"
    if op.op == "abs":
        return f"metal::abs({arguments[0]})"
    if op.op == "sigmoid":
        return f"1.0f / (1.0f + metal::exp(-{arguments[0]}))"
    if op.op == "tanh":
        return f"metal::tanh({arguments[0]})"
    if op.op == "relu":
        return f"metal::max(0.0f, {arguments[0]})"
    if op.op == "exp":
        return f"metal::exp({arguments[0]})"
    if op.op == "erf":
        return f"tsc_erf({arguments[0]})"
    if op.op == "sqrt":
        return f"metal::sqrt({arguments[0]})"
    if op.op == "rsqrt":
        return f"metal::rsqrt({arguments[0]})"
    if op.op == "reciprocal":
        return f"1.0f / {arguments[0]}"
    raise NotImplementedError(f"cannot emit fused pointwise op: {op.op}")


def _select_matmul_microtile(reduction: int) -> tuple[int, int]:
    if reduction >= _MATMUL_LARGE_K_THRESHOLD:
        return _MATMUL_LARGE_K_MICROTILE
    return _MATMUL_MICROTILE


def _emit_register_matmul_source(
    op: Operation,
    rows: int,
    reduction: int,
    columns: int,
    micro_rows: int,
    micro_columns: int,
    graph: Graph,
) -> str:
    if op.op == "addmm":
        bias_id, _, _ = op.inputs
        bias_type = graph.value(bias_id).type
        output_shape = graph.value(op.outputs[0]).type.shape
        alpha = float(op.attrs.get("alpha", 1.0))
        beta = float(op.attrs.get("beta", 1.0))
        left_position = 1
        right_position = 2
    else:
        bias_id = None
        bias_type = None
        output_shape = ()
        alpha = 1.0
        beta = 1.0
        left_position = 0
        right_position = 1

    exact_tiles = rows % micro_rows == 0 and columns % micro_columns == 0
    lines = [
        f"uint row_base = thread_position_in_grid.y * {micro_rows}u;",
        f"uint col_base = thread_position_in_grid.x * {micro_columns}u;",
    ]
    if not exact_tiles:
        lines.extend(
            [
                f"if (row_base >= {rows}u || col_base >= {columns}u) {{",
                "    return;",
                "}",
            ]
        )
        for row in range(micro_rows):
            lines.append(f"bool row_ok_{row} = row_base + {row}u < {rows}u;")
        for column in range(micro_columns):
            lines.append(f"bool col_ok_{column} = col_base + {column}u < {columns}u;")

    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(f"float acc_{row}_{column} = 0.0f;")

    lines.append(f"for (uint k = 0u; k < {reduction}u; ++k) {{")
    for row in range(micro_rows):
        load = (
            f"float(in_{left_position}["
            f"(row_base + {row}u) * {reduction}u + k])"
        )
        if not exact_tiles:
            load = f"row_ok_{row} ? {load} : 0.0f"
        lines.append(f"    float av_{row} = {load};")
    for column in range(micro_columns):
        load = (
            f"float(in_{right_position}["
            f"k * {columns}u + col_base + {column}u])"
        )
        if not exact_tiles:
            load = f"col_ok_{column} ? {load} : 0.0f"
        lines.append(f"    float bv_{column} = {load};")
    for row in range(micro_rows):
        for column in range(micro_columns):
            lines.append(
                f"    acc_{row}_{column} = fma("
                f"av_{row}, bv_{column}, acc_{row}_{column});"
            )
    lines.append("}")

    for row in range(micro_rows):
        for column in range(micro_columns):
            linear = f"linear_{row}_{column}"
            if not exact_tiles:
                lines.append(f"if (row_ok_{row} && col_ok_{column}) {{")
            indent = "    " if not exact_tiles else ""
            lines.append(
                f"{indent}uint {linear} = (row_base + {row}u) * {columns}u "
                f"+ col_base + {column}u;"
            )
            accumulator = f"acc_{row}_{column}"
            if bias_id is not None:
                assert bias_type is not None
                bias_offset = _offset_expression(bias_type.shape, output_shape, linear)
                value = (
                    f"{beta!r} * float(in_0[{bias_offset}]) "
                    f"+ {alpha!r} * {accumulator}"
                )
            else:
                value = accumulator
            lines.append(f"{indent}out[{linear}] = (T)({value});")
            if not exact_tiles:
                lines.append("}")

    return "\n".join(f"        {line}" for line in lines)


def _evaluate_pointwise_mlx(op: Operation, env: dict[str, mx.array]) -> mx.array:
    arguments = [env[value_id] for value_id in op.inputs]
    if op.op == "silu":
        value = arguments[0]
        return value / (1.0 + mx.exp(-value))
    if op.op == "gelu":
        value = arguments[0]
        if op.attrs.get("approximate") == "tanh":
            inner = 0.7978845608028654 * (
                value + 0.044715 * value * value * value
            )
            return 0.5 * value * (1.0 + mx.tanh(inner))
        return 0.5 * value * (1.0 + mx.erf(value * 0.7071067811865476))
    if op.op in {"mul", "div", "add", "sub", "rsub"}:
        left, right = _binary_mx_operands(op, arguments)
        if op.op == "mul":
            return left * right
        if op.op == "div":
            return left / right
        alpha = float(op.attrs.get("alpha", 1.0))
        if op.op == "add":
            return left + alpha * right
        return left - alpha * right
    if op.op == "neg":
        return -arguments[0]
    if op.op == "abs":
        return mx.abs(arguments[0])
    if op.op == "sigmoid":
        return 1.0 / (1.0 + mx.exp(-arguments[0]))
    if op.op == "tanh":
        return mx.tanh(arguments[0])
    if op.op == "relu":
        return mx.maximum(arguments[0], 0.0)
    if op.op == "exp":
        return mx.exp(arguments[0])
    if op.op == "erf":
        return mx.erf(arguments[0])
    if op.op == "sqrt":
        return mx.sqrt(arguments[0])
    if op.op == "rsqrt":
        return 1.0 / mx.sqrt(arguments[0])
    if op.op == "reciprocal":
        return 1.0 / arguments[0]
    raise NotImplementedError(f"no Apple runtime lowering for {op.op}")


def _binary_mx_operands(
    op: Operation, arguments: list[mx.array]
) -> tuple[mx.array | float, mx.array | float]:
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    if len(arguments) != 1 or "scalar" not in op.attrs:
        raise ValueError(f"malformed binary pointwise op: {op.op}")
    scalar = float(op.attrs["scalar"])
    if op.attrs.get("scalar_position") == "lhs":
        return scalar, arguments[0]
    return arguments[0], scalar
