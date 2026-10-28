# Transformer Subgraph Compiler

[English](README.md) | 简体中文

一个原生支持 Apple Silicon 和 Metal 的实验性 Transformer 模型编译器。

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#当前状态)

Transformer Subgraph Compiler 把 PyTorch 模型捕获成 FX 图，经过带类型和 shape 的 SSA IR，生成由 Metal kernel 和 MLX fallback 共同执行的计算计划。

> [!IMPORTANT]
> 这个项目编译 Transformer 模型计算。请求调度、continuous batching、分页 KV 分配、采样、分布式执行和 serving 由外部运行时负责。

## 为什么做这个项目

主流 Transformer 编译栈大多围绕 CUDA 和成熟的 NVIDIA kernel 生态构建。这个项目走另一条路：保留与后端无关的图 IR，再把模型计算原生编译到 Apple Silicon，而不是直接把 MLX 当成完整运行时。

编译器负责图捕获、合法化、布局简化、算子融合、公共子表达式消除、tensor 生命周期分析、kernel selection 和 Metal code generation。MLX 提供 fallback 算子，并在原生覆盖范围扩大前作为数值执行基线。

## 当前状态

项目已经可以运行静态 shape 的端到端 Transformer 实验，但还不是稳定的编译器产品。

| 模块 | 当前状态 |
|---|---|
| PyTorch 前端 | 可编译静态 shape 的 callable 和 `nn.Module`；模型参数与 buffer 会绑定到执行计划 |
| 图编译器 | 使用 typed SSA IR，支持 shape/dtype 校验、合法化、布局折叠、融合和 CSE |
| 存储规划 | 已生成 live interval、释放点、峰值活跃字节和可复用 buffer slot，执行时使用真实 arena |
| Kernel 选择 | 为每个 lowered op 记录原生 Metal 或 MLX 实现，以及 fallback 原因 |
| 编译产物 | 版本化 `.tsca` archive 包含图、kernel、selection、storage metadata 和模型 state，可在新进程加载执行 |
| Autotune | 可选的 profile 流程会测量原生 kernel 和 MLX 候选，测量结果与最终选择写入 artifact |

当前回归用例把两层 HuggingFace Llama 从 220 个 FX 算子降到 136 个 lowered op，logits 相对 PyTorch 参考实现的最大绝对误差为 `8.941e-08`。释放原始 module 后，编译计划仍可执行。

## 编译流程

```text
PyTorch FX / nn.Module
        │
        ▼
Typed SSA IR
        │  legalization、layout folding、fusion、CSE
        ▼
Liveness analysis / storage plan
        │
        ▼
Kernel selection / fallback planning
        │
        ├── 生成的 Metal kernel
        └── MLX fallback 算子
        ▼
CompiledPlan
        │
        ├── save：manifest + 图 + kernel + selection + state
        │
        ▼
版本化 artifact → 重新加载并构建 kernel → Apple GPU
```

## 环境要求

- Apple Silicon Mac，macOS
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+，用于 FX 捕获

## 安装

```bash
git clone https://github.com/jaykay233/transformer-subgraph-compiler.git
cd transformer-subgraph-compiler

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

只编译 Python callable 时可以安装 `.[fx]`。`dev` 和 `hf` extras 会额外安装 HuggingFace Transformers，用于模型级编译和测试。

## 快速开始

下面的例子编译一个小型 `addmm + SiLU` 函数。执行前可以检查图、存储计划和 kernel selection。

```python
import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def fn(x, weight, bias):
    hidden = torch.addmm(bias, x, weight, alpha=2.0)
    return torch.nn.functional.silu(hidden)


torch.manual_seed(0)
example_inputs = (
    torch.randn(4, 16),
    torch.randn(16, 32),
    torch.randn(32),
)

compiled = compile_callable(fn, example_inputs, codegen_matmul=True)
inputs = tuple(mx.array(value.numpy()) for value in example_inputs)
output = compiled.plan(*inputs)[0]
mx.eval(output)

print("peak live bytes:", compiled.plan.storage_plan.peak_live_bytes)
for selection in compiled.plan.selections:
    print(selection.op, selection.implementation, selection.reason)
```

### 编译 HuggingFace 模型

`compile_module` 会把模型参数和 buffer 作为显式 state 参与 tracing。返回的计划不持有原始 `nn.Module`。

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from transformer_subgraph_compiler import compile_module


model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        use_cache=False,
    )
).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = compile_module(
    model,
    (input_ids,),
    example_kwargs={"use_cache": False},
    codegen_matmul=True,
    codegen_attention=True,
)

logits = compiled(input_ids)[0]
mx.eval(logits)
```

### 命令行与编译产物

安装后可以用 `tsc` 编译 callable、module 或 HuggingFace 模型。产物也可以在不导入原始模型的情况下检查和执行。

```bash
tsc compile \
  --module 'my_module:build_model' \
  --input-file inputs.json \
  --output model.tsca \
  --codegen-matmul \
  --autotune

tsc inspect model.tsca
tsc verify model.tsca --require-device-match
tsc run model.tsca --input-file inputs.json --output outputs.json
```

`inputs.json` 是按位置排列的 tensor 规格。标量使用空 shape 和单元素 data：

```json
[
  {
    "shape": [1, 4],
    "dtype": "int64",
    "data": [[1, 2, 3, 4]]
  },
  {
    "shape": [],
    "dtype": "float32",
    "data": [1.0]
  }
]
```

也可以重复传入 `--input '<json>'`，不写输入文件。`--callable MODULE:ATTR` 捕获函数，`--module MODULE:ATTR` 接受 `nn.Module`、类或 factory，`--hf-model` 加载 HuggingFace causal language model。console script 不在 `PATH` 时，用 `python -m transformer_subgraph_compiler` 执行相同命令。

artifact 使用 format version 3 和 ABI version 1.2。两张图、kernel table、selection table、storage plan 和每个 state tensor 都有 SHA-256 digest。加载时会校验 ABI、target、图、入口布局、state shape/dtype、序列化 kernel、重新构建的 selection 和 storage plan。kernel 根据校验后的图和已保存的 selection 重建，因此 artifact 不需要原始 Python 模型，加载时也不会重新 autotune。

是否要求 GPU 完全相同由调用方决定。`--require-device-match` 或 `require_device_match=True` 会拒绝为其他 Apple GPU 型号编译的 artifact；默认允许在其他 Apple Silicon 设备上重新构建 kernel 后执行。

## 原生 Metal Code Generation

pointwise、softmax、reduction、LayerNorm 和 RMSNorm 默认生成 Metal kernel。matmul 和 attention 默认使用 MLX，开启下面的选项后才生成原生 kernel：

```python
compiled = compile_callable(
    fn,
    example_inputs,
    codegen_matmul=True,
    codegen_attention=True,
)
```

原生实现不保证更快。目前已测试的 shape 上，自定义 matmul 和 attention kernel 仍慢于 MLX 的调优实现，因此两个选项默认关闭。`CompiledPlan.selections` 会记录每个算子实际使用的路径以及 fallback 原因。

## 支持范围

| 类别 | 覆盖范围 |
|---|---|
| 模型 | decoder-only Transformer forward；tiny Llama/Qwen 风格模型；固定容量静态 KV Cache prefill/decode |
| 矩阵计算 | `mm`、`bmm`、`addmm`、`baddbmm`，包括支持范围内的 batch broadcasting |
| 索引 | embedding、`index_select`、advanced indexing、`gather` 和 `scatter` |
| Transformer 计算 | scaled dot-product attention、分解后的 causal attention、softmax、sum/mean、cumsum、LayerNorm、RMSNorm、SiLU、GELU、SwiGLU 和常见 pointwise 算子 |
| 布局与数据移动 | view/reshape、transpose/permute、slice、expand、cat/stack、clone、cast 和静态常量 |
| 数据类型 | 支持范围内的 FP16、BF16、FP32、bool、I32 和 I64 |
| KV Cache | 静态 shape 的 prefill/decode 图契约和布局；分配与分页仍由外部运行时负责 |

算子覆盖范围小于 PyTorch。不支持的 target 会在捕获或构造 IR 时失败，不会静默绕过编译计划。逐算子的 lowering 表和边界见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- shape 和 dtype 在编译期固定，不支持动态 shape、符号维度和控制流。
- tensor-valued keyword arguments 目前不会提升为图输入。
- 共享参数 storage alias 目前会按多个 state entry 序列化。数值结果不受影响，但 artifact 还没有去重这些 aliased storage。
- artifact ABI 只接受 `apple-metal` target，尚未实现跨厂商 target 和 host runtime ABI 兼容。
- 原生 kernel 的覆盖和性能仍弱于 MLX。当前自定义 matmul、attention 不保证更快，因此 `codegen_matmul` 和 `codegen_attention` 默认关闭。
- 原生 attention 要求 rank-3/4 输入、Q/K/V dtype 一致，并在文档规定的 mask 和 head dimension 约束内支持 MHA/GQA/MQA；其他情况回退到 MLX SDPA。
- 仓库不实现 scheduler、continuous batching、抢占、paged KV cache、prefix cache、采样策略、PD 分离、EP/MoE dispatch、EPLB 或 serving API。
- 不支持训练和 autograd。

## 测试

```bash
python -m pytest -q
python examples/tiny_llama_compile.py
```

其他示例覆盖 embedding、MLP、matmul kernel、softmax、reduction、normalization、attention 和静态 KV Cache 生成：

```bash
python examples/tiny_transformer.py
python examples/tiny_transformer_generate.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

首次执行包含 MLX kernel JIT 和 GPU dispatch，首轮耗时不能作为 steady-state benchmark。

## Roadmap

1. 在 state ABI 中保留共享 storage alias，而不是重复序列化相同参数的视图。
2. 增加符号 shape 和有限控制流，同时保留静态 shape 的严格校验。
3. 把逐 op autotune 扩展成 shape-aware backend policy 和可复现 benchmark report。
4. 提升原生 matmul、attention 的性能，再扩展模型、dtype、算子和 GPU 覆盖；PyTorch 数值对齐仍是验收条件。

## 仓库结构

```text
src/transformer_subgraph_compiler/   编译器前端、IR、passes、planner、selection 和 Metal backend
tests/                               IR、lowering、数值对齐、生成和 planner 测试
examples/                            可运行的模型、kernel 和静态 KV Cache 示例
docs/architecture.md                 详细实现说明与已知边界
```

## 参与贡献

欢迎提交 issue 和 pull request。修改 lowering 行为时，请附上明确的动机、可复现测试和数值对齐结果。

## License

MIT，详见 [LICENSE](LICENSE)。
