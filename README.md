# Transformer Subgraph Compiler

一个面向 Transformer 子图的实验性编译器。当前版本以 Apple Silicon 为第一目标，把 PyTorch FX 图转换成自定义 SSA IR，融合 pointwise 子图，生成 Metal Shading Language，并通过 MLX 在 Apple GPU 上执行。

```text
PyTorch FX
  -> SSA IR
  -> pointwise fusion
  -> Metal Shading Language
  -> mx.fast.metal_kernel
  -> Apple GPU
```

## 当前能力

支持的主要 ATen 节点：

| 类别 | 节点 |
|---|---|
| 矩阵乘 | `aten.mm`、`aten.addmm` |
| 激活 | `silu`、`gelu`、`sigmoid`、`tanh`、`relu` |
| 一元运算 | `neg`、`abs`、`exp`、`erf`、`sqrt`、`rsqrt`、`reciprocal` |
| 二元运算 | `mul`、`add`、`sub`、`rsub`、`div` |

编译器已经实现：

- 用 `make_fx` 捕获静态 shape 的 PyTorch 计算图
- 构建带 dtype/shape 校验的 SSA IR
- 把相邻 pointwise op 融合成一个 DAG region
- 按静态 shape 生成 broadcast 索引
- 把 tensor 与标量混合的 pointwise 运算编码成 IR attribute
- 生成每个 thread 处理 4 个元素的 Metal kernel
- FP16、BF16、FP32 输入和输出
- 可选的 Metal matmul/addmm codegen

默认情况下，`matmul` 和 `addmm` 调用 MLX，只有 pointwise 区域由本项目生成 MSL。传入 `codegen_matmul=True` 后，矩阵乘和 bias 也会生成自定义 Metal kernel：

```python
result = compile_callable(fn, example_inputs, codegen_matmul=True)
```

这条 matmul 路径目前是正确性实现：一个 thread 计算一个输出元素，K 维循环累加，没有 tiling、shared memory 或 autotune。默认开关关闭，是因为 MLX 的矩阵乘实现明显更适合真实模型。

## 为什么没有直接改 Triton

这个项目先验证 `FX -> IR -> fusion -> MSL -> GPU` 的完整链路。Triton 的主要编译栈面向 NVIDIA GPU；在 Apple 后端可用之前先绑定 Triton，会把前端、IR、Apple codegen 和上游兼容性绑在一起。当前实现保留独立 IR，后续可以增加 Triton/CUDA backend，而不用重做前端和融合 pass。

## 环境

- macOS，Apple Silicon
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+，只用于 FX 捕获

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

如果 PyTorch 来自已有的 conda 环境，可以改用：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 测试与示例

```bash
python -m pytest -q
python examples/mlp.py
python examples/metal_mlp.py
```

- `examples/mlp.py` 使用 MLX matmul 和生成的 pointwise kernel，矩阵规模更接近实际 MLP。
- `examples/metal_mlp.py` 使用小矩阵开启 `codegen_matmul=True`，完整验证 FX、IR、自定义 MSL matmul、融合 pointwise 和 runtime。

示例会打印生成的 MSL 或 kernel 类型。首次执行包含 MLX kernel JIT 和 GPU dispatch，耗时不能当作 steady-state 性能。

## Python API

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
result = compile_callable(fn, example_inputs, codegen_matmul=True)

inputs = tuple(mx.array(value.numpy()) for value in example_inputs)
output = result.plan(*inputs)[0]
mx.eval(output)
```

编译结果是静态的：shape 和 dtype 来自 `example_inputs`，运行时必须匹配。当前不支持动态 shape、控制流、自动微分、reduction、layout pass、kernel cache 和训练。

精确 GELU 的 `erf` 使用 kernel header 中的数值近似，误差在测试覆盖范围内低于 `1e-6`。模型精度要求更高时应重新评估这个实现。

架构与实现细节见 [docs/architecture.md](docs/architecture.md)。
