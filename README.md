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

已支持的图节点：

- `aten.mm` / `aten.addmm`
- `aten.silu`
- `aten.mul.Tensor`
- `aten.add.Tensor`

已实现：

- 用 `torch.fx.experimental.proxy_tensor.make_fx` 捕获静态 shape 的计算图
- 编译期内联一段只含 `silu`、`mul`、`add` 的 DAG
- 将融合区域生成一个 Metal kernel
- FP16、BF16、FP32 输入和输出
- MLX runtime 的参数个数、shape、dtype 校验

当前 `matmul` 和 `addmm` 使用 MLX 的 `mx.matmul`，由 MLX 调用 Apple GPU。编译器目前只为自己的融合 pointwise 区域生成 MSL。矩阵乘 kernel、tiling、autotune 和性能缓存尚未实现。

## 为什么没有直接改 Triton

这个 MVP 先要验证 `FX -> IR -> fusion -> MSL -> GPU` 的完整链路。Triton 的主要编译栈面向 NVIDIA GPU；在 Apple 后端可用之前先绑定 Triton，会把前端、IR、Apple codegen 和上游兼容性绑在一起。当前实现保留独立 IR，后续可以增加 Triton/CUDA backend，而不用重做前端和融合 pass。

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
```

`examples/mlp.py` 会捕获下面的 MLP：

```python
gate = x @ gate_w
up = x @ up_w
return (silu(gate) * up) @ down_w
```

编译后的图是：

```text
matmul
matmul
fused_elementwise
matmul
```

只有 `fused_elementwise` 会生成自定义 MSL。示例还会打印生成的源码和首次执行耗时。首次执行包含 MLX kernel JIT 和 GPU dispatch，耗时不代表 steady-state 性能。

## Python API

```python
import mlx.core as mx
import torch

from transformer_subgraph_compiler import compile_callable


def fn(x, weight):
    return torch.nn.functional.silu(x) @ weight


torch.manual_seed(0)
example_inputs = (torch.randn(4, 16), torch.randn(16, 8))
result = compile_callable(fn, example_inputs)

x = mx.array(example_inputs[0].numpy())
weight = mx.array(example_inputs[1].numpy())
output = result.plan(x, weight)[0]
mx.eval(output)
```

编译结果是静态的：shape 和 dtype 来自 `example_inputs`，运行时必须匹配。当前不支持动态 shape、PyTorch 自动微分回传、控制流、scalar/broadcast pointwise 或自定义 matmul kernel。

架构与实现细节见 [docs/architecture.md](docs/architecture.md)。
