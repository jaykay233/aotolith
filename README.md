# Transformer Subgraph Compiler

一个面向 Transformer 子图的实验性编译器。当前版本以 Apple Silicon 为第一目标，把 PyTorch FX 图转换成自定义 SSA IR，消除冗余布局与公共子表达式，融合 attention 和 pointwise 子图，再通过生成的 Metal Shading Language kernel 与 MLX runtime 在 Apple GPU 上执行。

```text
PyTorch FX
  -> SSA IR
  -> layout simplification / matmul canonicalization / attention and pointwise fusion / CSE
  -> Metal Shading Language + MLX runtime lowering
  -> mx.fast.metal_kernel
  -> Apple GPU
```

## 当前能力

支持的主要 FX target 和折叠模式：

| 类别 | 节点 |
|---|---|
| 矩阵乘 | `aten.mm`、`aten.bmm`、`aten.addmm` |
| 视图、布局与 dtype | `view`、`reshape`、`transpose`、`permute`、`expand`、`slice`、`unsqueeze`、`squeeze`、`clone`、`alias`、`_to_copy` |
| 激活 | `silu`、`gelu`、`sigmoid`、`tanh`、`relu` |
| 一元运算 | `neg`、`abs`、`exp`、`erf`、`sqrt`、`rsqrt`、`reciprocal` |
| 二元运算 | `mul`、`add`、`sub`、`rsub`、`div` |
| Softmax | `aten._softmax.default`、`aten._safe_softmax.default`，支持任意单个维度 |
| Attention | `F.scaled_dot_product_attention`，以及 `QK^T -> scale/mask -> softmax -> PV` 的分解图 |
| Reduction | `sum`、`mean`，支持任意非空维度集合、`keepdim` 和输出 dtype 覆盖 |
| LayerNorm | `aten.native_layer_norm.default`，当前使用第 0 个输出 |
| RMSNorm | `F.rms_norm` 的 `pow/mean/add_/rsqrt/mul` 分解，折叠为专用 `rms_norm` |

编译器已经实现：

- 用 `make_fx` 捕获静态 shape 的 PyTorch 计算图
- 构建带 dtype/shape 校验的 SSA IR
- 消除恒等 layout、连续逆 `transpose/permute` 和 `unsqueeze/squeeze`，折叠相邻 `reshape`
- 把 PyTorch rank-N matmul 产生的 `reshape -> bmm -> reshape` 恢复为规范 `matmul`
- 把 `QK^T -> scale/mask -> softmax -> PV` 融合成单个 `attention` op，支持 rank-3/4、additive mask、causal mask、显式 scale，以及 GQA/MQA
- 识别 PyTorch 低精度 attention 在内部转 FP32、最后转回原 dtype 的 `_to_copy` 链；显式用户 dtype cast 仍保留为独立 IR op
- 直接识别 `F.scaled_dot_product_attention` 的 CPU flash、CUDA flash、efficient attention 和通用 ATen target
- 识别 PyTorch 分解图中 `repeat_kv` 产生的 `unsqueeze -> expand -> reshape`，融合后不再物化复制后的 K/V
- 把相邻 pointwise op 融合成一个 DAG region
- 消除 op、输入、输出 shape/dtype 和 attributes 均相同的公共子表达式，并保持 graph output alias
- 在 schedule 前校验静态视图和布局变换，Runtime 通过 MLX 执行 view 语义
- 支持多头 attention 的 `transpose/reshape/expand/bmm/softmax` 组合
- 按静态 shape 生成 broadcast 索引
- 把 tensor 与标量混合的 pointwise 运算编码成 IR attribute
- 生成每个 thread 处理 4 个元素的 Metal kernel
- FP16、BF16、FP32 输入和输出
- 可选的 Metal matmul/addmm codegen
- 可选的 fused Metal attention codegen，支持 rank-3/4、MHA/GQA/MQA、causal mask、additive mask 和 FP16/BF16/FP32
- matmul shared-memory tiling：`64×64×32` threadgroup block；FP16/BF16 完整块使用 SIMD matrix，其他 shape 用寄存器 microtile
- softmax reduction：每个 softmax slice 一个 threadgroup，在 threadgroup memory 中完成 max 和 sum，尾维走连续寻址 fast path
- sum/mean reduction：每个输出元素一个 threadgroup，在 float 中归约并按请求的输出 dtype 写回；连续 block 使用紧凑寻址，交错维度使用通用静态索引
- LayerNorm：每行一个 threadgroup，在 float 中做两遍 mean/variance reduction，并支持可选 affine
- RMSNorm：匹配 FX 分解链并折叠为专用 kernel，在 float 中做平方和 reduction，支持可选 weight
- 进程内 LRU kernel cache：相同 MSL 规格复用同一个 `mx.fast.metal_kernel` 对象，最多缓存 256 项

默认情况下，`matmul`、`addmm` 和融合后的 attention 调用 MLX，pointwise、softmax、sum/mean、LayerNorm 和 RMSNorm 由本项目生成 MSL。attention 使用 `mx.fast.scaled_dot_product_attention`，不会进入 `CompiledPlan.kernels`；该列表只包含本项目生成的 Metal kernel。传入 `codegen_matmul=True` 后，rank-2 矩阵乘和 bias 会生成自定义 Metal kernel：

```python
result = compile_callable(fn, example_inputs, codegen_matmul=True)
```

matmul kernel 使用 `64×64×32` threadgroup tile。每个 block 先把 A 的 64×32 和 B 的 32×64 片段协作加载到 threadgroup memory。FP16/BF16 输入且输出行列都能整除 64 时，8 个 SIMD group 使用 `simdgroup_*8x8` 指令做矩阵乘并保持 FP32 accumulator；FP32 或尾块继续使用每线程 4×4/8×4 的寄存器 microtile。当前仍没有 vectorized load、Metal 4 MPP、split-K 或 autotune，实测仍慢于 MLX，所以默认开关保持关闭。

传入 `codegen_attention=True` 会为满足条件的 attention 生成原生 MSL kernel：

```python
result = compile_callable(fn, example_inputs, codegen_attention=True)
```

kernel 为每个 `(batch, query head, query position)` 启动一个 threadgroup，在 threadgroup memory 中完成 QK 点积，再用 online softmax 累积 V。GQA/MQA 直接按 `kv_head = query_head / group_size` 读取 K/V，不物化重复后的 head。当前实现以正确性为先，还没有针对 MLX SDPA 做性能调优，因此默认开关关闭。`head_dim > 256` 或 mask 形式不在原生路径范围内时会自动回退到 MLX。

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
python examples/softmax.py
python examples/reduction.py
python examples/layer_norm.py
python examples/rms_norm.py
python examples/metal_matmul_benchmark.py
```

- `examples/mlp.py` 使用 MLX matmul 和生成的 pointwise kernel，矩阵规模更接近实际 MLP。
- `examples/metal_mlp.py` 使用小矩阵开启 `codegen_matmul=True`，完整验证 FX、IR、自定义 MSL matmul、融合 pointwise 和 runtime。
- `examples/softmax.py` 用超过 256 列的输入验证尾维和中间维的 Metal threadgroup max/sum reduction。
- `examples/reduction.py` 验证 last-dim sum、连续维 mean、交错维 sum、`keepdim` 和 FP16 输出 dtype 覆盖。
- `examples/layer_norm.py` 验证带 affine 参数和多线程跨步归约的 LayerNorm kernel。
- `examples/rms_norm.py` 验证 FX pattern folding、weight affine 和 RMSNorm reduction kernel。
- `examples/metal_matmul_benchmark.py` 对比 FP32 tiled 和 FP16 SIMD matmul kernel 与 MLX matmul，并检查 PyTorch CPU 数值误差。

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

编译结果是静态的：shape 和 dtype 来自 `example_inputs`，运行时必须匹配。当前可以编译并执行包含 RMSNorm、多头 attention、残差连接和 SwiGLU MLP 的完整 decoder block，但不支持动态 shape、控制流、自动微分、跨进程 kernel cache 和训练。Layout op 已具备静态校验、恒等与逆变换消除和 MLX lowering，但还没有布局传播。LayerNorm 只支持主输出，mean/rstd 不能作为图输出使用。

attention 当前要求 Q/K/V dtype 相同、rank 为 3 或 4，mask dtype 与输入相同。rank-4 输入支持 GQA/MQA：Q head 数必须是 K/V head 数的整数倍，MQA 是 K/V 只有 1 个 head 的特例。默认 lowering 使用 `mx.fast.scaled_dot_product_attention`；开启 `codegen_attention=True` 后，Q/K 和 V 的 head dim 均不超过 256 的常见形式会使用生成的 MSL attention kernel，其余形式回退到 MLX。暂不支持 attention dropout，也不会在 softmax 概率被其他节点观察时做激进融合。

精确 GELU 的 `erf` 使用 kernel header 中的数值近似，误差在测试覆盖范围内低于 `1e-6`。模型精度要求更高时应重新评估这个实现。

架构与实现细节见 [docs/architecture.md](docs/architecture.md)。
