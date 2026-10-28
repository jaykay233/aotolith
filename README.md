# Transformer Subgraph Compiler

一个面向 Transformer 子图的实验性编译器。当前版本以 Apple Silicon 为第一目标，把 PyTorch FX 图转换成自定义 SSA IR，消除冗余布局与公共子表达式，融合 attention 和 pointwise 子图，再通过生成的 Metal Shading Language kernel 与 MLX runtime 在 Apple GPU 上执行。

```text
PyTorch FX
  -> SSA IR
  -> layout simplification / matmul layout folding and canonicalization / attention and pointwise fusion / CSE
  -> Metal Shading Language + MLX runtime lowering
  -> mx.fast.metal_kernel
  -> Apple GPU
```

## 当前能力

支持的主要 FX target 和折叠模式：

| 类别 | 节点 |
|---|---|
| 矩阵乘 | `aten.mm`、`aten.bmm`、`aten.addmm`、`aten.baddbmm` |
| 索引读写 | `aten.embedding.default`、`aten.index_select.default`、`aten.gather.default`、`aten.scatter.src`、`aten.scatter.value`，支持 `int32/int64` indices，并由 MLX gather/scatter 执行 |
| 视图、布局与 dtype | `view`、`reshape`、`transpose`、`permute`、`expand`、`slice`、`select`、`flip`、`roll`、`triu`、`tril`、`split`、`chunk`、`unbind`、`unsqueeze`、`squeeze`、`clone`、`alias`、`_to_copy` |
| 张量组合 | `aten.cat.default`；`aten.stack.default` 展开为 `unsqueeze + cat`，校验输入 shape/dtype 后交给 MLX |
| 激活与边界 | `silu`、`gelu`、`softplus`、`mish`、`logsigmoid`、`sigmoid`、`tanh`、`relu`、`leaky_relu`、`elu`、`hardswish`、`hardsigmoid`、`threshold`、`clamp`、`clamp_min`、`clamp_max`、`hardtanh` |
| 一元运算与谓词 | `neg`、`abs`、`exp`、`expm1`、`exp2`、`log`、`log2`、`log10`、`log1p`、`erf`、`sin`、`cos`、`floor`、`ceil`、`round`、`trunc`、`sign`、`square`、`sqrt`、`rsqrt`、`reciprocal`、`isnan`、`isinf` |
| 二元运算 | `mul`、`add`、`sub`、`rsub`、`div`、`pow`、`maximum`、`minimum`、`logaddexp`、`atan2` |
| 条件与选择 | `eq`、`ne`、`lt`、`le`、`gt`、`ge`、`logical_and`、`logical_or`、`logical_xor`、`logical_not`、`where`、`masked_fill`，以及布尔 tensor 的 `bitwise_*` 和 bool scalar 变体 |
| 常量生成 | `aten.scalar_tensor.default`、`aten.zeros_like.default`、`aten.ones_like.default`、`aten.full_like.default`；静态 shape/dtype 来自输出 metadata |
| Softmax | `aten._softmax.default`、`aten._safe_softmax.default`，支持任意单个维度 |
| Attention | `F.scaled_dot_product_attention`，以及 `QK^T -> scale/mask -> softmax -> PV` 的分解图 |
| Reduction | `sum`、`mean`，支持不指定维度的全量归约、任意非空维度集合、`keepdim` 和输出 dtype 覆盖 |
| LayerNorm | `aten.native_layer_norm.default`，当前使用第 0 个输出 |
| RMSNorm | `F.rms_norm` 的 `pow/mean/add_/rsqrt/mul` 分解，折叠为专用 `rms_norm` |

编译器已经实现：

- 用 `make_fx` 捕获静态 shape 的 PyTorch 计算图
- 构建带 dtype/shape 校验的 SSA IR
- 支持 `aten.embedding.default` 的 token-id gather，接受 `int32/int64` indices 和 FP16/BF16/FP32 embedding table，并可与后续 matmul、pointwise 子图组合
- 支持 `aten.index_select.default` 沿任意单轴 gather，保留负轴归一化、输出 dtype 和静态 shape；运行时校验索引非负且未越界
- 支持 `aten.gather.default` 的逐元素单轴 gather；前端校验 index rank 和静态 shape，运行时裁掉非 gather 轴的多余范围后交给 MLX `take_along_axis`
- 支持 `aten.scatter.src` 和 `aten.scatter.value` 的单轴写入；校验 source、index、update 的静态 shape，运行时复制 source 后通过 MLX `put_along_axis` 局部更新，不修改输入
- 消除恒等 layout、连续逆 `transpose/permute`、`flip`、`roll` 以及 `unsqueeze/squeeze`，折叠相邻 `reshape`
- 把 PyTorch rank-N matmul 产生的 `reshape -> bmm -> reshape` 恢复为规范 `matmul`
- 把 `matmul/addmm` 输入侧单消费者的最后一维 `transpose/permute` 折叠为 `transpose_a`/`transpose_b`；Runtime 和生成的 Metal kernel 直接按逻辑布局取数
- 消除仍能由 matmul batch broadcast 恢复语义的单消费者 `expand`，避免为广播输入生成额外拷贝
- 支持沿正负轴拼接 tensor，可用于 RoPE 的 `rotate_half`、QKV 拼接等常见结构；`cat` 本身由 MLX runtime 执行，前后 pointwise 子图仍分别融合
- 把 `stack` 展开为每个输入的 `unsqueeze` 加一个 `cat`；把 `flip`、`roll`、`triu`、`tril` 保留为可校验 op，由 MLX 执行而不生成额外 Metal kernel
- 把 `aten.select.int` 展开为单元素 `slice` 加 `squeeze`，支持负轴和负索引，并能输出 rank-0 scalar
- 支持 `zeros_like`、`ones_like` 和 `full_like` 生成静态 FP16/BF16/FP32/bool 常量；模板 tensor 只用于捕获 shape/dtype，不进入运行时 IR，常量可与相邻 pointwise op 融合
- 把静态 `split`、`chunk` 和 `unbind` 展开为已有的 `slice`/`squeeze` op；Q/K/V 投影结果拆开后仍可共享 layout、CSE 和 pointwise 融合
- 把 `QK^T -> scale/mask -> softmax -> PV` 融合成单个 `attention` op，支持 rank-3/4、additive/bool mask、causal mask、显式 scale，以及 GQA/MQA
- 识别 PyTorch 低精度 attention 在内部转 FP32、最后转回原 dtype 的 `_to_copy` 链；显式用户 dtype cast 仍保留为独立 IR op
- 直接识别 `F.scaled_dot_product_attention` 的 CPU flash、CUDA flash、efficient attention 和通用 ATen target
- 识别 PyTorch 分解图中 `repeat_kv` 产生的 `unsqueeze -> expand -> reshape`，融合后不再物化复制后的 K/V
- 把相邻 pointwise op 融合成一个 DAG region
- 消除 op、输入、输出 shape/dtype 和 attributes 均相同的公共子表达式；融合 region 会先对内部 SSA value 做结构化编号，并同步改写上游 alias，因此内部临时 ID 不同的等价 region 也能合并
- 在 schedule 前校验静态视图和布局变换；未折叠的 layout op 由 Runtime 执行 MLX view 语义
- 支持多头 attention 的 `transpose/reshape/expand/bmm/softmax` 组合
- 按静态 shape 生成 broadcast 索引；`pow` 支持 tensor exponent、标量指数和标量底数，`clamp`/`clamp_min`/`clamp_max` 支持标量和可广播 tensor 边界，并与新增的一元数学算子、`leaky_relu`、`elu`、`hardswish` 一起参与 pointwise 融合
- 支持 `softplus`、`mish` 和 `logsigmoid` 的稳定 Metal lowering；`softplus` 保留 `beta`/`threshold`，`log_sigmoid_forward` 只接受第 0 个输出，消费 backward buffer 会明确报错
- 支持 `maximum`/`minimum` 的 NaN 传播和 signed-zero 语义、`logaddexp` 的稳定计算，以及 `atan2` 的零和无穷边界；这些算子与 `hardsigmoid`、`threshold`、`exp2` 一起参与 pointwise 融合
- 把 tensor 与标量混合的 pointwise 运算编码成 IR attribute
- 支持 `torch.bool` IR value、比较/logical/NaN/Inf 谓词、`where`、`masked_fill`、标量常量生成，并允许 bool predicate/condition 与 FP16/BF16/FP32 数值分支融合在同一个 kernel
- 生成每个 thread 处理 4 个元素的 Metal kernel
- FP16、BF16、FP32 数值计算，bool 中间值和输出，以及供 embedding/index_select/gather/scatter indices 使用的 `int32/int64` 输入
- 可选的 Metal matmul/addmm codegen，支持 rank-2 和 rank-3/4 的标准右对齐 batch broadcast
- 可选的 tiled online-softmax Metal attention codegen，支持 rank-3/4、MHA/GQA/MQA、causal mask、additive/bool mask 和 FP16/BF16/FP32
- matmul shared-memory tiling：`64×64×32` threadgroup block；rank-N batch 映射到 grid 的 y 维，FP16/BF16 完整块使用 SIMD matrix，其他 shape 用寄存器 microtile
- softmax reduction：每个 softmax slice 一个 threadgroup，在 threadgroup memory 中完成 max 和 sum，尾维走连续寻址 fast path
- sum/mean reduction：每个输出元素一个 threadgroup，在 float 中归约并按请求的输出 dtype 写回；连续 block 使用紧凑寻址，交错维度使用通用静态索引
- LayerNorm：每行一个 threadgroup，在 float 中做两遍 mean/variance reduction，并支持可选 affine
- RMSNorm：匹配 FX 分解链并折叠为专用 kernel，在 float 中做平方和 reduction，支持可选 weight
- 进程内 LRU kernel cache：相同 MSL 规格复用同一个 `mx.fast.metal_kernel` 对象，最多缓存 256 项

默认情况下，`matmul`、`addmm` 和融合后的 attention 调用 MLX，pointwise、softmax、sum/mean、LayerNorm 和 RMSNorm 由本项目生成 MSL。attention 使用 `mx.fast.scaled_dot_product_attention`，不会进入 `CompiledPlan.kernels`；该列表只包含本项目生成的 Metal kernel。PyTorch 可能先把 bool mask 分解成 `where(mask, 0, -inf)`，此分解后的 additive mask 与直接传入的 bool mask 都可进入原生 attention kernel。传入 `codegen_matmul=True` 后，rank-2 以及 batch 前缀可广播的 rank-3/4 矩阵乘会生成自定义 Metal kernel。编译器会把可由另一操作数或输出 shape 证明安全的单消费者 batch `expand` 折入 matmul；变体无法保持原输出 shape 时保留 `expand`，由 MLX Runtime 执行：

```python
result = compile_callable(fn, example_inputs, codegen_matmul=True)
```

matmul kernel 使用 `64×64×32` threadgroup tile。每个 block 先把 A 的 64×32 和 B 的 32×64 片段协作加载到 threadgroup memory。FP16/BF16 输入且输出行列都能整除 64 时，8 个 SIMD group 使用 `simdgroup_*8x8` 指令做矩阵乘并保持 FP32 accumulator；reduction 和输出列也满足 8 元素对齐时，A/B shared tile 按 16-byte 对齐并使用 `vec<T, 8>` 加载，尾 K、尾输出块和 F32 仍走标量加载。FP32 或尾块继续使用每线程 4×4/8×4 的寄存器 microtile。当前仍没有 Metal 4 MPP、split-K 或 autotune，实测仍慢于 MLX，所以默认开关保持关闭。

传入 `codegen_attention=True` 会为满足条件的 attention 生成原生 MSL kernel：

```python
result = compile_callable(fn, example_inputs, codegen_attention=True)
```

kernel 每个 threadgroup 处理 8、16 或 24 个 query。短序列使用 32-key tile；query sequence 达到 768 后，FP16/BF16 且 QK/PV 都满足 SIMD matrix 条件时优先尝试 `24×64` tile、768 个 thread，再依次回退到 `16×64`、`24×32` 和 `16×32`。64-key 路径用 16 或 24 个 SIMD group 并行计算 QK，并在 shared memory 中复用 score 与 probability workspace；PV 按 64 个 value 维度分块，避免超过 31 KiB。Q/K/V tile 先协作加载到 32-byte 对齐的 threadgroup memory；head dim 都能被 16 整除时使用 16 元素向量加载，否则使用 8 元素向量加载。warp 内用 `simd_max`/`simd_sum` 做 online softmax。GQA/MQA 直接按 `kv_head = query_head / group_size` 读取 K/V，不物化重复后的 head。当前还没有运行时 autotune，实测仍慢于 MLX SDPA，因此默认开关关闭。`head_dim > 256` 或 mask 形式不在原生路径范围内时会自动回退到 MLX。

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
python examples/embedding.py
python examples/metal_mlp.py
python examples/softmax.py
python examples/reduction.py
python examples/layer_norm.py
python examples/rms_norm.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

- `examples/mlp.py` 使用 MLX matmul 和生成的 pointwise kernel，矩阵规模更接近实际 MLP。
- `examples/embedding.py` 从 token id 查表，再执行 projection 和 SiLU，验证整数输入、MLX gather 与后续融合。
- `examples/metal_mlp.py` 使用小矩阵开启 `codegen_matmul=True`，完整验证 FX、IR、自定义 MSL matmul、融合 pointwise 和 runtime。
- `examples/softmax.py` 用超过 256 列的输入验证尾维和中间维的 Metal threadgroup max/sum reduction。
- `examples/reduction.py` 验证 last-dim sum、连续维 mean、交错维 sum、`keepdim` 和 FP16 输出 dtype 覆盖。
- `examples/layer_norm.py` 验证带 affine 参数和多线程跨步归约的 LayerNorm kernel。
- `examples/rms_norm.py` 验证 FX pattern folding、weight affine 和 RMSNorm reduction kernel。
- `examples/metal_matmul_benchmark.py` 对比 FP32 tiled 和 FP16 SIMD matmul kernel 与 MLX matmul，并检查 PyTorch CPU 数值误差。
- `examples/metal_attention_benchmark.py` 对比 tiled online-softmax attention kernel 与 MLX SDPA，并检查 PyTorch CPU 数值误差。

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

编译结果是静态的：shape 和 dtype 来自 `example_inputs`，运行时必须匹配。当前可以编译并执行包含 RMSNorm、多头 attention、残差连接和 SwiGLU MLP 的完整 decoder block，但不支持动态 shape、控制流、自动微分、跨进程 kernel cache 和训练。`split`、`chunk`、`unbind` 要求由 example inputs 求出的每个输出 shape 都是静态且非空的。Layout 优化已覆盖恒等、逆变换、matmul 输入转置和单消费者 batch `expand` 折叠；`reshape` 跨转置以及非 matmul 边界的一般广播布局传播尚未实现。LayerNorm 只支持主输出，mean/rstd 不能作为图输出使用。

attention 当前要求 Q/K/V dtype 相同、rank 为 3 或 4，mask 可以是同 dtype 的 additive mask 或 bool mask。rank-4 输入支持 GQA/MQA：Q head 数必须是 K/V head 数的整数倍，MQA 是 K/V 只有 1 个 head 的特例。默认 lowering 使用 `mx.fast.scaled_dot_product_attention`；开启 `codegen_attention=True` 后，Q/K 和 V 的 head dim 均不超过 256 的常见 additive-mask 或 bool-mask 形式会使用生成的 MSL attention kernel。bool mask 的 `True` 表示允许该 key 参与 attention，整行都为 `False` 时输出为零。暂不支持 attention dropout，也不会在 softmax 概率被其他节点观察时做激进融合。

精确 GELU 的 `erf` 使用 kernel header 中的数值近似，误差在测试覆盖范围内低于 `1e-6`。模型精度要求更高时应重新评估这个实现。

`round` 使用 `metal::rint` 保持 PyTorch 的 ties-to-even 语义。Apple MSL 没有可直接调用的 `expm1`，编译器在 |x| <= 0.5 时使用七阶 Taylor 展开，其他范围使用 `exp(x) - 1`；测试覆盖的输入上最大误差小于 `1e-7`。

架构与实现细节见 [docs/architecture.md](docs/architecture.md)。
