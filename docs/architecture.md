# 架构

当前实现是一条最小但可运行的 Apple Metal 编译链。目标不是覆盖完整 Transformer，而是把前端、IR、融合、Metal codegen 和 runtime 的接口固定下来，并持续增加可验证的 lowering。

## 1. FX 捕获

`capture_fx` 调用 `make_fx(fn)(*example_inputs)`。`make_fx` 执行一次函数并记录 ATen 算子，同时为每个节点保留具体的 tensor metadata。

前端当前只接受以下 ATen 目标：

| FX target | IR op |
|---|---|
| `aten.mm.default` | `matmul` |
| `aten.addmm.default` | `addmm` |
| `aten.silu.default` | `silu` |
| `aten.gelu.default` | `gelu` |
| `aten.mul.Tensor` | `mul` |
| `aten.add.Tensor` | `add` |
| `aten.sub.Tensor` | `sub` |
| `aten.rsub.Scalar` | `rsub` |
| `aten.div.Tensor` | `div` |
| `aten.neg.default` | `neg` |
| `aten.abs.default` | `abs` |
| `aten.sigmoid.default` | `sigmoid` |
| `aten.tanh.default` | `tanh` |
| `aten.relu.default` | `relu` |
| `aten.exp.default` | `exp` |
| `aten.erf.default` | `erf` |
| `aten.sqrt.default` | `sqrt` |
| `aten.rsqrt.default` | `rsqrt` |
| `aten.reciprocal.default` | `reciprocal` |

每个 placeholder 变成 `argN`，每个计算节点变成 `vN`。shape 和 dtype 从 `node.meta["val"]` 读取，动态维度和空维度会直接报错。

`addmm` 的 `alpha`/`beta`、`add`/`sub` 的 `alpha` 和 `gelu` 的 `approximate` 会写入 IR attribute。二元 pointwise 支持一个 tensor 和一个 Python scalar；scalar 以及它位于左侧还是右侧也保存在 attribute 中。`aten.rsub.Scalar` 单独映射为 `rsub`，因为它的 FX 参数顺序是 `(tensor, scalar)`，语义却是 `scalar - tensor`。

## 2. SSA IR

IR 定义在 `ir.py`：

- `Value` 有唯一 ID 和静态 `TensorType`
- `Operation` 只产生一个输出，输入是已定义的 value ID
- `Graph.validate()` 检查输入唯一性、use-before-def、重复定义、输出类型是否存在

这是限制过的 SSA 形式。后续加入 layout、tile、schedule 时，可以继续扩展 operation attribute，而不需要改前端的 value 命名规则。

## 3. Pointwise 融合

`fuse_pointwise` 找出图的融合区域：

1. 把支持的一元和二元运算标记为 pointwise op。
2. 从 region root 反向收集 producer。
3. producer 的输出必须只被当前 region 使用，否则停止跨越该 producer。
4. region 至少包含两个 op 才替换成 `fused_elementwise`。
5. `fused_elementwise` 的 `attrs["body"]` 保存原始 pointwise op 序列。

region root 是满足以下条件之一的 pointwise op：

- 它的输出是 graph output
- 它有一个非 pointwise consumer

只有单个 pointwise op 时不做 IR fusion，但 Apple backend 仍会为它生成一个独立 Metal kernel。

融合把 region 内部的中间 value 消掉，只把外部输入放进 `fused_elementwise.inputs`，输出 ID 保持为 root 原来的输出 ID。

例如：

```text
v3 = silu(v1)
v4 = mul(v3, v2)
```

变成：

```text
v4 = fused_elementwise(v1, v2) {
  temp_0 = silu(v1)
  temp_1 = mul(temp_0, v2)
  return temp_1
}
```

## 4. Broadcast 索引

pointwise kernel 使用输出 tensor 的扁平索引 `elem`。每个输入先检查静态 broadcast 规则：

- 输入 rank 不能大于输出 rank
- 从右向左对齐
- 每一维必须等于输出维，或等于 1

对需要 broadcast 的输入，codegen 根据输出 stride 还原坐标，再按输入 stride 计算线性 offset。例如输出 `[2, 3]`、输入 `[1, 3]` 时：

```text
input_offset = (elem % 3) * 1
```

这避免了依赖 MLX 在 kernel 调用前自动 broadcast，也把索引推导留在编译器里。

## 5. Metal pointwise codegen

`AppleMetalBackend._compile_elementwise` 生成 elementwise Metal kernel：

- grid 大小为 `ceil(输出元素数 / 4)`
- 每个 thread 依次处理最多 4 个元素
- threadgroup 最大为 256
- 输入转成 `float` 计算
- 最终结果转回模板类型 `T`
- `mx.fast.metal_kernel` 在首次调用时完成 JIT

精确 GELU 和独立的 `erf` 使用 `_MSL_HEADER` 中的 Abramowitz-Stegun 近似。Apple MSL 没有 `metal::erf`，所以不能直接生成标准库调用。

## 6. Metal matmul/addmm codegen

`codegen_matmul=True` 时，backend 为 rank-2 `matmul` 和 `addmm` 生成独立 Metal kernel：

```text
C[row, col] = sum_k A[row, k] * B[k, col]
```

`addmm` 在同一 kernel 中计算：

```text
beta * bias + alpha * (A @ B)
```

bias 可以是 rank-0、一维或二维 tensor，只要它满足到输出 shape 的静态 broadcast 规则。FP16/BF16 输入用 float 累加，最后 cast 回输出 dtype。

### 寄存器 tiling

每个 thread 计算一个 microtile，而不是只计算一个输出：

- 默认使用 4×4 microtile，即每个 thread 持有 16 个 float accumulator
- reduction 维达到 1024 时使用 8×4 microtile，增加大 K 下的指令级并行
- grid 是 `ceil(columns / micro_columns) × ceil(rows / micro_rows)`
- K 循环里加载 microtile 对应的 A/B 行和列，用 `fma` 更新所有 accumulator
- row/column 能被 microtile 整除时，step 直接生成无 predicate 的加载和写回
- 只有尾部 tile 才生成边界检查，避免常见整除 shape 在 K 循环里付分支成本

该实现没有 threadgroup memory、vectorized load、SIMD matrix 或 split-K。它在同一台机器上比旧的一线程一输出 kernel 快约 2–4 倍，但仍明显慢于 MLX，因此默认 `codegen_matmul=False`。基准可用 `examples/metal_matmul_benchmark.py` 复现。

## 7. Runtime

`CompiledPlan` 按 SSA 顺序执行 IR：

- 有生成 kernel 的 op：调用对应的 `mx.fast.metal_kernel`
- 默认模式下的 `matmul`、`addmm`：调用 MLX
- 未生成 kernel 的 pointwise op：调用 MLX fallback

每个 graph input 在进入计划前会检查 shape 和 dtype。输出统一以 tuple 返回，即使图只有一个输出。

## 8. 已知边界

- 只支持静态、非空 shape
- 不支持动态 shape、控制流、list/dict 输出和 `ones_like` 等常量生成节点
- pointwise 要求输入输出 dtype 相同
- reduction、softmax、normalization 尚未 lowering
- 没有 kernel cache、autotune、layout propagation
- matmul 还没有 shared-memory tile、vectorized load、SIMD matrix 或 split-K
- 没有反向传播和训练支持
- NVIDIA Triton/CUDA backend 尚未实现

下一步最有价值的方向是 reduction/normalization lowering、kernel cache，以及用 SIMD matrix 或 MPS 路径继续优化 matmul。继续扩大 pointwise 算子覆盖的收益已经开始下降。
