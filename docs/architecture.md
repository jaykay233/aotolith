# 架构

当前实现是一条最小但可运行的 Apple Metal 编译链。目标不是覆盖完整 Transformer，而是把前端、IR、融合、Metal codegen 和 runtime 的接口固定下来，并持续增加可验证的 lowering。

## 1. FX 捕获

`capture_fx` 调用 `make_fx(fn)(*example_inputs)`。`make_fx` 执行一次函数并记录 ATen 算子，同时为每个节点保留具体的 tensor metadata。

前端当前处理以下 FX target 和折叠模式：

| FX target | IR op |
|---|---|
| `aten.mm.default` | `matmul` |
| `aten.addmm.default` | `addmm` |
| `aten._softmax.default` | `softmax` |
| `aten.sum.dim_IntList` | `sum` |
| `aten.mean.dim` | `mean` |
| `aten.native_layer_norm.default` | `layer_norm`，仅第 0 个输出 |
| `F.rms_norm` 的分解链 | `rms_norm`，由前端 pattern folding 生成 |
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

`addmm` 的 `alpha`/`beta`、`add`/`sub` 的 `alpha`、`gelu` 的 `approximate`、softmax 的 `dim`，以及 sum/mean 的 `dims`/`keepdim` 会写入 IR attribute。二元 pointwise 支持一个 tensor 和一个 Python scalar；scalar 以及它位于左侧还是右侧也保存在 attribute 中。`aten.rsub.Scalar` 单独映射为 `rsub`，因为它的 FX 参数顺序是 `(tensor, scalar)`，语义却是 `scalar - tensor`。

`F.layer_norm` 会被 `make_fx` 展开成 `native_layer_norm` 和三个 `getitem`。当前编译器把第 0 个输出折叠成单输出 `layer_norm` IR op，normalized shape、eps 以及是否带 weight/bias 保存在 attribute 中。如果调用方继续使用 mean 或 rstd，前端会明确拒绝，而不是返回错误语义。

`F.rms_norm` 没有单独的 ATen target，在当前 PyTorch 中会展开成 `pow -> mean -> add_ -> rsqrt -> mul -> optional weight mul`。前端只在中间节点单消费者、reduction 覆盖连续尾维且 shape 吻合时折叠成 `rms_norm`，避免把任意的平方、均值、rsqrt 组合误认为归一化。FP16/BF16 下 `make_fx` 还会插入 float32 `_to_copy`，折叠范围包含这组 cast。

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

softmax 是 reduction op，不会被收进 pointwise region。后面的 pointwise 链会单独融合，因此 `softmax -> mul -> add` 仍然产生两个 kernel，而不是把 reduction 和逐元素计算混在同一个表达式里。

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

## 6. Metal softmax reduction

softmax 当前只支持最后一维。backend 把前面的维度展平为 `rows`，最后一维作为连续的 `columns`：

- 每个 threadgroup 处理一行
- threadgroup 线程数是 `next_power_of_two(columns)`，上限为 256
- 每个线程跨步遍历该行，先在寄存器中得到局部 max
- threadgroup memory 中做树形 max reduction，再用相同结构做 exp-sum reduction
- 使用 max 平移保证数值稳定，最终在 float 中归一化并 cast 回 `T`

grid 大小是 `rows * threads`，每个 threadgroup 的大小固定为 `threads`。当 `columns > 256` 时，每个线程处理多个列；当 `columns` 不是 2 的幂时，线程数向上补齐，全局内存访问由循环条件 `col < columns` 限制。

非最后一维的 softmax 会在 backend 编译阶段抛 `NotImplementedError`，不会静默退化成 MLX。FP16/BF16 输入在 float 中完成 reduction，输出保持输入 dtype。

## 7. Metal sum/mean reduction

sum/mean 接受任意非空 reduction 维集合。backend 把未被 reduction 的维度压平为输出元素，把 reduction 维压平为 `columns`：

- 每个输出元素对应一个 threadgroup
- threadgroup 线程数复用 softmax 的选择规则，最多 256 个线程
- 每个线程跨步累加局部结果，再在 threadgroup memory 中做树形加法
- sum 直接写回归约值；mean 在 float 中除以 `columns` 后写回
- reduction 维组成连续 block 时，backend 把输入拆成 `outer × reduction_block × inner`，用紧凑地址映射；尾维还省略除法和取模
- reduction 维交错时，backend 从 `output_index` 生成非 reduction 维的输入基址，从 `col` 生成各 reduction 维坐标，再组合成静态输入偏移
- 只有 `tid == 0` 写一个输出元素，`keepdim` 只影响输出 shape，不影响线性写入位置

输入输出必须保持相同 dtype。显式传入不同的 reduction dtype 会在前端被拒绝。每个输出元素启动一个 threadgroup，因此输出元素很多而 reduction 很小时 dispatch 开销会偏高；交错维的通用索引还会为每次读取增加整数除法和取模。

## 8. Metal LayerNorm 和 RMSNorm lowering

两种归一化都按 normalized shape 的最后一组维度处理。backend 把 normalized shape 之前的维度展平为 `rows`，把 normalized shape 展平为连续的 `columns`：

- 每个 threadgroup 处理一行
- threadgroup 大小与 softmax 使用同一套选择规则，最多 256 个线程
- reduction 和归一化都使用 float，最后 cast 回输入 dtype
- weight 与 normalized shape 完全一致，并且 dtype 与输入相同

LayerNorm 分两遍读取输入：先计算行均值，再基于 `(x - mean)^2` 计算 variance，避免 `E[x^2] - E[x]^2` 的相消误差。最后用 `rsqrt(variance + eps)` 归一化，并可在同一 kernel 中应用 weight/bias。bias 与 weight 可以分别缺省。

RMSNorm 只做一遍平方和 reduction，得到 `mean(x^2)` 后用 `rsqrt(mean(x^2) + eps)` 缩放输入，再应用可选 weight。kernel 不接受 bias，因为当前折叠的 `F.rms_norm` 语义没有 bias。

两者都支持 normalized shape 占满最后一维、多个连续尾维，或整个 tensor。若 normalized shape 不是连续尾维，PyTorch 不会产生对应的归一化 FX 结构。

## 9. Metal matmul/addmm codegen

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

## 10. Runtime

`CompiledPlan` 按 SSA 顺序执行 IR：

- 有生成 kernel 的 op：调用对应的 `mx.fast.metal_kernel`
- 默认模式下的 `matmul`、`addmm`：调用 MLX
- 未生成 kernel 的 pointwise op：调用 MLX fallback
- softmax：调用生成的 last-dim reduction kernel
- sum/mean：调用生成的 static reduction kernel
- layer_norm：调用生成的两遍 mean/variance reduction kernel
- rms_norm：调用生成的平方和 reduction kernel

每个 graph input 在进入计划前会检查 shape 和 dtype。输出统一以 tuple 返回，即使图只有一个输出。

## 11. 已知边界

- 只支持静态、非空 shape
- 不支持动态 shape、控制流、list/dict 输出和 `ones_like` 等常量生成节点
- pointwise 要求输入输出 dtype 相同
- softmax 只支持最后一维
- sum/mean 支持任意非空维度集合，但不支持 reduction dtype override
- LayerNorm/RMSNorm 只支持连续尾维 reduction
- LayerNorm 只支持主输出，不支持直接使用 mean/rstd
- RMSNorm 依赖当前 PyTorch FX 分解形态；上游改变分解链后需要同步更新 pattern
- 没有 kernel cache、autotune、layout propagation
- matmul 还没有 shared-memory tile、vectorized load、SIMD matrix 或 split-K
- 没有反向传播和训练支持
- NVIDIA Triton/CUDA backend 尚未实现

下一步最有价值的方向是 kernel cache，以及用 SIMD matrix 或 MPS 路径继续优化 matmul。继续扩大 pointwise 算子覆盖的收益已经开始下降。
