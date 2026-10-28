# 架构

当前实现是一条最小但可运行的 Apple Metal 编译链。目标不是覆盖完整 Transformer，而是把前端、IR、融合、Metal codegen 和 runtime 的接口固定下来，并持续增加可验证的 lowering。

## 1. FX 捕获

`capture_fx` 调用 `make_fx(fn)(*example_inputs)`。`make_fx` 执行一次函数并记录 ATen 算子，同时为每个节点保留具体的 tensor metadata。

前端当前处理以下 FX target 和折叠模式：

| FX target | IR op |
|---|---|
| `aten.mm.default` | `matmul` |
| `aten.bmm.default` | `matmul` |
| `aten.addmm.default` | `addmm` |
| `aten.view.default`、`aten._unsafe_view.default`、`aten.reshape.default` | `reshape` |
| `aten.transpose.int` | `transpose` |
| `aten.permute.default` | `permute` |
| `aten.expand.default` | `expand` |
| `aten.slice.Tensor` | `slice` |
| `aten.unsqueeze.default` | `unsqueeze` |
| `aten.squeeze.dim` | `squeeze` |
| `aten.clone.default`、`aten.alias.default` | `contiguous`、`alias` |
| `aten._to_copy.default` | `cast` |
| `aten._softmax.default`、`aten._safe_softmax.default` | `softmax`、`safe_softmax` |
| `aten.scaled_dot_product_attention.default` 及 CPU/CUDA flash、efficient attention target | `attention` |
| `aten.sum.dim_IntList` | `sum` |
| `aten.mean.dim` | `mean` |
| `aten.native_layer_norm.default` | `layer_norm`，仅第 0 个输出 |
| `F.rms_norm` 的分解链 | `rms_norm`，由前端 pattern folding 生成 |
| `aten.silu.default` | `silu` |
| `aten.gelu.default` | `gelu` |
| `aten.mul.Tensor`、`aten.mul.Scalar` | `mul` |
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

每个 placeholder 变成 `argN`，每个计算节点变成 `vN`。shape 和 dtype 从 `node.meta["val"]` 读取，动态维度和空维度会直接报错。`bmm` 和 `mm` 统一为 `matmul` IR op，后面的 canonicalization pass 会把 rank-N matmul 被 PyTorch 展开的 `reshape -> bmm -> reshape` 还原成规范形式。Runtime 用同一个 batched matmul 语义执行。

Layout op 保留为独立 IR 节点。Backend 在下发前检查 reshape 的元素数、transpose/permute 的轴置换、expand 的 broadcast 合法性、slice 的静态输出 shape、unsqueeze 的插入轴，以及 squeeze 删除的 singleton 轴。它们不生成 Metal kernel，而是由 Runtime 映射到 MLX 的 reshape、swapaxes、transpose、broadcast、切片、expand-dims 和 squeeze。`clone` 使用 `mx.contiguous` 建立独立布局，`alias` 直接复用输入数组；两者保持 dtype、shape 和值不变。

`addmm` 的 `alpha`/`beta`、`add`/`sub` 的 `alpha`、`gelu` 的 `approximate`、softmax 的 `dim`，以及 sum/mean 的 `dims`/`keepdim`/`dtype` 会写入 IR attribute。`_to_copy` 的 dtype 直接由 FX 节点的输出 metadata 表示，backend 用 MLX `astype` 执行。二元 pointwise 支持一个 tensor 和一个 Python scalar；scalar 以及它位于左侧还是右侧也保存在 attribute 中。`aten.rsub.Scalar` 单独映射为 `rsub`，因为它的 FX 参数顺序是 `(tensor, scalar)`，语义却是 `scalar - tensor`。

`F.layer_norm` 会被 `make_fx` 展开成 `native_layer_norm` 和三个 `getitem`。当前编译器把第 0 个输出折叠成单输出 `layer_norm` IR op，normalized shape、eps 以及是否带 weight/bias 保存在 attribute 中。如果调用方继续使用 mean 或 rstd，前端会明确拒绝，而不是返回错误语义。

`F.rms_norm` 没有单独的 ATen target，在当前 PyTorch 中会展开成 `pow -> mean -> add_ -> rsqrt -> mul -> optional weight mul`。前端只在中间节点单消费者、reduction 覆盖连续尾维且 shape 吻合时折叠成 `rms_norm`，避免把任意的平方、均值、rsqrt 组合误认为归一化。FP16/BF16 下 `make_fx` 还会插入 float32 `_to_copy`，折叠范围包含这组 cast。

`F.scaled_dot_product_attention` 可能落到通用 ATen op，也可能被拍平成 CPU/CUDA flash 或 efficient attention 的内部 target。这些 target 的参数顺序和输出个数不同，前端按 target 分别读取 `attn_mask`、`dropout_p`、`is_causal`、`scale` 和关键字参数 `enable_gqa`。当前只接受主输出；辅助 logsumexp、debug mask 等输出被使用时直接报错。dropout 非零也会报错。`enable_gqa` 只做布尔校验，实际 head 分组关系由 Q/K/V 的静态 shape 决定；部分 CPU SDPA 路径在 tracing 后会丢掉这个参数并落到 flash target。

FP16/BF16 的 CPU SDPA 分解图常把 Q/K/V 转成 FP32 计算，再把结果转回输入 dtype。frontend 将这些节点保留为 `cast`。attention fusion 只在输入 cast 来自同一种低精度 dtype、输出 cast 又回到同一 dtype 时吸收整条内部 round-trip；显式 `to(bfloat16)` 或 `to(float32)` 等用户 cast 仍作为独立节点执行，不能被错误地改写成另一种 attention dtype。

## 2. SSA IR

IR 定义在 `ir.py`：

- `Value` 有唯一 ID 和静态 `TensorType`
- `Operation` 只产生一个输出，输入是已定义的 value ID
- `Graph.validate()` 检查输入唯一性、use-before-def、重复定义、输出类型是否存在

这是限制过的 SSA 形式。后续加入 layout、tile、schedule 时，可以继续扩展 operation attribute，而不需要改前端的 value 命名规则。

## 3. Layout 简化、matmul 规范化与 attention 融合

`simplify_layouts` 先删除不改变数据的 layout op，包括 shape 相同的 `reshape/expand/squeeze`、交换同一轴的 `transpose`、恒等 `permute`、完整 slice、`alias` 和 `contiguous`。连续且互逆的 `transpose/permute` 会折叠掉，连续的 `unsqueeze/squeeze` 在恢复原 shape 时也会被消除，相邻 `reshape` 会合并成从最早输入到最终 shape 的一次 reshape。删除前仍会解析轴和检查静态 shape，非法轴不会被当成恒等操作跳过。

`canonicalize_batched_matmul` 识别 PyTorch 为 rank-N matmul 生成的折叠模式：输入先 reshape 成二维 batch，执行 `bmm`，输出再 reshape 回原始 batch shape。匹配成功后替换成一个 `matmul`，避免后续 attention pattern 只看到低秩 `bmm`。

`fuse_attention` 从 PV matmul 反向匹配 `QK^T -> optional scale -> optional additive mask -> softmax -> PV`。它也能恢复 PyTorch 把 batch-head 拍平后执行的 `reshape -> mm -> reshape`。只有内部 value 都只有单个 consumer，且没有被 graph output 观察时才会替换成 `attention`。`scale`、`has_mask`、`causal` 和 `safe_softmax` 存在 operation attribute 中，Q/K/V 保持原 graph 中的 rank-3 或 rank-4 形式。softmax probability 被其他节点使用时不会融合，因为替换会丢掉该中间结果。

GQA/MQA 有两种入口。直接调用 `F.scaled_dot_product_attention(..., enable_gqa=True)` 时，前端直接生成 `attention`；显式 `repeat_kv` 的分解图则通过 `unsqueeze -> expand -> reshape` 识别出原始 K/V。后一种模式会去掉物化复制，把 K/V head 数低于 Q head 数的 `attention` 交给 MLX。rank-4 下要求 Q head 数是 K/V head 数的整数倍；MQA 是 K/V head 数为 1 的特例。

Apple backend 默认让 `attention` 调用 `mx.fast.scaled_dot_product_attention`，GQA/MQA 同样复用这个 MLX 实现。rank-3 输入在调用前临时增加 head 轴，调用后 squeeze 回去。当 `safe_softmax=True` 且存在 additive mask 时，MLX lowering 会把整行都被 mask 的输出显式置零，避免 fused SDPA 产生 NaN。此时 `CompiledPlan.kernels` 只记录项目生成的 Metal kernel，不记录 MLX 内置的 attention 实现。开启 `codegen_attention=True` 后，满足条件的 attention 会改为执行下面描述的生成 kernel；不满足条件时仍回退到 MLX。MLX fused SDPA、生成 kernel 与逐算子 `matmul/softmax` 存在浮点路径差异，测试按 dtype 使用 `2e-3` 到 `2e-2` 的容差。

## 4. Pointwise 融合

`fuse_pointwise` 找出图的融合区域：

1. 把支持的一元和二元运算标记为 pointwise op。
2. 找出 region root：输出被 graph output 或非 pointwise consumer 观察的 pointwise op。
3. 从每个 root 开始迭代式反向闭包，只有 producer 没有被 graph output 观察、且它的全部 consumer 都已经在 region 内时才纳入 producer。
4. region 至少包含两个 op 才替换成 `fused_elementwise`。
5. `fused_elementwise` 的 `attrs["body"]` 保存原始 pointwise op 序列。

只有单个 pointwise op 时不做 IR fusion，但 Apple backend 仍会为它生成一个独立 Metal kernel。

闭包规则允许共享 producer 的 DAG 进入同一个 region。例如 `s = sigmoid(x); (s + s) * s` 中，`sigmoid` 的三个 consumer 都在 region 内，因此整条链只需一个 kernel。仍然被 graph output 观察的中间值不会从图中删除；backend 按 region 内 value ID 做局部 CSE，让同一表达式只生成一次代码。

softmax 是 reduction op，不会被收进 pointwise region。后面的 pointwise 链会单独融合，因此 `softmax -> mul -> add` 仍然产生两个 kernel，而不是把 reduction 和逐元素计算混在同一个表达式里。

融合把 region 内部的中间 value 消掉，只把外部输入放进 `fused_elementwise.inputs`，输出 ID 保持为 root 原来的输出 ID。替换后的 region 安排在 root 原来的位置，而不是最早成员的位置；这样即使中间插入了一个会产生 boundary input 的非 pointwise 节点，融合 op 也不会提前引用尚未定义的值。

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

## 5. 图级公共子表达式消除

attention 和 pointwise 融合完成后，`eliminate_common_subexpressions` 按 SSA 顺序遍历剩余的纯 op。每个 op 的结构 key 包含 op 类型、已经重映射的输入 ID、输出 shape/dtype 和递归冻结的 attributes。key 相同的后续 op 会被删除，其输出 ID alias 到第一次出现的输出；随后所有 consumer 和 graph output 都改写到保留的 value ID。

输出类型进入 key 是必要的。例如同一输入分别 cast 到 FP32 和 BF16 时，两个 `cast` 的输入和 attributes 相同，但输出 dtype 不同，不能合并。嵌套在 `fused_elementwise.attrs["body"]` 中的 op 也参与稳定序列化，因此相同 region 可以安全比较。CSE 放在 fusion 之后，避免先合并单个 op 再改变 pointwise region 的边界。

当前 CSE 不会为两个 `fused_elementwise` region 内部的临时 value 重新编号。因此两个 region 只有在内部 SSA value ID 也一致时才会被识别为同一结构；单个 matmul、attention、norm、cast 等独立 op 的重复消除不受这个限制。

## 6. Broadcast 索引

pointwise kernel 使用输出 tensor 的扁平索引 `elem`。每个输入先检查静态 broadcast 规则：

- 输入 rank 不能大于输出 rank
- 从右向左对齐
- 每一维必须等于输出维，或等于 1

对需要 broadcast 的输入，codegen 根据输出 stride 还原坐标，再按输入 stride 计算线性 offset。例如输出 `[2, 3]`、输入 `[1, 3]` 时：

```text
input_offset = (elem % 3) * 1
```

这避免了依赖 MLX 在 kernel 调用前自动 broadcast，也把索引推导留在编译器里。

## 7. Metal pointwise codegen

`AppleMetalBackend._compile_elementwise` 生成 elementwise Metal kernel：

- grid 大小为 `ceil(输出元素数 / 4)`
- 每个 thread 依次处理最多 4 个元素
- threadgroup 最大为 256
- 输入转成 `float` 计算
- 最终结果转回模板类型 `T`
- `mx.fast.metal_kernel` 在首次调用时完成 JIT

精确 GELU 和独立的 `erf` 使用 `_MSL_HEADER` 中的 Abramowitz-Stegun 近似。Apple MSL 没有 `metal::erf`，所以不能直接生成标准库调用。

## 8. Metal softmax reduction

softmax 支持任意单个维度。backend 把 softmax 轴之前和之后的维度分别压平为 `outer` 和 `inner`，把 softmax 轴作为 `columns`：

- 每个 threadgroup 处理一个 softmax slice
- threadgroup 线程数是 `next_power_of_two(columns)`，上限为 256
- 每个线程跨步遍历该行，先在寄存器中得到局部 max
- threadgroup memory 中做树形 max reduction，再用相同结构做 exp-sum reduction
- 使用 max 平移保证数值稳定，最终在 float 中归一化并 cast 回 `T`
- slice 的输入基址由 `outer × columns × inner` 计算，softmax 轴上的元素步长是 `inner`
- `inner == 1` 时直接按连续尾维寻址，省略除法和取模

grid 大小是 `outer * inner * threads`，每个 threadgroup 的大小固定为 `threads`。当 `columns > 256` 时，每个线程处理多个列；当 `columns` 不是 2 的幂时，线程数向上补齐，全局内存访问由循环条件 `col < columns` 限制。FP16/BF16 输入在 float 中完成 reduction，输出保持输入 dtype。

## 9. Metal sum/mean reduction

sum/mean 接受任意非空 reduction 维集合。backend 把未被 reduction 的维度压平为输出元素，把 reduction 维压平为 `columns`：

- 每个输出元素对应一个 threadgroup
- threadgroup 线程数复用 softmax 的选择规则，最多 256 个线程
- 每个线程跨步累加局部结果，再在 threadgroup memory 中做树形加法
- sum 直接写回归约值；mean 在 float 中除以 `columns` 后写回
- reduction 维组成连续 block 时，backend 把输入拆成 `outer × reduction_block × inner`，用紧凑地址映射；尾维还省略除法和取模
- reduction 维交错时，backend 从 `output_index` 生成非 reduction 维的输入基址，从 `col` 生成各 reduction 维坐标，再组合成静态输入偏移
- 只有 `tid == 0` 写一个输出元素，`keepdim` 只影响输出 shape，不影响线性写入位置

输出 dtype 默认与输入一致，也可以通过 PyTorch 的 `dtype=` 指定为 FP16、BF16 或 FP32。kernel 始终用 float 累加，input 和 output 使用不同的模板类型，最后 cast 到请求的输出 dtype。每个输出元素启动一个 threadgroup，因此输出元素很多而 reduction 很小时 dispatch 开销会偏高；交错维的通用索引还会为每次读取增加整数除法和取模。

## 10. Metal LayerNorm 和 RMSNorm lowering

两种归一化都按 normalized shape 的最后一组维度处理。backend 把 normalized shape 之前的维度展平为 `rows`，把 normalized shape 展平为连续的 `columns`：

- 每个 threadgroup 处理一行
- threadgroup 大小与 softmax 使用同一套选择规则，最多 256 个线程
- reduction 和归一化都使用 float，最后 cast 回输入 dtype
- weight 与 normalized shape 完全一致，并且 dtype 与输入相同

LayerNorm 分两遍读取输入：先计算行均值，再基于 `(x - mean)^2` 计算 variance，避免 `E[x^2] - E[x]^2` 的相消误差。最后用 `rsqrt(variance + eps)` 归一化，并可在同一 kernel 中应用 weight/bias。bias 与 weight 可以分别缺省。

RMSNorm 只做一遍平方和 reduction，得到 `mean(x^2)` 后用 `rsqrt(mean(x^2) + eps)` 缩放输入，再应用可选 weight。kernel 不接受 bias，因为当前折叠的 `F.rms_norm` 语义没有 bias。

两者都支持 normalized shape 占满最后一维、多个连续尾维，或整个 tensor。若 normalized shape 不是连续尾维，PyTorch 不会产生对应的归一化 FX 结构。

## 11. Metal matmul/addmm codegen

`codegen_matmul=True` 时，backend 为 rank-2 `matmul` 和 `addmm` 生成独立 Metal kernel：

```text
C[row, col] = sum_k A[row, k] * B[k, col]
```

`addmm` 在同一 kernel 中计算：

```text
beta * bias + alpha * (A @ B)
```

bias 可以是 rank-0、一维或二维 tensor，只要它满足到输出 shape 的静态 broadcast 规则。FP16/BF16 输入用 float 累加，最后 cast 回输出 dtype。

### Shared-memory tiling

每个 threadgroup 计算输出矩阵的一个 `64×64` block，并沿 reduction 维按 `32` 分块：

- 每个 K step 由 block 内线程协作加载 `A[64×32]` 和 `B[32×64]` 到 threadgroup memory
- barrier 后，每个 thread 从共享片段读取数据并用寄存器 microtile 计算多个输出
- reduction 小于 1024 时使用 4×4 microtile，共 256 个 thread；较大的 reduction 使用 8×4 microtile，共 128 个 thread
- 每次 K step 结束时再次 barrier，避免下一轮加载覆盖仍在使用的共享数据
- row/column 能被 `64` 整除时跳过写回 predicate，reduction 能整除 `32` 时跳过 tail load predicate
- 尾部 block 用零填充越界的 A/B 元素，不执行越界读取

该标量路径还没有 vectorized load、SIMD matrix、split-K 或 autotune，在当前 Apple Silicon 测试中比 MLX 慢约 1.8–5.6 倍，因此默认 `codegen_matmul=False`。基准可用 `examples/metal_matmul_benchmark.py` 复现。

### SIMD matrix fast path

FP16/BF16 输入且 output 的行、列都能整除 `64` 时，kernel 改用 8 个 SIMD group：

- 每个 SIMD group 负责 `16×32` 输出，使用 2 个 A fragment 和 4 个 B fragment 覆盖一次 `16×8 × 8×32` MMA
- A/B 仍先协作加载到 threadgroup memory，尾 K 用零填充，避免 `simdgroup_load` 越界
- reduction 能整除 `32` 且输出列能整除 `8` 时，A/B shared tile 按 16-byte 对齐，并用 `vec<T, 8>` 一次搬运 8 个 FP16/BF16 元素；尾 K 继续使用逐元素加载和边界判断
- accumulator 保持 `simdgroup_float8x8`，只在最终写回时转换到 FP16/BF16
- 输出先存入 `threadgroup float` tile，再统一执行 alpha/beta 和 bias broadcast epilogue
- FP32 SIMD matrix 在当前 Apple GPU 上观察到精度降级，因此不启用；带尾行或尾列的低精度输出也回退到标量 tiled 路径

在当前 M5 Pro 上交替启动进程，对 `512×2048 @ 2048×512` 的 FP16 matmul 各运行 5 次、每次取 101 轮中位数：

| `64×64×32` SIMD path | 耗时 |
| --- | ---: |
| scalar load | 0.437 ms |
| `vec<T, 8>` load | 0.344 ms |

向量加载使该 shape 的 kernel 耗时下降约 21%。SIMD 路径仍比 MLX 慢，并且没有使用 Metal 4 MPP、split-K 或 autotune。

## 12. Metal attention codegen

`codegen_attention=True` 时，backend 为常见 attention 生成 FlashAttention 风格的 fused MSL kernel。每个 threadgroup 处理同一 `(batch, query_head)` 下的 8、16 或 24 个 query。短序列使用 32-key tile；FP16/BF16 且 QK、PV 都能使用 SIMD matrix 时，query sequence 达到 768 会依次尝试 `24×64`、`16×64`、`24×32`、`16×32` tile。`24×64` 和 `16×64` 分别使用 768 和 512 个 thread；如果估算 shared memory 超过 31 KiB，则继续回退。shared memory 仍不足时使用 `8×32` tile 和 256 个 thread。

kernel 的执行方式：

- Q tile 和当前 K tile 由 threadgroup 协作加载到 shared memory；FP16/BF16 下，32-key tile 为每个 query fragment 分配 4 个 SIMD group，64-key tile 分配 8 个。`24×64` 正好使用全部 24 个 warp。64-key 路径的 shared tile 按 32-byte 对齐，head dim 都能被 16 整除时用 `vec<T, 16>` 搬运 Q/K/V，否则用 `vec<T, 8>`；短序列的 32-key 路径继续逐元素加载，避免小工作集产生额外开销。FP32 或非 8 对齐的 head dim 回退到 lane 串行点积
- 32-key 路径中，每个 lane 持有一个 score；64-key 路径中，每个 lane 持有相隔 32 的两个 score。`simd_max` 和 `simd_sum` 在 warp 内得到 block max 与 block sum，再与 running max、denominator 合并成 online softmax 状态
- 64-key 路径的 shared-memory workspace 先存放 float QK score，所有 warp 读完 score 后，再把前半段复用于 FP16/BF16 probability，后半段作为 PV 输出 tile。V tile 在 QK 完成后加载，并与 probability 写入共用一次同步
- PV 按 64 个 value 维度分块，再按 query/value 的 `8×8` fragment 分配 SIMD group，计算 `query_tile×64 @ 64×value_chunk` 的 PV tile，并把结果合并回每个 query 的 online accumulator
- key tile 之间只重标定 running sum 和 V accumulator，不物化完整 attention matrix
- query tile 和 key tile 的尾部用 `-INFINITY` score 或零 probability 补齐，最后一个 tile 不越界读写
- GQA/MQA 通过静态的 `kv_head = query_head / group_size` 寻址，不展开 K/V head
- rank-4 additive mask 按 `(batch, head, query, key)` 生成静态 offset，支持 batch/head/query/key 维广播
- causal mask 在 score 写入 softmax 状态前应用

score 和 online-softmax 状态使用 float，低精度 PV 使用 FP16/BF16 概率与 V fragment，最后写回 FP16、BF16 或 FP32。当前 M5 Pro 上交替启动进程，每项各运行 5 次、每次取 21 轮中位数：

| 输入 | `24×64` scalar load | `24×64` vector load | 降幅 |
| --- | ---: | ---: | ---: |
| `1×8×768×128` | 1.086 ms | 0.833 ms | 23.3% |
| `1×8×1024×128` | 1.720 ms | 1.314 ms | 23.6% |
| `1×8×2048×128` | 6.211 ms | 4.612 ms | 25.7% |

数值对比中最大绝对误差为 `9.766e-04`。该 kernel 仍明显慢于 MLX SDPA，且 GPU 时钟波动较大，单次绝对耗时不宜直接横向比较。QK/PV 已使用 SIMD matrix，64-key tile 也已向量化加载，但 tile 仍由 sequence 和静态 shared-memory 预算选择，没有运行时 autotune、split-key 或更细的流水；默认 `codegen_attention=False`。Q/K head dim 或 V head dim 超过 256，或者 attention 的 mask rank 不适合静态 offset 时，backend 不创建 kernel，Runtime 自动回退到 MLX。

## 13. Runtime

`CompiledPlan` 按 SSA 顺序执行 IR：

- 有生成 kernel 的 op：调用对应的 `mx.fast.metal_kernel`
- 默认模式下的 `matmul`、`addmm`：调用 MLX
- attention：优先调用生成的 MSL kernel；未生成或发生 fallback 时调用 `mx.fast.scaled_dot_product_attention`
- layout op：调用 MLX 的 view、broadcast 和 slicing 语义
- cast：调用 MLX `astype`，保留源和目标 dtype、shape
- 未生成 kernel 的 pointwise op：调用 MLX fallback
- softmax：调用生成的单轴 reduction kernel
- sum/mean：调用生成的 static reduction kernel
- layer_norm：调用生成的两遍 mean/variance reduction kernel
- rms_norm：调用生成的平方和 reduction kernel

每个 graph input 在进入计划前会检查 shape 和 dtype。输出统一以 tuple 返回，即使图只有一个输出。

backend 用进程内 LRU cache 复用 `mx.fast.metal_kernel` 对象，key 由 kernel name、input/output 参数名、MSL source 和 header 组成，容量为 256。它只避免同一 Python 进程内重复构造相同 kernel，不是跨进程磁盘缓存；MLX 自身的编译缓存行为不由这里控制。

## 14. 已知边界

- 只支持静态、非空 shape
- 不支持动态 shape、控制流、list/dict 输出和 `ones_like` 等常量生成节点
- pointwise 要求输入输出 dtype 相同
- softmax 支持任意单轴，但不支持一次对多个轴做 softmax
- sum/mean 支持任意非空维度集合，以及 FP16/BF16/FP32 输出 dtype 覆盖
- LayerNorm/RMSNorm 只支持连续尾维 reduction
- LayerNorm 只支持主输出，不支持直接使用 mean/rstd
- RMSNorm 依赖当前 PyTorch FX 分解形态；上游改变分解链后需要同步更新 pattern
- kernel cache 仅限当前进程，没有跨进程磁盘缓存、autotune 或 layout propagation
- matmul 的 SIMD matrix 只覆盖 FP16/BF16 完整输出块；完整 reduction 和 8 对齐输出列使用 8 元素向量加载，尾块与 F32 仍走标量路径，且还没有 Metal 4 MPP、split-K 或 autotune
- 没有反向传播和训练支持
- layout op 已做恒等与逆变换消除，但还没有布局传播
- graph CSE 不会规范化两个 fused region 内部的临时 SSA value ID，因此内部 ID 不同的等价 region 未必合并
- attention 只支持相同 dtype 的 rank-3/4 Q/K/V 和 additive mask；rank-4 支持 GQA/MQA，但 Q head 数必须是 K/V head 数的整数倍
- attention 不支持 dropout 或辅助输出；原生 MSL attention 要求 Q/K 和 V 的 head dim 不超过 256，超出后回退到 MLX SDPA
- 原生 attention 仅在 FP16/BF16 且 head dim 能被 8 整除时使用 SIMD matrix 计算 QK/PV；64-key tile 仅在同样条件成立时使用 8/16 元素向量加载，不满足条件的单个阶段回退到标量路径
- attention tile 只按 sequence 和静态 shared-memory 预算在 `8×32`、`16×32`、`24×32`、`16×64`、`24×64` 间选择，还没有按实际 GPU 和 profile 做运行时 autotune
- NVIDIA Triton/CUDA backend 尚未实现

下一步可以把低精度 matmul 接到 Metal 4 MPP，继续做 split-K 与 shape autotune；attention 已使用 SIMD matrix 计算 QK/PV，后续重点是 tile autotune、split-key 和更细的流水。编译器侧仍可推进布局传播与跨进程 kernel cache。
