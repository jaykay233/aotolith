# 架构

当前实现是一条最小但可运行的 Apple Metal 编译链。目标不是覆盖完整 Transformer，而是先把编译产物送到 Apple GPU，并用测试固定住前端、IR、融合、代码生成和 runtime 的接口。

## 1. FX 捕获

`capture_fx` 调用 `make_fx(fn)(*example_inputs)`。`make_fx` 执行一次函数并记录 ATen 算子，同时为每个节点保留具体的 tensor metadata。

前端当前只接受以下 ATen 目标：

| FX target | IR op |
|---|---|
| `aten.mm.default` | `matmul` |
| `aten.addmm.default` | `addmm` |
| `aten.silu.default` | `silu` |
| `aten.mul.Tensor` | `mul` |
| `aten.add.Tensor` | `add` |

每个 placeholder 变成 `argN`，每个计算节点变成 `vN`。shape 和 dtype 从 `node.meta["val"]` 读取。动态维度和空维度目前直接报错。`addmm` 的 `alpha`/`beta` 和 `add` 的 `alpha` 会写入 IR attribute；未知 kwargs 直接报错。

## 2. SSA IR

IR 定义在 `ir.py`：

- `Value` 有唯一 ID 和静态 `TensorType`
- `Operation` 只产生一个输出，输入是已定义的 value ID
- `Graph.validate()` 检查输入唯一性、use-before-def、重复定义和输出有效性

这是刻意限制的 SSA 形式。后续加入 layout、tile、schedule 时，可以继续扩展 attribute，而不需要改前端的 value 命名规则。

## 3. Pointwise 融合

`fuse_pointwise` 找出图的融合区域：

1. 把 `silu`、`mul`、`add` 标记为 pointwise op。
2. 从 region root 反向收集 producer。
3. producer 的输出必须只被当前 region 使用，否则停止跨越该 producer。
4. region 至少包含两个 op 才替换成 `fused_elementwise`。
5. `fused_elementwise` 的 `attrs["body"]` 保存原始 pointwise op 序列。

region root 是满足以下条件之一的 pointwise op：

- 它的输出是 graph output
- 它有一个非 pointwise consumer

融合把 region 内部的中间 value 消掉，只把外部输入放进 `fused_elementwise.inputs`，输出 ID 保持为 root 原来的输出 ID。这样下游 op 不需要改写。

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

## 4. MSL 生成

`AppleMetalBackend._compile_pointwise` 生成一个 elementwise Metal kernel：

- grid 大小为输出元素数
- threadgroup 最大为 256
- 每个 thread 处理一个输出元素
- 输入先转成 `float` 计算
- 最终结果转回模板类型 `T`
- `mx.fast.metal_kernel` 在首次调用时完成 JIT

当前要求融合 region 的所有输入和输出具有完全相同的 shape 与 dtype。这是为了避免在还没有 broadcast/layout lowering 时生成错误代码。

`matmul` 和 `addmm` 不经过自定义 MSL，而是由 `CompiledPlan` 调用 `mx.matmul`。因此当前示例的端到端路径是“MLX matmul + 自定义 Metal pointwise kernel”，不是全图由本项目的 codegen 执行。

## 5. Runtime

`CompiledPlan` 按 SSA 顺序解释 IR op：

- `matmul`、`addmm`：MLX runtime
- `fused_elementwise`：预编译的 `mx.fast.metal_kernel`
- 未融合的 `silu`、`mul`、`add`：MLX runtime

每个 graph input 在进入计划前会检查 shape 和 dtype。输出以 tuple 返回，即使图只有一个输出。

## 6. 已知边界

- 只支持静态、非空 shape
- FX tensor 输入和输出之外的控制流、常量参数、list/dict 输出未支持
- pointwise 不支持 broadcast 和混合 dtype
- `addmm` 的 `alpha`、`beta` 和 `add` 的 `alpha` 已保留；其他 ATen kwargs 会被拒绝
- 没有 kernel cache、autotune、tiling、layout propagation
- 没有反向传播和训练支持
- Metal kernel 每次执行一个元素，尚未做向量化和 occupancy 调优
- NVIDIA Triton/CUDA backend 尚未实现

这些问题应在增加模型覆盖前逐项解决，尤其是 `addmm` 参数语义、broadcast lowering 和 kernel cache。
