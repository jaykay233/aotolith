# 架构

当前实现是一条可运行的 Apple Metal 模型编译链。项目负责把 Transformer 计算图转换为带 ABI 的执行计划；请求调度、cache 分配、采样策略和 serving 留在外部运行时。静态、多层 decoder-only forward 和固定容量静态 KV Cache 的 prefill/decode 已经有端到端验证。MLX 在这里提供回退算子，也承担编译结果的数值验证执行。

## 0. 目标与边界

项目目标是构建一个 PyTorch Transformer 模型编译器。输入是模型、state 和静态 input spec，输出是版本化、自包含、可重新加载的编译产物。编译产物描述模型计算的 ABI、IR、权重布局、内存计划、backend plan 和 kernel；请求级控制流留在外部推理系统。

编译器负责：

- FX 捕获、参数与 buffer 绑定、模型级输入输出 ABI
- typed SSA IR、shape/dtype/layout/alias 校验和 legalization
- 常量折叠、DCE、CSE、布局传播、算子融合
- liveness、buffer planning 和静态内存复用
- backend capability 查询、kernel selection、Metal codegen、fallback 和 autotune
- manifest、产物版本、序列化/反序列化、跨进程加载和 kernel 重建
- 数值对齐、可复现性和编译产物兼容性验证

目标按三个验收门推进：

1. 模型前端：tiny Llama/Qwen 风格 `nn.Module` 可以直接编译；logits 与 PyTorch 对齐；编译计划不持有或依赖原始 Python 模型对象。
2. 编译产物：manifest、图、ABI、权重布局和 kernel 可以落盘，并在新进程中重新加载；版本不兼容时明确失败；同一输入和 compiler version 产生可复现结果。
3. 编译器质量：可以查看 liveness/buffer plan、kernel 选择和 fallback 原因；原生 kernel 与 fallback 在支持的 dtype 上数值一致（f32 matmul 是唯一例外：原生标量 kernel 与 PyTorch 逐位一致，而这条 GPU 上 MLX 自己的 f32 matmul 只有 bf16 精度，所以两边不相等，见第 14 节）；autotune 保存 profile 依据和最终选择。

请求调度、continuous batching、paged KV allocator、prefix cache、抢占、采样策略、PD 分离和 serving API 由外部推理系统负责。EP/MoE 的 rank 放置、expert dispatch、EPLB、通信调度和负载均衡也不是编译器职责；编译器只表达 MoE 计算和静态布局契约。

KV Cache 在编译器中的语义是图输入、图输出和 buffer layout contract。编译器可以 lower cache 读写和规划静态内存，但不管理 page table、分配、回收或跨请求复用。

## 1. FX 捕获

`capture_fx` 调用 `make_fx(fn)(*example_inputs)`。`make_fx` 执行一次函数并记录 ATen 算子，同时为每个节点保留具体的 tensor metadata。

模型级入口 `compile_module` 先用 `torch.func.functional_call` 把 `named_parameters()` 和 `named_buffers()` 作为显式参数传入 tracing function，再把 state 转为图中的私有输入。返回的 `CompiledModulePlan` 只保留 state 名称和 MLX 副本，不持有原始 module。当前按 parameter/buffer 全名绑定，`remove_duplicate=False`，因此共享 tensor 会作为多个状态项进入图；它们在运行期和 artifact 里共享同一份 buffer 和 payload，见第 15 节。

HuggingFace 模型输出默认选择 `.logits`，也可用 `output_key` 指定 mapping key 或对象属性。tensor-valued `example_kwargs` 会被提升为图输入：它们在 `example_inputs` 之后、state 之前进入图，调用时按名字传递（`plan(input_ids, position_ids=...)`），因此像 `position_ids` 这种调用方会变化的参数不会被固化成常量。其余关键字参数仍是 tracing 期常量；值里嵌套 tensor（list/dict 等）目前明确报错而不是静默当成常量。

两层随机 Llama 的当前回归结果是 220 个 FX 源算子降到 136 个 lowered op，logits 最大绝对误差约 `9e-08`；同一计划在删除原 PyTorch module 后仍可执行。

前端当前处理以下 FX target 和折叠模式：

| FX target | IR op |
|---|---|
| `aten.mm.default` | `matmul` |
| `aten.bmm.default` | `matmul` |
| `aten.addmm.default` | `addmm` |
| `aten.baddbmm.default` | `addmm`，batch 维按 broadcasting 语义执行 |
| `aten.embedding.default` | `embedding` |
| `aten.index_select.default` | `index_select`，沿单个静态轴 gather |
| `aten.index.Tensor` | `index`，支持连续多轴 advanced indexing 和 index broadcasting |
| `aten.cumsum.default` | `cumsum`，沿单个静态轴执行前缀和 |
| `aten.t.default` | `transpose`，rank-2 矩阵转置 |
| `aten.gather.default` | `gather`，逐元素沿单个静态轴 gather |
| `aten.scatter.src`、`aten.scatter.value` | `scatter`，沿单个静态轴写入 tensor 或 scalar |
| `aten.add_.Tensor` | `add`，仅在原值没有其他别名观察者时做 SSA legalization |
| `aten.view.default`、`aten._unsafe_view.default`、`aten.reshape.default` | `reshape` |
| `aten.transpose.int` | `transpose` |
| `aten.permute.default` | `permute` |
| `aten.expand.default` | `expand` |
| `aten.slice.Tensor` | `slice` |
| `aten.select.int` | 在前端展开为单元素 `slice + squeeze` |
| `aten.flip.default` | `flip` |
| `aten.roll.default` | `roll` |
| `aten.triu.default`、`aten.tril.default` | `triu`、`tril`，支持 batch 维和 `diagonal` |
| `aten.unsqueeze.default` | `unsqueeze` |
| `aten.squeeze.dim` | `squeeze` |
| `aten.clone.default`、`aten.alias.default` | `contiguous`、`alias` |
| `aten._to_copy.default` | `cast` |
| `aten._softmax.default`、`aten._safe_softmax.default` | `softmax`、`safe_softmax` |
| `aten.scaled_dot_product_attention.default` 及 CPU/CUDA flash、efficient attention target | `attention` |
| `aten.sum.default`、`aten.sum.dim_IntList` | `sum` |
| `aten.mean.default`、`aten.mean.dim` | `mean` |
| `aten.native_layer_norm.default` | `layer_norm`，仅第 0 个输出 |
| `F.rms_norm` 的分解链 | `rms_norm`，由前端 pattern folding 生成 |
| `aten.silu.default` | `silu` |
| `aten.gelu.default` | `gelu` |
| `aten.softplus.default` | `softplus`，保留 `beta` 和 `threshold` |
| `aten.mish.default` | `mish` |
| `aten.log_sigmoid_forward.default` | `logsigmoid`，仅第 0 个输出 |
| `aten.mul.Tensor`、`aten.mul.Scalar` | `mul` |
| `aten.add.Tensor` | `add` |
| `aten.sub.Tensor` | `sub` |
| `aten.rsub.Scalar` | `rsub` |
| `aten.div.Tensor` | `div` |
| `aten.floor_divide.default`、`aten.floor_divide.Tensor`、`aten.floor_divide.Scalar` | `floor_divide` |
| `aten.remainder.Tensor`、`aten.remainder.Scalar` | `remainder` |
| `aten.pow.Tensor_Tensor`、`aten.pow.Tensor_Scalar`、`aten.pow.Scalar` | `pow` |
| `aten.maximum.default`、`aten.minimum.default` | `maximum`、`minimum` |
| `aten.logaddexp.default` | `logaddexp` |
| `aten.atan2.default` | `atan2` |
| `aten.neg.default` | `neg` |
| `aten.abs.default` | `abs` |
| `aten.sigmoid.default` | `sigmoid` |
| `aten.tanh.default` | `tanh` |
| `aten.relu.default` | `relu` |
| `aten.leaky_relu.default` | `leaky_relu` |
| `aten.elu.default` | `elu` |
| `aten.hardswish.default` | `hardswish` |
| `aten.hardsigmoid.default` | `hardsigmoid` |
| `aten.threshold.default` | `threshold`，保留 `threshold` 和 `value` |
| `aten.exp.default` | `exp` |
| `aten.expm1.default` | `expm1` |
| `aten.exp2.default` | `exp2` |
| `aten.log.default` | `log` |
| `aten.log2.default` | `log2` |
| `aten.log10.default` | `log10` |
| `aten.log1p.default` | `log1p` |
| `aten.erf.default` | `erf` |
| `aten.sin.default` | `sin` |
| `aten.cos.default` | `cos` |
| `aten.floor.default` | `floor` |
| `aten.ceil.default` | `ceil` |
| `aten.round.default` | `round` |
| `aten.trunc.default` | `trunc` |
| `aten.sign.default` | `sign` |
| `aten.square.default` | `square` |
| `aten.sqrt.default` | `sqrt` |
| `aten.rsqrt.default` | `rsqrt` |
| `aten.reciprocal.default` | `reciprocal` |
| `aten.clamp.default`、`aten.clamp.Tensor`、`aten.hardtanh.default` | `clamp` |
| `aten.clamp_min.default`、`aten.clamp_min.Tensor` | `clamp_min` |
| `aten.clamp_max.default`、`aten.clamp_max.Tensor` | `clamp_max` |
| `aten.eq`、`aten.ne`、`aten.lt`、`aten.le`、`aten.gt`、`aten.ge` 的 Tensor/Scalar 变体 | `eq`、`ne`、`lt`、`le`、`gt`、`ge` |
| `aten.logical_and/or/xor/not` | `logical_and`、`logical_or`、`logical_xor`、`logical_not` |
| `aten.bitwise_and/or/xor/not` 的 Tensor/Scalar 变体 | `bitwise_and`、`bitwise_or`、`bitwise_xor`、`bitwise_not`，并按输入 dtype 走 bool 或整数 lowering |
| `aten.isnan.default`、`aten.isinf.default` | `isnan`、`isinf` |
| `aten.where.self` | `where` |
| `aten.masked_fill.Scalar` | `masked_fill`，bool mask 与标量填充值进入 IR |
| `aten.scalar_tensor.default` | `scalar_tensor`，标量值和输出 dtype 进入 IR |
| `aten.arange.default`、`aten.arange.start`、`aten.arange.start_step` | `arange`，参数规范化为 `start/end/step` attributes |
| `aten.zeros.default`、`aten.ones.default`、`aten.full.default`、`aten.new_zeros.default`、`aten.new_ones.default`、`aten.new_full.default`、`aten.zeros_like.default`、`aten.ones_like.default`、`aten.full_like.default` | `zeros`、`ones`、`full`、`new_zeros`、`new_ones`、`new_full`、`zeros_like`、`ones_like`、`full_like`，shape/dtype 来自输出 metadata，填充值进入 IR |
| `aten.cat.default` | `cat`，tensor 输入序列和拼接轴进入 IR |
| `aten.stack.default` | 在前端展开为多个 `unsqueeze` 后接一个 `cat` |
| `aten.split.Tensor`、`aten.split_with_sizes.default` | 在前端展开为多个 `slice` |
| `aten.unbind.int` | 在前端展开为多个 `slice + squeeze` |

每个 placeholder 变成 `argN`，每个计算节点变成 `vN`。shape 和 dtype 从 `node.meta["val"]` 读取，动态维度和空维度会直接报错。IR 的数值 dtype 包括 FP16、BF16 和 FP32，另外支持 `DType.BOOL` 表示谓词、logical/bitwise 结果、bool mask 和 bool 输出；`DType.I32`、`DType.I64` 用于整数常量、整数 pointwise、token id 以及 embedding/index/gather/scatter indices。`bmm` 和 `mm` 统一为 `matmul` IR op，后面的 canonicalization pass 会把 rank-N matmul 被 PyTorch 展开的 `reshape -> bmm -> reshape` 还原成规范形式。Runtime 用同一个 batched matmul 语义执行。

`cat` 保留所有 tensor 输入，并把归一化前的轴写入 `dim` attribute。Backend 要求输入 rank、非拼接轴 shape 和非拼接轴目标 shape 完全一致，且所有输入与输出 dtype 相同；拼接轴长度必须等于各输入该轴长度之和。Runtime 用 `mx.concatenate` 执行，不生成 Metal kernel。

`stack` 没有独立的 IR op。前端先检查所有输入的 shape 和 dtype 一致，再把 `dim` 归一化到输出 rank 范围，生成每个输入一个 `unsqueeze`，最后用一个 `cat` 合并。这样 `stack` 直接复用已有的 layout 校验、CSE 和 MLX runtime，不需要额外的 tuple 或 sequence 输入语义。

`get_attr` 常量和零长度 `cat` 操作数。`use_cache=True` 是 HuggingFace 模型的默认设置，它的 KV cache 从 `torch.tensor([])` 起步，`make_fx` 会把它提升为图里的 `get_attr` 常量，并插入一个 `lift_fresh_copy` 恒等拷贝。IR 的 `TensorType` 要求每一维为正，因此空 tensor 无法成为 plan 输入；但零长度的 `cat` 操作数对结果没有任何贡献，所以前端在拼接处把它折掉：只剩一个操作数时整条 `cat` 直接退化成该操作数，剩两个及以上时按剩余操作数生成 `cat`。`lift_fresh_copy`、`clone`、`detach`、`alias`、`_to_copy`、`contiguous` 这类不携带算术的拷贝，如果输出为空 tensor 也一并跳过。非空的常量 tensor 仍然被拒绝，报错会带上常量名和 shape，并提示 `use_cache=True` 是最常见的来源——这才是真正需要用户改写模型调用的情况。

`embedding` 保留为一个 IR op，输入顺序与 ATen 一致，为 `weight, indices`。前端要求 weight 是 rank-2 的 FP16/BF16/FP32 tensor，indices 是 `int32/int64`，输出 shape 为 `indices.shape + (embedding_dim,)`。`padding_idx` 只影响反向梯度，推理时允许并校验范围；`scale_grad_by_freq` 和 `sparse` 同样只影响反向传播。`max_norm` 会生成 `embedding_renorm_` 节点，当前不进入这个 lowering。Runtime 用 `mx.take(weight, indices, axis=0)` 执行 gather，不生成 Metal kernel。

`index_select` 同样保留为独立 IR op，输入为 `source, indices`，归一化后的轴写入 `dim` attribute。前端要求 source rank 至少为 1，indices rank 为 1 且 dtype 为 `int32/int64`，输出只把 source 的 `dim` 轴替换为 indices 长度，并保持 source dtype。Runtime 映射到 `mx.take(source, indices, axis=dim)`；由于 MLX 会回绕负索引并把越界索引填成零，而 PyTorch 会报错，Runtime 在 gather 前显式检查 `[0, source.shape[dim])`。

`gather` 的输入同样为 `source, indices`，但 indices 必须与 source 同 rank，输出 shape 和 dtype 分别等于 indices shape 和 source dtype。非 gather 轴上，indices 对应维度不能大于 source；gather 轴长度不受此限制，索引值仍须落在 source 该轴范围内。Runtime 先按 indices shape 截取 source 的非 gather 轴，再调用 `mx.take_along_axis`。这样既满足 MLX 对非 gather 轴的严格形状要求，也保留 PyTorch 允许索引子区域的行为。`sparse_grad` 只影响反向传播，推理 lowering 会忽略。

`scatter` 输入为 `source, indices`，`aten.scatter.src` 额外接收 update tensor，`aten.scatter.value` 把 scalar 写入 `value` attribute。source、indices 和 update 必须同 rank；indices 的每个非 scatter 轴不能大于 source；update 的每一维不能小于 indices，与 PyTorch 的 `index.size(d) <= src.size(d)` 约束一致。输出始终保持 source 的 shape 和 dtype。Runtime 检查索引范围，把 update 截到 indices shape，调用 `mx.put_along_axis`，再复制 source 并写回更新区域，因此不会修改输入。重复索引的胜者与 PyTorch 一样不做确定性保证。

`aten.select.int` 没有独立 IR op。前端把负轴和负索引归一化，校验索引范围，再生成只包含目标元素的 `slice` 和删除该轴的 `squeeze`。输入 rank 为 1 时输出 rank-0 scalar；其他情况下输出 shape 等于输入删除目标轴后的 shape。

`split`、`chunk`、`unbind` 在 FX 中产生 tuple 和 `getitem` 节点。前端根据 `make_fx` 给出的静态输出 metadata，把每个被使用的 tuple 元素直接映射成一个单输出 IR value：`split/chunk` 生成带累计 `start/end` 的 `slice`，`unbind` 生成单元素 `slice` 后接 `squeeze`。因此 IR 仍保持单输出 op 约束，后续 layout 化简、CSE 和 pointwise 融合不需要理解 tuple。这里不能用只有 `squeeze` 的表示，因为多个同形 unbind 输出会被 CSE 误认为相同值。

除已折叠到矩阵乘的转置和 batch `expand` 外，Layout op 保留为独立 IR 节点。Backend 在下发前检查 reshape 的元素数、transpose/permute 的轴置换、expand 的 broadcast 合法性、slice 的静态输出 shape、flip 的轴集合、roll 的 shift/dim 对齐、unsqueeze 的插入轴，以及 squeeze 删除的 singleton 轴。它们不生成 Metal kernel，而是由 Runtime 映射到 MLX 的 reshape、swapaxes、transpose、broadcast、切片、flip、roll、expand-dims 和 squeeze。`clone` 使用 `mx.contiguous` 建立独立布局，`alias` 直接复用输入数组；两者保持 dtype、shape 和值不变。

`addmm` 的 `alpha`/`beta`、`add`/`sub` 的 `alpha`、`leaky_relu` 的 `negative_slope`、`elu` 的 `alpha`、`gelu` 的 `approximate`、`threshold` 的 `threshold`/`value`、softmax 的 `dim`，以及 sum/mean 的 `dims`/`keepdim`/`dtype` 会写入 IR attribute；不指定 dimension 的 sum/mean 会在前端规范化为全部轴，`keepdim=False`。`_to_copy` 的 dtype 直接由 FX 节点的输出 metadata 表示，backend 用 MLX `astype` 执行。二元 pointwise 支持两个 tensor，或一个 tensor 和一个 Python scalar；scalar 以及它位于左侧还是右侧也保存在 attribute 中，比较和 bool `bitwise_*` op 还接受 bool scalar。`pow` 覆盖 tensor exponent、tensor 底数配标量指数，以及标量底数配 tensor 指数。`clamp`/`clamp_min`/`clamp_max` 的标量边界保存在 attribute 中，tensor 边界作为可广播的普通输入，并对 NaN 输入或边界保持与 PyTorch 一致的传播语义。`aten.rsub.Scalar` 单独映射为 `rsub`，因为它的 FX 参数顺序是 `(tensor, scalar)`，语义却是 `scalar - tensor`。`masked_fill.Scalar` 保留原 tensor 和 bool mask 两个输入，填充值进入 attribute。`scalar_tensor`、`zeros`、`ones`、`full`、`new_zeros`、`new_ones`、`new_full`、`zeros_like`、`ones_like` 和 `full_like` 都是无 tensor 输入的常量节点：值和输出 dtype 分别来自 IR attribute 与节点 metadata。`new_*` 和 `*_like` 的模板 tensor 只参与 FX 捕获，不进入 op 的运行时输入。浮点/bool 常量可以单独生成 Metal kernel，也可以与相邻 pointwise op 融合；I32/I64 常量走独立的整数 kernel。CSE 的 key 包含输出 shape/dtype，因此同一模板上的不同 dtype 覆盖或不同整数宽度不会被错误合并。`arange` 的三个 ATen target 统一规范化为 `(start, end, step)` attributes，当前只接受整数参数和 I32/I64 输出。

`F.layer_norm` 会被 `make_fx` 展开成 `native_layer_norm` 和三个 `getitem`。当前编译器把第 0 个输出折叠成单输出 `layer_norm` IR op，normalized shape、eps 以及是否带 weight/bias 保存在 attribute 中。如果调用方继续使用 mean 或 rstd，前端会明确拒绝，而不是返回错误语义。

`F.rms_norm` 没有单独的 ATen target，在当前 PyTorch 中会展开成 `pow -> mean -> add_ -> rsqrt -> mul -> optional weight mul`。前端只在中间节点单消费者、reduction 覆盖连续尾维且 shape 吻合时折叠成 `rms_norm`，避免把任意的平方、均值、rsqrt 组合误认为归一化。FP16/BF16 下 `make_fx` 还会插入 float32 `_to_copy`，折叠范围包含这组 cast。

整数 `torch.square` 在 `make_fx` 中会落到 `aten.pow.Tensor_Scalar(x, 2)`，前端将这个特殊形式规范化为 `square` IR op。其他整数 `pow` 仍不在当前支持范围，避免把浮点指数或负数指数的语义差异隐式带入整数路径。

`F.scaled_dot_product_attention` 可能落到通用 ATen op，也可能被拍平成 CPU/CUDA flash 或 efficient attention 的内部 target。这些 target 的参数顺序和输出个数不同，前端按 target 分别读取 `attn_mask`、`dropout_p`、`is_causal`、`scale` 和关键字参数 `enable_gqa`。当前只接受主输出；辅助 logsumexp、debug mask 等输出被使用时直接报错。dropout 非零也会报错。`enable_gqa` 只做布尔校验，实际 head 分组关系由 Q/K/V 的静态 shape 决定；部分 CPU SDPA 路径在 tracing 后会丢掉这个参数并落到 flash target。

FP16/BF16 的 CPU SDPA 分解图常把 Q/K/V 转成 FP32 计算，再把结果转回输入 dtype。frontend 将这些节点保留为 `cast`。attention fusion 只在输入 cast 来自同一种低精度 dtype、输出 cast 又回到同一 dtype 时吸收整条内部 round-trip；显式 `to(bfloat16)` 或 `to(float32)` 等用户 cast 仍作为独立节点执行，不能被错误地改写成另一种 attention dtype。

## 2. SSA IR

IR 定义在 `ir.py`：

- `Value` 有唯一 ID 和静态 `TensorType`
- `Operation` 只产生一个输出，输入是已定义的 value ID
- `Graph.validate()` 检查输入唯一性、use-before-def、重复定义、输出类型是否存在

这是限制过的 SSA 形式。后续加入 layout、tile、schedule 时，可以继续扩展 operation attribute，而不需要改前端的 value 命名规则。

## 3. Layout 简化与折叠、matmul 规范化与 attention 融合

`simplify_layouts` 先删除不改变数据的 layout op，包括 shape 相同的 `reshape/expand/squeeze`、交换同一轴的 `transpose`、恒等 `permute`、完整 slice、只翻转 singleton 轴的 `flip`、所有 shift 对对应维取模后为零的 `roll`、`alias` 和 `contiguous`。连续且互逆的 `transpose/permute`、重复的 `flip` 以及 shift 相消的 `roll` 会折叠掉，连续的 `unsqueeze/squeeze` 在恢复原 shape 时也会被消除，相邻 `reshape` 会合并成从最早输入到最终 shape 的一次 reshape。删除前仍会解析轴和检查静态 shape，非法轴不会被当成恒等操作跳过。

`canonicalize_batched_matmul` 识别 PyTorch 为 rank-N matmul 生成的折叠模式：输入先 reshape 成二维 batch，执行 `bmm`，输出再 reshape 回原始 batch shape。匹配成功后替换成一个 `matmul`，避免后续 attention pattern 只看到低秩 `bmm`。

`fold_matmul_input_layouts` 处理单个 consumer 的最后一维转置和 batch `expand`。若 `matmul/addmm` 的操作数来自 `transpose` 或交换最后两维的 `permute`，pass 会把该 op 收进矩阵乘的 `transpose_a`/`transpose_b` attribute。若操作数来自只改变 batch 前缀的 `expand`，且折叠后 matmul 仍推导出原输出 shape，pass 会直接改用 `expand` 的输入，让 backend 在取数时计算 broadcast offset。候选改写会先做完整的 shape 推导；像 `(1, 2, M, K)` 和 `(1, 2, K, N)` 都显式展开到 `(3, 2, M, N)` 时，两个 `expand` 都必须是保留节点，否则 matmul 只能推导出 `(1, 2, M, N)`。只要 layout 结果还被 graph output 或其他节点使用，pass 同样保留原节点。该 pass 在 `fuse_attention` 之后运行，不改变 attention pattern 对 `K^T` 的匹配。

`fuse_attention` 从 PV matmul 反向匹配 `QK^T -> optional scale -> optional additive mask -> softmax -> PV`。它也能恢复 PyTorch 把 batch-head 拍平后执行的 `reshape -> mm -> reshape`。只有内部 value 都只有单个 consumer，且没有被 graph output 观察时才会替换成 `attention`。`scale`、`has_mask`、`causal` 和 `safe_softmax` 存在 operation attribute 中，Q/K/V 保持原 graph 中的 rank-3 或 rank-4 形式。softmax probability 被其他节点使用时不会融合，因为替换会丢掉该中间结果。

GQA/MQA 有两种入口。直接调用 `F.scaled_dot_product_attention(..., enable_gqa=True)` 时，前端直接生成 `attention`；显式 `repeat_kv` 的分解图则通过 `unsqueeze -> expand -> reshape` 识别出原始 K/V。后一种模式会去掉物化复制，把 K/V head 数低于 Q head 数的 `attention` 交给 MLX。rank-4 下要求 Q head 数是 K/V head 数的整数倍；MQA 是 K/V head 数为 1 的特例。

Apple backend 默认让 `attention` 调用 `mx.fast.scaled_dot_product_attention`，GQA/MQA 同样复用这个 MLX 实现。rank-3 输入在调用前临时增加 head 轴，调用后 squeeze 回去。显式 mask 可以是与 Q/K/V 同 dtype 的 additive mask，也可以是 bool mask；bool mask 中 `True` 表示允许该 key 参与 attention。当 `safe_softmax=True` 且存在显式 mask 时，MLX lowering 会把整行都被 mask 的输出显式置零：additive mask 判断整行是否为 `-inf`，bool mask 判断整行是否为 `False`，避免 fused SDPA 产生 NaN 或错误结果。此时 `CompiledPlan.kernels` 只记录项目生成的 Metal kernel，不记录 MLX 内置的 attention 实现。开启 `codegen_attention=True` 后，满足条件的 additive-mask 或 bool-mask attention 会改为执行下面描述的生成 kernel；不满足条件的形状仍回退到 MLX。MLX fused SDPA、生成 kernel 与逐算子 `matmul/softmax` 存在浮点路径差异，测试按 dtype 使用 `2e-3` 到 `2e-2` 的容差。

## 4. Pointwise 融合

`fuse_pointwise` 找出图的融合区域：

1. 把支持的一元和二元运算标记为 pointwise op；含 I32/I64 输入或输出的 op 不参与融合。
2. 找出 region root：输出被 graph output 或非 pointwise consumer 观察的 pointwise op。
3. 从每个 root 开始迭代式反向闭包，只有 producer 没有被 graph output 观察、且它的全部 consumer 都已经在 region 内时才纳入 producer。
4. region 至少包含两个 op 才替换成 `fused_elementwise`。
5. `fused_elementwise` 的 `attrs["body"]` 保存原始 pointwise op 序列。

只有单个 pointwise op 时不做 IR fusion，但 Apple backend 仍会为它生成一个独立 Metal kernel。当前 pointwise emitter 使用 float 中间值，因此整数 op 被排除在融合之外，避免超过 `2**53` 的 I64 值发生静默舍入；普通整数算术目前仍不支持。

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

输出类型进入 key 是必要的。例如同一输入分别 cast 到 FP32 和 BF16 时，两个 `cast` 的输入和 attributes 相同，但输出 dtype 不同，不能合并。CSE 放在 fusion 之后，避免先合并单个 op 再改变 pointwise region 的边界。

`fused_elementwise` 的 key 不只递归冻结 body：它把 boundary input 映射为参数序号，把 body 中每个临时输出按 SSA 顺序重新编号，并记录每个中间值的类型和 attributes。这样，即使两个 region 使用了不同的 value ID，只要数据流和算子属性等价，仍然会被合并。CSE 还会在删除上游重复 op 后改写 region body 的 boundary 引用，避免保留的 region 指向已从执行序列中删除的 value。

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
- 输入和中间值按 IR dtype 声明为 `float`、`bool`、`int` 或 `long`；I32/I64 路径不会把大整数转换成 float
- 比较必须产生 bool；logical op 只接受 bool 输入并产生 bool；bool tensor 的 bitwise op 也走 bool lowering；`where` 的第一个输入必须是 bool condition
- 普通数值 op 不接受 bool operand；数值分支与 bool condition 可以处于同一个 DAG region
- 中间临时值和最终结果的 MSL 类型由各自 IR 输出的 dtype 决定，最终写回模板类型 `T`
- `mx.fast.metal_kernel` 在首次调用时完成 JIT

混合 dtype 融合让 `(x > y) & mask -> where(condition, x + y, x - y)` 仍只生成一个 kernel。bool 输出使用 `mx.bool_` 分配，FP16/BF16/FP32 数值路径仍以 float 计算并 cast 回目标 dtype。

精确 GELU 和独立的 `erf` 使用 `_MSL_HEADER` 中的 Abramowitz-Stegun 近似。Apple MSL 没有 `metal::erf`，所以不能直接生成标准库调用。

`round` 生成 `metal::rint`，与 PyTorch 一样采用 ties-to-even。Apple MSL 同样没有可用的 `expm1`，因此 `_MSL_HEADER` 提供 `tsc_expm1`：|x| <= 0.5 时使用七阶 Taylor 展开，其他范围回退到 `exp(x) - 1`。`log1p` 使用 MSL 的全局函数，不在 `metal` 命名空间中。

`softplus` 使用 threshold 分段公式：`beta * x > threshold` 时直接返回 `x`，否则计算 `log1p(exp(beta * x)) / beta`。`logsigmoid` 使用 `min(x, 0) - log1p(exp(-abs(x)))`，避免正输入时直接计算 `log(sigmoid(x))` 溢出。`mish` 计算 `x * tanh(softplus(x, 1, 20))`，在 `x > 20` 时直接返回 `x`。`hardsigmoid` 计算 `min(max(x + 3, 0), 6) / 6`；`threshold` 仅在 `x > threshold` 时保留输入，否则返回 `value`，两者都显式传播 NaN。这些路径都复用 `_MSL_HEADER` 中的 helper，并能与其他 pointwise op 融合。

`maximum`/`minimum` 在比较相等时选择左操作数，以匹配 PyTorch 对 signed zero 的行为，并显式传播任一操作数的 NaN。`logaddexp` 使用 `max(a, b) + log1p(exp(-abs(a - b)))`，避免两个同号大数直接相加后溢出；任一输入为 NaN 时返回 NaN，最大值为正无穷时返回正无穷，两个 `-inf` 返回 `-inf`。`exp2` 直接生成 `metal::exp2`。

Apple Metal 的 `atan2` 在 signed zero 和双无穷象限上存在平台差异，因此 `tsc_atan2` 显式处理 NaN、`x == ±0`、`y == ±0`、`x`/`y` 为无穷以及 `(-inf, -inf)` 等象限边界，其余有限值回退到系统 `atan2`。这些规则与 `torch.atan2` 的参数顺序一致。

整数常量和 `arange` 生成无 tensor 输入的 Metal kernel，直接把 I64 字面量写入 `long`，再 cast 到模板类型 `T`。`arange` 使用 `long` 计算 `start + elem * step`；参数和输出长度在编译期校验，因此 `2**53` 以上的值仍能精确生成。`-2**63` 用 `(-9223372036854775807LL - 1LL)` 表示，避免整数类型的正数上限无法容纳该字面量。当前不 lowering 浮点 `arange`，因为 PyTorch 使用 double 语义，而 Metal 没有稳定可用的 GPU double 路径。

I32/I64 pointwise 走同一融合框架，但中间变量使用 `int`/`long`，输入和输出保持原 dtype。tensor 与 Python 整数标量混合时，前端保留标量原始值，backend 再按 tensor 宽度取模并解释为二补码有符号数；因此 I32 的 `2**32 - 1` 等价于 `-1`，I64 的 `2**64 - 1` 也等价于 `-1`，不会先转成 float。加法、减法、乘法和一元负号按二补码回绕。`floor_divide` 先做 C 截断除法，再在余数非零且符号不一致时向负无穷调整；`remainder` 的结果跟除数同号。除数为 `-1` 时单独处理，避免 `INT_MIN / -1` 触发未定义行为，并按 PyTorch 回绕到 `INT_MIN`。

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

`codegen_matmul=True` 时，backend 为 `matmul` 和 `addmm` 生成独立 Metal kernel。rank-2 使用单个矩阵，rank-3/4 使用相同的最后两维作为矩阵维度，并把前面的维度展平为 batch：

```text
C[batch, row, col] = sum_k A[batch, row, k] * B[batch, k, col]
```

`addmm` 在同一 kernel 中计算：

```text
beta * bias + alpha * (A @ B)
```

bias 可以是 rank-0、一维或二维 tensor，只要它满足到输出 shape 的静态 broadcast 规则。FP16/BF16 输入用 float 累加，最后 cast 回输出 dtype。

### Batched layout

rank-N matmul 按 NumPy/PyTorch 的右对齐规则广播 `A` 和 `B` 的 batch 前缀，输出 batch 展平后映射到 `threadgroup_position_in_grid.y`。grid 的 x 维仍然负责输出 tile，因此同一个 kernel 可以并发处理多个 batch。

输出 batch 是连续扁平索引。每个输入按自己的 batch shape 和 stride 还原对应的 batch 坐标，再生成矩阵基址偏移。例如 `A: (2, 1, M, K)`、`B: (1, 3, K, N)` 产生 `(2, 3, M, N)` 输出时，A 的偏移包含 `((batch / 3) % 2) * M * K`，B 的偏移包含 `(batch % 3) * K * N`，C 使用 `batch * M * N`。这些表达式在 codegen 时静态生成，不需要在 kernel 里查 shape。如果 `transpose_a` 或 `transpose_b` 已折叠进 matmul，先按逻辑转置调整矩阵索引，再加对应的 broadcast offset。

单消费者 batch `expand` 只有在折叠后 matmul 的输出 shape 仍与原图一致时才会删除；否则保留为 Runtime 的 MLX broadcast。当前支持标准右对齐 batch broadcast，不支持需要任意轴重排的非标准广播。

### Shared-memory tiling

每个 threadgroup 计算输出矩阵的一个 `64×64` block，并沿 reduction 维按 `32` 分块：

- 每个 K step 由 block 内线程协作加载 `A[64×32]` 和 `B[32×64]` 到 threadgroup memory
- barrier 后，每个 thread 从共享片段读取数据并用寄存器 microtile 计算多个输出
- reduction 小于 1024 时使用 4×4 microtile，共 256 个 thread；较大的 reduction 使用 8×4 microtile，共 128 个 thread
- 每次 K step 结束时再次 barrier，避免下一轮加载覆盖仍在使用的共享数据
- 每个 threadgroup 的 y 坐标选择输出 batch；A/B 加静态 broadcast offset，C 加连续 batch stride
- row/column 能被 `64` 整除时跳过写回 predicate，reduction 能整除 `32` 时跳过 tail load predicate
- 尾部 block 用零填充越界的 A/B 元素，不执行越界读取

该标量路径还没有 vectorized load、SIMD matrix、split-K 或 autotune，在当前 Apple Silicon 测试中比 MLX 慢约 1.8–5.6 倍，因此默认 `codegen_matmul=False`。基准可用 `examples/metal_matmul_benchmark.py` 复现。

### SIMD matrix fast path

FP16/BF16 输入且 output 的行、列都能整除 `64` 时，kernel 改用 8 个 SIMD group：

- 每个 SIMD group 负责 `16×32` 输出，使用 2 个 A fragment 和 4 个 B fragment 覆盖一次 `16×8 × 8×32` MMA
- A/B 仍先协作加载到 threadgroup memory，尾 K 用零填充，避免 `simdgroup_load` 越界
- reduction 能整除 `32` 且输出列能整除 `8` 时，A/B shared tile 按 16-byte 对齐，并用 `vec<T, 8>` 一次搬运 8 个 FP16/BF16 元素；batched kernel 会在向量地址中加入 A/B 各自的 broadcast offset，尾 K 继续使用逐元素加载和边界判断
- accumulator 保持 `simdgroup_float8x8`，只在最终写回时转换到 FP16/BF16
- 输出先存入 `threadgroup float` tile，再统一执行 alpha/beta 和 bias broadcast epilogue
- FP32 SIMD matrix 在当前 Apple GPU 上观察到精度降级，因此不启用；带尾行或尾列的低精度输出也回退到标量 tiled 路径

在当前 M5 Pro 上交替启动进程，对 `512×2048 @ 2048×512` 的 FP16 matmul 各运行 5 次、每次取 101 轮中位数：

| `64×64×32` SIMD path | 耗时 |
| --- | ---: |
| scalar load | 0.437 ms |
| `vec<T, 8>` load | 0.344 ms |

向量加载使该 shape 的 kernel 耗时下降约 21%。这条 SIMD 路径本身没有 tile autotune，也没有 split-K；更快的 MPP 路径见下一节。

### Tensor-core fast path（MPP）

`matmul`、`addmm` 的 GEMM 和 prefill attention 各有一条 `mpp::tensor_ops::matmul2d` 路径，用 Apple 的 tensor core 取代 `simdgroup_float8x8`：

- GEMM 的 block tile 按输出 shape 和右操作数布局从测量得到的 tile 阶梯里选：右操作数转置时 K 连续，偏好宽的 column block；右操作数 row-major 时沿 K 跨步，偏好更高的 reduction step。warp 内用三个 cooperative tensor 分别承接左右操作数和 FP32 累加器，epilogue（bias、alpha/beta、`silu_mul`）在写回时执行
- 这条路径只服务 FP16/BF16。硬件没有 float32 tensor core，`T = float` 的 `matmul2d` 实际按 bfloat16 精度执行：`512x2048x512` 的 f32 GEMM 因此偏离 PyTorch 1.6e-1，与 MLX 自己 f32 matmul 的偏差相同，而标量 tiled kernel 与 PyTorch 逐位一致（偏差 0）。所以 `_supports_mpp_matmul` 排除 F32，f32 留在标量路径上，代价是该 shape 比 MLX 慢 3.1x
- attention 用一个 simdgroup 负责 `16` 行 query fragment，`Q` 只暂存一次到 threadgroup memory，`K`/`V` 直接从 device memory 读进 cooperative tensor，online softmax 的 `max`/`sum` 仍在寄存器里用 `simd_shuffle` 归约
- 两条路径都以编译器自己的 `__HAVE_TENSOR__` 宏为前提（Apple GPU family 9 及以上）。更早的机型走上一节的 SIMD 路径
- 这个宏只回答「GPU 有没有 tensor op」，不回答「这个 metallib 装得上」。半虚拟化 GPU（托管 CI、云 Mac）会定义宏，却以 `unsupported deferred-static-alloca-size function body` 拒绝加载。因此 `compile` 在选路之前会真正加载并求值一个已经生成好的 tensor-core kernel；失败就把整个 graph 重新 lower 一次，改走 SIMD 路径，并把原因写进 `selections` 的 `reason`。代价是每次编译多一次 launch（kernel 已缓存时约 0.25 ms），换来的是首次 matmul 不会抛出一句带 mangled 符号的 `RuntimeError`



单 token 的 `matmul` 走 warp-per-column 的 GEMV：一个 warp 负责一个输出列，沿 reduction 维读取该列对应的一行权重。gated MLP 的 `gate` 和 `up` 共享同一个 normalized 输入，于是 `fuse_gate_up_projection` 把它们合成一个 `matmul`，并标记 `epilogue="silu_mul_paired"`。

kernel 在同一个循环里读 gate 行和 up 行（`in_1`、`in_2`），各自累加 4 个 float，最后写出 `silu(gate) * up`。两次乘加共用同一组已归一化的 `x0..x3`，所以第二个权重的带宽开销只是多读 8.72 MB，而 launch 从两次减到一次。gate 在 `silu` 前先 cast 回存储 dtype，得到的正是独立投影原本会写下的那个值，因此结果与未融合版本逐位一致。保留 FP32 累加精度反而会改变结果：在 24 token 的 greedy 运行中，第 4 个 token 就因为这点差异翻转了。

该 epilogue 只由 transposed GEMV 实现。`fuse_gate_up_projection` 只在左侧操作数只有一行、权重物理布局为 `[out, in]`（`transpose_b=True`）、且两个投影的 weight shape/dtype 完全一致时才改写；taller 的操作数走 tiled kernel，两个投影保持独立，`silu_mul` 仍由原有的单向量 epilogue 处理。MLX fallback 用两次 `mx.matmul` 加一次逐元素表达式实现同一语义。

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
- rank-4 additive/bool mask 按 `(batch, head, query, key)` 生成静态 offset，支持 batch/head/query/key 维广播；bool mask 的 `True` 保留对应 score，`False` 将其置为 `-inf`
- causal mask 在 score 写入 softmax 状态前应用

score 和 online-softmax 状态使用 float，低精度 PV 使用 FP16/BF16 概率与 V fragment，最后写回 FP16、BF16 或 FP32。当前 M5 Pro 上交替启动进程，每项各运行 5 次、每次取 21 轮中位数：

| 输入 | `24×64` scalar load | `24×64` vector load | 降幅 |
| --- | ---: | ---: | ---: |
| `1×8×768×128` | 1.086 ms | 0.833 ms | 23.3% |
| `1×8×1024×128` | 1.720 ms | 1.314 ms | 23.6% |
| `1×8×2048×128` | 6.211 ms | 4.612 ms | 25.7% |

数值对比中最大绝对误差为 `9.766e-04`。该 kernel 仍明显慢于 MLX SDPA，且 GPU 时钟波动较大，单次绝对耗时不宜直接横向比较。QK/PV 已使用 SIMD matrix，64-key tile 也已向量化加载，但 tile 仍由 sequence 和静态 shared-memory 预算选择，没有运行时 autotune、split-key 或更细的流水；默认 `codegen_attention=False`。Q/K head dim 或 V head dim 超过 256，或者 attention 的 mask rank 不适合静态 offset 时，backend 不创建 kernel，Runtime 自动回退到 MLX。

## 13. CompiledPlan 执行器

本节中的 Runtime 只指编译结果的最小执行器。它按照 `CompiledPlan` 调用生成的 Metal kernel 或 MLX 回退算子，不实现请求调度、batching、采样或 serving。

`CompiledPlan` 按 SSA 顺序执行 IR：

- 有生成 kernel 的 op：调用对应的 `mx.fast.metal_kernel`
- 默认模式下的 `matmul`、`addmm`：调用 MLX，并按 `transpose_a`/`transpose_b` 执行对应的 swapaxes
- attention：优先调用生成的 MSL kernel；未生成或发生 fallback 时调用 `mx.fast.scaled_dot_product_attention`，并对安全 softmax 的整行 mask 补零
- embedding/index_select：调用带范围检查的 MLX `mx.take`，输出保持 source dtype
- gather：检查索引范围，截取非 gather 轴后调用 MLX `mx.take_along_axis`
- scatter：检查索引范围，把 update 截到 indices shape，复制 source 后用 `mx.put_along_axis` 更新
- `cat`：校验后的 tensor 输入交给 `mx.concatenate`
- layout op：调用 MLX 的 view、broadcast 和 slicing 语义
- cast：调用 MLX `astype`，保留源和目标 dtype、shape
- 未生成 kernel 的 pointwise op：调用 MLX fallback
- softmax：调用生成的单轴 reduction kernel
- sum/mean：调用生成的 static reduction kernel
- layer_norm：调用生成的两遍 mean/variance reduction kernel
- rms_norm：调用生成的平方和 reduction kernel

每个 graph input 在进入计划前会检查 shape 和 dtype。输出统一以 tuple 返回，即使图只有一个输出。

backend 用进程内 LRU cache 复用 `mx.fast.metal_kernel` 对象，key 由 kernel name、input/output 参数名、MSL source 和 header 组成，容量为 256。该 cache 只避免同一 Python 进程内重复构造相同 kernel，不提供跨进程磁盘缓存。MLX 自身的编译缓存行为不由这里控制。

### 静态 KV Cache 生成

`examples/tiny_transformer_generate.py` 把 prefill 和 decode 编译成两个静态计划，并由 Python host loop 串起 greedy 生成。这个例子用于验证 KV Cache 的图 ABI；生成循环仍由示例代码提供。batch 和 cache shape 固定，但每个 batch row 可以通过 mask 和 position 使用不同的有效长度。cache 布局是 `[batch, kv_heads, capacity, head_dim]`，每层分别返回 K 和 V。

- prefill 图接收左侧 padding 的 prompt token、RoPE position、逐请求 bool mask，以及模型权重；它把 padded prompt 的 K/V 右对齐写入固定容量 cache。对应的 mask 为 `[B, 1, prompt_len, capacity]`，padding query 被整行 mask，每个有效 query 只能看到本请求有效区域中的 causal prefix。
- decode 图接收每行一个 token、各自当前位置、`[B, 1, 1, capacity]` mask、每层旧 cache 和权重；它沿 sequence 维把 cache 左移一格，在末尾拼接当前 token 的 K/V，再返回 logits 和全部更新后的 cache。
- RoPE position 通过 `F.embedding` 从静态 cos/sin 表选行。这个操作已经能被现有 embedding lowering 执行，不需要新增 cache 专用算子。
- 测试覆盖 `B=1` 和左侧 padding 的 `B=2`，第二个请求的有效 prompt 更短。连续三步 decode 会逐请求对照完整序列 forward 的 logits 和贪心 token，验证 padding、mask、position 和 cache 更新没有改变模型语义。

这个实现优先保证布局容易验证。每一步都会复制并重排完整固定容量 cache，attention 也读取包括 padding 在内的整个 capacity；block table、slot mapping、paged attention、prefix sharing 和 cache eviction 属于外部推理运行时，不在编译器边界内。

## 14. 编译产物、校验与加载

`CompilationResult.save()`、`CompiledModulePlan.save()` 或 `save_compiled()` 写出 `.aoto` archive。当前 archive 使用 format version 4、ABI version 1.3，包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | format/ABI/compiler version、target、entrypoint、compile options、device profile、payload 路径和 digest |
| `input_graph.json` | FX 捕获后、优化前的 typed SSA IR |
| `lowered_graph.json` | legalization、layout folding、fusion、CSE 之后的 IR |
| `kernels.json` | 生成 kernel 的名称、MSL source/header、输入输出和 tensor type |
| `selections.json` | 每个 lowered op 的实现、fallback 原因、autotune benchmark samples 和最终选择 |
| `storage_plan.json` | live interval、slot、release point、峰值活跃字节和 arena 大小 |
| `state/*.bin` | module 参数和 buffer 的 MLX 原始字节，每个 state entry 记录名称、shape、dtype、nbytes 和 digest |

ABI 1.3 增加了 `entrypoint.kwargs`，记录被提升为图输入的 tensor 关键字参数。图输入顺序是 public inputs、这些关键字输入、state；`entrypoint.input_count` 仍然只数 public inputs，因此读取 1.2 产物（没有该字段）时布局与旧版一致。

两张图、kernel table、selection table、storage plan 和每个 state tensor 都使用 SHA-256 digest。`read_artifact_manifest()` 只校验 manifest 中的格式、ABI、target 和可选 device profile；`load_compiled_module()` 还会读取并校验全部 payload，反序列化 IR，检查入口的 public input、关键字输入和 state 布局，再根据 lowered graph 和已保存的 selection 重建 kernel。

加载时使用 `autotune=False` 和 `selection_overrides`。这样可以验证序列化 selection 仍与当前 backend 的可重建 plan 一致，同时不重新执行 benchmark。校验内容包括 kernel artifact、selection key、storage plan、state shape/dtype 和 graph inputs/outputs；任一不匹配都会抛出 `ArtifactCompatibilityError`。

默认允许 artifact 在其他 Apple Silicon GPU 上加载，因为 Metal kernel 会从受校验的图和 MSL source 重新构建。`require_device_match=True` 会同时比较编译时和当前设备的 `device_name`、`architecture`，用于必须绑定特定 GPU 型号的场景。当前 target 只有 `apple-metal`。

CLI 提供四个命令：`compile` 生成 artifact，`inspect` 读取 manifest，`verify` 完整重建并校验 artifact，`run` 加载 artifact 并写出 JSON 结果。Python API 与 CLI 使用同一个 loader。

## 15. 已知边界

- 只支持静态、非空 shape
- 不支持动态 shape、控制流和 list/dict 输出；常量生成支持 `scalar_tensor`、`zeros`、`ones`、`full`、`new_zeros`、`new_ones`、`new_full`、`zeros_like`、`ones_like` 和 `full_like` 的 float/bool/I32/I64 输出
- `arange` 目前只支持整数 start/end/step 和 I32/I64 输出；浮点 `arange` 尚未实现
- bool tensor 支持比较、logical、bitwise、`isnan`/`isinf`、bool mask、`where` 和 `masked_fill`；I32/I64 tensor 除作为 embedding/index_select/gather/scatter indices 外，也支持上述整数 pointwise 集合
- 整数 pointwise 当前要求输入、tensor 边界和输出同宽；I32 与 I64 混合、除 `torch.square` 特殊形式外的整数 `pow`、true division 和不在支持集合内的算子会明确报错。`floor_divide`/`remainder` 不生成逐元素除零检查，调用方必须保证除数非零
- scatter 只支持默认写入语义，不支持 `reduce`、`scatter_add` 或 `scatter_reduce`
- softmax 支持任意单轴，但不支持一次对多个轴做 softmax
- sum/mean 支持任意非空维度集合，以及 FP16/BF16/FP32 输出 dtype 覆盖
- LayerNorm/RMSNorm 只支持连续尾维 reduction
- LayerNorm 只支持主输出，不支持直接使用 mean/rstd
- RMSNorm 依赖当前 PyTorch FX 分解形态；上游改变分解链后需要同步更新 pattern
- `mx.fast.metal_kernel` 对象只在当前进程内做 LRU cache；跨进程时会从 artifact 中的 MSL source 重建，MLX 自身的编译缓存不在本项目的 ABI 内
- matmul codegen 支持 rank-2 和 batch 前缀可广播的 rank-3/4；SIMD matrix 只覆盖 FP16/BF16 完整输出块，完整 reduction 和 8 对齐输出列使用 8 元素向量加载，尾块与 F32 仍走标量路径。MPP 路径需要 `__HAVE_TENSOR__`，并且编译期会实际加载一个 kernel 来确认它装得上；两条路径都还没有 split-K
- 没有反向传播和训练支持
- 静态 KV Cache 只有固定 batch、固定容量 prefill/decode 示例：每步复制整段 cache，attention 读取 padded capacity；有效长度只能通过 static-shape mask 和 position 表达。分页 cache、prefix sharing、请求动态进出 batch、采样 API 和 serving 属于外部推理运行时，不是本项目的待实现功能
- Qwen2 静态 cache 适配层在构造时校验它所依赖的语义：MLP 激活必须是 SiLU（在构造好的 `mlp.act_fn` 上做数值比较，而不是读 `hidden_act`，因为库会把它归一化甚至替换掉），且不能有 sliding-window 层（读 `model.model.has_sliding_layers`，旧版本回退到 `config.layer_types`）。不满足时抛 `NotImplementedError`。这两种情况此前会静默产出错误数值：融合后的 MLP 硬编码了 `silu(gate) * up`，mask 则由适配层自己构造为普通 causal mask
- layout 优化已覆盖恒等、逆变换、矩阵乘输入转置、`flip/roll` 相消和可证明的单消费者 batch `expand` 折叠；跨 reshape 以及其他非 matmul 边界的一般布局传播尚未实现
- attention 只支持相同 dtype 的 rank-3/4 Q/K/V，以及同 dtype additive mask 或 bool mask；rank-4 支持 GQA/MQA，但 Q head 数必须是 K/V head 数的整数倍
- attention 不支持 dropout 或辅助输出；原生 MSL attention 要求 Q/K 和 V 的 head dim 不超过 256，超出后回退到 MLX SDPA
- 原生 attention 仅在 FP16/BF16 且 head dim 能被 8 整除时使用 SIMD matrix 计算 QK/PV；64-key tile 仅在同样条件成立时使用 8/16 元素向量加载，不满足条件的单个阶段回退到标量路径
- attention tile 只按 sequence 和静态 shared-memory 预算在 `8×32`、`16×32`、`24×32`、`16×64`、`24×64` 间选择；autotune 当前比较 op 级别的 Metal/MLX 实现，还没有搜索 tile 参数
- 共享 parameter storage alias 会在运行时和 artifact 两处共享：`compile_module` 对同一个 tensor 只转换一次，别名共享一个 MLX buffer；`save_compiled` 让别名指向同一个 `state/*.bin` 和 digest，`load_compiled_module` 按文件名缓存，恢复同样的共享关系。图里仍然每个名字一个输入，因为 tracing 就是按位置产生 placeholders 的。视图（共享 storage 但 strides 不同，如 `t` 与 `t.t()`）不合并
- 当前 artifact target 只有 `apple-metal`，没有跨厂商 target 或 host runtime ABI
- NVIDIA Triton/CUDA backend 尚未实现

## 16. 编译器路线

模型前端、静态内存复用、kernel selection、autotune 和 artifact 加载链已经打通。后续工作集中在 ABI 完整性和编译器质量：

1. 增加符号 shape 和有限控制流，同时保留静态 shape 路径的严格 verifier。
2. 把 op 级 autotune 扩展成 shape-aware backend policy，生成可复现的 benchmark report。
3. 改进原生 matmul、attention 的 tile 和流水，扩大真实模型、dtype、算子与 Apple GPU 覆盖。
4. 在核心 ABI 稳定后增加其他 backend；当前不为跨厂商兼容性做推测性抽象。
