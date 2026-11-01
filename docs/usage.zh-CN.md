# 使用说明

README 只放两个最关键的调用，这一页涵盖其余接口：tensor 关键字参数、尺寸桶、`aoto` 命令、编译产物和越界检查。

## tensor 关键字参数

tensor 类型的关键字参数会变成图输入，因此调用方可以每次传入不同的值，而不是把它固化进图里：

```python
position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_ids": position_ids},
)
logits = compiled(input_ids, position_ids=position_ids)[0]
```

提升是递归的：值里嵌在 list、tuple、mapping 中的 tensor 各自成为一个图输入，调用方按同样的嵌套结构传回：

```python
cos, sin = position_embeddings  # 新版 HF 层就是这样传的一对 tensor
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_embeddings": (cos, sin)},
)
logits = compiled(input_ids, position_embeddings=(cos, sin))[0]
```

非 tensor 的关键字参数在 tracing 期固定。不是 tensor 的叶子——比如 `use_cache=False`，或者同一个 mapping 里的一个 float——仍然是 tracing 期常量；调用时给它传不同的值会报错，而不是被静默忽略。`past_key_values=[(k, v), ...]` 的提升方式相同。

用 `use_cache=True` 编译模型是可以的，尽管一个 plan 就是一次前向：cache 初始为空，而零长度的 concatenate 操作数不贡献任何内容，前端会把它折掉，所以编出来的图和 `use_cache=False` 完全一样。要做带 cache 的 decode 请用下面的静态 cache adapter。

## 尺寸桶

prompt 长度固定意味着服务循环每换一个长度就要重编译一次。`compile_qwen2_bucketed` 改为每个桶宽度编译一对计划，更短的 prompt 左填充进能装下它的最小桶：

```python
from aotolith.models.qwen2 import compile_qwen2_bucketed

runtime = compile_qwen2_bucketed(
    model,
    prompt_lengths=[128, 512, 2048],
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids)  # 落到 128、512 或 2048
```

左填充是保持 decode 布局的前提：decode 的 mask 取 cache 的尾部窗口，真实 token 必须落在尾部。但 `is_causal=True` 分不清 pad 和 prompt token，所以 prefill 图额外接收一个 rank-4 mask，把 causal 和 padding 合成一个张量。桶宽正好等长时它就是纯粹的 causal mask，因此一张计划同时服务填充和等长两种情况。

只有桶宽度会改变 shape。权重在各桶的计划之间按对象身份共享，所以三桶运行时只保留一份模型，而不是三份。`save(prefix)` 仍然是每个桶一份 archive（`prefix.128.prefill.aoto` 等），每份 archive 各自带一份 state。

Qwen2 适配层有一套管尺寸桶的做法，编译器把它也做成了通用的：`compile_bucketed` 沿指定轴为每个宽度编译一份计划，返回的桶集合把任意不超过最大桶的请求派给能装下它的最小桶。

```python
from aotolith import compile_bucketed

plans = compile_bucketed(
    model,                        # 任意 eval 状态的 nn.Module
    (torch.randn(1, 16),),        # 追踪时用的 shape
    axis=0,                       # 每个请求会变的轴
    sizes=(1, 4, 8),              # 要编译的宽度
)
plans(mx.array(rows_3_by_16))     # 3 行，交给 4 的桶
```

让它能直接上的有三件事。追踪宽度相同的那几个输入就是请求维度，每个桶按宽度重新展开；同一座上宽度不同的输入（一张表、一份权重）保持不动，张量关键字参数同理——它如果没跟着追踪宽度走，就当作常量。填充是*平铺*，重复请求自己的那些行，而不是填常量，所以填充区上的归约不会变成 NaN，填充出来的行最后再从输出里切掉。最后一条是它成立的前提契约：把这一轴变宽不能改变原本那些行的值。batch 轴满足这条；causal 块的序列轴不满足，所以那种轴需要调用方自己的填充：

```python
plans = compile_bucketed(
    model, (torch.randn(1, 8),), axis=1, sizes=(8, 32),
    pad=lambda inputs, width: (left_pad(inputs[0], width), mask_for(width)),
    unpad=lambda outputs, requested: (outputs[0][:, -requested:],),
)
```

输出按 shape 切：某一座上长度正好等于桶宽的输出会被当作请求维度切回请求宽度。唯一需要 `unpad` 钩子说明的是那种「模块本来就返回一个恰好等于某个桶宽的定长张量」的情况。

跨桶派发不是逐位一致的，因为后端自己的 matmul 也不是逐行一致的：这份 MLX 上同一行放进 1 行和 4 行的 `float32` matmul，结果差 6.8e-3，因为 tile 形状决定了累加顺序。桶集合的差异不超过这个噪声，`tests/test_buckets.py` 断言的就是这个噪声本身，而不是手挑的容差。

请求来了先问清楚：

```python
plans.widths                   # (1, 4, 8)
plans.footprint()              # state 字节、最坏活集、最坏上界
plans.check_memory_budget(budget_bytes)   # MemoryBudgetError（AOTO-2004）
plans.memory_report("minimal")            # 最宽那个桶的报告
plans.stats.snapshot()         # 各桶命中次数，以及为此浪费的填充量
```

比所有桶都宽的请求抛 `BucketRangeError`（`AOTO-2005`），而不是在请求里现编：一次编译要几百毫秒且无法给上界，这正好是准入决策最不需要的东西。

桶集合本身也是产物。`plans.save("model.aoto")` 写出 `model.1.aoto`、`model.4.aoto` 和 `model.buckets.json`；`load_bucketed` 读回索引，每份 archive 照常走 ABI 与设备检查，然后把权重重新共享。索引里记了每个桶的 planned peak 作为指纹，所以从别的构建里换进来的产物会被拒绝，而不是把某个宽度派给一份按别的宽度编出来的计划。自定义 `pad` 钩子是宿主代码、无法序列化，因此用了钩子的集合拒绝保存。

## 模型族

静态 Cache 适配层是一个引擎加一个族，不是一个模型一份适配层。`aotolith.models.decoder` 装着图、cache ABI 和桶派发；`aotolith.models.qwen2` 与 `aotolith.models.llama` 是薄前端，负责固定族名并把 checkpoint 对着这个族校验。

| 族 | 入口 | 它坚持什么 |
| --- | --- | --- |
| `qwen2` | `compile_qwen2_static_cache`、`compile_qwen2_bucketed` | q/k/v/o 投影带 bias，打包投影 kernel 把它折进 epilogue |
| `llama` | `compile_llama_static_cache`、`compile_llama_bucketed` | attention 无 bias；`num_key_value_heads` 等于或整除 `num_attention_heads`；RoPE 底数来自 `rope_theta`，设了 `rope_scaling` 按名字拒绝 |

```python
from aotolith.models.llama import compile_llama_static_cache

runtime = compile_llama_static_cache(
    model, prompt_length=128, max_new_tokens=64, quantize=64
)
tokens = runtime.greedy_generate(prompt_ids)
```

校验之后的一切都是共享的：同一套 cache 布局、同一个 `prefill_mask`、同一份 int4 打包、同一个 `greedy_generate`。这也正是那条拒绝的意义——`check_qwen2_model` 与 `check_llama_model` 遇到另一个族的 checkpoint 会抛 `UnsupportedModelError`（AOTO-1004），而不是悄悄编出一份按别的 checkpoint 排布的投影。族从 `model.config.model_type` 读，所以包一层自己的类是允许的，而写错了 `config` 不是。

族也会被记在事后读得到的地方：`runtime.family` 给出族名，基准门禁为每个族各编一个 prefill 和一个 decode 负载，所以免 bias 投影 kernel 的回归会失败，而不是悄悄变成一次交换。

## 后端

`target=` 选择由谁来产这份计划，编译器入口和两个适配层都接受：

| target | 发射什么 | 跑在哪 |
| --- | --- | --- |
| `apple-metal`（默认） | 逐 op 选中的 Metal kernel，其余回退 MLX | 任意 Apple GPU；kernel 在加载时从 MSL 重建 |
| `mlx-reference` | 什么都不发射，每个节点都走 MLX 解释器 | MLX 能跑的地方，包括没有 Metal 设备的 host |

```python
compiled = aoto.compile_module(model, (input_ids,), target=aoto.REFERENCE_TARGET)
assert aoto.TARGETS == ("apple-metal", "mlx-reference")
```
```bash
aoto compile ... --target mlx-reference
```

参考计划是一份真正的计划：同一张 IR、同一个存储计划、同一份 manifest、同一张 selections 表——每一项只是写着 `mlx`，这正好是一份 kernel 全部回退的计划会记录的样子。由此有两件事：

- **它让那条缝可测。** 编译器、产物格式和运行时不问计划是谁产的。一份参考后端的产物走的是与 Metal 产物完全相同的保存、加载和校验代码。
- **它是第二种意见，不是数值更紧的 oracle。** MLX 在 Apple GPU 上不用 IEEE float32 求 batched `matmul`：M5 + MLX 0.32 上同一个乘积相对 float32 参考差约 1e-3，只有单行路径是精确的。生成的 kernel 用 float32 累加，是两条路里更紧的那条。两个 target 之间比结构、shape、dtype、cache 内容和 token 决策；精度去和 kernel 比。

`autotune=True` 会被明确拒绝（`ValueError`），而不是被忽略——什么都没有生成的时候没有东西可测。适配层上的 `target=` 会一路传给集合里的每一份计划，桶索引只记一次，所以为一个 target 编出来的集合被当成另一个加载时，拒绝发生在集合这一层——每份 archive 本来也会各自拒绝，但只有索引能在加载任何东西之前说出来。

## 命令行

安装后可以用 `aoto` 编译 callable、module 或 HuggingFace 模型。产物也可以在不导入原始模型的情况下检查和执行。

```bash
aoto compile \
  --module 'my_module:build_model' \
  --input-file inputs.json \
  --output model.aoto \
  --codegen-matmul \
  --autotune

aoto inspect model.aoto
aoto verify model.aoto --require-device-match
aoto run model.aoto --input-file inputs.json --output outputs.json
aoto --log-level INFO compile ...   # 或 AOTOLITH_LOG=INFO
```

`--log-level`（以及 `AOTOLITH_LOG`）打开[日志](#日志)一节描述的包日志，参数要放在子命令之前。失败输出形如 `aoto: error: [AOTO-4001] …`，code 在最前面，包装脚本可以直接匹配，退出码为 `1`；命令行本身写错时退出码为 `2`。

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

也可以重复传入 `--input '<json>'`，不写输入文件。`--callable MODULE:ATTR` 捕获函数，`--module MODULE:ATTR` 接受 `nn.Module`、类或 factory，`--hf-model` 加载 HuggingFace causal language model。console script 不在 `PATH` 时，用 `python -m aotolith` 执行相同命令。

tensor 关键字参数会变成图输入，写作 `NAME=<json spec>`：`aoto compile ... --forward-kwarg position_ids='{"shape":[1,4],"dtype":"int64","data":[[0,1,2,3]]}'`，运行时用 `aoto run ... --kwarg position_ids=...`。`aoto inspect` 会把它们列在 `entrypoint.kwargs` 下。

一次编译也可以产出一套尺寸桶而不是一份产物：

```bash
aoto compile --module 'my_module:build_model' \
  --input '{"shape":[1,16],"dtype":"float32","data":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}' \
  --output model.aoto --bucket-sizes 1,4,8
aoto inspect model.buckets.json
aoto run model.buckets.json --input '{"shape":[3,16],...}'   # 落到 4 的桶
```

它会写出 `model.1.aoto`、`model.4.aoto`、`model.8.aoto` 和索引 `model.buckets.json`；`inspect`、`verify`、`run` 在原本接受产物的位置都接受这个索引。`--bucket-axis` 选择轴（默认 `0`），宽度必须严格递增。比所有桶都宽的请求以 `AOTO-2005` 退出 `1`，包装脚本可以直接按 code 分支。

`aoto bench` 是编译器自身的回归门禁：测量一组固定 shape（projection、尺寸桶派发、gated MLP、RMSNorm、causal attention、int4 GEMV，和 Qwen2 prefill／decode 适配层），并可与记录的 baseline 比对：

```bash
aoto bench --baseline benchmarks/baselines --output perf-report.json
aoto bench --update-baseline benchmarks/baselines        # 按本机芯片命名写入
aoto bench --check-baselines benchmarks/baselines        # 不需要 GPU
```

`benchmarks/baselines` 是一个芯片矩阵，不是单个文件：每份文件按它记录的芯片命名（`Apple M5 Pro` 对应 `apple-m5-pro.json`）。`--baseline` 指向目录时会选本机那份，所以在第二台 Mac 上记录是**加**一份文件，而不是替换第一份。在没有对应文件的芯片上，门禁照样比较 op 数与计划峰值——这两个由图形状决定、与 GPU 无关——并说明用了谁的文件、为什么跳过毫秒。`--check-baselines` 不测量任何东西，只校验提交进仓库的文件：schema 版本、文件名是否还和它记录的设备一致、同一块芯片有没有记两次。显式给文件路径仍然照用。

它精确比较哪些数字、只在同型号 GPU 上比较哪些数字，见[性能笔记](performance.zh-CN.md#数字是怎么测的)。`aoto run --memory latency|balanced|minimal` 选择这次调用的峰值内存策略；Python 里是同一个开关 `memory=`，见[内存](#内存)。

## 内存

MLX 是惰性的。只构图不 evaluate 的 plan 调用会把每个中间值都留在原地，等 caller evaluate 输出时，evaluator 又会先给每个节点分配输出、再释放用完的那些。在一条活集峰值 134 MB 的八段 concat 链上，天真的峰值是 201 MB——是所有中间值之和，而不是最大活集。storage plan 早就知道哪个值在哪一层死掉，只是 runner 之前没法照着做。

`memory=` 就是这个口子。三档策略：

| 策略 | 做什么 | 峰值 |
| --- | --- | --- |
| `"latency"` | 只构图，绝不同步 | 所有中间值之和 |
| `"balanced"`（默认） | 死掉的字节累积到 `max(8 MiB, transient/8)` 时在释放点 evaluate 并等待 | planned peak + 该阈值 |
| `"minimal"` | 每个真的释放了东西的点都 evaluate 并等待 | planned peak |

同步点是真正的屏障，不只是 `mx.eval`。evaluate 只是把工作排进队列就返回，而一个 storage plan 认为已经用完的 buffer，在 GPU 可能还在写它的时候无法还给分配器——于是下一个算子会叠在只是名义上死掉的 buffer 上分配。跨十二个新进程实测，同一条 134 MB 的 plan 峰值有时 134 MB、有时 160 MB、有时 134 MB + 256 B。现在 runner 先 evaluate、再等待、最后删除，十二个进程测到的都正好是 `peak_bound_bytes`。

默认档在本仓库自带的 workload 上一分钱不花：它们每一个的 transient 都小于 8 MiB，所以 `"balanced"` 不会发出任何同步，生成的函数体与 `"latency"` 完全一致。它在内存开始要紧的地方才生效——上面那条链上，它用 4 次同步把峰值压在 134 MB，而 `"minimal"` 用 9 次拿到同样的 134 MB。

```python
compiled.plan(inputs, memory="balanced")   # 或 "latency" / "minimal"
compiled(inputs, memory="minimal")          # module 入口同样接受
```

跑之前先问：

```python
report = compiled.plan.memory_report("balanced")
report.planned_peak_bytes   # 最忙那一刻的所有活字节
report.peak_bound_bytes     # 上界：planned + 阈值
report.sync_count           # 展开后的函数体会发出几次同步
report.transient_bytes      # plan 自己产生、可以还回去的字节
```

`peak_bound_bytes` 是上界，而且是被 CI 卡的：`aoto bench` 每个 workload 一列，`tests/test_memory.py` 还会在一个分配器干净的子进程里实测这个上界。

另外，`plan.footprint` 按"谁负责"把内存拆开——caller 常驻的输入（权重）、这次调用分配又还回去的激活、输出，以及 storage plan 需要的那块 arena：

```python
print(compiled.plan.footprint.describe())
# inputs 16.0 MiB, activations 256.0 KiB, outputs 128.0 KiB, peak 16.2 MiB
```

接请求的服务用这些数字在真正分配之前就做决定。static-cache runtime 上还有一个预算检查：

```python
runtime.check_memory_budget(budget_bytes)   # 放不下就抛 MemoryBudgetError（AOTO-2004）
```

它比较的是"一个 decode step"的占用，而一步里同时有两份 cache：它先把要返回的 cache 分配出来，才丢掉传进来的那份。这个翻倍正是"看着放得下的桶"跑到序列中途仍然会失败的原因。`runtime.memory_footprint()` 会报出拆分：权重、打包后的 KV cache、prefill mask、一步的 scratch，以及一步的峰值。

### 冷启动

plan 的第一次调用要付两笔钱：把图展开成 Python 函数，以及让 Metal 编译用到的 kernel。`compiled.prepare()`（或 `plan.prepare()`）把展开放到启动时做，于是第一个请求只剩 kernel 编译：

```python
compiled.prepare()   # 只构建 runner；kernel 仍在首次使用时编译
```

`aoto bench` 两笔都报：`first_call_ms` 是含 kernel 编译的首次调用，`compile_ms` 是编译本身。两者都对着记录的 baseline 卡，允许 2 倍——未改动代码的两次运行之间，Metal 的 shader compiler 实测会差 1.5 倍。

## 编译产物

artifact 使用 format version 5 和 ABI version 1.3。format 4 仍然能读：reader 本来就支持它的 kernel 编码，是一条 manifest 检查在拦；读 format 4 时会打一条 warning。更新的 format 会被拒绝。两张图、kernel table、selection table、storage plan 和每个 state tensor 都有 SHA-256 digest。加载时会校验 ABI、target、图、入口布局、state shape/dtype、序列化 kernel、重新构建的 selection 和 storage plan。kernel 根据校验后的图和已保存的 selection 重建，因此 artifact 不需要原始 Python 模型，加载时也不会重新 autotune。

是否要求 GPU 完全相同由调用方决定。`--require-device-match` 或 `require_device_match=True` 会拒绝为其他 Apple GPU 型号编译的 artifact；默认允许在其他 Apple Silicon 设备上重新构建 kernel 后执行。

## 越界检查

PyTorch 对越界的 embedding／gather 下标抛出 `IndexError`，而 MLX 会静默 wrap 或填值，因此后端需要复现该检查。下标值在设备上，所以校验它必然要同步：逐个算子检查会让每个 decode step 产生六次设备同步，改成把越界检查排队累积、最后合并成一次 reduction 之后只剩一次。

如果调用方已经知道下标一定在范围内，可以去掉这最后一次同步——`plan(..., verify_index_bounds=False)`，或 `runtime.decode_token(..., verify_token=False)`；greedy 主循环正是这么做的，因为它回填的 token 全部来自词表上的 `argmax`。默认的严格行为没有变化，越界仍然抛 `IndexError`。

具体抛出的是 `IndexBoundsError`（`AOTO-2002`），它是 `IndexError`，并且会说出越界的值和被接受的范围：

```
index out of range: 7 is outside the valid range [0, 5] for a dimension of size 6
```

高级索引接受负下标，所以它的范围写作 `[-6, 5]`；`F.embedding` 不接受，范围写作 `[0, 5]`。文案跟着算子实际接受的范围走，而不是跟着 shape 猜。

## 错误

每个主动拒绝都是带稳定 code 的 `AotolithError`，按调用方的决策分组：`CompilationError` 是「这个模型按现在写法编译不了」，`PlanInputError` 是「这次调用与 plan 不符」，`DeviceError` 是「这台机器跑不了」，`ArtifactError` 是已存文件的问题，`PlatformError` 是这个安装在这台机器上不成立。每个类同时继承上一版抛的内置异常，所以已有的 handler 继续有效：

```python
from aotolith import errors

try:
    compiled = aoto.compile_module(model, (input_ids,))
except errors.UnsupportedOperationError as exc:
    print(exc.code)   # AOTO-1001
    print(exc.hint)   # 下一步怎么做
```

完整目录，包括每个 code 对应哪种反应，见 [docs/errors.zh-CN.md](errors.zh-CN.md)。

## 日志

包往 `logging.getLogger("aotolith")` 写，只挂一个 `NullHandler`，所以 import 不会打印。想看一次编译决定了什么：

```python
import logging
logging.getLogger("aotolith").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
```

脚本里也可以用 `aotolith.configure_logging("INFO")`。`aoto` 命令接受 `--log-level`，`AOTOLITH_LOG=DEBUG` 不用加参数也一样。`INFO` 是每次编译一行（原生与 MLX 的计数、GPU），`DEBUG` 是逐算子一行，`WARNING` 是每一次降级 —— tensor core 被拒、autotune 没有赢家、读了旧格式产物。

## 多线程调用 plan

`CompiledPlan` 可以并发调用。唯一的惰性步骤 —— 首次调用时把图展开成直线 Python 函数 —— 在锁里完成，所以两个线程同时打到一个新 plan 也只会展开一次，之后都是每次调用自己的状态。设备仍然按提交顺序使用单条 stream；要重叠 GPU 工作用 `mx.async_eval`（Qwen2 adapter 就是这么做的），而不是再加线程。

## 编译一个 callable

README 里的例子编译一个 `addmm + SiLU` 函数。`compile_module` 会把模型的参数和 buffer 作为显式 state 追踪，返回的 plan 不再持有原始 `nn.Module`：

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import aotolith as aoto

model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
).eval()

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
compiled = aoto.compile_module(
    model,
    (input_ids,),
    codegen_matmul=True,
    codegen_attention=True,
)
logits = compiled(input_ids)[0]
mx.eval(logits)
```
