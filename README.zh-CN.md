# Aotolith

[English](README.md) | 简体中文

一个原生支持 Apple Silicon 和 Metal 的实验性 Transformer 模型编译器。

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml/badge.svg)](https://github.com/jaykay233/aotolith/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white)](#环境要求)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#当前状态)

Aotolith 把 PyTorch 模型捕获成 FX 图，经过带类型和 shape 的 SSA IR，生成由 Metal kernel 和 MLX fallback 共同执行的计算计划。

> [!IMPORTANT]
> 这个项目编译 Transformer 模型计算。请求调度、continuous batching、分页 KV 分配、采样、分布式执行和 serving 由外部运行时负责。

## 为什么做这个项目

主流 Transformer 编译栈大多围绕 CUDA 和成熟的 NVIDIA kernel 生态构建。这个项目走另一条路：保留与后端无关的图 IR，再把模型计算原生编译到 Apple Silicon，而不是直接把 MLX 当成完整运行时。

编译器负责图捕获、合法化、布局简化、算子融合、公共子表达式消除、tensor 生命周期分析、kernel selection 和 Metal code generation。MLX 提供 fallback 算子，并在原生覆盖范围扩大前作为数值执行基线。

## 当前状态

项目已经可以运行静态 shape 的端到端 Transformer 实验，但还不是稳定的编译器产品。

| 模块 | 当前状态 |
|---|---|
| PyTorch 前端 | 可编译静态 shape 的 callable 和 `nn.Module`；模型参数与 buffer 会绑定到执行计划 |
| 图编译器 | 使用 typed SSA IR，支持 shape/dtype 校验、合法化、布局折叠、融合和 CSE |
| 存储规划 | 已生成 live interval、释放点、峰值活跃字节和可复用 buffer slot，执行时使用真实 arena |
| Kernel 选择 | 为每个 lowered op 记录原生 Metal 或 MLX 实现，以及 fallback 原因 |
| 编译产物 | 版本化 `.aoto` archive 包含图、kernel、selection、storage metadata 和模型 state，可在新进程加载执行 |
| Autotune | 可选的 profile 流程会测量原生 kernel 和 MLX 候选，测量结果与最终选择写入 artifact |

当前回归用例把两层 HuggingFace Llama 从 220 个 FX 算子降到 136 个 lowered op，logits 相对 PyTorch 参考实现的最大绝对误差为 `8.941e-08`。释放原始 module 后，编译计划仍可执行。

## 编译流程

```text
PyTorch FX / nn.Module
        │
        ▼
Typed SSA IR
        │  legalization、layout folding、fusion、CSE
        ▼
Liveness analysis / storage plan
        │
        ▼
Kernel selection / fallback planning
        │
        ├── 生成的 Metal kernel
        └── MLX fallback 算子
        ▼
CompiledPlan
        │
        ├── 首次调用把图展开成直线 Python 代码
        ├── save：manifest + 图 + kernel + selection + state
        │
        ▼
版本化 artifact → 重新加载并构建 kernel → Apple GPU
```

一次生成里要重复成千上万次的 step 不该每次都重新解析自己的操作数。因此首次调用会把 plan 编译成一个 Python 函数，函数体里每个算子对应一行赋值，kernel 调用点直接绑定成常量；之后的调用只校验输入，然后执行这个函数。kernel 本身没有任何变化，变的只是它们周围的簿记开销。

## 环境要求

- Apple Silicon Mac，macOS
- Python 3.11+
- MLX 0.32
- PyTorch 2.7+，用于 FX 捕获

## 安装

```bash
git clone https://github.com/jaykay233/aotolith.git
cd aotolith

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

只编译 Python callable 时可以安装 `.[fx]`。`dev` 和 `hf` extras 会额外安装 HuggingFace Transformers，用于模型级编译和测试。

## 快速开始

下面的例子编译一个小型 `addmm + SiLU` 函数。执行前可以检查图、存储计划和 kernel selection。

```python
import mlx.core as mx
import torch

import aotolith as aoto


def fn(x, weight, bias):
    hidden = torch.addmm(bias, x, weight, alpha=2.0)
    return torch.nn.functional.silu(hidden)


torch.manual_seed(0)
example_inputs = (
    torch.randn(4, 16),
    torch.randn(16, 32),
    torch.randn(32),
)

compiled = aoto.compile_callable(fn, example_inputs, codegen_matmul=True)
inputs = tuple(mx.array(value.numpy()) for value in example_inputs)
output = compiled.plan(*inputs)[0]
mx.eval(output)

print("peak live bytes:", compiled.plan.storage_plan.peak_live_bytes)
for selection in compiled.plan.selections:
    print(selection.op, selection.implementation, selection.reason)
```

### 编译 HuggingFace 模型

`compile_module` 会把模型参数和 buffer 作为显式 state 参与 tracing。返回的计划不持有原始 `nn.Module`。

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

上面的 config 里 `use_cache` 保持默认的 `True`。这是可以编译的：cache 初始为空，而零长度的 concatenate 操作数不贡献任何内容，前端会把它折掉，所以编出来的图和 `use_cache=False` 完全一样。一个 plan 就是一次没有隐藏状态的前向，模型返回的 cache 不属于它；要做带 cache 的 decode 请用下面的静态 cache adapter。

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

非 tensor 的关键字参数在 tracing 期固定。提升是递归的：值里嵌在 list、tuple、mapping 中的 tensor 各自成为一个图输入，调用方按同样的嵌套结构传回：

```python
cos, sin = position_embeddings  # 新版 HF 层就是这样传的一对 tensor
compiled = aoto.compile_module(
    model,
    (input_ids,),
    example_kwargs={"position_embeddings": (cos, sin)},
)
logits = compiled(input_ids, position_embeddings=(cos, sin))[0]
```

不是 tensor 的叶子——比如 `use_cache=False`，或者同一个 mapping 里的一个 float——仍然是 tracing 期常量；调用时给它传不同的值会报错，而不是被静默忽略。

### 编译 Qwen2.5 静态 KV Cache

`aotolith.models.qwen2` 提供 Qwen2/Qwen2.5 的单请求静态图适配层。它把预填和逐 token 解码拆成两张图，用固定容量的 KV Cache 连接，并包含 Qwen2 的 RMSNorm、RoPE、GQA、SwiGLU 和 LM head。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aotolith.models.qwen2 import compile_qwen2_static_cache


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
).eval()
prompt_ids = tokenizer.encode("用一句话解释 KV Cache", add_special_tokens=False)[:128]

runtime = compile_qwen2_static_cache(
    model,
    prompt_length=len(prompt_ids),
    max_new_tokens=64,
)
tokens = runtime.greedy_generate(prompt_ids, max_new_tokens=64)
print(tokens.tolist())
```

这个适配层固定 batch、prompt 长度和最大生成长度，只实现 greedy 单请求生成；host 侧负责 token 循环，scheduler、batching、分页 KV 和采样仍不包含在编译器中。`runtime.save("qwen")` 会保存 prefill 和 decode 两个 `.aoto`，每张图当前各自包含一份模型 state。

query、key、value 共享同一份归一化输入，因此适配层把三者按层存成一个打包后的投影并只做一次矩阵乘。拆分结果只需两个 view 操作，却省掉两次 kernel launch：128 行的 key/value 投影对 GPU 来说太小，带宽利用率不到融合后投影的五分之一。

gated MLP 没有可以这样打包的权重，因为 `gate` 和 `up` 是相乘而不是拼接。`fuse_gate_up_projection` 补上了另一半：它识别 `silu(gate) * up`，且两个操作数都回溯到读取同一份激活的投影，然后把这对投影替换成一个带 `silu_mul_paired` epilogue 的矩阵乘。kernel 在同一个循环里流式读取 decode token 的 gate 行和 up 行，一次写出 `silu(gate) * up`，于是 step 读的还是同样的 17.4 MB，但 launch 从两次变成一次。它只在激活只有一行时生效；更高的操作数走 tiled 矩阵乘，那里没有成对 epilogue，两个投影保持独立。gate 在 `silu` 之前先舍入到 BF16，正是独立投影原本会写下的值，所以融合结果是逐位一致的，而不只是接近——保留 FP32 累加精度的版本在 24 token 的 greedy 运行里，第 4 个 token 就翻转了。

仓库内 benchmark 使用相同的 prompt、精度和 `128 + 64` token 参数比较 PyTorch MPS、MLX 和 Aotolith：

```bash
python examples/qwen_static_cache_benchmark.py --output /tmp/qwen-aotolith.json
python examples/qwen_static_cache_benchmark.py \
  --codegen-matmul \
  --mlx-baseline \
  --output /tmp/qwen-aotolith-codegen.json
```

`--mlx-baseline` 会在同一个进程里用同一段 prompt 给 `mlx_lm` 自己的 greedy decode 计时，原生基线因此是在跑 benchmark 的这台机器上测出来的，而不是从别处引用。在这台机器上（M5 Pro，MLX 0.32）多次运行读到 **190.5-195.0 token/s**。

| 配置 | prefill | decode（每步同步） | decode（异步） | 相对 `mlx_lm` |
| --- | --- | --- | --- | --- |
| 仅 MLX fallback | 10 000 token/s | 119.0 token/s | **134.6 token/s** | 0.70x |
| `--codegen-matmul` | 13 000 token/s | 178.4 token/s | **240.1 token/s** | **1.23x** |
| `--codegen-matmul --codegen-attention` | 12 400 token/s | 176.8 token/s | 233.0 token/s | 1.21x |

（Qwen2.5-0.5B-Instruct、128 prompt token、greedy、batch 1、BF16、11 次重复取中位数；每一行的 64 个生成 token 都与 PyTorch 参考实现一致。这份构建多次运行下来，`--codegen-matmul` 的 decode 落在 240.0-240.2 token/s——这个配置稳定到可以读到小数点后一位——而 `--codegen-attention` 那行在 229.8-238.2 之间波动，prefill 在 11.6k-12.8k 之间波动。不加原生 attention 时 prefill 稳定在 13.0k 左右，加上后降到约 12.4k，这是 attention kernel 在 prefill 形状上 tile 更慢所致，不是 decode 的代价。`mlx_lm` 基线在同样几次运行里读到 190.5-195.0 token/s。）

token 一致不等于数字一致，两条路径离参考实现的距离也不同：在同一个模型的 24 层前向上，原生 kernel 的 logits 相对 PyTorch 平均 0.043、最大 0.38，而全走 MLX fallback 是平均 0.081、最大 1.5。也就是说打开 codegen 是把算术语义往 PyTorch 靠，而不是相反——所以它是个吞吐决策，不是精度决策。

和 `mlx_lm` 对比应该看 `decode_async`，因为两者都把 host 循环摊到了多个 token 上；`decode_sync_each_step` 每个 token 都要同步一次设备，任何服务循环都不会这么做。异步路径此前落后于同步路径，现在反超，因为每个 step 只强制一次设备同步。

decode 现在受权重带宽限制，而不是受 host 限制。构建一个 step 只需 0.98 ms Python，GPU 需要约 4.1 ms，所以 host 完全被异步队列盖住；在 plan 展开之前 host 要 6.2 ms，那时它才是瓶颈。一个 step 要流式读取 991 MB 权重（24 层的 `qkv` 2.07 MB、`o_proj` 1.61 MB、一个 17.4 MB 的成对 gate/up 矩阵、8.73 MB 的 `down`，再加 272 MB 的 LM head），按实测 4.17 ms 每个 token 算是 238 GB/s。

剩下的余量在哪，看下面的单次速率就知道：272 MB 的 LM head 单次能跑到 412 GB/s，而 17.4 MB 的 MLP 单次只有 279 GB/s。这一节早先的解释是「输出只有 4864 列的一次 launch 填不满设备」，这个解释经不起实测检验，见下面的[为什么小分桶修不动](#为什么小分桶修不动)。

单个 decode step 的分桶 GPU 开销。测量时把同一组 kernel 连着回放足够多遍，让设备队列一直是满的，于是墙钟时间就是设备时间，host 提交被盖住：

| 分桶 | launch 次数 | 每次搬运 | 单次 GPU 时间 | 带宽 |
| --- | --- | --- | --- | --- |
| LM head GEMV | 1 | 272.3 MB | 662 us | 412 GB/s |
| 带残差的 MLP down GEMV | 24 | 8.73 MB | 32.6 us | 268 GB/s |
| MLP gate+up GEMV + RMSNorm + `silu_mul` | 24 | 17.44 MB | 62.5 us | 279 GB/s |
| 融合 QKV GEMV + RMSNorm | 24 | 2.07 MB | 10.8 us | 193 GB/s |
| attention（单 query） | 24 | 0.10 MB | 7.3 us | 14 GB/s |
| 带残差的 o_proj GEMV | 24 | 1.61 MB | 7.8 us | 207 GB/s |
| RoPE q / k | 48 | 1 KB | 7.8 us | - |
| 加性 attention mask | 1 | 0.2 KB | 26.6 us | - |

加起来是 3.96 ms 设备时间、991 MB，而整个 step 实测 4.17 ms。LM head 加 MLP 矩阵占了其中 74%，而它们是在搬字节，不是在等 launch；其余分散在 121 次每次最多搬 2 MB 的 launch 上。一个 step 带着 170 次有依赖的 launch。删掉便宜的 launch 不影响总量：把 query 和 key 的旋转合成一次 16 头旋转省掉了 24 次 launch 和 24 个图节点，decode 却是 232.9 token/s 对之前的 232.5，落在运行间波动内，所以没有保留。删掉搬运真实数据的 launch 就会影响：把 `gate` 和 `up` 合成一次 launch 读的仍是同样的 17.4 MB，但 launch 从两次变成一次，交替 A/B 测到 1.031x-1.038x。区别在于这次 launch 搬多少，而不在于有多少次。

有三种做法会把这个表测成“什么都没测”，而且看起来都很正常。一是每轮丢掉输出的数组，惰性图在到达设备之前就被释放，墙钟时间只覆盖 host 提交，带宽那一列会算出 1 TB/s 以上。二是传全零权重，驱动可以跳过读取，小分桶因此被抬高。一个分桶必须让每一轮的输出都活着，并且喂真实数值，它的数字才有意义；任何超过 `mx.sum` 读同样字节能达到的读数，测的是 harness 而不是 kernel。三是回放本身会往反方向偏：表里同一个分桶的 24 个实例是连着跑的，而真实 step 里每一层都要等上一层，所以 RoPE 和 attention 的单次数字是上界，整表的和只是近似整个 step。

这个方向上有两个实验被测过并放弃。把新的 key/value 直接写到 slot `positions`、用 mask 挑出有效 slot，读起来比滚动窗口更顺，但 MLX 会把 scatter 降级成对整窗口做一遍下标比较，而 concatenate 只是同样字节数的一次拷贝：每个 step 慢 0.18 ms，所以保留滚动。另外，decode token 的旋转位置是 `valid_length - 1` 而不是 `valid_length`；`Qwen2StaticCacheRuntime.decode_token` 曾把后者写进文档并短暂使用，代价是相对参考实现 0.7 的 logit 最大偏差，token 也无法完全对齐。现在 64 个 token 全部一致。

### 为什么小分桶修不动

三个测量指向同一个结论：MLP 分桶没有留下可捡的东西，而且每一个测量都排除了另一种解释。

第一个是 threadgroup 大小。如果 4864 列的输出真的「太窄填不满设备」，给它更多更小的 threadgroup 就该有改善。把同一份已编译的 plan 从 64 个线程扫到 1024 个，这个分桶只动了 2%：

| 每 threadgroup 线程数 | 4864×896 | 151936×896 |
| --- | ---: | ---: |
| 1024（窄分桶当前选择） | 284 GB/s | 395 GB/s |
| 256 | 286 GB/s | 413 GB/s |
| 128 | 289 GB/s | 410 GB/s |
| 64 | 284 GB/s | 408 GB/s |
| 128，宽分桶仍用 64 | 282 GB/s | 291 GB/s |

4864 那个分桶是平的，而 151936 那个在整个扫描里反向动了 30%：它要的是 128 线程的 threadgroup，而窄分桶用的 1024 线程设置会拿走它三分之一的带宽。所以占用率是个真实效应，但它不是区分这两个分桶的原因。

第二个是 MLX 自己的归约。`mx.sum` 读同样的字节，走的是同一条曲线，而且在下面：

| 读取字节数 | `mx.sum` | 本项目的 GEMV |
| --- | ---: | ---: |
| 8 MB | 82 GB/s | 268 GB/s（8.7 MB） |
| 17.4 MB | 180 GB/s | 279 GB/s |
| 35 MB | 225 GB/s | — |
| 140 MB | 295 GB/s | — |
| 280 MB | 296 GB/s | 412 GB/s（272 MB） |

这个投影在每个尺寸上都快过 MLX 的归约，所以这不是本 kernel 实现上的偶然。

第三个是上面同一个表里的受控对比：同一份 warp-per-column kernel，在 272 MB 矩阵上跑到 412 GB/s，在 17.4 MB 上只有 279。差别只有字节数。

三者合起来说明：一次 launch 能达到的速率取决于它读多少字节，而不是它的网格怎么划分。要让整个 step 都跑在 412 GB/s，那 991 MB 必须以几百 MB 一次的 launch 到达，但每一层都被切成五个严格串行的投影 —— `qkv`、`o_proj`、成对 gate/up、`down` —— 而最大的只有 17.4 MB，因为中间的非线性是真屏障。把 `gate` 和 `up` 配对正是利用了这一点，这也是为什么那一次改动测到 1.03x，而删掉 24 次 1 KB 的 launch 什么也没测到。所以每个 token 4.17 ms 就是这台设备上的 BF16 地板：要明显更快只能少读字节，也就是量化权重，而不是写更好的 kernel。

自动调优也不是出路，而且在此之前它对这种模型连跑都跑不起来。`--autotune` 会给每个 op 都求一次 MLX fallback 当参考，但归一化、旋转和 attention 只降级到生成 kernel，所以编译一个 Transformer 会在第一个独立 `rms_norm` 上抛 `NotImplementedError`。现在它对这类 op 只测量自己，记录单条 benchmark，并在 selection reason 里说明这是唯一有效候选。

### 命令行与编译产物

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
```

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

artifact 使用 format version 4 和 ABI version 1.3。两张图、kernel table、selection table、storage plan 和每个 state tensor 都有 SHA-256 digest。加载时会校验 ABI、target、图、入口布局、state shape/dtype、序列化 kernel、重新构建的 selection 和 storage plan。kernel 根据校验后的图和已保存的 selection 重建，因此 artifact 不需要原始 Python 模型，加载时也不会重新 autotune。

是否要求 GPU 完全相同由调用方决定。`--require-device-match` 或 `require_device_match=True` 会拒绝为其他 Apple GPU 型号编译的 artifact；默认允许在其他 Apple Silicon 设备上重新构建 kernel 后执行。

## 原生 Metal Code Generation

pointwise、softmax、reduction、LayerNorm 和 RMSNorm 默认生成 Metal kernel。matmul 和 attention 默认使用 MLX，开启下面的选项后才生成原生 kernel：

```python
compiled = aoto.compile_callable(
    fn,
    example_inputs,
    codegen_matmul=True,
    codegen_attention=True,
)
```

生成的原生 kernel 在 matmul 上快于 MLX，但在 attention 上仍落后于 MLX 的 SDPA，两个选项默认关闭的原因因此并不相同：

| 基准 | Aotolith | MLX | 倍率 |
|---|---|---|---|
| matmul 512x2048x512 f32 | 367.4 us | 118.1 us | 0.32x |
| matmul 512x2048x512 f16 | 60.9 us | 80.9 us | 1.33x |
| matmul 512x2048x512 bf16 | 62.4 us | 82.2 us | 1.32x |
| matmul 1024x4864x896 f16 | 373.0 us | 519.3 us | 1.39x |
| matmul 4096x4096x4096 f16 | 5.53 ms | 5.66 ms | 1.02x |
| matmul 128x896x896 bf16（prefill） | 18.4 us | 16.5 us | 0.89x |
| matmul 128x896x4864 bf16（prefill） | 53.5 us | 54.3 us | 1.02x |
| matmul 128x4864x896 bf16（prefill） | 56.9 us | 83.6 us | 1.47x |
| attention 1x8x512x64 f16 causal | 34.2 us | 28.5 us | 0.83x |
| attention 1x8x2048x128 f16 causal | 530.8 us | 466.3 us | 0.88x |
| attention 1x64x4096x128 f16 causal | 14.8 ms | 13.3 ms | 0.90x |

matmul kernel 用 MPP tensor-core 的 `matmul2d` 跑分块操作数。分块配置由输出 shape 和右操作数的布局共同决定：右操作数转置时 K 方向连续，适合更宽的列分块；右操作数行主序时沿 K 方向跨步，更适合更高的归约步长。在 43 组 shape 与布局组合上，选择器对每个 shape 都保持在实测最优分块的 5% 以内；相对 MLX 平均为 1.16x（右操作数转置）和 1.10x（行主序），没有 shape 低于 1.01x。最大的几个 shape 收敛到持平，因为两边都逼近同一套 tensor core 吞吐。上面三行 128 行的数据是 Qwen2.5-0.5B 尺寸下 128 token prefill 的实测，也是唯一诚实的例外：最窄的那个比 MLX 慢 11%，摊到整个 prefill 上（每层一个 `o_proj`，共 24 个，每个约 3 us）大约是 1%——下面那张 prefill 实测表里看不出开启原生 matmul 有任何差别；而真正主导 prefill 的 `128x4864x896` 则快 1.47x。

f32 那一行是唯一的例外，而且是刻意的：硬件没有 float32 tensor core，`T = float` 的 `matmul2d` 实际按 bfloat16 精度执行，会把这个 shape 的结果偏离 PyTorch 1.6e-1——正好等于 MLX 自己 f32 matmul 的偏差。所以 f32 留在标量 tiled kernel 上，与 PyTorch 逐位一致，代价就是上面那 3.1x。FP16/BF16 不受影响。

attention kernel 是寄存器驻留的 FlashAttention：一个 simdgroup 负责 16 行 query 片段和整个 value 宽度，online-softmax 的累加器始终留在寄存器里，`Q` 只在开头暂存一次到 threadgroup memory，`K`/`V` 直接从 device memory 流式读取。长序列上能跑到 18.6 TFLOPS，prefill 形态上比 MLX 的 SDPA 慢 10%–17%。剩下的差距用 `examples/metal_attention_benchmark.py --ablate` 量化：去掉 `QK` matmul 省下 42%–45%，去掉 `PV` 省下 33%–43%，两个收缩就是这个 pass 的 88%，而 device operand 读取已经被它们掩盖——把 `K`/`V` 指向同一个常驻 head 而不是各自的数据，总时间变化不到 3%。所以差距在 MMA 利用率，不在流式读取；针对它试过的写法都实测更慢：把 `K`/`V` 暂存进 threadgroup memory 慢 1.9–2.6 倍（且共享同一份 tile 的 warp 越多损失越小，这正是 occupancy 受限的特征——限制常驻 threadgroup 数量的就是这份 tile），把 query 片段搬进寄存器会溢出、慢 20%，把 `matmul2d` 的 reduction 从 16 加到 32 慢 2.7 倍。真正有效的是 operand 地址运算：`K`/`V` 的行号在逐元素上都被钳到最后一个 key，而当 key 长度和 query tile 都是 32-key tile 的整数倍时这个钳位可证明是恒等运算，因此只在 tile 真的会越过末尾时才生成。在不会触发的情况下省掉它实测快 1.02–1.13 倍，上表的比值是改动之后的。decode 是另一个问题：单行 query 用不上分块 online softmax，所以后端会另外发射一个单 query kernel，把整个 head 的分数留在 threadgroup memory 里，用两级归约求 `max` 和 `sum(exp)`，再做 value 累加而不需要重新缩放。因此 `--codegen-attention` 不再损失 decode 吞吐：它每层只有一次 launch、覆盖 192 个 cache slot，与 MLX 的 SDPA 落在同一噪声范围内。`CompiledPlan.selections` 会记录每个算子实际使用的路径以及 fallback 原因。两个 kernel 都只覆盖 FP16/BF16，其他精度走 MLX。

tensor core 需要 Apple GPU family 9 及以上，所以两条 tensor-core 路径都以编译器自己的 `__HAVE_TENSOR__` 宏为准。更早的机型（M1、M2）以及任何虚拟化环境都走 simdgroup 路径——也就是 fallback 测试里会和 PyTorch 对拍的那份代码——而不是发射一个装不上的 kernel。只看这个宏还不够：半虚拟化 GPU（托管 CI、云 Mac）会定义它，却仍然以 `unsupported deferred-static-alloca-size function body` 拒绝该 metallib，最终变成用户第一次 matmul 时一句看不懂的 `RuntimeError`。因此 `compile` 在选路之前会真正加载并求值一个已生成的 tensor-core kernel；设备拒绝时就把整张图重新 lower 一次改走 simdgroup，并把原因记进 `plan.selections`。代价是多一次 launch（kernel 已缓存时约 0.25 ms），且只在确实生成了 tensor-core kernel 时才付。


## 支持范围

| 类别 | 覆盖范围 |
|---|---|
| 模型 | decoder-only Transformer forward；tiny Llama/Qwen 风格模型；Qwen2/Qwen2.5 单请求静态 KV Cache prefill/decode；已验证 Qwen2.5-0.5B-Instruct |
| 矩阵计算 | `mm`、`bmm`、`addmm`、`baddbmm`，包括支持范围内的 batch broadcasting |
| 索引 | embedding、`index_select`、advanced indexing、`gather`、`scatter`、`scatter_add`、`index_add`，以及 `scatter_reduce` 的 `sum`/`prod`/`amax`/`amin`/`mean` |
| Transformer 计算 | scaled dot-product attention、分解后的 causal attention、softmax、log_softmax、logsumexp、sum/mean/prod/amax/amin/nansum/nanmean/var/std/count_nonzero 归约、`all`/`any`、`vector_norm`、`F.normalize` 与 `F.cosine_similarity`、cumsum/cumprod、LayerNorm（含 mean/rstd）、RMSNorm、SiLU、GELU、SwiGLU 和常见 pointwise 算子 |
| 排序与选择 | `argmax`、`argmin`、带 `dim` 的 `max`/`min`（values/indices）、`sort`（values/indices）、`topk`（values/indices，最大或最小），按被读取的输出分别 lowering |
| 布局与数据移动 | view/reshape、transpose/permute、slice、`split`/`chunk`/`unbind`、`squeeze`（含省略轴的形式）、expand、`repeat`/`tile`、作为 strided view 的 `diagonal`、cat/stack、constant/reflect/replicate padding、flip/roll、clone、cast 和静态常量 |
| 数据类型 | 支持范围内的 FP16、BF16、FP32、bool、I32 和 I64 |
| KV Cache | 静态 shape 的 prefill/decode 图契约和布局；分配与分页仍由外部运行时负责 |

算子覆盖范围小于 PyTorch。不支持的 target 会在捕获或构造 IR 时失败，不会静默绕过编译计划。逐算子的 lowering 表和边界见 [docs/architecture.md](docs/architecture.md)。

## 已知限制

- shape 和 dtype 在编译期固定，不支持动态 shape、符号维度和控制流。
- tensor-valued `example_kwargs` 会被提升为图输入、按名字传入，而且是递归提升的：list、tuple、mapping 里嵌套的 tensor 各自成为一个图输入，所以 `position_embeddings=(cos, sin)` 是一个关键字对应两个输入，`past_key_values=[(k, v), ...]` 同理；调用方按 tracing 时的嵌套结构传回。同一个值里非 tensor 的叶子仍是 tracing 期常量，传入不同的值会报错而不是被静默替换。
- 共享参数只绑定一次：同一个 tensor 在两个名字下共享同一个 MLX buffer 和同一份 artifact payload。图里仍然每个名字一个输入，因为 tracing 就是这么产生的。只共享 storage 但内容不同的视图（如 tensor 和它的 `.t()`）不会被合并。
- 非空常量 tensor 不再直接拒绝，而是绑定进 plan。`compile_module` 会把 tracing 期捕获成图常量的 tensor——`forward` 里手写的 `torch.tensor([...])`，或者模块上没 `register_buffer` 的普通 tensor 属性——按图里出现的顺序提升为 plan 的私有状态：字面量在 `state_names` 里以 `<constant N>` 编号出现，模块属性则用它自己的名字。模块本身不会被改动：普通属性只在捕获期间临时注册成 buffer，结束后（包括捕获失败时）原样还原。artifact 会保留这些常量。裸调 `capture_fx` 仍然拒绝常量，因为它没有可挂载的状态绑定，报错信息里也说明了这一点。
- artifact ABI 只接受 `apple-metal` target，尚未实现跨厂商 target 和 host runtime ABI 兼容。
- advanced indexing 走 MLX，一个 `index` op 覆盖 `source` 和所有非 `None` 的索引 tensor。索引轴相邻时沿用 PyTorch 的规则，把 broadcast 维度插在那段轴原来的位置；多个索引 tensor 被固定轴隔开时 broadcast 维度会落到最前面，前端用一个把被索引轴移到最前的 `permute` 来表达，所以 `x[i, :, j]` 和 `x[i, j]` 共用同一段 Runtime。索引 tensor 之间会互相 broadcast，越界检查覆盖每一维。
- `scatter`、`scatter_add`、`scatter_reduce` 和 `index_add` 走 MLX 而不是生成 kernel。归约覆盖 `sum`/`prod`/`amax`/`amin`/`mean`，都支持 `include_self=False`；`mean` 会在累加数值的同时累加写入次数再相除，因此重复索引取平均而不是求和；`amax`/`amin` 在 `include_self=False` 时需要浮点输入，因为它们的单位元是 ±inf。重复索引按设备调度顺序累加，这正是 CUDA `scatter_add` 给出的保证，而 PyTorch 的 CPU 路径给不了。原地写法（`scatter_`、`scatter_add_`、`scatter_reduce_`、`index_add_`）在被改写的 tensor 于图内没有其他观察者时降级成同一组算子；编译后的调用返回更新后的 tensor，不会回写调用方仍然持有的那块 tensor。
- `F.pad` 的默认 `constant` 模式降级成 `pad` layout op，由 MLX 执行。padding 对按 PyTorch 的定义从最内层维开始，填充值保持源 dtype。全零的 pad 会被当作恒等操作消除；负数条目按 ATen 的语义裁剪展开后的 extent：只裁剪的 pad 变成 `slice`，混合的正负条目先 `slice` 再 `pad`（负数按 0 处理）；裁剪会把某个轴切成空维度时直接拒绝，因为静态 IR 不表示零长度维度。其他 mode 在捕获期报错。
- `F.pad(..., mode="reflect")` 和 `mode="replicate"` 展开成不生成 kernel 的视图链：每一维取内层窗口再 `flip`（reflect），或用 `expand` 复制边沿（replicate），最后由 `cat` 拼接回主轴两侧。负数条目按 ATen 的语义裁剪**展开后**的 extent，所以镜像窗口始终锚定在未裁剪的边界上；同一轴上既有 padding 又有裁剪也能表达，反射窗口宽度不小于轴长度时前端报错。
- 整数 pointwise 不再要求操作数 dtype 相同：PyTorch 自己会提升的组合，前端插入 `cast` 节点而不是让整张图失败——I32 与 I64 混合提升到 I64，`div` 的整数操作数提升到输出的 float dtype，整数与浮点混合提升到那个 float dtype。`where` 的两个分支按同样规则提升，所以 `where(mask, int32, int64)` 会以 I64 select 执行。bool 与数值 dtype 混合仍会被拒绝。整数 `pow` 是精确的：kernel 用 square-and-multiply 循环，乘法按位宽回绕，所以 `2**70` 和 PyTorch 一样回绕成 `0`，`0**0` 是 `1`。负的标量指数在编译期直接拒绝；只出现在 tensor 里的负指数不做逐元素检查（PyTorch 会在运行时报错），kernel 对它返回 `0`。
- `dot`、`mv`、`trace` 都是既有算子的组合，没有新 kernel：`dot` 是 `mul` 加整轴 `sum`，`mv` 把向量 reshape 成 `(n, 1)` 后走既有 `matmul`，`trace` 复用 `diagonal` 的 strided view（整块 flatten 后按 `columns + 1` 取步长）再做 `sum`，所以矩形输入取的是较短的那条对角线，和 ATen 一致。`round` 也保留 `decimals`：kernel 先乘 `10**decimals`、按 ties-to-even 取整再除回去，这正是 ATen 在没有十进制浮点类型时的实现方式。
- `logit`、`erfc`、`frac`、`sinc`、`hypot`、`copysign`、`xlogy`、`nan_to_num` 都融进 pointwise kernel。`logit` 是 `log(x / (1 - x))`，带可选 `eps` 时按 ATen 规则钳到 `[eps, 1 - eps]`。MSL 的 `metal` 命名空间没有 `erfc`，而 `1 - erf(x)` 会丢掉尾部有效位，所以 `erfc` 直接把 `erf` 那套多项式当作尾项使用；`hypot` 和 `log1p` 一样是全局声明，生成时不加 `metal::` 前缀。`frac` 是 `x - trunc(x)`，保留输入符号，而不是 floor 语义的 `fract`。`xlogy` 只在左操作数为 0 且右操作数不是 NaN 时短路成 0（与 ATen 一致），因此 `xlogy(0, -1)` 是 0 而 `xlogy(0, nan)` 仍是 NaN。`sinc` 在 `1e-4` 以内改用级数展开，因为 GPU 的 `sin` 会把这么小的参数冲成 0，`sinc(1e-8)` 就会变成 0 而不是 ATen 的 1。`nan_to_num` 保留 `nan`/`posinf`/`neginf` 三个替换值，缺省值按 dtype 的有限上限补齐（F16 是 65504，BF16/F32 取各自 `max`）；整数 tensor 没有非有限值可替换，lowering 直接返回原值。这些算子只接受浮点 tensor，整数输入会按 PyTorch 的 opmath 提升插入 `cast` 到输出 dtype。
- `floor_divide`、`remainder` 和 `fmod` 除了整数也接受浮点操作数，并与相邻 pointwise op 融进同一个 kernel。`floor_divide` 就是 `floor(a / b)`；`remainder` 沿用 ATen 的「`fmod` 之后再同号修正」，所以结果始终与除数同号，而 `fmod` 的符号跟随被除数。`lerp` 同时支持 tensor 和标量 `weight` 两种拼法，并照搬 ATen 的两个分支（`|weight| < 0.5` 用 `start + weight * (end - start)`，否则用 `end - (end - start) * (1 - weight)`），选中的分支正是抵消误差较小的那个。`addcmul`/`addcdiv` 保留 `value` 参数，`addcdiv` 和 ATen 一样从左到右求值 `value * t1 / t2`，因为换一种结合顺序浮点结果会变。`lerp`、`addcmul`、`addcdiv` 只接受浮点 tensor，整数拼法在 lowering 阶段报错。
- `prod`、`amax`、`amin` 和 `log_softmax` 走 MLX 而不是生成 kernel。三个归约支持任意非空轴集合和 `keepdim`；`amax`/`amin` 与 PyTorch 逐位一致，`prod` 因为乘法顺序不同在浮点上可能相差几个 ulp，整数乘积按位宽回绕。`log_softmax` 是输入减去自身的 log-sum-exp，只接受浮点输入，重复轴在 lowering 期报错。
- `native_layer_norm` 的三个输出都能用。归一化结果仍走专用 kernel；mean 和 rstd 只在图真的读取时才展开成一个 `mean` 加一段两次遍历的方差链（`sub`/`mul`/`mean`/`add eps`/`rsqrt`），因为 IR 的每个 op 只写一个值。它们与 PyTorch 内部的 Welford 统计相差几个 ulp（f32 测试实测在 `1e-5` 以内），只读 `rstd`、不读归一化输出的图也能编译。
- `logsumexp` 是组合 lowering 而不是新 op：`amax`（keepdim）先做稳定化平移，`masked_fill` 在相减前把无穷大的最大值换成 0，再由 `sum`/`log`/`add` 完成归约。这与 ATen 的实现一致，所以 `+inf` 行保持 `+inf`、全 `-inf` 行保持 `-inf`，不会变成 `nan`；这些算子会和相邻 pointwise 区域一起融合。
- `cumprod` 和 `cumsum` 一样走 MLX，同样只带一个 `dim` attribute 并保持 dtype。`diagonal` 降级成 strided view：把两根轴 permute 到末尾、合并成一条，对角线就是这条平轴上的每 `columns + 1` 个元素，因此 `torch.diagonal` 和矩阵上的 `torch.diag` 不复制、不生成 kernel。若 offset 使结果为空，因为静态 IR 不建模零长度维度而报错。`masked_fill` 的 tensor 值（0 维）降级成 `where(mask, value, input)`，就是把 select 的两个操作数换位。
- `elu` 保留 ATen 的 `alpha`/`scale`/`input_scale`，所以 `selu`（scale 1.0507、alpha 1.6733）和手写的 `elu(x, alpha, scale, input_scale)` 都能编译。只有负半支会被 `input_scale` 缩放，与 ATen 一致，测试里直接断言了这一点。
- `count_nonzero`、`all`、`any` 走「先布尔化再计数」的组合 lowering，因为 IR 没有布尔归约：`count_nonzero` 是 `ne(0) -> sum`，输出固定 I64；`all`/`any` 在同一条计数归约后与归约元素数（合取）或 0（析取）比较，输出 BOOL。NaN 按 PyTorch 的布尔转换算作真，输入已经是 BOOL 时省掉 `ne`，因此 `count_nonzero(x > 0)` 只多出一条 `sum`。省略轴的 `x.squeeze()` 是一条删掉所有长度 1 轴的 `reshape`；空目标 shape 只在输入恰好一个元素时接受，与 `argmax()` 全轴归约的 rank-0 规则一致。
- `repeat`/`tile` 展开成 `reshape -> expand -> reshape`，不生成 kernel、不复制：每根轴拆成 `(repeat, size)` 后广播，所以 repeats 比 rank 短或长都按 PyTorch 规则左补 1，全 1 的 repeat 整条消失，`repeat=0` 因为 IR 不建模空维度而在捕获期被拒绝。`vector_norm` 走 MLX：`mx.linalg.norm` 只接受单个轴，所以先把被归约的轴 permute 到末尾合并、调用后再按 `keepdim` reshape 回去，一份实现覆盖任意轴集合。`F.normalize` 就是 `vector_norm -> clamp_min -> expand -> div`，半精度输入先提升到 float32 做归约；`F.cosine_similarity` 是两条 `vector_norm` 加一个 `dot / max(范数乘积, eps)` 的融合 pointwise 区域，它分母上的钳位在 ATen 分解里是 `clamp_min_`，走原地 clamp 的 legalization。
- `var` 和 `std` 走 MLX。三条 ATen 拼法（`correction`、`dim`、`default`）统一归一化成 `dims`/`keepdim`/`correction`，`unbiased` 折算进 `correction`。`correction` 可以是小数：MLX 的 `ddof` 只接受整数，所以 backend 先算有偏方差，再按 ATen 的定义乘上 `n / (n - correction)`（`n` 是实际被归约的元素数）；`correction` 大于等于归约元素数时与 PyTorch 一样得到 `inf`，非有限的 `correction` 在捕获期被拒绝。半精度输入会先提升到 float32 做归约再 cast 回输入 dtype，与 PyTorch 的累加宽度一致。
- `nansum` 与 `nanmean` 同样是组合 lowering。`nansum` 采用 ATen 自己的定义 `masked_fill(isnan(x), 0) -> sum`，所以全 NaN 的切片求和为 0，整数输入直接跳过掩码；带 `dtype` 时先把操作数 `cast` 到目标 dtype，这正是 ATen 的累加宽度。`nanmean` 由 PyTorch 自己展开成 `nansum` 除以对 `x != x` 求出的 `max(count, 1)`，因此落在两个归约加一次融合除法上，全 NaN 的切片保持 NaN。也正因为这条分解，`sum.dim_IntList` 接受 `dim=None` 表示全轴归约：ATen 的分解用 `None` 而不是省略参数。
- `argmax`、`argmin`、`sort`、`topk` 走 MLX 而不是生成 kernel，返回 tuple 的 op 会按实际被读取的输出分别 lowering，所以只读 `sort(...).values` 的图不会计算索引。带 `dim` 的 `torch.max`/`torch.min` 复用同一套机制：values 选择降级成与无 `dim` 形式相同的 `amax`/`amin`，indices 选择降级成 `argmax`/`argmin`，所以 `values, indices = x.max(dim=-1)` 是两次归约，只读 `.values` 就只算一次。索引输出固定 I64，`topk` 两个方向都支持：MLX 的升序窗口会被翻转成 `torch.topk` 的排序输出，`k` 等于轴长时退回排序，因为 MLX 的 select 路径不保证全量窗口的顺序。ties 的索引顺序沿用 MLX，`sort(stable=True)` 的 values 仍一致但索引顺序不一定与 PyTorch 相同；`topk(sorted=False)` 仍返回排序结果，是 PyTorch 承诺的子集。NaN 的位置两个方向都与 PyTorch 一致。
- RMSNorm 的折叠识别 `F.rms_norm` 和它周围常见的手写形态：平方用 `pow(x, 2)` 或 `x * x`，eps 用 `add_` 或非原地 `add`，缩放用 `rsqrt` 或 `sqrt` 加 `div`，weight 在乘法的任意一侧。其他写法会退回普通 elementwise 运算而不是报错，所以上游分解变化只损失一次融合，不影响可编译性。
- 原生 kernel 的覆盖范围仍窄于 MLX。`codegen_matmul` 仍默认关闭只是因为它还在实验阶段，在实测的 FP16/BF16 matmul shape 上都快于 MLX；f32 则是刻意比 MLX 慢——tensor core 没有 f32 模式，所以它留在与 PyTorch 逐位一致的标量 kernel 上。`codegen_attention` 在 prefill 形态上比 MLX 的 SDPA 慢 10%–17%，但在单 query 的 decode 路径上与它持平，因为专用 kernel 用整 head 的 threadgroup 归约取代了分块 online softmax。
- `arange` 对 int32/int64 和 float16/bfloat16/float32 都生成 Metal 常量 kernel。整数输出要求整数边界，走 `long` 路径，因此 2**53 以上的值仍然精确。浮点输出接受任意有限边界：Metal 没有 double，所以编译期把 `start` 和 `step` 拆成 float 头和剩余 float 尾，kernel 用 fma 的乘积残差加 two-sum 重建 `start + index * step`，等价于 PyTorch 那条 double 表达式的一次舍入。整数步长以及 0.1、0.25 这类步长与 PyTorch 逐位一致，其余小数步长与序列相差不超过 1 ulp；PyTorch 自己的 CPU kernel 舍入的是另一个中间量，同样会偏离精确实数。
- 原生 attention 要求 rank-3/4 输入、Q/K/V dtype 一致，并在文档规定的 mask 和 head dimension 约束内支持 MHA/GQA/MQA；其他情况回退到 MLX SDPA。
- Qwen2 静态 cache 适配层只实现 SiLU MLP 激活和普通 causal mask。如果 config 选了别的激活或 sliding-window 层，会在编译期直接拒绝，而不是按错误的语义编译出结果。这些检查读的是构造好的层，而不是 config 字符串——库会把其中一部分字段归一化掉。激活用数值方式校验，因此连 config 描述不到的自定义 `act_fn` 也能抓到。
- 仓库不实现 scheduler、continuous batching、抢占、paged KV cache、prefix cache、采样策略、PD 分离、EP/MoE dispatch、EPLB 或 serving API。
- 不支持训练和 autograd。

- `heaviside` 融合进 pointwise kernel，比较顺序与 ATen 一致：先判 `x == 0` 取边界值，再判 `x > 0` 得 1，其余为 0。因此 `-0.0` 取边界值，NaN 落到 0 而不是边界值。浮点与整数操作数都支持，整数 kernel 使用整数字面量，输出保持操作数的 dtype；bool 操作数会被拒绝。
- bool 上的 `mul`、`add` 就是逻辑与、逻辑或，在同一个 kernel 里生成 `&&`/`||`。ATen 自己就用这两个拼写合并谓词——`isclose` 是两个比较、一个缩放、一个偏移再配合 bool 的 `*` 与 `&`——所以 `torch.isclose` 现在会编译成单个 pointwise kernel，并保持 ATen 的 NaN、无穷和容差规则（`equal_nan=True` 只翻转 NaN 那一路比较）。`add`、`sub`、`div`、`mul` 的 `.Scalar` 拼写，以及被除数为标量的 `remainder`（`2.0 % x`，即 `Scalar_Tensor` schema，符号仍跟随除数）与对应的 tensor 拼写走同一套 lowering。
- 原地拼写与原地 scatter 家族用同一套 legalization：`add_`、`sub_`、`mul_`、`div_`、`remainder_`、`bitwise_and_`/`bitwise_or_`/`bitwise_xor_`/`bitwise_not_`、`logical_not_`，以及 `clamp_`/`clamp_min_`/`clamp_max_`，每种的 `.Tensor` 与 `.Scalar` 形式都支持。被改写的值在图里不能有其他消费者，编译后的调用返回更新后的值，绝不回写调用方仍然持有的 tensor。

## 测试

```bash
python -m pytest -q
ruff check src tests examples
python examples/tiny_llama_compile.py
```

CI 在每次 push 和 pull request 上跑的就是这些：`ruff` 跑在 Linux 上；Apple Silicon 的 macOS runner 上则做 `dev` extra 的可编辑安装、跑测试、跑 Llama 示例、再跑 `python -m build`。托管 runner 的 GPU 是半虚拟化的，装不上 tensor-core kernel，所以那里的 `compile` 会回退到 simdgroup 路径，相关测试也会自行跳过——`pytest -rs` 会打印原因，绿灯只覆盖 simdgroup 路径。


其他示例覆盖 embedding、MLP、matmul kernel、softmax、reduction、normalization、attention 和静态 KV Cache 生成：

```bash
python examples/tiny_transformer.py
python examples/tiny_transformer_generate.py
python examples/metal_matmul_benchmark.py
python examples/metal_attention_benchmark.py
```

首次执行包含 MLX kernel JIT 和 GPU dispatch，首轮耗时不能作为 steady-state benchmark。

PyTorch 对越界的 embedding／gather 下标抛出 `IndexError`，而 MLX 会静默 wrap 或填值，因此后端需要复现该检查。下标值在设备上，所以校验它必然要同步：逐个算子检查会让每个 decode step 产生六次设备同步，改成把越界检查排队累积、最后合并成一次 reduction 之后只剩一次。如果调用方已经知道下标一定在范围内，可以去掉这最后一次同步——`plan(..., verify_index_bounds=False)`，或 `runtime.decode_token(..., verify_token=False)`；greedy 主循环正是这么做的，因为它回填的 token 全部来自词表上的 `argmax`。默认的严格行为没有变化，越界仍然抛 `IndexError`。

## Roadmap

1. 增加符号 shape 和有限控制流，同时保留静态 shape 的严格校验。
2. 把逐 op autotune 扩展成 shape-aware backend policy 和可复现 benchmark report。
3. 提升原生 matmul、attention 的性能，再扩展模型、dtype、算子和 GPU 覆盖；PyTorch 数值对齐仍是验收条件。

## 仓库结构

```text
src/aotolith/   编译器前端、IR、passes、planner、selection 和 Metal backend
tests/                               IR、lowering、数值对齐、生成和 planner 测试
examples/                            可运行的模型、kernel 和静态 KV Cache 示例
docs/architecture.md                 详细实现说明与已知边界
```

## 参与贡献

欢迎提交 issue 和 pull request。修改 lowering 行为时，请附上明确的动机、可复现测试和数值对齐结果。

## License

MIT，详见 [LICENSE](LICENSE)。
