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

`aoto bench` 是编译器自身的回归门禁：测量一组固定 shape（projection、gated MLP、RMSNorm、causal attention、int4 GEMV，和 Qwen2 prefill／decode 适配层），并可与记录的 baseline 比对：

```bash
aoto bench --baseline benchmarks/baselines/apple-m5-pro.json --output perf-report.json
aoto bench --update-baseline benchmarks/baselines/apple-m5-pro.json
```

它精确比较哪些数字、只在同型号 GPU 上比较哪些数字，见[性能笔记](performance.zh-CN.md#数字是怎么测的)。

## 编译产物

artifact 使用 format version 5 和 ABI version 1.3。两张图、kernel table、selection table、storage plan 和每个 state tensor 都有 SHA-256 digest。加载时会校验 ABI、target、图、入口布局、state shape/dtype、序列化 kernel、重新构建的 selection 和 storage plan。kernel 根据校验后的图和已保存的 selection 重建，因此 artifact 不需要原始 Python 模型，加载时也不会重新 autotune。

是否要求 GPU 完全相同由调用方决定。`--require-device-match` 或 `require_device_match=True` 会拒绝为其他 Apple GPU 型号编译的 artifact；默认允许在其他 Apple Silicon 设备上重新构建 kernel 后执行。

## 越界检查

PyTorch 对越界的 embedding／gather 下标抛出 `IndexError`，而 MLX 会静默 wrap 或填值，因此后端需要复现该检查。下标值在设备上，所以校验它必然要同步：逐个算子检查会让每个 decode step 产生六次设备同步，改成把越界检查排队累积、最后合并成一次 reduction 之后只剩一次。

如果调用方已经知道下标一定在范围内，可以去掉这最后一次同步——`plan(..., verify_index_bounds=False)`，或 `runtime.decode_token(..., verify_token=False)`；greedy 主循环正是这么做的，因为它回填的 token 全部来自词表上的 `argmax`。默认的严格行为没有变化，越界仍然抛 `IndexError`。

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
