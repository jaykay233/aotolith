# 错误契约

这个包**主动**抛出的每个失败都是 `AotolithError`，并带一个稳定的 code。类回答「调用方该做什么」，code 用来写日志、打 metric 标签、填 issue，改文案时不会变。

```python
import aotolith as aoto
from aotolith import errors

try:
    compiled = aoto.compile_module(model, (input_ids,))
except errors.UnsupportedOperationError as exc:
    print(exc.code, exc)          # AOTO-1001 unsupported FX target: aten.mode.default
    print(exc.hint)               # 下一步怎么做
    fall_back_to_eager()
except errors.CompilationError:
    # 前端或中端做出的任何拒绝
    report_unsupported_model()
```

`format_error(exc)` 把一个 `AotolithError` 渲染成 `[code] message` 加一行 `hint: …`，`aoto` 命令打印的就是这个。

## 错误码

| code | 类 | 含义 | 调用方该做什么 |
| --- | --- | --- | --- |
| `AOTO-0000` | `AotolithError` | 只是基类，不直接抛 | 捕获它表示「编译器主动拒绝」 |
| `AOTO-1000` | `CompilationError` | 这个模型按现在写法编译不了 | 改模型、改编译选项，或者不要用它编这个模型 |
| `AOTO-1001` | `UnsupportedOperationError` | 没有 lowering 的算子、追踪行为或模式 | 把这段放到编译区域外，或补一个 lowering |
| `AOTO-1002` | `DynamicShapeError` | 尺寸或结果长度只有拿到数据才知道 | 从输入 shape 传入 extent，或把算子移出去 |
| `AOTO-1003` | `UnsupportedDTypeError` | 该算子下后端不支持的 dtype | 在边界前 cast，或去掉这个算子 |
| `AOTO-1004` | `UnsupportedModelError` | 这个构建编不了的 checkpoint：另一个模型族、RoPE scaling 模式、sliding window attention | 用有对应 frontend 的族来编，或等适配层；hint 会说明是哪一种 |
| `AOTO-1010` | `IRValidationError` | 图违反了算子契约 | 编译器 bug，或手写的图；请报 issue |
| `AOTO-2001` | `PlanInputError` | 调用的参数与 plan 签名不匹配 | 按 `plan.graph.inputs` 修调用：个数与 shape |
| `AOTO-2002` | `IndexBoundsError` | index 会读到自己索引的张量之外 | 修 index，或确认无误后传 `verify_index_bounds=False` |
| `AOTO-2003` | `PlanDTypeError` | 某个参数 dtype 不被 plan 接受 | cast 这个参数，或用那个 dtype 重编 |
| `AOTO-2004` | `MemoryBudgetError` | 这组配置的内存放不进给定的预算 | 缩短序列、用更小的 `quantize=<group size>` 编，或提高预算 |
| `AOTO-2005` | `BucketRangeError` | 请求比最宽的桶还长 | 编译更宽的桶，或拆分请求；运行期不会重编译 |
| `AOTO-3001` | `DeviceError` | plan 有效，但这台机器跑不了 | 换受支持的 Apple GPU，或离开屏蔽 Metal 的沙箱 |
| `AOTO-4001` | `ArtifactError` | 产物损坏或结构不合法 | 重新编译，或恢复文件 |
| `AOTO-4002` | `ArtifactCompatibilityError` | 这个构建读不了这份产物 | 用写它的编译器，或重新编译 |
| `AOTO-5001` | `PlatformError` | 这个安装在这台机器上不成立 | 修安装：平台，或缺失的 extra |
| `AOTO-6000` | `BenchmarkError` | 性能门禁跑不起来，或用不了给它的东西 | 修门禁的输入，而不是编译器：模型本身没有问题 |
| `AOTO-6001` | `BaselineError` | 基线文件或基线目录不再可信地描述某台机器 | 重新记录、改成它真正记录的那块芯片的文件名，或按当前 schema 重记 |

code 按区域分组：`1xxx` 编译期决定，`2xxx` 调用期决定，`3xxx` 由机器决定，`4xxx` 由已存产物决定，`5xxx` 由安装决定，`6xxx` 由门禁自己的输入决定。

`6xxx` 刻意放在另外五个区域之外：门禁失败不是模型坏、不是调用错、不是设备不行、不是产物损坏、也不是安装不全。常见的 `BaselineError` 是一份提交进仓库的文件不再描述它自称的那块芯片——被改名了、按旧 schema 记录的、或者设备名被手工改过——而修法在文件上。它继承 `ValueError`，因为门禁在有这个类之前抛的就是它。

## 与内置异常的兼容

每个类同时继承上一版抛的内置异常，所以按旧版写的代码继续有效：

| 类 | 同时也是 |
| --- | --- |
| `CompilationError` | `ValueError` |
| `UnsupportedOperationError` | `ValueError`、`NotImplementedError` |
| `UnsupportedModelError` | `ValueError`、`NotImplementedError` |
| `PlanInputError` | `ValueError` |
| `PlanDTypeError` | `TypeError` |
| `IndexBoundsError` | `IndexError` |
| `ArtifactError`、`DeviceError` | `RuntimeError` |
| `PlatformError` | `ImportError` |
| `BenchmarkError`、`BaselineError` | `ValueError` |

这是承诺而不是巧合：编译调用外面的 `except ValueError` 依然能接住所有拒绝。

## 什么不是 `AotolithError`

MLX、Torch 和文件系统自己的异常原样透出 —— 磁盘满了的 `OSError` 不是编译器的决定，包一层只会把它藏起来。唯一的例外是缺少可选依赖：没装 Torch 时 `from aotolith.capture import capture_fx` 抛 `PlatformError`（`AOTO-5001`），因为「安装不完整」比「No module named torch」有用。

## 日志

包往 `logging.getLogger("aotolith")` 写，每个模块一个子 logger。它只挂一个 `NullHandler`，所以 import 不会打印，也不会碰 root logger。宿主按普通库的方式配置：

```python
import logging

logging.getLogger("aotolith").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)   # 或往这个 logger 挂自己的 handler
```

脚本可以用 `aotolith.configure_logging(level)` 这个捷径：挂一个 stderr handler 并设级别，别的什么都不做。`aoto` 命令就是调它，所以 `aoto --log-level INFO compile …` 有效；`AOTOLITH_LOG=DEBUG` 不用加参数也有同样效果。

各层级在这里的含义：

| 级别 | 写什么 |
| --- | --- |
| `DEBUG` | 逐算子决策：每个 value 用了哪个实现、为什么；哪些算子族没有原生候选 |
| `INFO` | 每个会话事件一行：编译摘要、产物写入、产物读取 |
| `WARNING` | 有东西被降级：tensor core 被拒、autotune 没有赢家、读了旧格式产物 |
| `ERROR` | 产出了一个预期会失败的 plan，或一次运行失败 |

同样的信息在进程内也有：`plan.selections` 带每个算子的理由。日志是为了「手上还没有 plan」的那种时刻。
