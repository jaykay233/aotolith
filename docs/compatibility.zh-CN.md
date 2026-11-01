# 兼容与弃用

版本之间调用方能依赖什么，以及必须改变时怎么办。一句话：编译产物的格式和 Python API 以不同速度演进，而且都不会悄悄变。

## 四个面

| 面 | 承诺 | 变化方式 |
| --- | --- | --- |
| **产物格式**（`.aoto`、`format_version` / `abi_version`） | 0.x 写出的文件，在同一大版本线内的后续 0.x 都能读，只要 reader 保留兼容分支 | 只有 reader 同时能读上一版时才写新格式版本；去掉 reader 分支需要次版本号加一条 `CHANGELOG` 记录 |
| **Python API**（`aotolith.__all__`：`compile_module`、`compile_callable`、`compile_graph`、`CompiledPlan`、错误类……） | 整个 0.x 线内存在且可用 | 先弃用，弃用后至少两个次版本才删除，补丁版本永不删除 |
| **错误契约**（`AotolithError.code`） | 一旦写进 `docs/errors.md`，code 的含义不变 | 同一个 code 永不复用于另一种失败；不再抛出的 code 先在目录里标弃用，再删除 |
| **CLI**（`aoto compile/inspect/verify/run/bench`、参数名、退出码） | 参数含义不变；`0` 成功，`1` 报告了失败，`2` 命令行写错了 | 删除参数前至少有一个版本在用到时警告 |

没有从 `aotolith/__init__.py` 导出的东西 —— passes、codegen 家族、`_` 开头的内部辅助 —— 都是实现细节，任何版本都可能变。架构测试约束的是这些模块的 import 方向，不是它们的内容。

桶索引（`*.buckets.json`）属于产物这个面，它自带 `buckets_index_version`，理由和 `.aoto` 带 `format_version` 一样：reader 拒绝自己读不懂的版本，而且只有上一版仍能被读时才写新版本。它点名的 archive 就是普通 `.aoto`，按普通规则演进；索引里的条目读不出来的表现，和单份产物读不出来一致。

## 什么算破坏性变更

下面这些是破坏性的，需要次版本号加 `CHANGELOG` 里的 **Changed** 或 **Removed** 条目：

- `aotolith.__all__` 里的函数或类改签名，导致原来能通过的调用被拒；
- 上一版写出的产物不再能加载；
- 错误在 `docs/errors.md` 的五个区域之间移动（`1xxx` → `2xxx`），因为那会改变调用方的分支；
- 同一产物、同一输入、同一设备，算出的数不同。

下面这些不算：

- 错误文案重写（类和 code 不变）；
- 一个拒绝换成同一区域内更具体的类；
- capture 或 legalization 新增一条检查，拒绝了后端本来就无法正确执行的图 —— 旧行为是错误答案或 kernel 里崩掉，不是一个能用的 plan；
- 某算子新增原生 kernel，改变的是速度不是结果；
- 启发式选择换了实现，这本来就由 `plan.selections` 逐次记录。

第三条要小心，边界是「上一版产出过正确的 plan」。如果一张图以前能编译并运行，新的拒绝就是破坏性的；如果它以前在更没用的地方失败，就不是。

## 弃用一个东西

1. 在读者看得见的地方留标记 —— Python 符号用 `DeprecationWarning`，运行时行为用 `WARNING` 日志，CLI 参数用帮助文本里的说明。没有提示就不叫弃用。
2. 在 `CHANGELOG.md` 的 **Deprecated** 下记一条，写清替代方案和会删除的版本。
3. 保持它能用。改变了行为的弃用就是删除。
4. 至少两个次版本后再删，并在删除的那个版本把条目挪到 **Removed**。

`DeprecationWarning` 默认不显示，所以同一件事也写日志：不用 `-W error::DeprecationWarning` 的用户也能在记录里看到一行。

## 版本号

`0.x` 是 1.0 之前，所以第二个数字承载一切：

- `0.63.0 → 0.63.1` —— 只修 bug。不改 API、不改产物、不改任何 plan 算出的数。
- `0.63.0 → 0.64.0` —— 其他一切：新功能、新拒绝、启发式变化、弃用、删除。
- `1.0.0` —— 从这一点起，产物格式和上面的 API 跨大版本线成为承诺，而不是跨次版本。

## 跨设备的产物

产物记录编译所在 GPU（`profile.device_name`、`profile.architecture`）。默认**不**要求匹配：Metal kernel 在加载时从校验过的图和存下的 MSL 重新构建，所以在 M3 Pro 上编译的 plan 能在 M5 Pro 上跑。会变的是速度 —— 启发式选择是按另一块 GPU 做的 —— 而记录会说明这一点。`load_compiled_module(..., require_device_match=True)` 是严格模式，给宁可失败也不跑未测量 plan 的部署用。

## 当前已知的限制

写在这里，因为这些是 API 里最可能变的部分：

- **静态 shape。** 每个 plan 按固定 shape 编译，prompt 长度和 KV cache 大小是产物的一部分。`aotolith.models.qwen2` 和 `aotolith.models.llama` 里有 prompt 桶；符号 shape 哪里都没有。
- **两个 target，都是 MLX 形状。** `apple-metal` 与 `mlx-reference` 是仅有的后端；`compile_graph(target=…)` 会直接拒绝别的，不假装支持。两者都离不开 MLX，所以「另一个后端」意味着「另一块设备」，这一条仍然敞着。
- **两个模型族。** `qwen2` 与 `llama` 是静态 Cache 适配层验证过的 decoder 族。第三个 `model_type` 会用 `UnsupportedModelError`（AOTO-1004）拒绝，而不是猜一个，所以增加一族属于「能编什么」的兼容性变化，不是悄悄放宽。
- **decoder adapter 的 batch 为 1。** IR 支持 batch 维；adapter 的 cache 和生成循环假设只有一条序列。
- **format 4 读支持** 存在是因为它只花一个分支。1.0 之后的某个次版本会删掉，删之前先有弃用日志。

## 0.64.0 的说明

- `memory=` 是 `CompiledPlan.__call__` 与 `CompiledModulePlan.__call__` 上新增的 keyword-only 参数。它有默认值，所以已有的调用不受影响；这个名字是保留的，模型不能把同名 tensor 关键字提升成输入（编译期会拒绝，和 `verify_index_bounds` 并列）。
- `Qwen2StaticCacheRuntime` 去掉了 `_decode_masks` 数组，新增 `decode_mask(valid_length)`。那个字段带下划线，按上面的规则属于实现；新方法返回的正是旧表里那一行。
- benchmark report schema 变成 2。`bench.compare` 遇到 schema 不同的 baseline 会直接拒绝，而不是去比两批对不上的数字，所以旧的 baseline 要用 `aoto bench --update-baseline` 重录。
- `aotolith.planning` 新增 `tensor_nbytes`，`DTYPE_ITEMSIZE` 从 `codegen.expr` 移到 `ir.types`（这两处都属于实现面）。

## 0.67.0 的说明

- **第二个模型族和第二个后端，都不改变已有的那一个。** `aotolith.models.llama` 是新增的；`aotolith.models.qwen2` 保留了它原有的全部名字，因为 Qwen2 的入口现在调用的是 `aotolith.models.decoder` 里的共享引擎。`Qwen2PrefillModule` 一类是 decoder 类的别名，所以调用方唯一可能察觉的是 `isinstance` 和 `type(...).__name__`，而公开面没有承诺过这两者。
- **`target=` 是新增参数**，在 `compile_graph`、`compile_callable`、`compile_module` 以及两个适配层上。它有默认值，所以已有调用不受影响。产物现在记录自己是为哪个 target 编的，并拒绝当作另一个加载；0.66.0 写出的产物没有 `target` 字段，会被读作 `apple-metal`，这正是它们本来的身份。
- **`AOTO-1004` / `UnsupportedModelError` 是新增的**，与 `AOTO-1001` 同在 `1xxx` 区域。它现在抛出的每一处以前都是裸的 `NotImplementedError`；新类同时继承 `NotImplementedError`，所以按 0.66.0 写的 handler 继续有效。这属于上面说的「同一区域内更具体的类」，不算破坏性。
- **新增 `AOTO-6000` / `BenchmarkError` 与 `AOTO-6001` / `BaselineError`**，单独占 `6xxx` 区域：基线文件或目录不可信是门禁自身的问题，不是编译器失败。两者都继承 `ValueError`，也就是 `bench.resolve_baseline` 和 `bench.validate_baselines` 在有这两个类之前抛的东西，所以没有既有的处理器会改变含义。
- **`benchmarks/baselines` 变成一个目录，`--baseline` 接受它。** 旧的那个单文件按路径给出时照旧可用，目录则按设备名解析，所以写成 `benchmarks/baselines/apple-m5-pro.json` 的工作流行为不变。`--update-baseline` 给目录时按芯片命名写入，而不是写到给的那个路径；给路径时与之前完全一致。
- **`prefill_mask` 的边界被拆开。** 非正的长度仍然抛 `PlanInputError`；比桶宽的长度现在抛 `BucketRangeError`（`AOTO-2005`），因为那是「请求装不下」而不是「参数写错了」。两者都是 `ValueError` 的子类，而 `BucketRangeError` 同时也是 `PlanInputError`，所以没有既有的 `except` 会改变含义。
- **基准门禁为每个模型族各加一项，为参考后端加了一项。** 0.67.0 之前录的 baseline 没有这些条目；`bench.compare` 把它们报成 `no baseline entry, not compared` 而不是失败，所以旧 baseline 仍然卡着它覆盖过的部分。要卡新的那几项，用 `aoto bench --rounds 3 --update-baseline` 重录。
