# msModelSlim 量化算法整体梳理

本文基于 `C:\workspace\msmodelslim` 当前代码仓，对仓内主要量化/压缩算法进行整体梳理，覆盖：

- AutoRound
- FA3 Quant
- Float Sparse
- GPTQ
- Histogram
- KV Cache Quant
- LAOS
- Linear Quant
- MinMax
- PDMIX
- SSZ

## 1. 总体架构

msModelSlim 中量化能力大致分为三层：

| 层级 | 作用 | 典型路径 |
|---|---|---|
| Processor | 负责算法流程编排、模块替换、校准/部署阶段切换 | `msmodelslim/processor/quant`、`msmodelslim/processor/sparse` |
| Quantizer | 负责具体量化参数统计、权重量化、激活伪量化 | `msmodelslim/core/quantizer` |
| IR / FakeQuant | 负责伪量化模块、部署态模块、保存格式 | `msmodelslim/ir`、`msmodelslim/core/quant_service` |

从使用视角看：

- `linear_quant` 是最通用的线性层量化处理器。
- `minmax`、`histogram`、`ssz`、`gptq` 是可被 `linear_quant` 调用的底层量化算法。
- `autoround` 是训练型低比特权重量化处理器。
- `fa3_quant`、`dynamic_cache`、`pdmix` 面向 Attention / KV Cache / 推理阶段差异等特定场景。
- `float_sparse` 是稀疏化压缩方案，不是传统整数低比特量化。
- `laos` 是由 Adapt Rotation 与 AutoRound 组合形成的 W4A4 方案。

## 2. 算法代码地图

| 算法/方案 | 代码主入口 | 核心类/对象 | 定位 |
|---|---|---|---|
| Linear Quant | `msmodelslim/processor/quant/linear.py` | `LinearQuantProcessor` | 线性层量化统一处理器 |
| MinMax | `msmodelslim/core/quantizer/impl/minmax.py` | `ActPerTensorMinmax`、`ActPerTokenMinmax`、`WeightPerChannelMinmax` | 基础统计量化 |
| Histogram | `msmodelslim/core/quantizer/impl/histogram.py` | `ActPerTensorHistogram` | 激活值直方图截断量化 |
| SSZ | `msmodelslim/core/quantizer/impl/ssz.py` | `WeightPerChannelSsz` | 权重量化参数迭代优化 |
| GPTQ | `msmodelslim/core/quantizer/impl/gptq.py` | `WeightPerChannelGPTQ`、`WeightPerGroupGPTQ` | Hessian 二阶误差补偿权重量化 |
| AutoRound | `msmodelslim/processor/quant/autoround.py` | `AutoroundQuantProcessor` | 低比特舍入参数训练 |
| FA3 Quant | `msmodelslim/processor/quant/fa3/processor.py` | `FA3QuantProcessor` | Attention 激活 per-head 量化 |
| KV Cache Quant | `msmodelslim/processor/quant/attention.py` | `DynamicCacheQuantProcessor` | KV Cache INT8 伪量化 |
| PDMIX | `msmodelslim/core/quantizer/impl/minmax.py` | `ActPDMixMinmax` | Prefill 动态 + Decode 静态激活量化 |
| Float Sparse | `msmodelslim/processor/sparse/float_sparse.py`、`msmodelslim/processor/sparse/admm.py` | `FloatSparseProcessor`、`AdmmPruner` | 浮点权重稀疏化 |
| LAOS | YAML 组合方案 | `adapt_rotation` + `autoround_quant` | Qwen3 W4A4 低比特方案 |

## 3. 核心算法对比

### 3.1 Linear Quant

`linear_quant` 是基础处理器，不是单一量化数学算法。它扫描模型中的 `torch.nn.Linear` 层，根据 `qconfig` 创建 `LinearQuantizer`，并在后处理阶段将量化模块 deploy 成部署态模块。

核心特点：

- 支持 `include` / `exclude` 通配符筛选层。
- 支持权重与激活分别配置量化粒度、数据类型、对称性和算法。
- 激活侧常见方法包括 `minmax`、`histogram`。
- 权重侧常见方法包括 `minmax`、`ssz`、`gptq`。
- 支持分布式共享层同步逻辑，但取决于底层量化器是否支持。

优点：

- 是 msModelSlim 中最通用的线性层量化入口。
- 配置灵活，能表达 W8A8、W4A8、PDMIX、MXFP 等多类方案。
- 处理逻辑清晰，便于叠加不同底层量化算法。

缺点：

- 只处理 `nn.Linear`，对 Attention 内部激活、KV Cache 等特殊路径无能为力。
- 合法配置组合依赖底层 registry 与校验，用户容易写出不支持的组合。
- include/exclude 如果未命中，只能通过日志发现，缺少结构化策略报告。

改进点：

- 输出每层最终量化策略报告，包含命中规则、量化配置、是否回退。
- 增加配置组合自动诊断工具。
- 增加推荐配置模板，例如 W8A8 静态、W8A8 动态、W4A8 SSZ、PDMIX 等。

### 3.2 MinMax

MinMax 是最基础的量化算法，通过统计张量最小值和最大值计算 scale / offset。

核心实现：

- `ActPerTensorMinmax`
- `ActPerTokenMinmax`
- `ActPerChannelMinmax`
- `WeightPerChannelMinmax`
- `MXWeightPerBlockMinmax`
- `MXActPerBlockMinmax`

优点：

- 计算开销最低，速度快。
- 实现简单，适合作为基线。
- 支持激活和权重，多种粒度，包括 per-tensor、per-token、per-channel、MX per-block。

缺点：

- 对离群值非常敏感。
- 在低比特场景下，大值会拉大量化区间，使主体数据有效精度下降。
- 不主动优化量化误差。

适用场景：

- 常规 W8A8 基线。
- 激活动态量化 per-token。
- 权重 int8 per-channel。
- 对精度要求不是极端苛刻、希望快速得到结果的场景。

改进点：

- 支持 percentile / clipping 策略。
- 与 SmoothQuant、IterSmooth、Rotation 等离群值抑制方案自动联动。
- 为不同模型结构提供默认 aggregation 策略。

### 3.3 Histogram

Histogram 是激活量化算法，通过统计激活值直方图并搜索更优截断区间，减少离群值对量化范围的污染。

核心实现：

- `msmodelslim/core/quantizer/impl/histogram.py`
- `msmodelslim/core/observer/histogram.py`
- 量化器：`ActPerTensorHistogram`

优点：

- 相比 MinMax 更能处理离群值。
- 静态激活量化场景下通常精度更好。
- 支持 L2 范数误差搜索，代码中也有 KL 散度搜索能力。

缺点：

- 当前主要作为激活 `per_tensor int8` 量化使用。
- 统计与搜索成本高于 MinMax。
- 不适合动态 per-token 场景。

适用场景：

- 静态 W8A8 激活量化。
- 激活分布存在明显离群值，但又希望避免训练型优化的场景。

改进点：

- 暴露 L2 / KL 搜索选择配置。
- 支持更多粒度，如 per-channel 或分组直方图。
- 输出截断比例和量化误差，方便分析。

### 3.4 SSZ

SSZ 是权重量化算法，通过 MinMax 得到初始量化参数后，迭代搜索更优 scale / offset，以降低量化误差。

核心实现：

- `msmodelslim/core/quantizer/impl/ssz.py`
- 核心函数：`ssz_calculate_qparam`
- 量化器：`WeightPerChannelSsz`

优点：

- 相比 MinMax 能进一步优化权重量化误差。
- 对权重分布不均的层更友好。
- 适合 int4 / int8 per-channel 权重量化。
- 不依赖激活 Hessian，复杂度低于 GPTQ。

缺点：

- 当前代码注册主要是 `int8_per_channel_sym` 和 `int4_per_channel_sym`。
- 不支持 per-group / per-tensor。
- 相比 MinMax 有额外迭代成本。

代码观察：

- 文档中提到默认迭代次数为 20、最小 scale 为 `1e-5`。
- 当前代码中 `SCALE_SEARCH_ITER_NUM = 50`，`SCALE_SEARCH_MIN_SCALE = 1e-30`。
- 当前代码支持通过 `config.ext.step` 覆盖迭代次数。

适用场景：

- W4A8 中的 int4 权重量化。
- 权重分布不均但不希望承担 GPTQ 激活统计成本的场景。

改进点：

- 修正文档与代码参数不一致问题。
- 支持 per-group。
- 增加每层 SSZ 收敛日志与误差改善统计。

### 3.5 GPTQ

GPTQ 是权重量化优化算法，利用激活值构建 Hessian 矩阵，通过逐列/分块量化和误差补偿降低输出误差。

核心实现：

- `msmodelslim/core/quantizer/impl/gptq.py`
- `WeightPerChannelGPTQ`
- `WeightPerGroupGPTQ`

核心参数：

- `percdamp`：Hessian 阻尼系数，默认 `0.01`。
- `block_size`：逐块处理列数，默认 `128`。
- `group_size`：per-group 量化分组大小，默认 `128`。

优点：

- 利用二阶信息，精度潜力高。
- 支持 per-channel 和 per-group。
- 支持对称和非对称量化。
- 对低比特权重量化有较强理论优势。

缺点：

- 依赖校准数据中的激活输入。
- 需要计算和分解 Hessian，速度慢、内存压力较大。
- `support_distributed()` 返回 `False`，不支持 DP 分布式量化。
- MoE 场景需要校准集覆盖所有专家，否则专家权重缺乏有效统计。

代码观察：

- 文档说当前暂不支持 int4。
- 但当前代码 registry 已注册：
  - `int4_per_channel_sym`
  - `int4_per_channel_asym`
  - `int4_per_group_sym`
  - `int4_per_group_asym`
- 因此需要统一文档、测试和用户说明。

适用场景：

- 对权重量化精度要求高的模型。
- 校准数据质量较好、可接受较大量化耗时的场景。

改进点：

- 补齐 int4 GPTQ 的测试与文档口径。
- 优化 Hessian 内存占用。
- 增加 MoE 专家覆盖率检查。
- 对 block_size / group_size 做自动建议。

### 3.6 AutoRound

AutoRound 是低比特权重量化优化算法，通过引入可学习的舍入偏移参数，并用 SignSGD 训练舍入方向，从而降低重构误差。

核心实现：

- `msmodelslim/processor/quant/autoround.py`
- `msmodelslim/processor/quant/autoround_utils`
- 核心处理器：`AutoroundQuantProcessor`

优点：

- 适合 W4A4、W4A8 等低比特场景。
- 能通过训练优化舍入方式，精度通常优于简单 round。
- 支持按层策略配置，可做混合量化。

缺点：

- 需要训练/迭代，量化耗时明显高于 MinMax/SSZ。
- 对显存和校准数据要求高。
- 单独使用风险较高，低比特场景通常需要配合离群值抑制。

适用场景：

- 极低比特权重量化。
- 对精度要求高且可接受额外量化成本的场景。
- 与 QuaRot、Adapt Rotation、IterSmooth 等组合使用。

改进点：

- 自动识别敏感层并回退到 W8A8 或 FP。
- 输出训练收敛曲线与误差改善报告。
- 默认提供与 Rotation/Smooth 的组合 recipe。

### 3.7 FA3 Quant

FA3 Quant 面向 Attention 激活，通常用于 Flash Attention 3 / MLA 场景，对 Q/K/V 等激活进行 per-head 量化。

核心实现：

- `msmodelslim/processor/quant/fa3/processor.py`
- `FA3QuantProcessor`
- `_FA3PerHeadObserver`
- `FA3QuantAdapterInterface`
- `FA3QuantPlaceHolder`

核心流程：

1. 通过模型 adapter 在 Attention 关键路径注入 `FA3QuantPlaceHolder`。
2. 校准阶段将占位模块替换为 per-head observer。
3. 根据每个 head 的 min/max 计算量化参数。
4. 后处理阶段替换为 `FakeQuantActivationPerHead` 或 `FakeQuantActivationPerToken`。

优点：

- 针对 Attention head 维度建模，粒度比全局激活量化更细。
- 适合长序列、MLA 架构、Attention 激活显存压力大的场景。
- 可与 linear quant 组合形成更完整的量化方案。

缺点：

- 强依赖模型 adapter，通用性弱于 linear_quant。
- 当前更偏 DeepSeek / MLA 适配。
- `support_distributed()` 返回 `False`。

适用场景：

- DeepSeek-V3 / R1 类 MLA 模型。
- 长序列 Attention 激活量化。

改进点：

- 扩展更多模型 adapter。
- 增加 adapter 注入点检查工具。
- 给出真实后端性能收益和显存收益验证。

### 3.8 KV Cache Quant

KV Cache Quant 对大模型推理时缓存的 `key_states` 和 `value_states` 做 INT8 量化，以降低长序列 KV Cache 显存压力。

核心实现：

- `msmodelslim/processor/quant/attention.py`
- `DynamicCacheQuantProcessor`
- `msmodelslim/core/quantizer/attention.py`
- `DynamicCacheQuantizer`

核心流程：

1. 检测 Attention 层。
2. 拦截 `transformers.cache_utils.DynamicCache.update()`。
3. 对 `key_states` / `value_states` 做量化统计。
4. 后处理阶段注入 fake quant cache 模块。

优点：

- 针对长序列推理中的关键显存瓶颈。
- 对上层推理逻辑侵入较低。
- 可与线性层量化组合。

缺点：

- 当前仅支持 `per_channel int8`。
- 依赖 `DynamicCache.update()` 接口和 `layer_idx`。
- 伪量化阶段不一定真实节省显存，真实收益需要底层算子和存储格式支持。
- 不支持分布式。

适用场景：

- Qwen2.5 / Qwen3 等使用 Transformers DynamicCache 的模型。
- 长上下文推理。

改进点：

- 支持真实 INT8 KV Cache 存储。
- 适配更多自定义 Cache。
- 改进 Attention 层识别逻辑，减少依赖命名模式。

### 3.9 PDMIX

PDMIX 是激活值阶段间混合量化方案：Prefilling 阶段使用动态 per-token 量化，Decoding 阶段使用静态 per-tensor 量化。

核心实现：

- `msmodelslim/core/quantizer/impl/minmax.py`
- `ActPDMixMinmax`
- 对应 IR：`msmodelslim/ir/w8a8_pdmix.py`

核心思想：

- Prefilling：上下文输入长、分布变化大，用 per-token 动态量化降低精度损失。
- Decoding：逐 token 输出阶段性能敏感，用 per-tensor 静态量化降低运行时开销。

优点：

- 在精度和性能之间折中。
- 相比纯静态量化，长上下文场景下精度更稳。
- 相比纯动态量化，decoding 阶段性能更好。

缺点：

- 当前配置组合较窄。
- 主要支持 W8A8 PDMIX。
- 当前文档说明主要支持 MindIE 推理部署。
- 阶段间权重量化方式必须一致，否则要存两份权重。

适用场景：

- 生成式大模型推理。
- 静态 W8A8 精度损失较大，但纯动态性能不满足要求的场景。

改进点：

- 支持更多后端。
- 支持更多权重量化算法组合。
- 自动判断哪些层适合 PDMIX、哪些层应保持动态或回退。

### 3.10 Float Sparse

Float Sparse 是浮点稀疏化方案，核心是基于 ADMM 和 Hessian 统计寻找稀疏权重模式。

核心实现：

- `msmodelslim/processor/sparse/float_sparse.py`
- `msmodelslim/processor/sparse/admm.py`
- `FloatSparseProcessor`
- `AdmmPruner`

核心流程：

1. 对 Linear 层安装 hook，收集输入激活统计。
2. 构建 Hessian 近似。
3. 通过 ADMM 迭代优化稀疏权重。
4. 结合重要性保护机制，避免关键权重被过度稀疏。

优点：

- 能提高模型压缩率。
- 有激活统计和 Hessian 信息辅助，较盲目剪枝更稳。
- 可服务于硬件压缩单元。

缺点：

- 量化/稀疏化耗时高。
- 需要额外显存存储 Hessian 和统计信息。
- 依赖特定硬件压缩单元才能获得真实部署收益。
- 当前只支持 `nn.Linear`。
- 不适合和 W8A8S 稀疏量化叠加使用。

适用场景：

- 高压缩率部署。
- 目标硬件支持稀疏压缩收益。

改进点：

- 自动搜索 sparse_ratio。
- 优化 Hessian 内存。
- 增加失败降级机制。
- 与量化方案形成互斥/兼容性检查。

### 3.11 LAOS

LAOS 是 W4A4 低比特组合方案，不是单个底层量化器。它的核心是：

1. 使用 Adapt Rotation 抑制离群值。
2. 使用 AutoRound 做低比特权重量化和舍入优化。
3. 通过分层策略对敏感层使用 W8A8，非敏感层使用 W4A4。

核心配置特点：

- `prior` 阶段执行 `adapt_rotation stage 1`。
- `process` 阶段执行 `adapt_rotation stage 2`。
- 再执行 `autoround_quant`。
- 针对 Qwen3 dense 模型配置层级策略。

优点：

- 专门解决 W4A4 下离群值与舍入误差问题。
- 精度通常优于单独 AutoRound 或单独 Rotation。
- 通过层级混合量化降低敏感层风险。

缺点：

- 计算成本高。
- 显存要求高，文档中说明需要 NPU 显存不低于 64G。
- 当前主要面向 Qwen3 dense，不保证泛化到其他模型。
- 配置复杂，对层名和模型结构强依赖。

适用场景：

- Qwen3 dense W4A4。
- 极致压缩且能接受较长量化时间的场景。

改进点：

- 将 LAOS recipe 抽象成更通用的策略生成器。
- 自动识别敏感层并生成 W8A8/W4A4 混合策略。
- 扩展到更多模型架构。

## 4. 方案差异总结

| 维度 | MinMax | Histogram | SSZ | GPTQ | AutoRound | FA3 Quant | KV Cache Quant | PDMIX | Float Sparse | LAOS |
|---|---|---|---|---|---|---|---|---|---|---|
| 类型 | 基础量化 | 激活截断量化 | 权重量化优化 | 权重量化优化 | 训练型低比特量化 | Attention 激活量化 | Cache 量化 | 阶段混合激活量化 | 稀疏化 | 组合方案 |
| 主要对象 | 权重/激活 | 激活 | 权重 | 权重 | 权重为主 | Q/K/V 激活 | K/V Cache | 激活 | 权重 | 全流程 |
| 是否依赖校准数据 | 低/中 | 高 | 低 | 高 | 高 | 高 | 高 | 高 | 高 | 高 |
| 计算成本 | 低 | 中 | 中 | 高 | 高 | 中 | 中 | 高 |
| 低比特友好度 | 一般 | 一般 | 较好 | 较好 | 很好 | 特定场景 | 特定场景 | W8A8 为主 | 非低比特量化 | 很好 |
| 通用性 | 高 | 中 | 中 | 中 | 中 | 低/中 | 中 | 中 | 中 | 低 |
| 主要收益 | 快速基线 | 抑制激活离群值 | 降低权重误差 | 二阶误差补偿 | 优化舍入 | Attention 激活优化 | 长序列显存优化 | 精度/性能折中 | 压缩率 | W4A4 精度 |

## 5. 推荐使用路径

### 5.1 快速基线

推荐：

- `linear_quant + minmax`
- W8A8 动态或静态量化

原因：

- 配置简单。
- 量化速度快。
- 方便作为后续算法对比基线。

### 5.2 激活离群值明显

推荐：

- 静态激活：`histogram`
- 动态激活：`per_token minmax`
- 配合 SmoothQuant / IterSmooth / Rotation 类算法

### 5.3 权重低比特但不想用训练型优化

推荐：

- W4A8：`linear_quant + weight ssz`
- 或在校准数据充足时尝试 `gptq`

### 5.4 极低比特 W4A4

推荐：

- `AutoRound`
- 更推荐 `LAOS = Adapt Rotation + AutoRound`

注意：

- 不建议在没有离群值抑制的情况下单独使用 AutoRound 做 W4A4。
- 需要充足校准数据和较高显存。

### 5.5 长序列推理

推荐：

- `fa3_quant`：优化 Attention 激活。
- `dynamic_cache`：优化 KV Cache。
- `pdmix`：平衡 prefill 精度与 decode 性能。

### 5.6 高压缩率部署

推荐：

- `float_sparse`

注意：

- 需要确认目标硬件和后端是否能利用稀疏压缩收益。
- 不建议与已有 W8A8S 稀疏量化叠加。

## 6. 当前代码与文档不一致点

### 6.1 GPTQ int4 支持口径不一致

文档中说明 GPTQ 当前暂不支持 int4，但代码中已注册：

- `int4_per_channel_sym`
- `int4_per_channel_asym`
- `int4_per_group_sym`
- `int4_per_group_asym`

建议：

- 如果 int4 GPTQ 已可用，应更新文档并补充测试。
- 如果仍不建议用户使用，应在代码校验层显式拦截，而不是只在文档中说明。

### 6.2 SSZ 默认参数不一致

文档中写到：

- `SCALE_SEARCH_ITER_NUM = 20`
- `SCALE_SEARCH_MIN_SCALE = 1e-5`

当前代码中为：

- `SCALE_SEARCH_ITER_NUM = 50`
- `SCALE_SEARCH_MIN_SCALE = 1e-30`

建议：

- 更新文档。
- 补充 `ext.step` 参数说明。

## 7. 总体改进建议

1. 建立量化能力矩阵：自动从 registry 枚举 `dtype/scope/symmetric/method` 支持关系，生成文档和测试。
2. 增加量化策略报告：每层命中哪个 processor、哪个 qconfig、最终是否部署成功。
3. 对高成本算法提供自动选择：先跑 MinMax 基线，再根据误差选择 Histogram、SSZ、GPTQ、AutoRound。
4. 强化 FA3 / KV Cache / PDMIX 的后端收益验证：区分伪量化精度验证与真实部署显存/性能收益。
5. 对 LAOS 这类组合方案 recipe 化：自动生成分层策略，降低手写 YAML 的难度。
6. 增加配置合法性提示：在 YAML 解析阶段给出更清晰的修复建议。
