# msModelSlim 离群值抑制算法梳理

本文基于 msmodelslim 代码仓，对量化流程中的离群值抑制算法进行整体梳理。覆盖算法包括：

- QuaRot
- SmoothQuant
- IterativeSmooth
- FlexSmoothQuant
- FlexAWQSSZ
- KV Smooth
- Adapt Rotation
- AWQ

## 1. v1 代码组织与入口

本文只梳理 `modelslim_v1` / Processor 体系下的离群值抑制算法。

v1 流程通常通过 YAML 中的 `spec.process` 或多阶段配置中的 `spec.prior` 调度 Processor。算法配置以 `type` 字段标识，例如：

| type | 算法 |
|---|---|
| `smooth_quant` | SmoothQuant |
| `iter_smooth` | IterativeSmooth |
| `flex_smooth_quant` | FlexSmoothQuant |
| `flex_awq_ssz` | FlexAWQSSZ |
| `awq` | AWQ |
| `quarot` | QuaRot |
| `adapt_rotation` | Adapt Rotation |
| `kv_smooth` | KV Smooth |

核心算法分布如下：

| 算法 | 代码路径 |
|---|---|
| SmoothQuant | `msmodelslim/processor/anti_outlier/smooth_quant/` |
| IterativeSmooth | `msmodelslim/processor/anti_outlier/iter_smooth/` |
| FlexSmoothQuant | `msmodelslim/processor/anti_outlier/flex_smooth/` |
| FlexAWQSSZ | `msmodelslim/processor/anti_outlier/flex_smooth/` |
| AWQ | `msmodelslim/processor/anti_outlier/awq/` |
| QuaRot | `msmodelslim/processor/quarot/` |
| Adapt Rotation | `msmodelslim/processor/adapt_rotation/` |
| KV Smooth | `msmodelslim/processor/kv_smooth/` |

文档路径：

```text
docs/zh/quantization_algorithms/outlier_suppression_algorithms/
```

## 2. 算法总览

| 算法 | 核心思想 | 主要处理对象 | 参数搜索 | 主要优势 | 主要代价 |
|---|---|---|---|---|---|
| QuaRot | 使用正交旋转矩阵扩散离群值 | 激活与权重 | 否 | 对低比特激活量化有效，数学等价 | 模型适配复杂，在线旋转可能影响性能 |
| Adapt Rotation | 基于校准数据优化 QuaRot 旋转矩阵 | 激活与权重 | 是 | 比固定 Hadamard 更贴合数据分布 | 两阶段流程复杂，耗时更高 |
| SmoothQuant | 将激活离群值平滑迁移到权重 | Norm-Linear 子图 | 否 | 简单、稳定、开销低 | 子图支持有限，参数固定 |
| IterativeSmooth | 扩展 SmoothQuant 到更多子图 | 多类相邻子图 | 否 | 覆盖更完整，适合复杂结构 | 参数仍偏经验，依赖子图适配 |
| FlexSmoothQuant | 自动搜索 alpha/beta | 多类相邻子图 | 是 | 更灵活，适配不同模型 | 保存激活张量，内存和时间开销更高 |
| FlexAWQSSZ | AWQ + SSZ，真实量化器评估 alpha | 多类相邻子图 | 是 | 更贴近最终量化误差 | 必须配置 qconfig，搜索较慢 |
| KV Smooth | 将 K 的离群值迁移到 Q，保持 `QK^T` 不变 | KVCache 的 Key/Query | 否 | 专门改善 KVCache 量化 | 依赖 Cache/RoPE 结构假设 |
| AWQ | 基于激活均值识别重要权重通道并缩放 | 权重量化相关子图 | 是 | 低比特权重量化成熟有效 | 搜索耗时，主要服务权重保护 |

## 3. 各算法详细梳理

### 3.1 QuaRot

核心路径：

```text
msmodelslim/processor/quarot/offline_quarot/quarot.py
msmodelslim/processor/quarot/common/hadamard.py
msmodelslim/processor/quarot/offline_quarot/laos_online.py
```

核心思想：

QuaRot 使用正交旋转矩阵，例如 Hadamard 矩阵，对激活和权重做等价变换，将少数通道上的极端离群值分散到多个通道中。

典型等价关系：

```text
XW = (XQ)(Q^T W)
```

其中 `Q` 为正交矩阵，满足：

```text
Q Q^T = I
```

优点：

- 能从分布形态上削弱激活离群值，对 W4A4、W8A8 等低比特激活量化很有价值。
- 正交变换保持数学等价，不改变浮点模型表达能力。
- 可与 AutoRound、LinearQuant 等后续量化算法组合。
- 支持离线旋转，也支持部分模型的在线旋转。

缺点：

- 适配成本较高，需要模型实现 `QuaRotInterface`。
- Hadamard 矩阵维度受支持列表限制，特殊 hidden size 可能需要补充矩阵。
- 在线旋转依赖推理引擎支持额外算子，可能带来推理性能下降。
- 张量并行部署时对 `tp_size` 有约束。

改进点：

- 增强 Hadamard 维度自动生成或降级能力。
- 扩展更多模型适配模板。
- 对在线旋转做按层选择、按收益启用，避免全量插入额外算子。
- 加强 TP 并行配置校验与导出侧提示。

适用场景：

- 极低比特激活量化。
- 激活离群值明显且 SmoothQuant 类算法效果不足。
- 可接受较高模型适配成本的高精度场景。

### 3.2 Adapt Rotation

核心路径：

```text
msmodelslim/processor/adapt_rotation/adapt_rotation.py
msmodelslim/processor/adapt_rotation/adapt_rotation_stage1.py
msmodelslim/processor/adapt_rotation/adapt_rotation_stage2.py
msmodelslim/processor/adapt_rotation/iterative_hadamard_optimization.py
```

核心思想：

Adapt Rotation 是 QuaRot 的增强版。它不是直接使用固定 Hadamard 矩阵，而是使用校准数据优化旋转矩阵，使旋转后的激活在量化和反量化后的重构误差更低。

两阶段流程：

1. Stage 1：收集指定层激活，基于初始 Hadamard 矩阵迭代优化旋转矩阵。
2. Stage 2：读取 Stage 1 产生的优化矩阵，替换 QuaRot 默认旋转矩阵，并执行 QuaRot 流程。

优点：

- 旋转矩阵由校准数据驱动，比固定 Hadamard 更贴合实际模型分布。
- 对 W4A4 等极低比特场景更有潜力。
- 复用 QuaRot 的旋转、融合和导出机制。

缺点：

- 流程复杂，需要 prior stage、ContextManager 和主阶段配合。
- 对校准集质量敏感。
- 需要模型实现 `AdaptRotationInterface`，该接口继承 `QuaRotInterface`，还需要提供 hidden dim。
- MoE 模型若匹配到大量专家层，激活收集和优化会明显变慢。

改进点：

- 增加早停机制，避免无收益迭代。
- 支持分层共享或分组共享旋转矩阵，降低计算量。
- 对 MoE 专家层增加默认过滤策略。
- 增加 Stage1/Stage2 配置一致性校验，例如 `quant_dtype` 与下游激活量化 dtype 是否一致。

适用场景：

- 对精度要求极高的 W4A4/W8A8 场景。
- QuaRot 已有效但仍需进一步压缩误差。
- 有足够校准数据和离线处理时间。

### 3.3 SmoothQuant

核心路径：

```text
msmodelslim/processor/anti_outlier/smooth_quant/
```

核心思想：

SmoothQuant 通过数学等价缩放，把激活中的离群值迁移到权重中，使激活更容易量化。

公式：

```text
Y = XW = (X / s)(sW)
scales = A_scale^alpha / W_scale^(1-alpha)
```

其中：

- `A_scale` 为激活每通道尺度，通常是绝对最大值。
- `W_scale` 为权重每通道尺度。
- `alpha` 控制激活和权重之间的平衡，默认常见值为 `0.5`。

优点：

- 算法简单、稳定、容易理解。
- 推理侧没有额外算子开销，缩放可融合进权重或归一化层。
- 对 W8A8 激活离群值抑制是非常常用的基础方案。
- 实现和调试成本低。

缺点：

- Processor 版本主要支持 `norm-linear` 子图，覆盖范围不如 IterativeSmooth。
- 固定 alpha 难以适配所有层。
- 将离群值迁移到权重后，可能增加权重量化难度。
- 对复杂结构，如 OV、Up-Down、Linear-Linear，表达能力有限。

改进点：

- 支持按层 alpha 或自动 alpha 搜索。
- 扩展更多子图类型。
- 与真实量化误差评估结合，而不是只依赖统计量。

适用场景：

- 标准 Transformer 的 W8A8 量化。
- 初始量化精度不够，需要低成本平滑激活。
- 模型结构比较标准，norm-linear 映射清晰。

### 3.4 IterativeSmooth

核心路径：

```text
msmodelslim/processor/anti_outlier/iter_smooth/
msmodelslim/processor/anti_outlier/common/scale_computation.py
```

核心思想：

IterativeSmooth 可以视为 SmoothQuant 的增强版。它仍然通过相邻模块间的等价缩放迁移离群值，但支持更多子图类型。

支持子图：

- `norm-linear`
- `linear-linear`
- `ov`
- `up-down`
- `non-fusion`

公式：

```text
scales = A_scale^alpha / W_scale^(1-alpha)
```

默认 `alpha` 在 Processor 配置中为 `0.9`。

优点：

- 子图覆盖远大于 SmoothQuant。
- 可处理 attention 的 V/O 投影关系和 MLP 的 Up/Down 投影关系。
- 支持非融合子图，可对无法融合到前置层的 Linear 通过 pre-hook 保持等价。
- 对复杂 Transformer 结构更实用。

非融合子图方案：

- 将模型的前向前面挂上钩子hook，然后走到时，将激活乘上一个1/scales，这样就等价于融合了

缺点：

- 子图适配要求更高，需要模型实现 `IterSmoothInterface`。
- alpha 仍然主要依赖经验配置。
- 多子图之间存在处理顺序和潜在交互，需要谨慎过滤 include/exclude。
- 非融合路径依赖 hook，部署导出时需要额外关注。

改进点：

- 引入按子图类型的默认 alpha 策略。
- 增加自动 alpha 搜索或小样本误差评估。
- 强化子图冲突检测，避免同一模块被多个子图重复缩放。
- 对 non-fusion 路径提供更明确的导出兼容性检查。

适用场景：

- SmoothQuant 覆盖不足的复杂模型。
- 需要对 attention 与 MLP 内部连接做离群值迁移。
- W8A8 或更低比特激活量化前处理。

### 3.5 FlexSmoothQuant

核心路径：

```text
msmodelslim/processor/anti_outlier/flex_smooth/
msmodelslim/processor/anti_outlier/flex_smooth/alpha_beta_search.py
msmodelslim/processor/anti_outlier/common/scale_computation.py
```

核心思想：

FlexSmoothQuant 将 SmoothQuant 中固定的 `alpha` 与隐含的 `1-alpha` 拆成两个可独立控制的参数 `alpha` 和 `beta`。

公式：

```text
scales = A_scale^alpha / W_scale^beta
```

如果用户没有配置 `alpha` 或 `beta`，算法会做二阶段搜索：

1. 搜索 alpha，此时 beta 通常取 `1-alpha`。
2. 固定最佳 alpha 后继续搜索 beta。

优点：

- 比 SmoothQuant 和 IterativeSmooth 更灵活。
- 自动搜索参数，能适配不同层、不同模型结构。
- 支持多类子图：`norm-linear`、`linear-linear`、`ov`、`up-down`、`non-fusion`。
- 对 MQA/GQA 等 attention 结构有专门 scale 规约逻辑。

缺点：

- 搜索需要保留激活张量，内存压力比只保存统计量更大。
- 搜索过程有额外计算开销。
- 搜索目标使用模拟量化，不一定完全等同最终真实量化器误差。
- 参数空间扩大后，结果更依赖校准数据代表性。

改进点：

- 对激活张量做采样、分块或流式统计，降低内存。
- 引入真实量化器评估或与 qconfig 联动。
- 支持按子图类型设置搜索范围和步长。
- 增加搜索结果缓存，避免重复校准。

适用场景：

- 固定 alpha 表现不稳定的模型。
- 不同层离群值分布差异较大。
- 有一定离线搜索时间预算，追求更稳精度。

### 3.6 FlexAWQSSZ

核心路径：

```text
msmodelslim/processor/anti_outlier/flex_smooth/
msmodelslim/processor/anti_outlier/flex_smooth/alpha_beta_search.py
```

核心思想：

FlexAWQSSZ 结合 AWQ 的激活感知思想和 SSZ 权重量化思想，使用真实 `LinearQuantizer` 评估不同 alpha 下的量化误差。

公式：

```text
scales = Act_Mean_Abs^alpha / Weight_Max_Abs^beta
```

在当前设计中：

- 激活尺度使用 `mean(abs(act))`，不是 max。
- `beta` 通常固定为 `0`。
- 如果未配置 `alpha`，在 `[0, 1]` 范围内搜索。
- 需要提供 `qconfig`，通常权重方法使用 `ssz`。

优点：

- 使用真实量化器评估，和最终量化部署更一致。
- 激活均值比激活最大值更稳定，适合低比特权重量化中的通道重要性评估。
- 搜索空间比 FlexSmoothQuant 更小，因为 beta 固定为 0。
- 适合与 SSZ 权重量化配合。

缺点：

- 必须提供完整 qconfig，配置成本高于 FlexSmoothQuant。
- 每个候选 alpha 都要调用量化器评估，耗时较高。
- 当前实现与 FlexSmoothQuant 共享接口，但语义不同，使用时容易混淆。
- 对 qconfig 的质量敏感，错误配置会直接影响搜索结果。

改进点：

- 增加 qconfig 推荐模板。
- 对常用 qconfig 组合做搜索结果缓存。
- 支持动态调整搜索步长，先粗搜再细搜。
- 提供更清晰的日志，区分 FlexSmoothQuant 与 FlexAWQSSZ 的参数含义。

适用场景：

- W4/W8 低比特权重量化。
- 权重量化方法使用 SSZ。
- 需要让离群值抑制参数直接服务最终量化误差最小化。

### 3.7 KV Smooth

核心路径：

```text
msmodelslim/processor/kv_smooth/
```

核心思想：

KV Smooth 专门处理 KVCache 量化中的 Key 离群值问题。它将 `key_states` 的离群值迁移到 `query_states`，保持注意力分数不变。

等价关系：

```text
K' = K / s
Q' = Q * s
Q'K'^T = QK^T
```

由于推理时通常量化写入 KVCache 的 `key_states` 和 `value_states`，而不量化 `query_states`，因此把离群值迁移到 Q 一侧可以降低 KVCache 量化误差。

优点：

- 专门针对 KVCache 量化，目标明确。
- 可以降低 Key 的动态范围，改善 KVCache int8/int4 量化效果。
- 保持 `QK^T` 数学等价。
- 对长序列推理和 KVCache 显存优化有意义。

缺点：

- 依赖模型 Attention 前向支持 `past_key_values` 或 `past_key_value`。
- 依赖 `Cache.update()` 观测 key states。
- 默认假设 RoPE 成对通道结构。
- 如果 query_states 也被量化，则迁移到 Q 的离群值可能产生新问题。
- 仅支持 `Linear/Norm -> RoPE -> KVCache` 两类融合路径。

改进点：

- 增强对自定义 Cache 实现的兼容。
- 增加非 RoPE 或变体 RoPE 的适配策略。
- 当 Q 也参与量化时，引入 Q/K 双侧约束优化。
- 对 smooth_factor 做自动调优。

适用场景：

- KVCache 量化。
- 长上下文推理。
- Key 离群值显著导致注意力退化。

### 3.8 AWQ

核心路径：

```text
msmodelslim/processor/anti_outlier/awq/
msmodelslim/processor/anti_outlier/awq/best_scales_search.py
```

核心思想：

AWQ 即 Activation-aware Weight Quantization。它认为并非所有权重通道同等重要，使用激活均值衡量通道重要性，并搜索缩放因子保护重要通道。

公式：

```text
scales = act_mean^ratio
scales = scales / sqrt(scales.max() * scales.min())
```

搜索流程：

1. 收集目标 Linear 输入激活的 `mean(abs(act))`。
2. 找到目标模块的最低公共祖先模块，用于块级误差评估。
3. 在 `ratio in [0, 1)` 上做网格搜索。
4. 对候选 scale 应用权重缩放、权重量化、反向缩放。
5. 比较块级输出与浮点输出的 MSE，选择最佳 ratio。

优点：

- 对低比特权重量化非常有效，尤其是 W4A16/W8A16 等场景。
- 使用激活均值度量通道重要性，比单纯看权重分布更合理。
- 块级输出 MSE 评估比单层权重误差更接近模型行为。
- 搜索参数少，主要是 ratio。

缺点：

- 搜索需要反复运行祖先模块，耗时较高。
- 主要服务权重量化，对激活量化离群值的直接抑制不如 SmoothQuant/QuaRot。
- 依赖 LCA 自动发现和参数缓存，模型结构特殊时可能失效。
- 对校准样本覆盖度有要求。

改进点：

- 与 FlexAWQSSZ 的真实量化器评估逻辑统一。
- 增加更鲁棒的祖先模块发现策略。
- 支持按层自适应 `n_grid`。
- 对高敏感层做细搜，对低敏感层做粗搜以降低耗时。

适用场景：

- W4A16、W8A16 等权重量化。
- 激活不量化或激活量化压力较小，但权重量化误差明显。
- 需要保护重要权重通道的低比特模型压缩。

## 4. 横向差异分析

### 4.1 旋转类与缩放类

旋转类：

- QuaRot
- Adapt Rotation

特点：

- 通过正交矩阵重新分布离群值。
- 对激活量化尤其有效。
- 模型适配和部署复杂度较高。

缩放类：

- SmoothQuant
- IterativeSmooth
- FlexSmoothQuant
- FlexAWQSSZ
- AWQ
- KV Smooth

特点：

- 通过相邻模块之间的等价缩放迁移离群值。
- 更容易融合进权重或归一化层。
- 适配成本相对可控。

### 4.2 固定参数与搜索参数

固定参数为主：

- SmoothQuant
- IterativeSmooth
- KV Smooth
- QuaRot

搜索参数为主：

- FlexSmoothQuant：搜索 alpha/beta。
- FlexAWQSSZ：搜索 alpha，beta 通常固定为 0。
- AWQ：搜索 ratio。
- Adapt Rotation：优化旋转矩阵。

### 4.3 激活统计方式差异

| 算法 | 激活统计 |
|---|---|
| SmoothQuant | 每通道 abs max |
| IterativeSmooth | 每通道 max/min/abs max/shift |
| FlexSmoothQuant | 保存激活张量并计算 abs max |
| FlexAWQSSZ | 使用 `mean(abs(act))` |
| AWQ | 使用 `mean(abs(act))` |
| KV Smooth | 观测 key_states 的每通道 abs max |
| QuaRot | 不以传统 scale 统计为核心，使用旋转矩阵扩散离群 |
| Adapt Rotation | 使用校准激活优化旋转矩阵 |

### 4.4 子图支持差异

| 算法 | 支持子图 |
|---|---|
| SmoothQuant | 主要为 `norm-linear` |
| IterativeSmooth | `norm-linear`、`linear-linear`、`ov`、`up-down`、`non-fusion` |
| FlexSmoothQuant | `norm-linear`、`linear-linear`、`ov`、`up-down`、`non-fusion` |
| FlexAWQSSZ | `norm-linear`、`linear-linear`、`ov`、`up-down` |
| AWQ | `norm-linear`、`linear-linear`、`ov`、`up-down` |
| KV Smooth | `state-rope-linear`、`state-rope-norm` |
| QuaRot | 由模型适配器提供旋转映射 |
| Adapt Rotation | 继承 QuaRot 映射，并额外需要 hidden dim |

### 4.5 典型精度与性能取舍

| 目标 | 推荐算法 |
|---|---|
| 低成本 W8A8 平滑 | SmoothQuant |
| 更完整的 Transformer 子图平滑 | IterativeSmooth |
| 自动搜索平滑参数 | FlexSmoothQuant |
| 低比特权重量化且使用 SSZ | FlexAWQSSZ |
| 经典低比特权重量化 | AWQ |
| 极低比特激活量化 | QuaRot |
| 极致精度旋转优化 | Adapt Rotation |
| KVCache 量化 | KV Smooth |

## 5. 选型建议

### 5.1 W8A8 基础量化

优先选择：

```text
SmoothQuant 或 IterativeSmooth
```

如果模型结构标准且只需要处理 Norm 到 Linear 的激活离群，SmoothQuant 足够简单稳定。

如果模型中 attention 和 MLP 内部投影也需要平滑，优先使用 IterativeSmooth。

### 5.2 固定 alpha 效果不稳定

优先选择：

```text
FlexSmoothQuant
```

它可以自动搜索 alpha/beta，适合不同层分布差异明显的模型。

### 5.3 W4A16 或低比特权重量化

优先选择：

```text
AWQ
```

AWQ 对权重量化通道保护更直接，尤其适合激活不量化或激活量化压力较小的场景。

### 5.4 W4/W8 权重量化并使用 SSZ

优先选择：

```text
FlexAWQSSZ
```

它使用真实量化器评估 alpha，对最终量化误差更敏感。

### 5.5 W4A4 或极低比特激活量化

优先选择：

```text
QuaRot 或 Adapt Rotation
```

QuaRot 适合作为旋转类基础方案。若有足够校准时间并追求更高精度，可以进一步使用 Adapt Rotation。

### 5.6 KVCache 量化

优先选择：

```text
KV Smooth
```

KV Smooth 是专门面向 KVCache 的方案，不应简单用普通 SmoothQuant 替代。

## 6. 总结

这些算法可以按目标分成四类：

| 类别 | 算法 | 主要目标 |
|---|---|---|
| 基础平滑 | SmoothQuant、IterativeSmooth | 降低激活离群值对 W8A8 等量化的影响 |
| 自动搜索平滑 | FlexSmoothQuant、FlexAWQSSZ、AWQ | 用校准数据搜索更优 scale |
| 旋转抑制 | QuaRot、Adapt Rotation | 从分布层面扩散离群值，服务低比特激活量化 |
| KVCache 专项 | KV Smooth | 降低 KVCache Key 的量化动态范围 |

简要结论：

- 要简单稳定：选 SmoothQuant。
- 要覆盖更多子图：选 IterativeSmooth。
- 要自动调参：选 FlexSmoothQuant。
- 要低比特权重量化：选 AWQ。
- 要结合 SSZ 和真实量化器：选 FlexAWQSSZ。
- 要极低比特激活量化：选 QuaRot 或 Adapt Rotation。
- 要 KVCache 量化：选 KV Smooth。
