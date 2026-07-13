# ops-nn 基础量化/反量化：完全掌握指南

> 生成日期：2026-07-13
> 分析模式：`code-reader-v2-cn` Deep / 策略 C
> 分析范围：`ops-nn/quant` 39 个算子族、1195 个文件、约 18.4 万行源码

## 理解验证状态

| 核心概念                                   | 能解释 WHAT | 能解释 WHY | 能手算示例 | 能关联 ops-nn | 建议状态     |
| ------------------------------------------ | ----------: | ---------: | ---------: | ------------: | ------------ |
| affine 量化公式                            |          ✅ |         ✅ |       必须 |          必须 | 第一优先级   |
| scale/zero-point 两套约定                  |          ✅ |         ✅ |       必须 |          必须 | 第一优先级   |
| 静态量化与动态量化                         |          ✅ |         ✅ |       必须 |          必须 | 第一优先级   |
| per-tensor/per-channel/per-token/per-block |          ✅ |         ✅ |       建议 |          必须 | 第一优先级   |
| INT8/INT4 与 FP8/FP4/MX                    |          ✅ |         ✅ |       建议 |          必须 | 第二优先级   |
| 反量化、INT32 累加和 bias                  |          ✅ |         ✅ |       必须 |          必须 | 第一优先级   |
| Fake Quant、Nudge、STE                     |          ✅ |         ✅ |       建议 |          必须 | 第二优先级   |
| API→Host→Tiling→Kernel                  |          ✅ |         ✅ |     不要求 |          必须 | 工程面试重点 |

## 1. 快速概览与面试学习路径

### 1.1 一句话心智模型

量化本质上是在做“坐标系变换”：用一个有限、低精度的编码集合表示原来的浮点数。`scale` 决定网格间距，`zero-point/offset` 决定网格原点落在哪里，`round` 决定连续值落到哪个格点，`clamp/saturate` 决定超出编码范围时如何处理。动态量化只是把“如何选网格”从离线阶段搬到运行时；MX 量化则让一组数共享一个指数尺度，再用 FP8/FP4 表示块内相对值。

### 1.2 建议学习顺序

| 阶段                     | 学习目标                                      | 对应算子                                                            | 面试验收问题                                         |
| ------------------------ | --------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------- |
| 第 1 阶段：线性量化数学  | 手算 scale、zero-point、round、clamp、dequant | `quantize`、`ascend_quant_v2`、`ascend_anti_quant_v2`         | 为什么两份文档里的 scale 一个做除法、一个做乘法？    |
| 第 2 阶段：量化粒度      | 理解精度、元数据量和计算开销的权衡            | `group_quant`、`dynamic_quant`、`dynamic_block_quant`         | per-token 为什么适合大模型激活？                     |
| 第 3 阶段：动态量化      | 理解运行时归约、amax/min/max、对称与非对称    | `dynamic_quant`、`dynamic_quant_v2`                             | 全零行怎样避免除零？非对称 offset 如何推导？         |
| 第 4 阶段：低比特浮点/MX | 理解 FP8/FP4、共享尺度、blocksize 和双轴      | `dynamic_mx_quant`、`dynamic_block_mx_quant`、`anti_mx_quant` | MX 与普通 INT8 per-block 有何本质差异？              |
| 第 5 阶段：训练与校准    | 理解 Fake Quant、Nudge、STE、mask             | `fake_quant_*`、`acts_ulq`、`ifmr`                            | 为什么假量化输出仍是浮点？梯度为什么只在范围内通过？ |
| 第 6 阶段：硬件工程      | 理解参数打包、Tiling、尾块、多核和融合        | `trans_quant_param*`、`dequant_swiglu_quant`                    | 为什么 scale 要转换成 UINT64？为什么融合能提速？     |
| 第 7 阶段：模拟面试      | 能用 3 分钟讲完整主线，用 15 分钟讲源码       | 全部代表算子                                                        | 从公式、算子、Kernel、测试四层回答同一问题           |

### 1.3 七天复习计划

1. **第 1 天：公式。** 手算三组 INT8 affine 量化：对称、非对称、含异常值。要求能够从实数区间推导 `scale` 和 `zero-point`，再做反量化并解释误差来自哪里。
2. **第 2 天：静态与动态。** 阅读 `quantize/README.md`、`ascend_quant_v2/README.md`、`dynamic_quant_v2/README.md`，画出三者参数来源不同但计算骨架相同的流程图。
3. **第 3 天：粒度。** 用一个 `[B, T, H]` 激活张量分别描述 per-tensor、per-channel、per-token、per-block 的 scale shape，比较精度和元数据成本。
4. **第 4 天：MX。** 阅读 `dynamic_mx_quant` 与 `anti_mx_quant`，讲清共享尺度、FP8_E8M0 scale、FP4 打包、blocksize=32 的原因和尾块处理。
5. **第 5 天：QAT。** 手推 Nudge，理解为什么零点需要被精确表示；阅读梯度算子，解释范围内 STE、范围外梯度为零的意义。
6. **第 6 天：工程。** 选一个算子沿 `op_api → op_graph → op_host → op_kernel → tests` 走通，重点看参数校验、tiling key 和尾块测试。
7. **第 7 天：模拟面试。** 对照文末题目闭卷回答。每题先用一句话结论，再给公式，再联系具体算子，最后补充工程边界。

## 2. 背景与动机

### 2.1 WHY 需要量化

大模型推理的瓶颈常常不是纯计算，而是权重和激活在显存、HBM、片上缓存之间的搬运。把 FP16/BF16 压到 INT8、INT4、FP8 或 FP4，可以降低存储与带宽压力，并让硬件使用低精度矩阵乘单元获得更高吞吐。如果没有量化，模型可能首先受制于容量和带宽，即使计算单元仍有余量也无法继续提速。

量化不是免费午餐。编码集合变少后会出现舍入误差、截断误差和饱和误差，所以工程问题不是“是否有误差”，而是“用什么粒度和格式，把误差控制在业务允许范围内，同时让硬件真正获得收益”。

### 2.2 WHY 有多种量化方案

- **静态量化**把 scale/offset 提前算好，运行时开销低、行为稳定，但面对输入分布漂移容易损失精度。
- **动态量化**在运行时统计当前 token/块的范围，精度适应性更好，但多了一次归约和 scale 计算。
- **对称量化**通常只需要 scale，零点固定，计算与硬件实现简单；但当分布明显偏斜时会浪费一侧编码空间。
- **非对称量化**利用 min/max 覆盖偏斜分布，编码利用率更高，但需要 offset，参数和计算更复杂。
- **INT8/INT4**是均匀网格，适合矩阵乘整数通路；**FP8/FP4**保留指数，动态范围更大，但有效精度随数值大小变化；**MX**再让块共享尺度，进一步平衡范围、精度和元数据。

### 2.3 WHY ops-nn 要拆成多个算子

量化粒度、目标 dtype、芯片能力、输入布局和融合上下游不同，会改变输出 shape、scale shape、Tiling 和 Kernel。把所有路径塞进一个巨型算子会让参数约束、二进制编译和性能选择变得不可维护。因此仓库使用“算子族 + 版本演进”的方式：基础公式相近，但在 API、图定义、Host Tiling 和 Kernel 模板上分别落地。

## 3. 核心概念网络

### 3.1 affine 量化的两套 scale 约定

这是本仓库最重要的面试陷阱。

**步长约定（dequant scale）**：

```text
q = round(x / s + z)
x_hat = (q - z) * s
```

这里 `s` 是相邻量化格点在实数域中的距离。`quantize` 使用的公式属于这一类：`out = round(x / scales + zeroPoints)`。

**乘法器约定（quant multiplier）**：

```text
q = round(x * m + offset)
```

这里 `m = 1 / s`。`ascend_quant_v2` 使用这一类公式。两种写法没有数学冲突，但 API 参数语义不同，不能看到变量都叫 scale 就直接代入同一公式。

**WHY 会同时存在？** 硬件量化指令往往天然执行乘法，传入乘法器可以省掉运行时除法；而算法文献和反量化接口更习惯把 scale 定义成实数步长。面试时应先声明约定，再写公式。

### 3.2 对称与非对称量化

对称动态量化常用：

```text
s = max(abs(x)) / qmax
q = round(x / s)
```

`dynamic_quant` 的 per-token 路径就是这一思路。对称量化的好处是零点固定为 0，矩阵乘和反量化更简单；缺点是分布若主要在正区间，负数编码会被浪费。

非对称动态量化常用：

```text
s = (xmax - xmin) / (qmax - qmin)
z = qmax - xmax / s
q = round(x / s + z)
```

这与 `z = qmin - xmin/s` 等价。`dynamic_quant_v2` 文档使用这一公式。实现必须对 `xmax == xmin` 做保护，否则 scale 为零；Kernel 中通常通过 epsilon 或最小 scale 钳位规避。

### 3.3 量化粒度关系矩阵

| 粒度        |   scale 数量 | 精度           | 运行/存储成本          | 代表算子                             | WHY 这样选                         |
| ----------- | -----------: | -------------- | ---------------------- | ------------------------------------ | ---------------------------------- |
| per-tensor  |            1 | 最低           | 最低                   | `quantize`、`ascend_quant`       | 分布较稳定或只追求简单快速         |
| per-channel |       通道数 | 较高           | 中等                   | `ascend_quant_v2`、`group_quant` | 权重各输出通道范围差异明显         |
| per-token   | token/row 数 | 高             | 需要在线归约           | `dynamic_quant`                    | LLM 不同 token 激活幅度差异大      |
| per-group   |    组/专家数 | 高             | 需要 group index       | `grouped_dynamic_*`                | MoE 专家分布不同，不能共用 scale   |
| per-block   |         块数 | 很高           | scale 元数据和归约更多 | `dynamic_block_quant`              | 局部范围比整行范围更紧，适合低比特 |
| MX block    |         块数 | 高且动态范围大 | 特殊 scale 格式与布局  | `dynamic_mx_quant`                 | 块共享尺度 + 浮点尾数适配 FP8/FP4  |

### 3.4 误差网络

| 误差              | 来源                        | 典型缓解方式                    | 对应算子线索                         |
| ----------------- | --------------------------- | ------------------------------- | ------------------------------------ |
| 舍入误差          | 连续值映射到离散格点        | 更细粒度 scale、合理 round mode | `ascend_quant_v2`                  |
| 饱和误差          | 超出 qmin/qmax              | 校准范围、clamp、异常值处理     | `fake_quant_*`、`ifmr`           |
| 范围浪费          | 对称量化遇到偏斜分布        | 非对称 offset                   | `dynamic_quant_v2`                 |
| 局部异常值污染    | 一个大值拉大整张量 scale    | per-token/per-block/group       | `dynamic_block_quant`              |
| scale 量化误差    | scale 自身用低精度格式表示  | MX 算法、指数向上取整           | `dynamic_mx_quant`                 |
| 累加溢出/精度损失 | INT8×INT8 通常累加到 INT32 | INT32 累加后再反量化            | `ascend_dequant`、`dequant_bias` |

### 3.5 INT8/INT4、FP8/FP4 与 MX 的关系

INT8/INT4 使用均匀整数编码，格点等距，特别适合整数矩阵乘；FP8/FP4 保留指数和尾数，格点不等距，靠近零的分辨率和大数范围之间做浮点式权衡。MX 不是简单的另一种 dtype，它给一个 block 配共享缩放尺度，块内元素再用 FP8/FP4 表示。这样既避免每个元素都携带完整高成本指数，又比全张量共享一个 scale 更能适应局部分布。

## 4. 算法与理论总览

### 4.1 静态量化

- **时间复杂度：** O(N)，每个元素一次缩放、偏移、舍入和饱和。
- **空间复杂度：** 输出 O(N)，附加元数据取决于粒度。
- **WHY 可接受：** 量化通常与读写内存同步发生，属于线性流式算子；与后续矩阵乘相比计算量小。
- **退化场景：** 校准数据不能代表线上分布，或者异常值使 scale 过大，导致主体数据分辨率下降。

### 4.2 动态量化

- **时间复杂度：** O(N)，但一般包含范围归约和量化两个逻辑阶段。
- **空间复杂度：** 输出 O(N)，scale 为 O(行数/块数)。
- **WHY 选择：** 对输入分布变化敏感的激活，动态 scale 能显著降低分布漂移造成的饱和或范围浪费。
- **退化场景：** 行很短时归约开销占比高；异常值仍会污染整行；极端小范围需要 epsilon 保护。

### 4.3 block/MX 量化

- **时间复杂度：** O(N)，每个 block 做归约并量化。
- **附加空间：** O(N/B)，B 为 blocksize。
- **WHY 选择：** 更小的 block 能收紧局部动态范围，使 FP4/FP8 在极少位宽下仍有可用精度。
- **退化场景：** block 太小会增加 scale 元数据和访存；block 太大又接近 per-tensor，局部精度优势消失；尾块和 scale 布局会增加实现复杂度。

### 4.4 Fake Quant 与 STE

Fake Quant 前向执行 quant-dequant，因此输出仍是浮点，但数值被限制在量化网格上。直接对 round 求导几乎处处为零，网络无法训练，所以反向通常使用 Straight-Through Estimator：在有效量化范围内把梯度近似当成 1，范围外设为 0。它不是严格数学导数，而是让训练能够感知量化误差的工程近似。

## 5. 工程设计模式

### 5.1 分层管线模式

`op_api → op_graph → op_host → op_kernel` 把公共接口、图语义、运行时策略和设备实现分开。WHY 要分层：同一个数学算子可能跨多个 SoC 和 dtype，但 API 语义应该稳定；Host 可以根据 shape 和平台选择不同 tiling key；Kernel 可以为全载、分片、尾块和不同 dtype 提供专门模板。若把这些逻辑混在一起，版本演进和性能调优会互相牵连。

### 5.2 策略选择模式

Tiling 根据输入 shape、axis、dtype、是否有 offset、量化粒度和 SoC 选择 Kernel 路径。它本质上是运行时策略选择：小 shape 可全载 UB，大 shape 要分片或双缓冲；尾轴量化和非尾轴量化的数据访问模式不同。缺少这一层，单一 Kernel 虽然功能正确，但很难在所有形状上获得稳定性能。

### 5.3 模板特化模式

Kernel 经常按输入 dtype、输出 dtype、量化类型、round mode 和 tiling key 做编译期特化。WHY 不完全运行时分支：AI Core 内核对寄存器、UB、指令路径和流水高度敏感，编译期裁剪能减少分支并让编译器生成更直接的指令序列。代价是二进制组合增多，因此 Host 配置必须严格限制合法组合。

### 5.4 算子融合模式

`DequantSwigluQuant`、`SwiGluQuant`、`DynamicQuantUpdateScatter` 等把多个逻辑步骤放进一次 Kernel。主要收益通常不是少做几次乘加，而是少一次中间 Tensor 落到全局内存、少一次 Kernel launch、少一次图调度。融合过度也会增加寄存器/UB 压力，因此应以数据流和硬件资源为边界。

### 5.5 面试回答模板

回答任何量化问题时按四层展开：

1. **一句话结论：** 先说明方案解决什么问题。
2. **公式：** 声明 scale 约定，再写 quant/dequant 公式。
3. **粒度和 dtype：** 说明 scale 的 shape、目标类型、精度与成本权衡。
4. **工程实现：** 联系 ops-nn 的 API、Tiling、Kernel、尾块和测试。

例如回答“动态量化为什么比静态量化准”：动态量化在运行时按 token 或 block 统计范围，因此当前输入能更充分使用低精度编码空间；公式仍是 affine quant，只是 scale/offset 的来源从离线校准变成在线归约。代价是增加归约和 scale 输出，工程上需要专门的 Tiling、多核归约和全零范围保护，`dynamic_quant`/`dynamic_block_quant` 就体现了这些权衡。

## 6. 关键代码深度解析与算子族专题

### 核心片段清单（6A）

| 编号 | 片段                                | 位置                                                       | 优先级 | 选择理由                                             |
| ---- | ----------------------------------- | ---------------------------------------------------------- | -----: | ---------------------------------------------------- |
| #1   | DynamicQuantV2 per-channel 参数生成 | `dynamic_quant_regbase_perchannel_full_load.h:372-421`   | ★★★ | 同时体现归约轴、对称/非对称和 scale 两套语义。       |
| #2   | DynamicBlockQuant 单行量化          | `dynamic_block_quant_single_row_kernel.h:330-392`        | ★★★ | 展示 block amax、scale 输出、内部乘法器和饱和 Cast。 |
| #3   | DynamicMxQuant OCP scale 生成       | `dynamic_mx_quant_tail_axis.h:797-862`                   | ★★★ | 展示 shared exponent 如何编码成 E8M0 scale。         |
| #4   | AntiMxQuant FP8 反量化              | `anti_mx_quant_tail_axis.h:476-517`                      | ★★★ | 展示 scale 解码、广播、尾轴反量化和 dtype 转换。     |
| #5   | FakeQuant Nudge 预计算              | `fake_quant_with_min_max_args_tiling_arch35.cpp:110-145` | ★★★ | 展示零点对齐、独立 scaleInv 和精度敏感的 Host 计算。 |

以下三个专题覆盖 `ops-nn/quant` 的全部 39 个算子目录。每个专题先建立算子族差异，再深入真实 Kernel/Host 片段，最后给出测试边界和面试问答。

### 专题一：01. 静态与动态整数数量化

> 本章目标：面试时不仅能背出 `q = round(x / scale + zero_point)`，还要能回答：这个仓库里的 `scale` 到底是乘数还是除数、量化参数按什么粒度共享、动态参数沿哪条轴归约、取整和饱和在哪里发生，以及 Host tiling 为什么要为同一公式准备多套 Kernel。

#### 1. 先建立统一心智模型

量化代码最容易产生误解的地方，不是公式本身，而是不同接口对 `scale` 一词使用了两套相反的约定。

- **反量化尺度（step、delta）**：

  $$
  q=\operatorname{round}(x/\Delta+z),\qquad \hat{x}=(q-z)\Delta
  $$

  `Quantize`、`DynamicQuant`、`DynamicQuantV2` 的输出 `scale`、`DynamicBlockQuant` 的输出 `scale`都应按这个方向理解。
- **量化乘数（multiplier）**：

  $$
  q=\operatorname{round}(x\cdot m+z),\qquad m=1/\Delta
  $$

  `AscendQuant`、`AscendQuantV2`、`GroupQuant`、`QuantMax`、`GroupedQuantMax` 的输入 `scale`更接近这个含义。

所以，看到 `scale` 后第一问必须是：**它在公式里乘 x，还是除 x？它是外部给定，还是运行时从极值计算出来？** 如果这两点不先说清楚，后面的“per-channel”“动态量化”很容易全部讲反。

##### 1.1 十个目录的覆盖地图

| 目录                            | 核心算子/接口                                         | 参数来源                        | 粒度                                                         | 对称性                              | 核心公式与定位                                                               |
| ------------------------------- | ----------------------------------------------------- | ------------------------------- | ------------------------------------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------- |
| `ascend_quant`                | `AscendQuant`                                       | 静态给定标量属性                | per-tensor                                                   | `offset=0` 可视为对称，否则非对称 | `round(x * scale^(1或2) + offset)`；旧式标量乘数接口                       |
| `ascend_quant_v2`             | `AscendQuantV2` / `aclnnAscendQuant(V3)`          | 静态给定 Tensor                 | per-tensor 或指定 axis 的 per-axis                           | offset 可选                         | 在 AscendQuant 基础上增加 Tensor scale/offset 和 axis，axis 主要限制在末两维 |
| `quantize`                    | `Quantize` / `aclnnQuantize`                      | 静态给定 Tensor                 | per-tensor 或 per-axis（tiling 再区分 per-channel/per-head） | zeroPoints 可选                     | 标准形式`round(x / scales + zeroPoints)`                                   |
| `group_quant`                 | `GroupQuant`                                        | 静态给定`[E,H]` scale         | per-group-per-channel                                        | offset 是全局标量，可选             | 按累计`group_index` 为行选择专家 scale，`round(x * scale[e,h] + offset)` |
| `dynamic_quant`               | `DynamicQuant` / `aclnnDynamicQuant`              | 在线统计                        | per-token                                                    | 对称                                | 每行`amax` 决定 `scale=amax/qmax`；可先乘 smooth scale                   |
| `dynamic_quant_v2`            | `DynamicQuantV2`，由 DynamicQuant V2/V3/V4 API 调用 | 在线统计                        | per-token / per-tensor / per-channel                         | 对称或非对称                        | V2 是动态量化主实现，支持 offset 输出、MOE smooth scale 和多种低比特类型     |
| `dynamic_block_quant`         | `DynamicBlockQuant(V2)`                             | 在线统计                        | per-block                                                    | 对称                                | 最后两维按`(rowBlockSize,colBlockSize)` 分块，每块输出一个反量化尺度       |
| `grouped_dynamic_block_quant` | `GroupedDynamicBlockQuant(V2)`                      | 在线统计                        | per-group-per-block                                          | 对称                                | 分组边界会重置行块，避免一个 block 跨越两个 group                            |
| `quant_max`                   | `QuantMax`                                          | 静态给定乘数，同时在线统计 amax | per-tensor                                                   | 由输入数据与乘数决定                | 一次遍历同时做`Cast(x*scale)` 和全局 `amax=max(abs(x))`                  |
| `grouped_quant_max`           | `GroupedQuantMax`                                   | 每组静态乘数，同时在线统计 amax | per-group                                                    | 由输入数据与乘数决定                | 沿 dim-0 分组，每组独立乘 scale、量化、统计 amax                             |

##### 1.2 调用链应该怎样读

这些算子基本都遵循同一条链路：

```text
README / docs
  -> op_api 参数校验、连续化、输出分配
  -> op_graph proto 定义图模式 schema
  -> op_host infer shape / infer dtype
  -> op_host tiling 根据 shape、dtype、SoC、UB、核数选择 tiling key
  -> op_kernel 按 tiling key 实例化具体模板并执行
```

面试中若问“Host 与 Kernel 如何协作”，可以回答：Host 不做数值计算，它把动态 shape 和硬件资源转换为 `coreNum/baseLen/blockFactor/tilingKey`；Kernel 依靠 tiling key 在编译期模板分支中选择 full-load、large-shape、single-row、MOE、per-channel 等实现，减少运行时分支并让向量寄存器、UB 搬运和数据类型转换都可专门化。

#### 2. 静态线性量化：四套接口为什么同时存在

##### 2.1 AscendQuant：标量乘数版本

`ascend_quant/op_graph/ascend_quant_proto.h:56-64` 注册的 `scale`、`offset` 是必选标量属性，而不是输入 Tensor；`op_host/ascend_quant_def.cpp:38-42` 也印证了这一点。当前目录的 Arch35 Kernel 入口直接从 tiling data 读取标量参数，属于典型 per-tensor 静态量化：

$$
q=\operatorname{cast}(x\cdot scale+offset)
$$

若 `sqrt_mode=true`，使用 `scale^2`。它的实际价值是兼容历史图和已经把反量化尺度倒数预先算好的上游。与标准 affine quantization 相比，它省掉 Kernel 内除法，但接口语义更容易和 `Quantize` 搞混。

取整模式支持 Round/Floor/Ceil/Trunc；proto 明确把 Round 定义为 C `rint`，即 round-to-nearest、ties-to-even（`ascend_quant_proto.h:40-46`）。Arch35 的 FP8/HiFloat8 CastTrait 使用 `SatMode::SAT`，而整数转换链使用 `NO_SAT`（`ascend_quant/op_kernel/arch35/ascend_quant.h:53-137`）。因此不要笼统回答“所有量化都会显式 clamp”：这里对浮点 8 bit 有硬件饱和 Cast，整数路径没有独立的 `Min/Max` clamp 代码。

代表性测试 `test_ascend_quant_tiling.cpp` 覆盖 Round/Floor/Ceil/Trunc、INT4 偶数尾维、空输入、非法 rank、输出 dtype 与 `dst_type` 不一致，以及 FP8/HiFloat8 的 round-mode 配对。Kernel UT 还分别覆盖 FP16/FP32 到 INT8，且测试了非零 offset。

##### 2.2 AscendQuantV2：把标量扩展为 axis-aware Tensor

V2 的核心不是换了一条数学公式，而是把参数广播能力做完整：scale/offset 可以是单元素，也可以沿指定 axis 展开。`aclnn_ascend_quant_v3.cpp:152-191` 要求 scale 为 1D，或与 x 同 rank 且除 axis 外其他维都为 1；axis 对应维长度必须等于 x 对应维或 1。

这意味着它可以表达：

- `scale.numel()==1`：per-tensor；
- axis 为最后一维：常见 per-channel；
- axis 为倒数第二维：注意力/MoE 中常被称为 per-head 或 per-row-axis；
- offset 省略：零点为 0；offset 存在：静态 affine 量化。

不同产品约束不同：310P 只允许最后一维；A2/A3/Ascend 950 允许最后两维。INT4 要求输入最后一维可被 2 整除；API 还支持一种 INT4 结果伪装为 INT32 的打包视图，8 个 INT4 共用一个 INT32 存储槽（`aclnn_ascend_quant_v3.cpp:198-237,328-343`）。这是面试中很好的“逻辑 shape 与物理存储 shape 不一定相同”的例子。

V2 的测试重点不是数值 golden，而是契约：输入/scale/offset dtype 兼容、axis 合法性、scale 广播 shape、INT4/INT32 打包 shape，以及 FP8 只能使用 round、HiFloat8 使用 round/hybrid、整数使用 round/floor/ceil/trunc。

##### 2.3 Quantize：最接近教科书和 ONNX 的形式

`Quantize` 使用：

$$
q=\operatorname{round}(x/scales+zeroPoints)
$$

它与 AscendQuant 最大区别是 scales 为反量化尺度 `Δ`，所以做除法，而不是乘量化乘数。`zero_points` 可选；省略时可以看作 `z=0`。公开 aclnn API 当前把 scales/zeroPoints 限制为 1D：长度为 1 时 per-tensor，长度等于 `x[axis]` 时 per-axis；进入 L0 前会 reshape 成 `[1,...,C,...,1]` 以便广播（`quantize/op_api/aclnn_quantize.cpp:163-205,244-309`）。

Host tiling 把 per-axis 又按布局分为 per-channel 和 per-head，并为尾维搬运选择普通 DataCopy 或 NDDMA 版本。数学上它们都是“参数沿一个轴变化”，工程上的区别在于内存是否连续、一次向量能否直接载入连续 scale。

公开接口没有 round-mode 参数，默认采用 nearest/rint 路径。Arch35 Kernel 对 FP8/HiFloat8 使用饱和 Cast，对整数路径使用多级 Cast 且未显式写 clamp（`quantize/op_kernel/arch35/quantize.h:134-226`）。测试覆盖 per-tensor、per-channel、per-head、负 axis、BF16 一致性、非法 scale/zeroPoint shape、ND 格式限制和空 Tensor 快速返回。

##### 2.4 GroupQuant：MOE 场景的静态 per-group-per-channel

输入可抽象为 `x[S,H]`，scale 为 `[E,H]`，`group_index[E]` 是累计结束位置。例如：

```text
group_index = [2, 5, 8]
第 0 组行 [0,2)，第 1 组行 [2,5)，第 2 组行 [5,8)
```

第 `s` 行先找到所属专家 `e`，再对每个通道 `h` 计算：

$$
q_{s,h}=\operatorname{rint}(x_{s,h}\cdot scale_{e,h}+offset)
$$

这里的 scale 是乘数。offset 虽然可选，但当前只支持标量 `[1]` 或 scalar，因此它不是 per-group zero point。Kernel 在 `group_quant_base.h:232-371` 主动检查 group_index 非负、不超过 S、非递减且最后一个值等于 S；INT32 用向量归约检查，INT64 用标量检查。真正计算在 `group_quant_base.h:550-628`：scale 广播后乘 x，加标量 offset，先 RINT 到 int32，再转换为 INT8/INT4。

易错点是把 `group_index` 当成“每行专家 id”。它实际是累计边界；如果传 `[0,0,1,1]` 这类 id，Kernel 的单调性可能满足，但最后值通常不等于 S，最终会 Trap。

#### 3. 动态量化：统计范围、生成参数、立即量化

##### 3.1 DynamicQuant：per-token 对称动态量化

把 x 的最后一维当作特征维 N，其余维展平成 token 数 S。对每行：

$$
u=x\quad\text{或}\quad u=x\odot smoothScales
$$

$$
\Delta_s=\max_j|u_{s,j}|/q_{max},\qquad q_{s,j}=\operatorname{rint}(u_{s,j}/\Delta_s)
$$

INT8 的 `qmax=127`，INT4 的 `qmax=7`。`dynamic_quant/op_kernel/dynamic_quant_base.h:40-46` 明确给出这些常数。输出 scale shape 等于 x 去掉最后一维后的 shape，所以它是 per-token，而不是 per-tensor。

smooth scale 的作用不是量化参数本身，而是先把难量化的通道幅值重新平衡。没有 MOE 时 smooth shape 为 `[N]`；MOE 时可为 `[E,N]`，由累计 `group_index` 为 token 选择专家 smooth 向量。旧 `aclnnDynamicQuant` 固定为 `isSymmetrical=true, quantMode="pertoken"`；Ascend 950 上 API 内部会复用 DynamicQuantV2 实现（`dynamic_quant/op_host/op_api/aclnn_dynamic_quant.cpp:395-404`）。

##### 3.2 DynamicQuantV2：三种粒度与对称/非对称统一

V2 把动态量化真正抽象成两件事：**选择归约域**，再选择**对称或 affine 参数生成**。

| quant_mode     | 归约域                                                 | scale/offset shape   | 典型用途               |
| -------------- | ------------------------------------------------------ | -------------------- | ---------------------- |
| `pertoken`   | 每个 token 的最后一维 N                                | `x.shape[:-1]`     | LLM activation 最常见  |
| `pertensor`  | 整个 Tensor                                            | `[1]`              | 参数开销最小、精度最粗 |
| `perchannel` | 对概念形状`[B,M,N]` 的 M 维归约，每个 N 通道一组参数 | `x` 去掉倒数第二维 | 通道范围差异大时       |

对称量化：

$$
\Delta=\max(|u|)/q_{max},\qquad q=\operatorname{rint}(u/\Delta)
$$

非对称 INT8：

$$
\Delta=(u_{max}-u_{min})/255
$$

$$
z=127-u_{max}/\Delta
$$

$$
q=\operatorname{rint}(u/\Delta+z)
$$

这个 zero point 写法把 `u_max` 映射到 127，理论上把 `u_min` 映射到 -128。INT4 相同，只把区间宽度换成 15、上界换成 7。对称模式不需要 offset，API 明确要求 `isSymmetrical=true` 时 offset 输出必须为 null；非对称模式 offset 必须存在（`aclnn_dynamic_quant.cpp:181-200`）。此外 per-channel 禁止 groupIndex，因为该模式的 smooth scale 沿 M 维，而 MOE groupIndex 的专家 smooth 语义与之冲突。

V2 的 infer-shape 代码很值得背：per-token 删除最后一维；per-tensor 固定 `[1]`；per-channel 删除倒数第二维，保留最后一维（`dynamic_quant_v2_infershape.cpp:75-97`）。这比背“per-channel”三个字更能说明你真的理解归约轴。

##### 3.3 DynamicBlockQuant：以二维 block 控制局部离群值

对最后两维按 `(R,C)` 分块。对每块 B：

$$
a_B=\max_{i\in B}|x_i|
$$

若把量化乘数记为 `m_B`，代码真实语义是：

$$
m_B=\max(dstMax/a_B, minScale)
$$

$$
scaleOut_B=1/m_B=\min(a_B/dstMax,1/minScale)
$$

$$
q_i=\operatorname{Cast}(x_i/scaleOut_B)
$$

这能解释一个看似矛盾的命名：属性叫 `min_scale`，但输出 `scaleOut` 却被 `1/minScale` 上界截断。原因是属性约束的是内部**量化乘数**的下限，而输出保存的是它的倒数，即反量化尺度。`docs/aclnnDynamicBlockQuant(V2).md` 和测试 golden 使用的是 `scaleOut=amax/dstMax`；910B golden 还显式执行 `round -> clamp(-127,127)`（`dynamic_block_quant_data_910b/gen_data.py:39-49`）。

块粒度的收益是离群值只污染所在块，而不是污染整行或整个 Tensor；代价是 scale 数量增多，且尾块、对齐、scale 布局更复杂。2D 输入的 scale shape 为 `[ceil(M/R),ceil(N/C)]`，3D 输入前面再保留 B。

##### 3.4 GroupedDynamicBlockQuant：块不能跨 group

它是在 DynamicBlockQuant 外再加一层累计 `group_list`。关键区别不是多一个输入，而是**每个 group 独立从块边界开始切分**。如果某组行数不能整除 rowBlockSize，组尾会产生 partial block；下一组不能与上组尾部拼成完整 block，否则两个专家的数据会共享量化尺度。

因此 scale 的行空间不是简单 `ceil(M/R)`。infer-shape 预留为：

$$
M//R+G
$$

代码注释说明多出的 `G` 是为每组不能整除 R 的尾块留空间（`grouped_dynamic_block_quant_infershape.cpp:43-58,74-94`）。这是面试高频题：“为什么 grouped block quant 的 scale shape 比普通 block quant 大？”

当前 `group_list_type` 只支持 0，即 cumsum 模式；group_list 必须非负、非递减、最后一个值等于 M。输出只支持 FP8/HiFloat8，不支持 INT8。FP8 只允许 rint，HiFloat8 允许 round/hybrid。

##### 3.5 QuantMax 与 GroupedQuantMax：量化和校准统计融合

`QuantMax` 不生成 scale，而是消费外部给定的 per-tensor 乘数：

$$
y=\operatorname{Cast}(x\cdot scale),\qquad amax=\max|x|
$$

Kernel 在同一向量循环里同时 `Abs/Max` 和 `Mul/Cast`（`quant_max_per_tensor_regbase.h:149-213`），避免为了 amax 再读一遍 x。各核先把局部最大值写 workspace，核 0 在 `SyncAll` 后做第二次 ReduceMax（同文件 `229-260`）。这是典型的“融合减少 HBM 带宽，但需要跨核归约 workspace”的设计。

`GroupedQuantMax` 沿 dim-0 用累计 group_list 切分；scale 和 amax shape 都是 `[num_groups]`。Kernel 要处理一个核的数据段横跨多个 group 的情况：先求核区间与 group 区间交集，再用该组 scale 量化，并把每核每组局部 amax 写入 workspace，最后归并。测试专门覆盖单组、三组、四组多核和不等长 `[3,7,10]` 分组。

二者只支持 FP8/HiFloat8 输出，FP8 Cast 使用饱和模式。`QuantMax` 支持 x rank 1-8；`GroupedQuantMax` 支持 2-8，group 数限制 1-384，且明确不支持空 x。

#### 4. Round、RINT、饱和：面试时要分开回答

| 路径                   | 取整                                         | 饱和/裁剪观察                                                                   |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------------------------------- |
| AscendQuant / V2 整数  | round(rint ties-to-even)、floor、ceil、trunc | 未看到独立 clamp；Arch35 整数 CastTrait 多为`NO_SAT`，不能泛化成“必然饱和”  |
| Quantize 整数          | 公共 API 固定 nearest/rint                   | 同样主要依赖 Cast 链，没有显式 Min/Max clamp                                    |
| GroupQuant INT8/INT4   | 固定`CAST_RINT`                            | 无显式 clamp；应由 scale 保证范围，越界语义不要凭经验猜                         |
| DynamicQuant INT8/INT4 | `CAST_RINT`                                | 动态 extrema 理论上把范围压入目标区间；浮点误差、常量张量仍值得测试             |
| DynamicBlockQuant INT8 | rint                                         | Arch35 CastTrait 明确`SatMode::SAT`；910B golden 明确 clamp 到 `[-127,127]` |
| FP8 E5M2/E4M3FN        | rint                                         | 多个 Kernel 的 CastTrait 明确`SatMode::SAT`                                   |
| HiFloat8               | round 或 hybrid                              | CastTrait 明确`SatMode::SAT`；hybrid 是 HiFloat8 专用近似模式                 |

`round` 这个字符串在不同 API 中不一定等同于 C/C++ `std::round`。AscendQuant proto 已明确其 Round 是 `rint`、ties-to-even；而块量化把字符串 `rint` 与 `round` 分开用于 FP8 和 HiFloat8。面试回答应引用具体接口契约，不要只说“四舍五入”。

#### 5. 两段最值得深读的真实代码

##### 5.1 片段一：DynamicQuantV2 的 per-channel 非对称参数生成

位置：`ops-nn/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_perchannel_full_load.h:372-421`

```cpp
        for (uint16_t bIdx = 0; bIdx < baBlockSize; bIdx++) {
            uint32_t sregN = nSize;
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::Duplicate<float>(vregColMax, NEG_INFINITY, preg0);
                MicroAPI::Duplicate<float>(vregColMin, POS_INFINITY, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + (j + bIdx * mLen_) * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype*)(smoothAddr + j));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Max<float>(vregColMax, vregInFp32, vregColMax, preg0);
                    MicroAPI::Min<float>(vregColMin, vregInFp32, vregColMin, preg0);
                }
                MicroAPI::Sub(vregResult, vregColMax, vregColMin, preg0);
                MicroAPI::Mul(vregOutScale, vregResult, vregOffsetDivVal, preg0);
                MicroAPI::Div<float, &divHighPrecisionMode>(vregDivScale, vregColMax, vregOutScale, preg0);
                MicroAPI::Sub<float>(vregOffset, vregMaxFactor, vregDivScale, preg0);

                for (uint16_t k = 0; k < mLoopNum; k++) {
                    auto addr = yAddr + i * REG_LEN + (k + bIdx * mLen_) * nSizeOut;
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + (k + bIdx * mLen_) * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype*)(smoothAddr + k));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Div<float>(vregDiv, vregInFp32, vregOutScale, preg0);
                    MicroAPI::Add<float>(vregOutFp32, vregDiv, vregOffset, preg0);

                    CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        addr = yAddr + (i * REG_LEN + (k + bIdx * mLen_) * nSizeOut) / 2;
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, pregH);
                    } else {
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, preg0);
                    }
                }
                MicroAPI::DataCopy<float>((__ubuf__ float*)scaleAddr + i * REG_LEN + bIdx * nSizeScale, vregOutScale,
                                          preg0);
                MicroAPI::DataCopy<float>((__ubuf__ float*)offsetAddr + i * REG_LEN + bIdx * nSizeScale, vregOffset,
                                          preg0);
```

读法如下：

1. `i` 遍历 N 维的向量段，`j` 遍历 M，因此 `Max/Min` 是沿倒数第二维做的，证明 per-channel scale 最后保留 N。
2. smooth scale 通过 `DIST_BRC_B16` 广播一个 M 位置的值；这解释了 per-channel smooth shape 为什么对应 x 的倒数第二维，而不是最后一维。
3. `vregOffsetDivVal` 是 `1/(qmax-qmin)`，所以 `vregOutScale=(max-min)/(qmax-qmin)`。
4. `vregOffset=qmax-max/scale`，随后第二遍循环计算 `x/scale+offset`。之所以分两遍，是因为必须先完整归约 M 得到参数，才能量化每个元素。
5. INT4 分支把输出地址除以 2，并用 pack store；这不是 shape 缩小，而是两个 INT4 共享一个字节。

贯穿示例：一个通道在 M 维上的值为 `[-2, 6]`，INT8 非对称量化时 `scale=8/255≈0.03137`，`offset=127-6/scale≈-64.25`，两端经取整后约映射到 `[-128,127]`。若改成对称量化，则 `scale=6/127≈0.04724`，负端只能映射到约 `-42`，这说明非对称量化在分布明显偏移时能更充分利用编码区间。

值得追问的边界：当前片段没有显式把 `max-min` 与 epsilon 做 Max。常量通道会产生零 scale，硬件 Div 的具体结果需要用数值测试确认；现有 UT 主要覆盖 tiling 分支而非该数值边界，这是合理的补测建议。

##### 5.2 片段二：DynamicBlockQuant 的 absmax、反量化尺度与饱和 Cast

位置：`ops-nn/quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_single_row_kernel.h:330-392`

```cpp
        if constexpr (IsSameType<IN_TYPE, float>::value) {
            // ===== float32 类型处理 =====
            AscendC::MicroAPI::RegTensor<uint32_t> vRegFp32Max;
            AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<uint32_t>&)vRegFp32Max, maxValue32_);

            for (uint16_t rowIdx = 0; rowIdx < vfRowBlockSize; rowIdx++) {
                for (uint16_t colIdx = 0; colIdx < normalColBlockLoop; colIdx++) {
                    AscendC::MicroAPI::Duplicate(vReg2, 0.0f);
                    curSize = tilingData_->blockSizeCol;
                    for (uint16_t vlLoopIdx = 0; vlLoopIdx < normalLoopNum; vlLoopIdx++) {
                        inputMaskReg = AscendC::MicroAPI::UpdateMask<IN_TYPE>(curSize);
                        AscendC::MicroAPI::DataCopy(vReg0, xLocal + rowIdx * inputColAlign +
                                                               colIdx * tilingData_->blockSizeCol + vlLoopIdx * VL);
                        AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint32_t>&)vReg3,
                                               (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg0,
                                               (AscendC::MicroAPI::RegTensor<uint32_t>&)vRegFp32Max, inputMaskReg);
                        AscendC::MicroAPI::Max<uint32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg2,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg2,
                            (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg3, inputMaskReg);
                    }

                    AscendC::MicroAPI::ReduceMax<uint32_t>((AscendC::MicroAPI::RegTensor<uint32_t>&)vReg3,
                                                           (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg2,
                                                           defaultMaskReg);

                    AscendC::MicroAPI::Duplicate(vReg4, vReg3, defaultMaskReg);

                    // brc(input_max) - 位模式直接参与计算
                    AscendC::MicroAPI::Div<float, &mode>(vReg8, (AscendC::MicroAPI::RegTensor<float>&)vReg4,
                                                         fp8MaxValue, defaultMaskReg);
                    AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                        scaleMaskReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)vReg8, infValue_, defaultMaskReg);
                    AscendC::MicroAPI::Min<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                        vReg8, vReg8, reciprocalScale, scaleMaskReg);

                    // copy out scale
                    MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleLocal, vReg8, ureg,
                                                                                           static_cast<uint32_t>(1));
                    MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleLocal, ureg,
                                                                                               static_cast<int32_t>(0));

                    curSize = tilingData_->blockSizeCol;
                    for (uint16_t vlLoopIdx = 0; vlLoopIdx < normalLoopNum; vlLoopIdx++) {
                        inputMaskReg = AscendC::MicroAPI::UpdateMask<IN_TYPE>(curSize);
                        AscendC::MicroAPI::DataCopy(vReg0, xLocal + rowIdx * inputColAlign +
                                                               colIdx * tilingData_->blockSizeCol + vlLoopIdx * VL);
                        AscendC::MicroAPI::Div<float, &mode>(vReg13, vReg0, vReg8, defaultMaskReg);

                        if constexpr (IsSameType<OUT_TYPE, hifloat8_t>::value) {
                            AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32toh8Zero>(vReg15, vReg13,
                                                                                          defaultMaskReg);
                        } else if constexpr (IsSameType<OUT_TYPE, int8_t>::value) {
                            AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vReg19, vReg13, defaultMaskReg);
                            AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vReg20, vReg19, defaultMaskReg);
                            AscendC::MicroAPI::Cast<OUT_TYPE, half, castTraitF16ToI8>(vReg15, vReg20, defaultMaskReg);
                        } else {
                            AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32tofp8>(vReg15, vReg13, defaultMaskReg);
                        }

                        AscendC::MicroAPI::DataCopy<OUT_TYPE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            outLocal + rowIdx * colSizeAlign + colIdx * tilingData_->blockSizeCol + vlLoopIdx * VL,
                            vReg15, inputMaskReg);
```

这里有三个很漂亮的工程点：

1. FP32 绝对值不是调用浮点 `Abs`，而是把数据重解释为 uint32，再与 `0x7fffffff` 按位与，直接清符号位。对 IEEE 754 正常数，这比额外浮点指令更直接。
2. `vReg8=input_max/dstMax`，随后与 `1/minScale` 取 Min，并把 `vReg8` 原样写到 scale 输出，证明输出是反量化尺度；量化阶段再执行 `x/vReg8`。
3. `dynamic_block_quant_common.h:61-75` 中 FP8 和 INT8 相关 CastTrait 都设置 `SatMode::SAT`。所以 minScale 强制更大乘数导致越界时，结果会饱和，而不是无限扩大编码。

示例：block 为 `[-1,3]`，目标 INT8，则 `scaleOut=3/127≈0.02362`，量化值约为 `[-42,127]`。若设置 `minScale=100`，内部乘数至少为 100，`scaleOut` 被压到 `0.01`，3 会得到 300 并最终饱和为 127；这说明 minScale 是一个可能主动引入 clipping、换取更细步长的策略参数。

#### 6. 测试揭示的边界与当前覆盖缺口

| 边界                | 测试/代码结论                                                                                               | 面试表达                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| INT4 尾维           | 多个 API 要求 x 最后一维为偶数；INT32 伪装输出要求尾维按 8:1 收缩                                           | INT4 的难点不仅是数值范围，还有物理打包              |
| axis                | AscendQuantV2 在不同 SoC 只允许最后一维或最后两维；Quantize axis 可为负并 wrap                              | axis 是参数广播轴，不必然等同框架里的 channel 轴     |
| group 索引          | GroupQuant、GroupedDynamicBlockQuant、GroupedQuantMax 都使用累计边界                                        | 必须非递减，最后值覆盖全部行；不是 expert-id 数组    |
| per-channel dynamic | DynamicQuantV2 禁止同时传 groupIndex；smooth 长度对应倒数第二维                                             | per-channel 是沿 M 归约、保留 N，不是沿 N 归约       |
| 空 Tensor           | DynamicQuant API 空输入直接 workspace=0；QuantMax API 也有空 x 快速返回；GroupedQuantMax Host 明确拒绝空 x  | 空 Tensor 行为是算子级契约，不能统一假设             |
| block 尾块          | DynamicBlockQuant 覆盖 single-row、normal、large-shape、3D 和各类尾块                                       | 尾块需要 mask 和独立 scale，不能用完整块假设读取越界 |
| round/dtype 配对    | FP8=rint，HiFloat8=round/hybrid；静态整数才有 floor/ceil/trunc                                              | 取整模式受目标类型约束，不是任意组合                 |
| rank                | DynamicQuant x 为 2-8D；DynamicBlockQuant 2-3D；QuantMax 1-8D；GroupedQuantMax 2-8D                         | 公式可泛化不代表实现契约无限制                       |
| 数值测试强度        | QuantMax/GroupedQuantMax Kernel 有数值 amax UT，但多项 API 正向测试被 DISABLED；GroupQuant API 数值覆盖较弱 | 参数校验覆盖强，不等于数值边界覆盖强                 |

建议补测三类场景：全零/常量输入（scale 为 0 的路径）、NaN/Inf（absmax 与 Compare/Min 的行为）、恰好半整数（ties-to-even 与 round/hybrid 差异）。这些用例比再增加一个普通随机 shape 更能暴露量化实现问题。

#### 7. 面试高频追问与参考回答

##### Q1：对称量化为什么常用 127 而不是 128？

INT8 表示范围是 `[-128,127]`，但对称量化希望正负共享同一个尺度，通常使用 `[-127,127]`，让 0 精确对应 0，正负最大幅值对称。代价是浪费 `-128` 一个编码；非对称量化则可以利用完整 255 个间隔。

##### Q2：per-token、per-channel、per-block 精度和开销怎么排序？

一般参数越细，局部动态范围越小、量化误差越低，但 scale 数量、归约次数和访存开销越大。per-tensor 参数最少；per-token 对 LLM activation 很合适；per-channel 适合各通道分布差异大；per-block 进一步隔离局部离群值，但需要处理大量 scale 与尾块。

##### Q3：动态量化为什么慢，为什么还值得用？

它必须在量化前扫描输入求 max 或 min/max，理论上多一次归约依赖。价值是参数随当前 token/batch 自适应，不依赖离线校准；Kernel 可以把归约、参数生成和量化融合，并在 UB/寄存器中复用数据，降低额外 HBM 流量。

##### Q4：smooth scale 做了什么？

它先把 activation 的通道幅值重新分配，再统计动态范围。目标是把极端通道的幅值压力转移到后续权重尺度中，使 activation 更容易低比特表示；它不是最终输出的 quant scale。

##### Q5：非对称量化一定比对称量化精度好吗？

不一定。分布偏移明显时，非对称能充分利用编码范围；但它需要 offset，矩阵乘累加时还可能引入额外修正项。硬件和融合算子更偏好 zero-point 为 0 的对称路径，因此要在精度收益与执行复杂度之间权衡。

##### Q6：为什么 DynamicBlockQuant 的 README 容易把 scale 讲反？

因为内部需要的是量化乘数 `m=dstMax/amax`，而输出给下游反量化的是 `1/m=amax/dstMax`。面试时最好同时写出 `m` 和 `scaleOut`，不要只用一个 scale 符号。

##### Q7：为什么 GroupedDynamicBlockQuant 的 scale 行数是 `M//R+G`？

每个 group 独立分块，组尾不足 R 的 partial block 不能与下一组拼接。每组最多多出一个尾块，所以用 `M//R+G` 预留安全空间。

##### Q8：Tiling key 的作用是什么？

它把 dtype、round mode、是否 smooth、对称性、量化粒度、shape 策略编码为 Kernel 模板选择。例如 DynamicQuantV2 Arch35 入口按 quantMode 分派 full-load、large-shape、MOE、pertensor、perchannel-recompute/split-M 等实现（`dynamic_quant_v2_apt.cpp:44-126`）。这样同一算子 schema 可以有多套针对硬件瓶颈优化的 Kernel。

##### Q9：QuantMax 为什么把 amax 和量化绑在一起？

两者都需要读取 x。融合后一次加载即可同时更新最大值并生成低比特输出，主要节省 HBM 带宽；代价是需要 workspace 保存每核局部最大值，并做跨核同步归约。

##### Q10：代码里没有 Clamp，能否说转换一定饱和？

不能。必须看具体 CastTrait 的 `SatMode` 或 Cast API 契约。这个仓库中 FP8/HiFloat8 和 DynamicBlockQuant INT8 明确使用 SAT；若整数静态量化路径写的是 NO_SAT 或普通 Cast，就不应擅自补上 clamp 语义。

#### 8. 面试学习路径

##### 第 1 阶段：先能手算（半天）

只练四组数字：

1. 对称 INT8：`[-2,6] -> Δ=6/127 -> q`；
2. 非对称 INT8：`[-2,6] -> Δ=8/255 -> z -> q`；
3. 静态乘数：给 `m=20,z=3` 算 `round(x*m+z)`；
4. block：把一个 `2x4` 矩阵按 `1x2` 分块，逐块算 amax、scaleOut 和 q。

达到的标准是：看到任何接口能立刻判断 scale 是乘数还是除数。

##### 第 2 阶段：掌握 shape 与归约轴（半天）

拿 `x[B,M,N]` 画出：

- per-token scale `[B,M]`；
- per-channel scale `[B,N]`；
- per-tensor scale `[1]`；
- per-block scale `[B,ceil(M/R),ceil(N/C)]`；
- grouped per-block 为什么多出 group 尾块。

这一阶段是面试区分“背定义”和“懂实现”的关键。

##### 第 3 阶段：按调用链读四个代表算子（1 天）

推荐顺序：

1. `quantize`：最标准的静态公式；
2. `dynamic_quant_v2`：动态对称/非对称与三种粒度；
3. `dynamic_block_quant`：分块、尾块、minScale、饱和；
4. `quant_max`：融合计算与跨核归约。

每个算子都回答五问：输入输出 shape、scale 方向、归约轴、round/saturate、tiling 为什么分支。

##### 第 4 阶段：专项理解 MOE/group（半天）

对比三种 group 输入：`GroupQuant.group_index`、`DynamicQuant.group_index`、`GroupedDynamicBlockQuant.group_list`。它们都常用累计边界，但驱动的对象不同：选择静态 scale、选择 expert smooth scale、重置 block 边界。

##### 第 5 阶段：做代码口述和反向实现（1 天）

不看源码，口述本章两段核心代码；然后用 NumPy/PyTorch 写 golden：

- per-token symmetric/asymmetric；
- per-channel reduce-M；
- per-block absmax；
- grouped cumulative slicing；
- rint 与普通 round 的半整数对比。

如果能写出 golden 并解释每个输出 scale 的 shape，基础量化面试通常已经够用。

#### 9. 自测清单

- [ ] 能解释乘数 scale 与反量化 scale 的互为倒数关系。
- [ ] 能根据 `x[B,M,N]` 写出 per-token/per-channel/per-tensor/per-block 的 scale shape。
- [ ] 能说明对称 INT8 用 127、非对称区间宽度用 255 的原因。
- [ ] 能解释 DynamicQuantV2 非对称 offset 公式。
- [ ] 能解释 INT4 为什么要求偶数尾维，以及 INT32 伪装打包的 8:1 关系。
- [ ] 能说明 GroupQuant 的 group_index 是累计边界，不是专家 id。
- [ ] 能说明 GroupedDynamicBlockQuant 为什么不能让 block 跨 group。
- [ ] 能区分 rint、floor、ceil、trunc、round、hybrid，并指出它们受 dst type 约束。
- [ ] 能从 CastTrait 判断是否饱和，而不是凭经验推断。
- [ ] 能解释 QuantMax 的一次遍历融合与跨核 amax 归约。

#### 10. 阅读过的关键文件索引

- 静态：`ascend_quant/README.md`、`op_graph/ascend_quant_proto.h`、`op_host/arch35/ascend_quant_tiling*.cpp`、`op_kernel/arch35/ascend_quant*.h` 及 host/kernel UT。
- Axis 静态：`ascend_quant_v2/README.md`、`docs/aclnnAscendQuant*.md`、`op_host/op_api/aclnn_ascend_quant*.cpp`、infer/tiling、各 dtype Kernel 与 UT。
- 标准 Quantize：`quantize/README.md`、`docs/aclnnQuantize.md`、`op_api/aclnn_quantize.cpp`、proto、infer/tiling、per-tensor/per-channel/per-head Kernel 与 UT。
- GroupQuant：README/docs、proto、`op_host/op_api/aclnn_group_quant.cpp`、tiling、`group_quant_base.h` 与 UT。
- DynamicQuant/V2：两目录 README/docs/proto、统一 aclnn API、infer/tiling、普通/DB/large/MOE/per-tensor/per-channel Kernel 与 UT。
- Block：DynamicBlockQuant 与 GroupedDynamicBlockQuant 的 README、两版 aclnn docs、proto、infer/tiling、single/small/large block Kernel、golden 脚本与 UT。
- QuantMax：QuantMax/GroupedQuantMax 的 README、aclnn API、proto、infer/tiling、Kernel 和数值/参数校验 UT。

### 专题二：02 MX 与低比特浮点量化：从 shared exponent 到 FP4/FP8 Kernel

> 本章面向面试复习，分析范围是 `ops-nn/quant` 下 9 组 MX/FP4/FP8 算子。结论先行：MX 不是“把浮点数当 INT8 做一次线性缩放”，而是让一个小块内的低比特浮点数共享一个只含指数的 `FP8_E8M0` scale。工程难点不只在公式，还在 scale 的成对存储、尾块补齐、双轴布局、分组边界以及 Host tiling 到 Kernel 模板分派的一致性。

#### 理解验证状态

| 核心问题                                       | 应达到的掌握程度                                              |
| ---------------------------------------------- | ------------------------------------------------------------- |
| 为什么 MX scale 是 2 的整数次幂                | 能从 shared exponent、位级实现和硬件乘法成本三方面解释        |
| `blocksize=32` 与 scale shape 为什么多一维 2 | 能手算 shape，并解释两个 E8M0 scale 打包为一组的布局          |
| FP4、FP8 与 INT8 的差别                        | 能从数值格式、动态范围、量化误差和硬件路径比较                |
| 单轴、双轴、32×32 block、grouped、dual-level  | 能说明它们分别解决什么数据布局或精度问题                      |
| 反量化如何恢复近似值                           | 能解释`x_q × scale`、E8M0 解码和块广播                     |
| Host/Kernel 分工                               | 能从 API 校验、shape 推导、tiling key、模板实例化讲完整调用链 |

#### 1. 代码覆盖地图

| 算子目录                                    | README / API                                              | Graph / Host                                                                                                          | Kernel                                                                                          | 代表性测试                                               | 一句话定位                                                              |
| ------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| `dynamic_mx_quant`                        | `README.md`；`docs/aclnnDynamicMxQuant{,V2,V3}.md`    | `op_graph/dynamic_mx_quant_proto.h`；`op_host/dynamic_mx_quant_{def,infershape}.cpp`；`op_host/arch35/*tiling*` | `op_kernel/dynamic_mx_quant.cpp`；`arch35/dynamic_mx_quant_{tail_axis,not_tail_axis,...}.h` | `test_dynamic_mx_quant.cpp`、infershape 与 tiling 测试 | 通用单轴 MX 量化，支持尾轴/非尾轴、FP4/FP8 和多种 scale 算法            |
| `dynamic_mx_quant_with_dual_axis`         | `README.md`；V1/V2 API 文档                             | proto、def、infershape、tiling                                                                                        | `dynamic_mx_quant_with_dual_axis.cpp`；`*_base.h`                                           | kernel 覆盖 scaleAlg 0/1/2；host 覆盖 FP4/FP8 与舍入     | 同一输入沿 -1、-2 轴各量化一次，输出两套数据和 scale                    |
| `dynamic_block_mx_quant`                  | `README.md`；`aclnnDynamicBlockMxQuant.md`            | proto、def、infershape、tiling                                                                                        | `dynamic_block_mx_quant.cpp`；`*_base.h`                                                    | `dstTypeMax=0/6/7`、非对齐列、3D 和失败用例            | 一个 32×32 二维块只求一个 scale，再广播成两种轴向布局                  |
| `dynamic_dual_level_mx_quant`             | `README.md`；`aclnnDynamicDualLevelMxQuant.md`        | proto、def、infershape、tiling                                                                                        | `dynamic_dual_level_mx_quant.cpp`；`*_base.h`                                               | kernel 基本路径；host shape/非法 shape                   | 先做 512 粒度 FP32 scale，再对中间值做 32 粒度 MX FP4 量化              |
| `grouped_dynamic_mx_quant`                | `README.md`；V1/V2 API；显式 `op_api`                 | proto、def、infershape、group-aware tiling                                                                            | `grouped_dynamic_mx_quant.cpp`；`*_not_tail_axis_fp8.h`                                     | API 四种 dtype 组合、shape/dtype/attr 失败；scaleAlg=1   | 由`group_index` 切分第 0 维，在每组内部独立做 FP8 MX 量化             |
| `anti_mx_quant`                           | `README.md`；显式 `op_api` 与 `aclnnAntiMxQuant.md` | proto、def、infershape、tail-axis tiling                                                                              | `anti_mx_quant.cpp`；`anti_mx_quant_tail_axis.h`                                            | FP4/FP8 到 FP16/BF16/FP32、尾块、多核、非法 scale shape  | 将 FP4/FP8 与 E8M0 scale 组合恢复为高精度近似值                         |
| `swiglu_mx_quant`                         | `README.md`；`aclnnSwigluMxQuant.md`                  | proto、def、infershape、tiling                                                                                        | `swiglu_mx_quant_apt.cpp`；axis-last/not-last kernel                                          | FP16/BF16→FP4/FP8、scaleAlg=1、小 shape、非法属性       | 融合 SwiGLU 和单轴 MX 量化，减少中间张量读写                            |
| `swiglu_mx_quant_with_dual_axis`          | `README.md`；API 文档                                   | def、arch35 tiling                                                                                                    | `*_apt.cpp`；`*_regbase.h`                                                                  | FP4/FP8、scaleAlg=1、非法 round/dst                      | SwiGLU 后同时沿行列方向量化，可带 cumsum group 边界                     |
| `swiglu_backward_mx_quant_with_dual_axis` | `README.md`；API 文档                                   | def、arch35 tiling                                                                                                    | 入口 cpp；`*_regbase.h`                                                                       | FP16/BF16、batch2、非法 OCP/round/dst                    | 计算 SwiGLU 反向梯度后双轴量化；当前有效实现范围明显窄于通用 OpDef 声明 |

调用链可概括为：

```text
aclnn 两段式 API
  → 参数、dtype、shape、连续性检查
  → Graph Op / OpDef 与 InferShape
  → Host Tiling 选择 axis、scaleAlg、roundMode、尾块和核分配
  → tiling key 实例化 Kernel 模板
  → GM→UB 搬运、块内 ReduceMax、生成 E8M0 scale、乘倒数、Cast、写回
```

API 层的典型工程结构可以在 `anti_mx_quant/op_api/aclnn_anti_mx_quant.cpp` 中看到：先检查空指针、dtype、属性和 shape，再把非连续输入转成 contiguous，调用 L0 `AntiMxQuant`，最后 `ViewCopy` 到用户输出。Grouped V2 也采用相同模式，并额外暴露 `scaleAlg`、`dstTypeMax`。

#### 2. MX 的核心心智模型

##### 2.1 MX scale 与 shared exponent

对一个块 \(V\) 求最大绝对值，并让块内所有低比特浮点数共享指数：

\[
e_{shared}=\lfloor \log_2(\max_i |V_i|)\rfloor-e_{max}
\]

\[
s=2^{e_{shared}}, \qquad q_i=\operatorname{cast}(V_i/s)
\]

这里的 `emax` 是目标低比特格式最大正规数所对应的指数能力：FP4 E2M1 为 2、FP4 E1M2 为 0、FP8 E4M3FN 为 8、FP8 E5M2 为 15。直觉上，`shared_exp` 把该块最大的数“平移”到目标格式最靠近上界的位置，使其尽量不溢出，同时让较小值利用目标格式的尾数位。

为什么 scale 使用 `FLOAT8_E8M0`？它没有尾数，天然表达 \(2^k\)。因此量化和反量化可以主要通过指数调整或乘一个精确的 2 的幂完成，scale 本身只占 1 字节。代价是 scale 只能取离散的 2 的幂，比任意 FP32 scale 更粗；这正是 MX 用 scale 精度换元数据带宽与硬件效率的选择。

##### 2.2 blocksize 与 scale 的打包 shape

单轴算子沿 axis 每 `blocksize` 个元素产生一个 scale。逻辑 scale 数是：

\[
n_s=\left\lceil \frac{D_{axis}}{blocksize}\right\rceil
\]

仓库把两个 8-bit scale 成对存放，因此物理 shape 把 axis 位置改为 \(\lceil n_s/2\rceil\)，并在最后追加维度 2。若最后只有一个 scale，第二个槽位 pad 0。

例：`x=[2048,2360]`、axis=-1、blocksize=32，则 \(n_s=74\)，`mxscale=[2048,37,2]`；`x=[2048,2370]` 时 \(n_s=75\)，pad 后为 `[2048,38,2]`。这两个边界都被 `dynamic_mx_quant` 和 `anti_mx_quant` 的 infershape 测试明确覆盖。

##### 2.3 FP4 / FP8 格式取舍

| 格式       | 大致特点                                   | 更适合的场景             | 主要风险                     |
| ---------- | ------------------------------------------ | ------------------------ | ---------------------------- |
| FP4 E2M1   | 指数比 E1M2 多，动态范围更大，尾数仅 1 bit | 激活、梯度中跨度较大的块 | 相邻可表示值稀疏，舍入误差大 |
| FP4 E1M2   | 尾数更多、局部精度较好，范围最窄           | 已经被良好归一化的块     | 更容易溢出或把小值冲成 0     |
| FP8 E4M3FN | 3 bit 尾数，精度较好；有限数语义面向计算   | 训练前向、权重/激活      | 范围不如 E5M2                |
| FP8 E5M2   | 5 bit 指数，范围更大                       | 梯度或长尾分布           | 尾数更少，局部误差更大       |

FP4 的两个数通常打包在一个字节中，因此最后一维必须为偶数；部分融合/反向路径进一步要求最后一维能被 4 或 64 整除，这是向量加载、两半拼接和 scale 配对共同造成的工程约束。

##### 2.4 三种 scale 算法

- `scale_alg=0`：OCP MX。直接从块内最大值的指数得到 `floor(log2(amax))-emax`，符合 shared exponent 定义。它最“MX 原生”，scale 是不大于理想比例的 2 的幂。
- `scale_alg=1`：仓库文档称 CuBALS/cuBLAS scale。先算 \(s_{fp32}=amax/Amax(dtype)\)，再把 scale 指数有条件向上取整为 E8M0。向上取整的意图是避免量化值越过目标上界；相较 OCP，它更显式地从目标最大有限值出发。DynamicMxQuant V3 的 `maxLowBound` 只在该算法生效，用 `amax=max(amax,maxLowBound)` 防止近零块产生过小 scale、放大噪声。
- `scale_alg=2`：DynamicDtypeRange，当前只服务 FP4 E2M1。`dst_type_max` 可为默认值或 6~12 的自定义上界；实现根据是否是 0/6/7 等快速值选择指数进位或 FP32 乘法路径。它允许业务用“有效饱和值”而非格式理论上限换取精度。

算法组合不是任意的：FP4 E1M2 只支持 OCP；FP4 E2M1 支持 0/2；FP8 支持 0/1；FP8 只允许 `rint`。面试中如果只说“scaleAlg 是三种算法”而不说 dtype 约束，通常会被继续追问。

#### 3. 九类算子的设计差异

##### 3.1 DynamicMxQuant：通用单轴基线

这是最应该先掌握的基线算子。`y.shape=x.shape`，`mxscale.rank=x.rank+1`，axis 可正可负，blocksize 支持 32 的倍数且最大 1024；FP32 输入当前只走 blocksize=32 的受限路径。V2 新增 `dstTypeMax`，V3 新增 `maxLowBound`。

Kernel 入口 `op_kernel/dynamic_mx_quant.cpp` 不直接计算，而是按 tiling 编译期参数分派：尾轴 block32、非尾轴 small/large-tail 优化、普通尾轴/非尾轴，以及奇数 scale 先写 workspace 再由 post kernel 重排。WHY：不同 axis 决定内存是否连续，不同尾块长度决定向量利用率；一个通用 kernel 会塞满运行时分支，既慢又难以让编译器展开。

##### 3.2 DynamicMxQuantWithDualAxis：一份输入、两种量化视图

输入前导维会合并成 batch，核心数据视为 `[B,M,N]`。沿 -1 轴每 32 列生成 `y1/mxscale1`，沿 -2 轴每 32 行生成 `y2/mxscale2`。`y1` 和 `y2` 数值不同，因为同一个元素分别除以“所在行的列块 scale”和“所在列的行块 scale”。这通常为后续矩阵乘两种转置/布局消费方式准备。

`mxscale2` 需要每两行交织，base kernel 在计算结束后显式 `Interleave`。双轴不是“求两次 scale 但共用一个 y”，而是必须输出两份量化张量；否则一个元素无法同时满足两种 scale。

##### 3.3 DynamicBlockMxQuant：32×32 共用一个二维块 scale

它与双轴量化最容易混淆。双轴算子在一个 32×32 区域内会有 32 个行向 scale 和 32 个列向 scale；block 算子只对整个 32×32 求一个最大值和一个 scale，然后把该 scale 广播为 `scale1(32×1)`、`scale2(1×32)`。因此两个 scale 输出在语义上相同，只是为下游轴向布局复制。

优点是 scale 元数据和计算更少，两个方向看到完全一致的数值；缺点是一个 outlier 会压低整个 1024 元素块的有效精度。测试对 3D、非对齐列、FP4/FP8、`dstTypeMax=0/6/7` 和大量非法组合均有覆盖。

##### 3.4 DynamicDualLevelMxQuant：二级量化不是“量化两遍输出”

第一级以 `level0_block_size=512` 求 FP32 scale：

\[
s_0=amax(x)/FP4\_E2M1\_MAX,\quad temp=x/s_0
\]

第二级再以 `level1_block_size=32` 对 `temp` 做 OCP shared-exponent：

\[
s_1=2^{\lfloor\log_2(amax(temp))\rfloor-emax},\quad y=cast_{FP4}(temp/s_1)
\]

最终近似恢复应理解为 \(\hat{x}=y\times s_1\times s_0\)，其中每个元素分别索引自己的 32 块和 512 块。一级 scale 是 FP32，负责长程幅值；二级 scale 是 E8M0，负责局部动态范围。相比单级 32 块，这种设计可把一部分尺度信息放在较粗粒度 FP32 元数据中，也便于融合 `x*=smooth_scale`。Kernel 入口用模板布尔参数把有/无 smooth scale 编成两条路径，base 内依次执行 `ComputeSmoothScaleLevel0Quant`、`ComputeLevel1Scale`、`ComputeY`。

##### 3.5 GroupedDynamicMxQuant：组边界不能被普通 block 穿过

`group_index` 是第 0 维的 cumsum 终点，例如 `[10,25,32]` 表示 `[0,10)`、`[10,25)`、`[25,32)`。Kernel 对每个 group 和后轴列块建立任务，组内再按 32 行处理。即使前一组尾部和后一组头部凑够 32，也不能共享 scale，否则不同 expert/token group 的统计量会互相污染。

scale shape 为 `[(m/(32*2)+g), n, 2]` 的本质是：每组独立向 64 行成对打包，组边界会额外引入 pad，所以总长度不是简单 `ceil(m/64)`。约束包括 group 值非递减、非负、最后值等于 m；实现同时支持 OCP 与 cuBLAS FP8，输出仅 E4M3FN/E5M2。

##### 3.6 AntiMxQuant：反量化是近似重构，不是恢复原值

公式为：

\[
x_{dq}=x_q\times 2^{sf-127}=x_q\times scale
\]

输入 FP4/FP8 先展开到计算类型，E8M0 scale 解码成 BF16/FP32，再按 1×32 块广播相乘，最后 cast 到 FP16/BF16/FP32。量化时的舍入、饱和和下溢已不可逆，所以 AntiMxQuant 只能恢复近似值。

一个重要实现边界：Graph/API 的 `axis` 看起来是一般属性，但当前 Host tiling 在 `anti_mx_quant_tiling_arch35.cpp:191-198` 明确拒绝非尾轴，Kernel 入口也只实例化 `TPL_AXIS_TAIL`。面试时应区分“接口模型”与“当前产品实现能力”。

##### 3.7 SwigluMxQuant：融合的价值在带宽

先沿 `activate_dim` 把输入一分为二，计算 `SiLU(gate)*hidden`，再沿 `axis` 做 MX 量化。若拆成两个算子，高精度 SwiGLU 输出需要写 GM，再读回做 ReduceMax 和量化；融合后中间值停留在 UB/寄存器中，省一次大张量写读，并能共用 tiling。

它支持 activate_dim/axis 为 -1 或 -2，并有 group_index 可选路径。Kernel 入口按 group index 类型、axis 是否尾轴、activate_dim 是否尾轴和 roundMode 编译期选择 `AxisLast`/`AxisNotLast` 类。测试验证 FP16/BF16→FP4/FP8、scaleAlg=1、小 shape，以及 FP8 非 rint、非法 axis/dst 的失败路径。

##### 3.8 SwigluMxQuantWithDualAxis：融合后生成行列两份表示

输入要求二维 `[M,2N]`，先生成 `[M,N]` SwiGLU 结果，再复用双轴量化逻辑。可选 `group_index` 是 cumsum 行边界，尤其影响 -2 轴 scale：有 group 时 shape 为 `[floor(M/64)+G,N,2]`，因为每组都要独立 pad；无 group 时为 `[ceil(M/64),N,2]`。

base kernel 的注释和流程显示：先按 group 分任务、`ComputeSwiglu`，随后调用 OCP/cuBLAS 两套双轴 scale 计算和 `ComputeY1/Y2`，最后对 scale2 交织。这是“算子融合 + 布局融合”，不是简单在 API 层串联两个现有 kernel。

##### 3.9 SwigluBackwardMxQuantWithDualAxis：当前只开放 FP8 E4M3FN + scaleAlg=1

反向公式为：

\[
g_A=g_y\,B\,\sigma(A)\,[1+A-A\sigma(A)],\qquad
g_B=g_y\,A\,\sigma(A)
\]

Kernel 先生成拼接后的 `grad_x`，再沿 -1/-2 轴输出两份量化梯度和 scale。`group_index` 同样保护行方向组边界。

这里有一个值得面试强调的源码事实：OpDef 预留了 FP4、E5M2 等 8 个 dtype 分支，但当前 Host tiling 的 `Y_SUPPORT_DTYPE_SET` 只有 `FLOAT8_E4M3FN`，并强制 `scaleAlg==1`、`activateLeft==true`，测试也专门验证 OCP 和其它 dst_type 失败。因此阅读算子不能只看 OpDef；真正可用能力要以 API 文档、Host Check 和 tiling 测试的交集为准。

#### 4. 核心代码片段 #1：OCP shared exponent 直接生成 E8M0 scale

位置：`ops-nn/quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis.h:797-862`（真实源码 66 行）。

```cpp
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScaleOcp(__ubuf__ uint16_t* maxExpAddr,
                                                                                __ubuf__ uint16_t* mxScaleLocalAddr,
                                                                                __ubuf__ uint16_t* recipScaleLocalAddr,
                                                                                uint16_t loopNum1VF,
                                                                                uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xExp0;
        Reg::RegTensor<T> xExp1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f4Emax_);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);

            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32,
                preMaskScale); // 128 个scale，占用 128 * 1 Btyes = VF_LEN_32 * sizeof(uint16_t)

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(recipScaleLocalAddr, halfScale, VF_LEN_16,
                                                                          preMaskScale);
        }
    }
    return;
}
```

##### 4.1 整体作用

上游已经把每个 32 元素块的最大绝对值化成“最大指数位”写入 `maxExpAddr`。本函数不再调用 `log2` 或 `pow`，而是直接做指数域减法，生成一份对外的 E8M0 scale 和一份只在 UB 中使用的 reciprocal scale。前者写回模型数据，后者马上用于 `x * reciprocalScale`。

##### 4.2 逐段解释

- 813-825 行准备常量：指数 mask、目标 `emax`、BF16 bias、0、NaN 和特殊阈值。使用寄存器常量让整个向量批次并行处理。
- 833-839 行加载块最大指数，并建立 INF/NAN 与过小指数 mask。过小值被钳到 `f4Emax_`，避免无符号指数减法下溢；零块随后由 mask 恢复为 scale 0 的约定表示。
- 841-848 行是核心：`sharedExp=xMaxExp-emax`，右移去掉 BF16 指数在 16-bit 容器中的位偏移，再以 `DIST_PACK_B16` 压成 1-byte E8M0。也就是说公式中的减法在硬件里就是指数位整数减法。
- 850-858 行同步构造 reciprocal 的 BF16 指数编码。量化阶段只需把输入乘以这个倒数，无需真正做逐元素除法。

##### 4.3 关键边界

全零块、INF/NAN 和指数特殊值不能走普通减法，否则会产生下溢、伪正常数或错误倒数。代码使用多个 mask 做无分支 Select，既保持 SIMD 效率，也明确规定特殊值传播。`totalScaleInUB` 控制最后一个向量批次的有效 lane，防止尾部垃圾写出。

##### 4.4 面试式自我解释

问：为什么 OCP 路径比“先转 FP32、除最大值、再 log2”快？答：因为浮点格式本身已经存了 \(\lfloor\log_2 |x|\rfloor\) 的信息；ReduceMax 找到最大指数后，shared exponent 只是整数减法，scale 与倒数都能通过重新编码指数得到。其代价是必须精细处理 subnormal、0、Inf、NaN。

#### 5. 核心代码片段 #2：FP8 反量化的展开、广播和相乘

位置：`ops-nn/quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tail_axis.h:476-517`（真实源码 42 行）。

```cpp
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ComputeData(__ubuf__ uint8_t* xLocalAddr,
                                                              __ubuf__ float* scaleBufAddr, __ubuf__ U* yLocalAddr,
                                                              uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFp32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFp8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint8_t> vdFp8_0, vdFp8_1;
        Reg::RegTensor<float> vdFp32_0_0, vdFp32_0_1, vdFp32_0_2, vdFp32_0_3;
        Reg::RegTensor<float> vdFp32_1_0, vdFp32_1_1, vdFp32_1_2, vdFp32_1_3;
        Reg::RegTensor<float> vdScale_0, vdScale_1;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B8>(
                vdFp8_0, vdFp8_1, xLocalAddr, vfLen8Double);

            Reg::Interleave(vdFp8_0, vdFp8_1, vdFp8_0, vdFp8_1);
            Reg::Cast<float, T, castTraitFp8ToFp32_0>(vdFp32_0_0, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_1>(vdFp32_0_1, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_2>(vdFp32_0_2, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_3>(vdFp32_0_3, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_0>(vdFp32_1_0, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_1>(vdFp32_1_1, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_2>(vdFp32_1_2, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_3>(vdFp32_1_3, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);

            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
                vdScale_0, scaleBufAddr, elementAfterReduce_);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
                vdScale_1, scaleBufAddr, elementAfterReduce_);

            Reg::Mul(vdFp32_0_0, vdFp32_0_0, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_1, vdFp32_0_1, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_2, vdFp32_0_2, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_3, vdFp32_0_3, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_1_0, vdFp32_1_0, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_1, vdFp32_1_1, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_2, vdFp32_1_2, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_3, vdFp32_1_3, vdScale_1, maskFp32);
```

`DIST_DINTLV_B8` 和后续 `Interleave` 先恢复 FP8 数据的寄存器顺序；8-bit 到 32-bit 扩展需要多组 cast layout，因此 512 个 FP8 值分到八个 FP32 寄存器。`DIST_E2B_B32` 的关键不是普通 load，而是把每个块 scale 扩展广播到对应数据 lane。510-517 行就是反量化本质：每个 FP32 量化值乘所属 32 元素块的 scale。后续代码再重排顺序并 cast/store 为 FP32、BF16 或 FP16。

FP4 路径与此思想相同，但要先从每字节拆出高低两个 4-bit 值，计算类型主要用 BF16，以降低寄存器和 UB 压力。

#### 6. 与 INT8 动态量化的对比

| 维度        | MX FP4/FP8                                | 常见对称 INT8                          |
| ----------- | ----------------------------------------- | -------------------------------------- |
| 块内表示    | 低比特浮点，各元素仍有自己的指数/尾数     | 整数码，块内共享线性 scale             |
| scale       | 常为 E8M0，即 2 的幂                      | FP16/FP32 任意实数                     |
| 公式        | `cast_fp(x / 2^e)`                      | `round(x / s)`，常夹到 [-127,127]    |
| 动态范围    | FP8/FP4 自身指数 + shared scale，适合长尾 | 由共享 scale 决定，块内均匀间隔        |
| 误差形态    | 相对误差更接近浮点，越大数间距越大        | 绝对量化步长固定                       |
| 元数据/硬件 | scale 1 byte，幂次缩放便于指数路径        | scale 更精确但占用更高，通常需要浮点乘 |
| 零点        | 本组算子没有 affine zero-point            | 可有 zero-point；对称量化通常为 0      |
| 适用        | 训练、激活/梯度、FP8 GEMM 流水线          | 推理权重/激活、整数 GEMM 生态成熟      |

面试中的标准回答应包含权衡：MX 不是无条件比 INT8 精确。若块内数值范围窄且均匀，INT8 有 255 级线性分辨率，可能优于 FP4/FP8；当块内跨度大、需要保留浮点动态范围或硬件原生支持 FP8 时，MX 更有吸引力。

#### 7. 边界条件与真实源码陷阱

1. **零块**：`amax=0` 时 `log2(0)` 不可直接计算，Kernel 用 mask 约定 scale/倒数表示；反量化必须与该约定匹配。
2. **NaN/Inf**：最大值归约会污染整个 block。OCP kernel 显式检测指数全 1 并写特殊 E8M0；应用侧要决定是否允许特殊值传播。
3. **尾块**：不足 blocksize 的元素逻辑上 pad 0，但物理搬运仍需 mask、对齐和安全地址；测试中的 2360/2370、600 列尾块专门覆盖这一点。
4. **FP4 打包**：最后一维至少为偶数；融合反向等路径可能要求能被 4、64 整除。shape 合法不代表任意 stride 都高效，API 通常先 contiguous。
5. **scale 偶数 pad**：scale 最后一维固定 2。反量化输入 scale shape 必须与量化公式完全一致，不能只按元素总数相等。
6. **group 边界**：必须非递减，最后值等于 M；空组虽 Kernel 有 `groupRows<=0` 防护，但 API/Host 约束应优先理解，不应依赖设备侧跳过。
7. **round mode**：FP8 只支持 rint；FP4 才允许 floor/round。`round` 与 `rint` 的 tie 行为不能凭名字猜，应以平台 Cast 语义为准。
8. **接口声明与实现能力不完全相同**：AntiMxQuant 当前只支持尾轴；SwiGLU backward dual-axis 当前只开放 E4M3FN、scaleAlg=1。判断支持范围要看 Host Check 和测试。
9. **双轴输出不是互换的**：`y1/mxscale1` 与 `y2/mxscale2` 必须配套消费，拿错 scale 会得到形状可能合法但数值完全错误的结果。
10. **二级量化反量化**：DualLevel 需要同时乘 `level1_scale` 和 `level0_scale`；只用 E8M0 level1 scale 会漏掉粗粒度幅值。

#### 8. 测试反向阅读结论

| 测试证据                                                    | 揭示的设计意图                                           |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| DynamicMxQuant kernel 分别测 scaleAlg 0、FP8 alg1、FP4 alg2 | 三条 scale 算法是独立实现路径，不只是属性换常量          |
| dual-axis alg2 测默认、自定义、dst=7+floor                  | 自定义有效上界会改变 scale 生成和舍入结果                |
| block quant 测非对齐列、3D、错误 scale dtype/shape          | 二维块语义允许 batch，但对尾块和两个 scale 布局要求严格  |
| grouped API 大量失败用例                                    | 参数检查是算子契约的重要组成，不应只研究 kernel 正常路径 |
| AntiMxQuant 测 FP8→FP32 tail、多核、FP4→不同输出          | 反量化的格式展开、尾块和核切分均是核心功能               |
| SwiGLU forward 测 invalid FP8 round、invalid axis/dst       | 融合算子没有放宽基础 MX 的 dtype/round 约束              |
| SwiGLU backward 测 invalid OCP                              | 当前反向融合仅支持 cuBLAS scale，不应从通用模板误判能力  |

#### 9. 面试高频问题与追问

##### 9.1 基础必答

**Q1：什么是 MX 量化？** 以小块为单位共享一个 2 的幂 scale，块内每个元素仍编码成 FP4/FP8。它把高精度张量表示成“低比特浮点数据 + E8M0 shared exponent”。

**Q2：为什么 `shared_exp=floor(log2(amax))-emax`？** `floor(log2(amax))` 是块最大值的实际指数；减去目标格式最大正规指数后，除以 \(2^{shared\_exp}\) 会把最大值移到目标格式可表示上界附近。用 floor 遵循 OCP shared exponent 定义，但与向上取整型 scale 算法在溢出保护上有不同细节。

**Q3：为什么 block 越小通常精度越高？** 小块内动态范围更一致，outlier 影响的元素少，更多值能利用尾数。代价是 scale 数量、ReduceMax 次数、带宽和调度开销上升。

**Q4：E4M3 与 E5M2 怎么选？** E4M3 多一位尾数，局部精度更好；E5M2 多一位指数，动态范围更大。前向激活常偏好 E4M3，梯度长尾更可能需要 E5M2，但最终要看分布与硬件支持。

##### 9.2 进阶追问

**Q5：双轴量化为什么输出两个 y？** 同一元素沿两个方向属于不同 block，scale 不同，因此归一化后的低比特码也不同。只输出一份 y 无法同时与两套 scale 保持一致。

**Q6：DynamicBlockMxQuant 和 dual-axis 的根本区别？** 前者 32×32 共用一个 scale，只把它广播成两种布局；后者分别对每行的 32 列和每列的 32 行求 scale，会得到两份真实不同的量化结果。

**Q7：二级量化解决什么问题？** 用 512 粒度 FP32 scale 保存粗尺度，再用 32 粒度 E8M0 scale 捕获局部范围，使 FP4 编码不必单独承担全部动态范围。它增加了一个 scale 查找与乘法，但改善极低比特表示的可控性，并可融合 smooth scale。

**Q8：量化 Kernel 为什么保存 reciprocal scale？** 量化公式是除法，但向量乘法通常吞吐更高。scale 为 2 的幂时，倒数也可通过指数重编码得到，避免真实除法和 FP32 `pow`。

**Q9：如何验证量化—反量化误差？** 按 block 重建 \(\hat{x}=dequant(q,s)\)，统计 max abs、MAE、RMSE、相对误差、余弦相似度，并单独观察 0、subnormal、最大有限值、尾块和 NaN/Inf。不能只用平均误差，因为少量饱和可能严重破坏下游结果。

**Q10：如何判断源码中算子真正支持什么？** 取 README/API 文档、OpDef/Proto、Host 参数检查、tiling key、binary config 和测试的交集。OpDef 的宽 dtype 列表可能只是预留，Host tiling 的失败分支才反映当前产品能力。

#### 10. 面试学习路径

##### 第一阶段：先会手算，不看 Kernel（半天）

1. 记住四种目标格式的 `emax` 和 E8M0 scale 含义。
2. 对 `[1,2,4,8]`、全零块、含 outlier 的 32 元素块手算 OCP scale、量化值、反量化值。
3. 手算 `[2048,2360]`、`[B,M,N]` 的单轴和双轴 scale shape。

检索练习：合上文档回答“为什么 scale shape 多一维 2”“为什么最后一个 scale 要 pad 0”。

##### 第二阶段：掌握基线实现（1 天）

阅读顺序：`DynamicMxQuant README/API → proto/def → infershape → tiling → kernel 入口 → tail_axis ComputeScaleOcp → tests`。目标是能画出从 aclnn 参数到 `ComputeScaleOcp/ComputeData` 的调用链，并解释 tail/non-tail 为什么分 kernel。

##### 第三阶段：做横向对比（1 天）

按以下顺序对比：

```text
单轴 DynamicMxQuant
  → 双轴 DynamicMxQuantWithDualAxis
  → 32×32 DynamicBlockMxQuant
  → group-aware GroupedDynamicMxQuant
  → 二级 DynamicDualLevelMxQuant
```

每看一个算子，只回答三个问题：block 是谁定义的？一个 block 产生几个 scale？scale shape 为什么这样排布？如果三问答不清，就不要继续读优化代码。

##### 第四阶段：补反量化与融合（1 天）

先读 AntiMxQuant，建立“量化值与 scale 必须配套”的闭环；再读 SwiGLU forward 单轴、forward 双轴、backward 双轴。重点不是背 sigmoid 公式，而是解释融合为什么省 GM 带宽、group 边界如何影响 scale2，以及 backward 当前为何只开放受限组合。

##### 第五阶段：模拟面试（持续复习）

用 5 分钟完成一次白板讲解：

1. 画一个 32 元素块，解释 amax→shared exponent→E8M0→FP8。
2. 画 `[M,N]` 的行块与列块，解释为什么双轴有两套 y/scale。
3. 写出 `x_hat=q*scale`，说明不可逆误差来源。
4. 比较 MX FP8 与 INT8。
5. 给出 5 个边界测试。

自我解释验收：不看源码，能否说明 OCP 核心代码为什么不需要 `log2`；能否从 `[M,2N]` 推导 SwiGLU dual-axis 四个输出 shape；能否指出 AntiMxQuant 和 backward fused 算子的当前实现限制。三项都能答，才算达到面试可用程度。

#### 11. 复杂度与性能总结

所有量化/反量化路径对元素数 \(N\) 都是时间复杂度 \(O(N)\)，额外 scale 存储约为 \(O(N/blocksize)\)。双轴需要读取/处理同一逻辑数据两种方向，计算与输出量接近单轴的两倍；32×32 block 减少 scale 归约与元数据，但可能损失精度；grouped 增加边界调度和 pad；dual-level 增加一级归约、中间归一化及 FP32 scale。

仓库的主要优化手段包括：GM/UB 双缓冲、向量 ReduceMax、指数位运算替代 log/pow、reciprocal 乘法替代除法、编译期模板消除 dtype/round/axis 分支、tail-size 专用 kernel、scale pack/interleave，以及融合 SwiGLU 避免中间张量落 GM。退化场景主要是大量小 group、极短尾块、非尾轴低连续性、scale 元数据占比过高，以及 outlier 导致大块有效精度下降。

#### 12. 质量与覆盖结论

- 9 个指定目录均覆盖了 README、公开 API 或 API 文档、Graph/OpDef、Host shape/tiling、Kernel 入口/核心类及代表性测试。
- 两段核心代码分别覆盖正向 OCP shared exponent 和 FP8 反量化广播乘法，均为真实源码且超过 30 行，标注了精确行号。
- 最重要的工程结论是：MX 的数学公式很短，但可用性由 dtype×scaleAlg×roundMode×axis×shape 的组合约束决定；这些约束分散在 API、Host tiling 和测试中。
- 最重要的面试结论是：先用“块、scale 数量、scale 布局”建立统一模型，再理解单轴、双轴、二维块、分组、二级和融合变体，不要逐个背算子名称。

### 专题三：03 反量化、QAT 与参数辅助：从公式到 Ascend 内核

> 面试目标：能够解释“INT32 累加值为什么不能直接转 FP16”“伪量化为什么前向离散、反向仍能训练”“Nudge 到底修正了什么”“scale 为什么还要打包成 UINT64”，并能从 Graph、Host、Kernel 和测试四层定位实现证据。

#### 理解验证状态

| 主题                                     | 结论                                                     |
| ---------------------------------------- | -------------------------------------------------------- |
| 反量化公式与 INT32 累加                  | 已验证                                                   |
| per-tensor / per-channel / per-head 广播 | 已验证                                                   |
| QAT quant-dequant 与 Nudge               | 已验证                                                   |
| STE、梯度 mask、min/max 梯度             | 已验证                                                   |
| FP19 scale 与 INT9 offset 打包           | 已验证                                                   |
| 融合收益与并发边界                       | 已验证                                                   |
| 20 个指定目录覆盖                        | 已完成；个别目录本身缺 README、Kernel 或测试，已明确标注 |

#### 项目完整地图与覆盖映射

| 目录                                              | 定位                                                    | 已读取的主要证据                                                      | 测试与边界                                                              |
| ------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| ascend_anti_quant                                 | 标量属性 per-tensor 反量化                              | README、proto、def/infer、arch35 tiling/kernel                        | 覆盖 sqrt/non-sqrt、INT8/FP8、dtype 失败；仅图模式                      |
| ascend_anti_quant_v2                              | Tensor scale/offset 反量化                              | README、aclnn 文档/API、proto、def/infer、regbase tiling、多种 kernel | 60 个左右用例覆盖 per-tensor/channel/head、无 offset、INT4 和错误 shape |
| ascend_dequant                                    | INT32 累加值配合硬件 deq_scale 恢复浮点                 | README、proto、def                                                    | 当前目录没有 AscendC kernel 和测试，是声明/注册型实现                   |
| dequant_bias                                      | INT32 反量化并融合 activation scale、weight scale、bias | README/docs/API、proto、def/infer/tiling、kernel                      | 覆盖多种 bias 类型与可选输入                                            |
| acts_ulq                                          | ULQ 伪量化训练辅助                                      | README、proto、graph infer、def/infer/tiling、kernel                  | 主要为 Host tiling 测试，覆盖 0~8 维广播                                |
| fake_quant_affine_cachemask                       | affine QDQ，同时缓存反向 mask                           | README/docs/API、proto、def/infer/tiling、kernel                      | 覆盖 per-tensor/per-channel、FP16/FP32、量化开关和 shape                |
| fake_quant_with_min_max_args                      | min/max 为属性的 per-tensor QAT 前向                    | README、proto、def/infer/tiling、kernel                               | 覆盖 2~16 bit、narrow range、NaN、非法 min/max                          |
| fake_quant_with_min_max_args_gradient             | Args 版本 STE 反向                                      | README、proto、def/infer/tiling、kernel                               | 覆盖范围内外、NaN、负零符号位                                           |
| fake_quant_with_min_max_vars                      | min/max 为 Tensor 的 per-tensor QAT 前向                | proto、def/infer/tiling、kernel                                       | 本目录无 README、无 tests；min/max 在设备端读取并 Nudge                 |
| fake_quant_with_min_max_vars_gradient             | Vars 版本三输出反向                                     | proto、def/infer/tiling、kernel                                       | Host 测试覆盖 rank、空 Tensor、shape 错误；无 kernel UT                 |
| fake_quant_with_min_max_vars_per_channel          | 尾轴逐通道 QAT 前向                                     | proto、def/infer/tiling、kernel                                       | 覆盖 rank1/2/4、大通道、block axis 和 bit 数                            |
| fake_quant_with_min_max_vars_per_channel_gradient | 逐通道 x/min/max 梯度                                   | proto、def/infer/tiling、kernel                                       | tests 目录只有构建文件，没有实际测试源，需视为测试缺口                  |
| ifmr                                              | 基于重建 MSE 搜索量化范围                               | README、proto、def/infer/tiling、kernel                               | 覆盖 percentile、search range/step 的合法与非法场景                     |
| trans_quant_param                                 | CPU 侧旧版参数打包 ACLNN 接口                           | README、docs/API、quant_util、UT                                      | 无 Graph/Kernel；输出长度除标量外按 16 对齐                             |
| trans_quant_param_v2                              | 设备侧 FP32→硬件 UINT64 参数打包                       | README/docs/API、proto、def/infer/tiling、kernel                      | 覆盖 scale/offset 广播、V2/V3、round_mode                               |
| dequant_swiglu_quant                              | Dequant+SwiGLU+Quant 融合                               | README/docs/API、proto、def/infer/tiling、多实现 kernel               | 大量用例覆盖动态/静态、bias、BF16、GPT-OSS 变体                         |
| swi_glu_quant                                     | SwiGLU+Quant 融合                                       | README/docs/API、proto、def/infer/tiling、kernel                      | 覆盖 FP16/BF16/FP32、动态、静态 per-tensor/channel                      |
| flat_quant                                        | 两次 Kronecker 小矩阵乘+量化                            | README/docs/API、proto、def/infer/tiling、cube/vector kernel          | 覆盖 INT4 per-token 与 FLOAT4 per-group                                 |
| dynamic_quant_update_scatter                      | INT8 动态量化后原地更新数据和 scale                     | README、proto、def/infer/tiling、多策略 kernel                        | 索引段不得重叠，否则多核写竞争导致不确定                                |
| dynamic_quant_update_scatter_v2                   | INT4 非对称量化后更新 data/scale/offset                 | README、proto、fusion pass、def/infer/tiling、kernel                  | 图融合测试覆盖 dtype、H 维、模式匹配与多 tiling key                     |

#### 1. 先建立总心智模型

量化链路可以压缩成五个动作：

<code>浮点值 → 选范围/求 scale、zero-point → 整数化 → 整数算子累加 → 反量化恢复浮点</code>

QAT 在训练阶段插入的是“模拟整数化再恢复浮点”的 QDQ；参数辅助算子把数学上的 scale/offset 变成硬件消费的位格式；融合算子则把中间结果留在寄存器或 UB 中，避免反复写回 GM。

#### 2. 反量化为什么常从 INT32 开始

设输入和权重量化为：

$$
q_x=\operatorname{round}(x/s_x)+z_x,\qquad q_w=\operatorname{round}(w/s_w)+z_w
$$

量化 MatMul 的累加器是：

$$
A_{m,n}=\sum_k(q_x-z_x)(q_w-z_w)
$$

单个乘积可能是 INT8×INT8，但 K 维求和很快超出 INT8/INT16，因此硬件通常用 INT32 累加。恢复真实尺度时：

$$
y_{m,n}\approx A_{m,n}\cdot s_x[m]\cdot s_w[n]
$$

这也解释了 <code>dequant_bias</code> 的二维广播：weight_scale 沿 N 维变化，activation_scale 沿 M 维变化。kernel 的 NDDMA 复制分别把列参数复制到多行、把行参数复制到多列。

##### bias 放在 scale 前还是后

<code>dequant_bias/op_kernel/arch35/dequant_bias_kernel.h:295-330</code> 给出了很适合面试的答案：

- bias 为 INT32 时，先执行 <code>A + bias_int32</code>，再 cast FP32 并乘 scale；该 bias 已处于累加器尺度。
- bias 为 FP16/BF16/FP32 时，先完成 <code>A × weight_scale × activation_scale</code>，再加浮点 bias。

所以三种 README 公式并非重复，而是表达不同 bias 域：

$$
y=A s_w s_a,\quad y=(A+b_q)s_w s_a,\quad y=A s_w s_a+b_f
$$

#### 3. 反量化算子族与广播

<code>ascend_anti_quant</code> 使用标量属性，公式为：

$$
y=\operatorname{cast}((x+offset)\cdot scale)
$$

当 sqrt_mode 为 true 时再乘一次 scale。它只支持 per-tensor，优点是参数无需作为 Tensor 搬运，缺点是不能适应不同通道分布。

<code>ascend_anti_quant_v2</code> 把 scale/offset 变为 Tensor，并在 Host 侧识别三种模式：

- per-tensor：参数只有一个元素。
- per-channel：非 1 维位于尾轴，对常见权重输出通道分别缩放。
- per-head：非 1 维位于倒数第二轴，对注意力 head 分别缩放；尾轴为 1 时会退化为 per-channel。

代码证据在 <code>ascend_anti_quant_v2/op_host/arch35/ascend_anti_quant_v2_regbase_tiling.cpp:126-170,236-266,305-335</code>。scale 可为 1 维，也可与 x 同 rank，但最多只能有一个非 1 维，且只能落在最后或倒数第二轴；offset 必须与 scale 同 shape、同 dtype。

<code>ascend_dequant</code> 更接近量化 MatMul 的后处理：输入明确为 INT32，deq_scale 可为 FP16 或已打包 UINT64，还可融合 sqrt_mode 和 ReLU。当前目录只有 Graph/Host 注册，不能把“接口存在”误说成“本仓已有 AscendC 内核”。

#### 4. QAT 前向：为什么输出仍是浮点

统一的 affine fake quant 公式是：

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max})
$$

$$
y=(q-z)s
$$

y 仍是浮点，因此训练时没有节省模型存储；它的价值是把裁剪、舍入和离散网格误差注入前向，使权重和激活提前适应部署时的整数路径。

<code>fake_quant_affine_cachemask</code> 直接接收 scale/zero_point，并额外输出：

$$
mask=(qval\ge q_{min})\land(qval\le q_{max})
$$

这个 mask 可在反向复用，避免再次做边界比较。<code>acts_ulq</code> 更进一步输出上下界 mask 和量化损失，适合训练算法显式利用裁剪/舍入误差。

#### 5. Nudge：不是“随便移动 min/max”

Nudge 的目的只有一个：让实数 0 精确落在整数网格上。对 Args 版本：

$$
s=(max-min)/(q_{max}-q_{min})
$$

$$
z^\*=q_{min}-min/s,\qquad z=\operatorname{clip}(\operatorname{round}(z^\*),q_{min},q_{max})
$$

$$
nudgedMin=(q_{min}-z)s,\qquad nudgedMax=(q_{max}-z)s
$$

例如 min=-6、max=6、8 bit、非 narrow range：scale=12/255，原始 zero-point=127.5，round 后为 128，因此 nudgedMin≈-6.02353、nudgedMax≈5.97647。范围略微平移，但 0 现在对应整数码 128。

##### 真实代码片段：Host 端精度敏感的 Nudge

位置：<code>quant/fake_quant_with_min_max_args/op_host/arch35/fake_quant_with_min_max_args_tiling_arch35.cpp:110-145</code>，共 36 行。

```cpp
ge::graphStatus FakeQuantWithMinMaxArgsTiling::CalcNudge()
{
    // TF Eigen Nudge() algorithm with all H1-H5 precision-sensitive cases handled.
    const float qMinF = narrowRange_ ? 1.0f : 0.0f;
    const float qMaxF = static_cast<float>((1ULL << static_cast<uint32_t>(numBits_)) - 1ULL);

    const float scale = (fMax_ - fMin_) / (qMaxF - qMinF); // 反量化
    // H1: scaleInv must be re-divided, NOT 1/scale.
    const float scaleInv = (qMaxF - qMinF) / (fMax_ - fMin_);

    // H2: zeroPointFromMin uses division (qMin - fMin / scale), NOT (qMin - fMin * scaleInv).
    const float zeroPointFromMin = qMinF - fMin_ / scale;

    // H3 & H4: closed-interval clip + std::round (round-half-away).
    float nudgedZeroPoint = 0.0f;
    if (zeroPointFromMin <= qMinF) {
        nudgedZeroPoint = qMinF;
    } else if (zeroPointFromMin >= qMaxF) {
        nudgedZeroPoint = qMaxF;
    } else {
        nudgedZeroPoint = std::round(zeroPointFromMin);
    }

    const float nudgedMin = (qMinF - nudgedZeroPoint) * scale;
    const float nudgedMax = (qMaxF - nudgedZeroPoint) * scale;

    // H5: quantZero must be recomputed from nudgedMin, NOT directly reuse nudgedZeroPoint.
    const float quantZero = std::floor(-nudgedMin * scaleInv + 0.5f);

    tilingData_.nudgedMin = nudgedMin;
    tilingData_.nudgedMax = nudgedMax;
    tilingData_.scale = scale;
    tilingData_.scaleInv = scaleInv;
    tilingData_.quantZero = quantZero;
    return ge::GRAPH_SUCCESS;
}
```

这段代码最值得追问的是“为什么不写 1/scale”。浮点除法与倒数再乘并不保证相同舍入，恰好落在半整数附近时可能让 zero-point 差 1，最终整段量化网格平移一个码。代码还刻意从 nudgedMin 重算 quantZero，说明兼容参考框架的逐 bit/逐舍入语义比代数化简更重要。

Args 版本在 Host 计算 Nudge，kernel 只做高吞吐逐元素 QDQ；Vars 版本的 min/max 是运行时 Tensor，只能由 kernel 从 GM 读取并在设备端计算 Nudge。这是“属性已知”与“数据依赖”的系统分层。

#### 6. STE 与梯度 mask

round 的真实导数几乎处处为 0，直接求导会让训练停止。QAT 使用 Straight-Through Estimator：

$$
\frac{\partial L}{\partial x}=g\cdot \mathbf{1}[nudgedMin\le x\le nudgedMax]
$$

ArgsGradient 的 min/max 是属性，因此只输出 x 梯度。VarsGradient 的 min/max 是可学习 Tensor，还输出：

$$
\frac{\partial L}{\partial min}=\sum_{x<nudgedMin}g,\qquad
\frac{\partial L}{\partial max}=\sum_{x>nudgedMax}g
$$

PerChannelGradient 对每个尾轴通道分别归约其他维度。其 kernel 按 dTileLen 分块，行切分时先写每核 workspace 再归并；单行大通道则按互不重叠的通道块切核，避免不必要的跨核 reduce。

ArgsGradient 的真实实现位于 <code>fake_quant_with_min_max_args_gradient_regbase.h:178-212</code>：先构造 GE/LE 两个谓词并合成 0/1 mask，再执行 gradient×mask。代码额外 OR 回上游梯度的 sign bit，修复乘法可能丢失 -0 的问题；NaN x 因比较均为 false 得到 mask=0，而 NaN gradient 仍按 IEEE754 传播。

#### 7. IFMR：量化参数也可以通过重建搜索

IFMR 不是普通前向 QAT，而是校准/参数搜索：

1. 通过 cumsum/CDF 和 percentile 得到初始 min/max。
2. 在 search_range 内按 search_step 生成候选裁剪上界。
3. 对每个候选执行 quant→round→clip→dequant。
4. 累加重建 MSE，多核归约后选损失最小的 scale/offset。

证据集中在 <code>ifmr/op_kernel/ifmr.h:347-390,413-435,445-503</code>。复杂度约为 O(NK)，K 是候选数；它比只看绝对最大值昂贵，但校准离线执行，换来更低的实际重建误差。异常值很多时 percentile+搜索通常优于 min-max；数据分布漂移时，离线最优参数也会退化。

#### 8. 参数辅助：scale 为什么打包成 UINT64

数学层的 FP32 scale 不是所有矩阵单元能直接消费的格式。<code>trans_quant_param_v2</code> 把它编码为硬件约定：

- FP32 的高 19 位放入结果对应位段，并设置 bit46 标志。
- round_mode=0 直接截断；round_mode=1 使用 round bit、sticky bits 和保留尾数最低位实现 round-to-nearest-even。
- offset 先 RINT，再裁剪到 [-256,255]，保留 9 位二补码，放到 bit37~45。

关键代码在 <code>trans_quant_param_v2/op_kernel/trans_quant_param_v2.h:23-24,270-320</code>。旧版 <code>trans_quant_param</code> 在 CPU 侧完成同类打包，并把非标量输出补齐到 16 个 UINT64；V2 将其做成图算子/设备 kernel，适合动态 Tensor 参数和图内流水。

面试时应强调：UINT64 不是数值意义上的“把 float 转整数”，而是位级协议容器；对它做普通整数算术没有意义。

#### 9. 代表性融合及收益

| 融合算子                        | 被合并链路                   | 核心收益                                                        | 关键边界                                                               |
| ------------------------------- | ---------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| dequant_swiglu_quant            | Dequant→SwiGLU→Quant       | INT32/浮点中间张量留在片上；减少 3 次 kernel 启动和中间 GM 读写 | INT32 输入必须有 weight/activation scale；输出尾轴、dtype、UB 总量受限 |
| swi_glu_quant                   | SwiGLU→静态/动态 Quant      | 激活结果直接求 row max/scale 并量化；适合 MoE 分组              | groupIndex 支持 cumsum/count；INT4 尾轴打包约束                        |
| flat_quant                      | 两次 Kronecker 变换→Quant   | 变换与低比特落盘一次完成                                        | INT4 per-token 与 FLOAT4 per-group 的 shape/scale 格式不同             |
| dynamic_quant_update_scatter    | DynamicQuant→两个 Scatter   | 不生成独立量化结果和 scale 临时张量                             | indices 更新段不得重合                                                 |
| dynamic_quant_update_scatter_v2 | DynamicQuantV2→三个 Scatter | 一次产生并原地写入 INT4、scale、offset                          | 只匹配 INT4 非对称路径，不支持 smooth_scale                            |

<code>dynamic_quant_update_scatter_v2</code> 还有真实图融合 Pass：在 <code>fusion_pass.cpp:211-251</code> 匹配 DynamicQuantV2、三个 Scatter 和 offset 的 Neg，在 <code>291-294</code> 拒绝非 INT4，在 <code>540-561</code> 用一个融合节点替换。收益不是减少算术量，而是减少中间 Tensor 的 GM 往返、调度开销和同步点。

#### 10. 测试反向揭示的边界

- AscendAntiQuantV2 的错误 shape 测试说明广播是白名单，不是任意 NumPy broadcast。
- FakeQuant Args/Gradient 测试专门覆盖 NaN、半整数舍入和 -0，表明“数值语义一致”是验收目标。
- IFMR 对 percentile、search_range、search_step 有大量失败用例，参数搜索必须先保证候选集合合法。
- DynamicQuantUpdateScatter 明确禁止重叠索引段；这不是数学错误，而是多核无序写的数据竞争。
- ascend_dequant、fake_quant_with_min_max_vars、per_channel_gradient 存在实现或测试缺口，面试/评审时必须区分“API 约束”“源码实现”和“测试证明”三种证据等级。

#### 11. 面试学习路径

##### 第一阶段：先会手算

1. 手算对称/非对称量化的 scale、zero-point。
2. 手算 -6~6、8 bit 的 Nudge，解释端点为什么改变。
3. 从 INT8×INT8 推出 INT32 accumulation 和双 scale 反量化。
4. 分别写出 INT32 bias 与浮点 bias 的公式。

验收题：给定 A[2,3]、activation_scale[2]、weight_scale[3]，说出每个输出元素使用哪个 scale。

##### 第二阶段：读基础算子

按 <code>ascend_anti_quant → ascend_anti_quant_v2 → ascend_dequant → dequant_bias</code> 阅读。每个算子都从 proto 看契约，从 Host 看 shape/tiling，再从 kernel 看 cast、广播和流水，最后用测试确认失败边界。

##### 第三阶段：掌握 QAT

按 <code>fake_quant_affine_cachemask → min_max_args → args_gradient → vars → vars_gradient → per_channel</code> 阅读。重点不是背公式，而是回答：Nudge 在哪一侧计算、mask 是否缓存、min/max 是否可学习、跨通道梯度如何归约。

##### 第四阶段：理解部署衔接

学习 IFMR 如何选参数，再学习 TransQuantParam 如何把参数编码给硬件。这样能把“训练得到范围”连到“推理硬件真正消费的位格式”。

##### 第五阶段：用融合题收尾

画出 Dequant→SwiGLU→Quant 与 DynamicQuantV2→Scatter×3 的未融合图，标出每个中间 Tensor 的写回和再次读取，再解释融合为何主要优化带宽和 launch，而不是改变数学公式。

#### 12. 高频追问与标准回答

1. **Fake quant 为什么不直接输出 INT8？** 训练图仍需浮点算子和梯度，QDQ 只模拟整数网格误差；真正 INT8 存储通常发生在部署转换后。
2. **Nudge 为什么要保证 0 可表示？** 卷积/MatMul 的 padding、ReLU 零点和稀疏零值都依赖精确零；零不可表示会引入系统性偏置。
3. **STE 是真实导数吗？** 不是，它是有偏梯度估计；优点是能训练，代价是优化目标与真实离散函数不完全一致。
4. **per-channel 为什么通常更准？** 每个通道动态范围不同，共享 scale 会让小范围通道浪费码字；代价是参数搬运、广播和硬件调度更复杂。
5. **INT32 bias 为什么先加？** 它已经按 s_x·s_w 量化到累加器域；先反量化再把它当浮点加会尺度错误。
6. **scale 打包为何保留 19 位？** 这是矩阵硬件接口精度/位宽协议，不是通用 IEEE 类型；round_mode=1 用 RNE 降低截断偏差。
7. **融合一定更快吗？** 通常减少 GM 流量和 launch，但可能增加 UB/寄存器压力、降低并行度；需要看 shape、dtype 和 tiling，不能只凭节点数判断。
8. **最危险的工程边界是什么？** 舍入模式不一致、错误广播、INT4 尾轴未对齐、重叠 scatter 索引、NaN/Inf、以及把“目录中有注册”误当成“已有可运行 kernel”。

#### 质量验证清单

- [X] 20 个指定目录均映射到功能和源码证据。
- [X] 讲清反量化公式、INT32 累加与二维 scale 广播。
- [X] 讲清 QDQ、Nudge、STE、gradient mask 与 min/max 梯度。
- [X] 讲清 FP19/INT9 硬件参数打包。
- [X] 包含 36 行真实代码及准确文件行号。
- [X] 说明融合收益、适用条件和并发边界。
- [X] 给出可执行的面试学习路径与高频追问。

#### 参考

- TensorFlow FakeQuantWithMinMaxVars：[https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_vars](https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_vars)
- Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference：[https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- Bengio et al., Estimating or Propagating Gradients Through Stochastic Neurons：[https://arxiv.org/abs/1308.3432](https://arxiv.org/abs/1308.3432)

## 7. 测试用例反向理解

### 7.1 测试覆盖矩阵

| 核心能力                | 代表测试                                                                                                                                                      | 测试揭示的真实约束                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 静态量化 API 参数校验   | [`test_aclnn_ascend_quant_v3.cpp`](../ops-nn/quant/ascend_quant_v2/tests/ut/op_host/op_api/test_aclnn_ascend_quant_v3.cpp)                                   | 不同 SoC 对 dtype、axis、shape、INT4/INT32 输出和 roundMode 的支持不同，公式正确不代表接口组合合法。 |
| 动态量化多核与尾块      | [`test_dynamic_quant_dav_2002.cpp`](../ops-nn/quant/dynamic_quant/tests/ut/op_kernel/test_dynamic_quant_dav_2002.cpp)                                        | 覆盖尾核数量、最后尾行和 innerLoopTail，说明性能实现必须显式处理“不整除”。                         |
| MX 量化 Tiling          | [`test_dynamic_mx_quant_tiling.cpp`](../ops-nn/quant/dynamic_mx_quant/tests/ut/op_host/arch35/test_dynamic_mx_quant_tiling.cpp)                              | 覆盖尾轴/非尾轴、FP4/FP8、FP16/BF16/FP32、优化路径和非法 dtype/rank。                                |
| MX 反量化 Kernel        | [`test_anti_mx_quant.cpp`](../ops-nn/quant/anti_mx_quant/tests/ut/op_kernel/test_anti_mx_quant.cpp)                                                          | 基础、600 列尾块和多核三种路径，scale 布局正确之外还要保证尾数据不越界。                             |
| Fake Quant 参数与 Nudge | [`test_fake_quant_with_min_max_args_tiling.cpp`](../ops-nn/quant/fake_quant_with_min_max_args/tests/ut/op_host/test_fake_quant_with_min_max_args_tiling.cpp) | 位宽 2/4/8/16、narrow range、空 shape、min≥max、NaN/Inf、非法 dtype 都有明确期望。                  |
| Fake Quant 数值正确性   | [`test_fake_quant_with_min_max_args.cpp`](../ops-nn/quant/fake_quant_with_min_max_args/tests/ut/op_kernel/test_fake_quant_with_min_max_args.cpp)             | 使用 CPU golden 比较真实输入，并验证 NaN 透传，说明边界语义也是算子契约。                            |
| 硬件参数转换并发        | [`test_trans_quant_param_v2_tiling.cpp`](../ops-nn/quant/trans_quant_param_v2/tests/ut/op_host/test_trans_quant_param_v2_tiling.cpp)                         | 除普通 shape 外还执行多线程 Tiling，要求 Host 注册和上下文处理不存在共享状态污染。                   |

### 7.2 从测试中必须学会的五个工程结论

1. **尾块不是小细节。** 量化 Kernel 大多按向量宽度和 UB 容量分块，shape 不整除是常态。尾块长度、最后一个核分到多少行、scale 输出如何 pad，都是正确性和越界风险的核心。
2. **dtype 组合受 SoC 约束。** 同一 API 在不同处理器上支持的输入、输出和 round mode 不完全相同。面试回答“支持 FP8/INT4”时应补一句“具体组合由 SoC 配置和接口文档约束”。
3. **shape 约束来自量化粒度。** per-channel 的 scale 必须和 axis 对应维度匹配；MX scale 常比输入多一维并带 pair/interleave 布局；INT4 输出可能要求尾轴为偶数。
4. **异常浮点值必须定义。** Fake Quant 的 Host 拒绝 NaN/Inf 范围，但 Kernel 对输入 NaN 选择透传。两者分别保护“参数不可计算”和“数据语义可预测”。
5. **round mode 会影响图融合。** 图融合测试会检查不同目标 dtype 对应的 round/rint 选择，说明融合 Pass 必须保持原始数值语义，而不只是把两个节点换成一个节点。

### 7.3 测试质量评价

- 正常路径：✅ API、Tiling、Kernel 都有代表性覆盖。
- dtype/shape/属性错误：✅ 静态量化和 MX 覆盖较充分。
- 尾块与多核：✅ 动态量化、AntiMxQuant 有专门用例。
- NaN/Inf：✅ Fake Quant 覆盖明确；其他算子可继续补充一致性测试。
- 跨算子互逆：⚠️ 建议增加 `quant → anti_quant`、`dynamic_mx_quant → anti_mx_quant` 的端到端误差界测试。
- 性能回归：⚠️ 单元测试主要验证选择和正确性，面试中要主动说明性能还需 ST/benchmark 和 profiling。

## 8. 应用迁移场景

### 8.1 LLM 激活：静态 per-tensor → 动态 per-token

**原始方案：** 整个激活张量共用离线 scale。优点是零运行时统计开销，缺点是不同 token 幅度差异会让小 token 分辨率不足，或大 token 饱和。

**迁移方案：** 沿最后一维对每个 token 计算 `amax`，得到每行 scale，再做对称量化。这对应 `dynamic_quant` 的核心路径。

**不变原理：** 都是 affine quant，最终仍执行缩放、舍入和饱和。

**需要修改：** scale 从一个标量变成去掉尾轴后的张量；Kernel 多一次行归约；下游矩阵乘必须读取对应 token 的 scale。

**WHY 值得：** LLM 激活天然随 token、层和请求变化，per-token 通常能用很小的元数据成本换取明显更稳定的精度。

### 8.2 量化矩阵乘：低精度乘法 + INT32 累加 + 反量化

典型推理链路是：权重与激活量化到 INT8，矩阵乘使用低精度乘法，但把部分和累加到 INT32，最后结合权重 scale、激活 scale 和 bias 恢复到 FP16/BF16。`ascend_dequant` 和 `dequant_bias` 体现的是末端恢复阶段。

**不变原理：** 反量化必须使用与前向量化一致的 scale/offset 约定。

**需要修改：** bias 在量化域还是浮点域加入，会改变公式；per-channel 权重 scale 与 per-token 激活 scale 的广播维度也必须匹配。

**WHY INT32 累加：** INT8 乘积和 K 维累加会迅速超出 INT8 范围。INT32 提供足够的累加动态范围，同时仍能使用整数矩阵乘通路。

### 8.3 PTQ → QAT

PTQ 在训练后校准并量化，成本低，但模型没有机会适应量化误差。若低比特导致精度下降，可以把真实量化替换成 Fake Quant：前向把值投影到量化网格，输出仍保持浮点以便继续使用训练算子；反向通过 STE 或 mask 传播近似梯度。

**不变原理：** scale、zero-point、round、clamp 与推理量化一致。

**需要修改：** 训练图中加入 Fake Quant 节点、范围参数学习或更新、梯度门控；导出部署模型时再替换成真实低精度算子。

## 9. 依赖关系与使用示例

### 9.1 内部依赖关系

| 关系                                 | WHY 这样依赖                                                               |
| ------------------------------------ | -------------------------------------------------------------------------- |
| API → Graph/OpDef                   | API 做用户侧参数校验和执行器构建，Graph/OpDef 固化框架可见的输入输出语义。 |
| Graph/OpDef → InferShape/InferDtype | scale 粒度和低比特打包会改变辅助输出 shape，必须在执行前确定。             |
| Host → Tiling                       | Host 根据 SoC、shape、axis、dtype 和 UB 容量选多核切分及 Kernel 模板。     |
| Tiling → Kernel                     | Kernel 不再做复杂动态决策，只消费紧凑 TilingData 执行高吞吐数据流。        |
| DynamicQuant → 下游量化 MatMul      | 量化值和运行时 scale 必须一起消费，否则无法正确解释低精度编码。            |
| DynamicMxQuant → AntiMxQuant        | MX scale 的 shape、pair 布局、axis 和 blocksize 必须完全一致。             |
| FakeQuant Forward → Gradient        | 反向需要前向范围或 mask，才能决定哪些值使用 STE 传播。                     |

### 9.2 手算示例：对称 per-token INT8

输入一行 `x = [-1.2, 0.2, 2.5]`，取 `qmax=127`：

```text
amax = 2.5
s = 2.5 / 127 ≈ 0.019685
q = round(x / s) = [-61, 10, 127]
x_hat = q * s ≈ [-1.2008, 0.1969, 2.5]
```

误差来自舍入。最大值 2.5 被精确映射到 127，而 0.2 只能落到最近格点。若改成全张量 scale，其他幅度更大的 token 可能把 `s` 拉大，使这一行误差继续增加。

### 9.3 手算示例：非对称范围

输入范围 `[-1, 5]`，使用无符号编码 `[0,255]`：

```text
s = (5 - (-1)) / 255 ≈ 0.023529
z = 255 - 5 / s ≈ 42.5
```

实现会按指定 round 规则把零点变为整数，并钳位到编码范围。非对称量化能同时覆盖 -1 和 5，不需要像对称量化那样把范围扩成 `[-5,5]`，因此对偏斜分布的有效分辨率更高。

## 10. 质量验证、面试题与复习清单

### 10.1 高频面试题速答

1. **什么是量化？** 用有限低精度编码近似浮点数据，核心是 scale、zero-point、round 和 clamp 的坐标映射。
2. **为什么变量都叫 scale，公式却有乘有除？** 有的 API 把 scale 定义成实数步长 `s`，有的定义成乘法器 `1/s`；必须先声明约定。
3. **静态和动态量化区别？** scale/offset 来自离线校准还是运行时当前输入；动态更适应分布，代价是在线归约。
4. **对称和非对称区别？** 对称零点固定、实现简单；非对称使用 min/max 和 offset，更适合偏斜分布。
5. **为什么权重常用 per-channel？** 不同输出通道范围差异大，而且权重是静态的，额外 scale 成本可控。
6. **为什么 LLM 激活常用 per-token？** 不同 token 幅度变化大，逐 token scale 能避免一个异常 token 污染所有 token。
7. **per-block 为什么适合更低位宽？** block 越小，局部动态范围越紧，有限编码能提供更细分辨率；代价是更多 scale 和归约。
8. **全零输入怎样处理？** 必须设置 epsilon/minScale 或特殊分支，防止 scale 为零和除零。
9. **为什么量化 MatMul 常用 INT32 累加？** 单个 INT8 乘积和长 K 维累加会超出 INT8，INT32 保证累加范围。
10. **bias 应该什么时候加？** 取决于 bias 位于量化域、INT32 累加域还是浮点域；位置不同，scale 组合公式不同。
11. **round、rint、floor、trunc 有何影响？** 它们在半整数和负数处结果不同，会产生不同偏差；硬件和 dtype 可能只支持部分模式。
12. **什么是饱和？** 超出目标 dtype 范围时钳到 qmin/qmax，避免溢出回绕，但会产生不可恢复的饱和误差。
13. **为什么 Fake Quant 输出还是浮点？** 它模拟量化网格误差，同时保持训练算子和自动微分使用浮点 Tensor。
14. **Nudge 做什么？** 调整 min/max 和零点，让零以及量化边界落在合法整数格点上，保证 quant-dequant 行为一致。
15. **round 不可导，QAT 怎么训练？** 使用 STE 或范围 mask，把有效区间内梯度近似直通，区间外阻断。
16. **MX 与普通 INT8 per-block 的区别？** MX 通常让 block 共享一个指数式 scale，块内使用 FP8/FP4；普通 INT8 block quant 是均匀整数网格。
17. **FP8 相比 INT8 的优势？** FP8 有指数，动态范围更大；INT8 格点均匀，固定范围内分辨率更稳定且整数矩阵乘生态成熟。
18. **算子融合为什么快？** 减少中间 Tensor 的全局内存读写、Kernel launch 和图调度，而不只是减少算术指令。
19. **Tiling 在量化算子中解决什么？** 决定多核分工、UB 分块、归约策略、尾块处理和不同 dtype/axis 的 Kernel 路径。
20. **如何测试一个量化算子？** 公式 golden、dtype/shape/axis/round 组合、全零/异常值、饱和边界、尾块、多核、quant-dequant 误差界和性能回归。

### 10.2 闭卷自我解释题

- 不看文档，分别写出 `Quantize` 与 `AscendQuantV2` 的公式，并解释两者 scale 的关系。
- 给定 `[B,T,H]`，写出 per-token 和 per-channel scale 的可能 shape。
- 推导非对称 zero-point 的两个等价公式。
- 解释为什么动态量化需要至少“归约 + 量化”两个逻辑阶段。
- 解释 MX scale 为什么会有额外维度、pair 和 interleave 布局。
- 说明 Fake Quant 的前向和反向为什么都不是普通 cast。
- 沿一个算子讲清 API、Graph、Host、Tiling、Kernel、Test 六层职责。

### 10.3 最终四能检查

- [ ] 能在 3 分钟内讲清量化主线，不混淆 scale 约定。
- [ ] 能手算一组对称和非对称量化，并估算误差。
- [ ] 能根据模型场景选择 per-channel/per-token/per-block/MX。
- [ ] 能解释一个 ops-nn 算子的工程调用链和尾块处理。
- [ ] 能从测试用例反推接口边界，而不是只背 README。
- [ ] 能说明 PTQ、QAT、Fake Quant、反量化和融合算子的关系。

## 覆盖率摘要

| 专题                   |       目录数 |                  覆盖率 | 代表算子                                                         |
| ---------------------- | -----------: | ----------------------: | ---------------------------------------------------------------- |
| 静态与动态整数数量化   |           10 |                   10/10 | AscendQuantV2、DynamicQuant、DynamicBlockQuant、QuantMax         |
| MX 与低比特浮点量化    |            9 |                     9/9 | DynamicMxQuant、DynamicBlockMxQuant、AntiMxQuant、SwigluMxQuant  |
| 反量化、QAT 与参数辅助 |           20 |                   20/20 | AscendAntiQuantV2、DequantBias、FakeQuant、IFMR、TransQuantParam |
| **合计**         | **39** | **39/39（100%）** | `ops-nn/quant` 全部算子目录                                    |

### 覆盖说明

- 核心数学、API 语义、Host/Tiling、Kernel 和代表性测试均有证据链。
- 对每个算子目录至少给出其在算子族中的定位；对五个代表片段进行源码级深读。
- `AscendDequant` 等仅有 Graph/Host 定义或实现不完整的目录也被纳入，并明确标注当前边界。
- 结论以当前工作区 `ops-nn` 源码为准；不同 SoC、后续版本的 dtype 和属性支持范围可能变化。
