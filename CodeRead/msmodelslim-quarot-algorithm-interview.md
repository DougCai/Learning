# msModelSlim QuaRot 算法深度理解分析

> 基于 `/home/caishengcheng/msmodelslim` 源码，聚焦 `processor/quarot/`、`ir/quarot.py`、各模型 `quarot.py` 适配及官方文档。
> 分析模式：**Deep** | 目标：面试可讲清 **算法原理 + 执行流程 + 文件地图 + 模型差异 + 代码对应关系**
>
> 配套文档：[DeepSeek 量化](./msmodelslim-deepseek-quant-interview.md) · [Qwen 量化](./msmodelslim-qwen-quant-interview.md) · [MXFP 量化](./msmodelslim-mxfp-quant-interview.md)

---

## 理解验证状态

| 核心概念                               | 自我解释 | 理解"为什么" | 应用迁移 | 状态     |
| -------------------------------------- | -------- | ------------ | -------- | -------- |
| 正交旋转不变性 XQ·QᵀW = XW           | ✅       | ✅           | ✅       | 已掌握   |
| Hadamard 矩阵构造                      | ✅       | ✅           | ✅       | 已掌握   |
| RMSNorm 融合（fuse_ln_linear）         | ✅       | ✅           | ✅       | 已掌握   |
| 左/右旋转语义                          | ✅       | ✅           | ✅       | 已掌握   |
| pre_run / preprocess / post_run 三阶段 | ✅       | ✅           | ✅       | 已掌握   |
| 离线旋转 vs LAOS 在线旋转              | ✅       | ✅           | ✅       | 已掌握   |
| HookIR → WrapperIR 推理导出链         | ✅       | ✅           | ✅       | 已掌握   |
| 模型适配器 RotatePair 映射             | ✅       | ✅           | ✅       | 已掌握   |
| AdaptRotation 可学习旋转               | ✅       | ⚠️         | ✅       | 基本掌握 |
| block_size / max_tp_size 约束          | ✅       | ✅           | ✅       | 已掌握   |

---

## 项目完整地图

### QuaRot 相关目录树

```
msmodelslim/
├── msmodelslim/processor/quarot/          # ★ 算法核心实现
│   ├── __init__.py                        # 对外导出 QuaRotProcessor 等
│   ├── common/
│   │   ├── quarot_utils.py                # 旋转矩阵创建、权重旋转、Norm 融合
│   │   ├── hadamard.py                    # Hadamard/Walsh 矩阵生成
│   │   └── hadamard_txt/                  # 非 2 幂维度的预置 Hadamard 矩阵
│   ├── offline_quarot/
│   │   ├── quarot.py                      # ★ QuaRotProcessor 主处理器
│   │   ├── quarot_interface.py            # QuaRotInterface / RotatePair 定义
│   │   └── laos_online.py                 # LAOS 在线旋转子处理器
│   └── online_quarot/
│       ├── online_quarot.py               # 通用 OnlineQuaRotProcessor
│       └── online_quarot_interface.py     # 模型自定义在线旋转配置
├── msmodelslim/ir/quarot.py               # ★ 推理侧 IR：Hook/Wrapper、旋转信息
├── msmodelslim/processor/adapt_rotation/  # AdaptRotation：数据驱动的旋转优化
│   ├── adapt_rotation_stage1.py           # 收集激活 + Hadamard 迭代优化
│   └── adapt_rotation_stage2.py           # 将优化后矩阵注入 QuaRotProcessor
├── msmodelslim/model/*/                   # 各模型 RotatePair 映射（模型结构差异）
│   ├── qwen3/model_adapter.py           # 标准 Dense：rot + rot_uv 两套
│   ├── deepseek_v3/quarot.py              # MLA：四套旋转矩阵
│   ├── kimi_k2/quarot.py / glm_5/quarot.py / ...
│   └── deepseek_v3_2/model_adapter.py   # OnlineQuaRotInterface（Indexer 路径）
├── msmodelslim/core/quant_service/modelslim_v1/save/
│   └── ascendv1.py                        # 导出 optional/quarot.safetensors
├── docs/zh/user_guide/quantization_algorithms/outlier_suppression_algorithms/quarot.md
├── lab_practice/*/                        # 各模型 QuaRot YAML 配置
└── test/cases/processor/quarot/           # 单元测试
```

### 文件清单（分类）

| 类别          | 文件路径                                                | 行数（约）  | 职责摘要                                                 |
| ------------- | ------------------------------------------------------- | ----------- | -------------------------------------------------------- |
| 主处理器      | `processor/quarot/offline_quarot/quarot.py`           | 257         | 三阶段调度：融合 Norm → 旋转权重 → 可选在线旋转        |
| 旋转工具      | `processor/quarot/common/quarot_utils.py`             | 267         | `create_rot` / `rotate_linear` / `fuse_ln_linear`  |
| Hadamard      | `processor/quarot/common/hadamard.py`                 | 156         | Walsh 递归 + 预置矩阵 + 随机 Hadamard                    |
| 接口定义      | `processor/quarot/offline_quarot/quarot_interface.py` | 144         | `QuaRotInterface` / `RotatePair` / `RotateCommand` |
| LAOS 在线     | `processor/quarot/offline_quarot/laos_online.py`      | 181         | o_proj / down_proj 在线 Kronecker 旋转                   |
| 通用在线      | `processor/quarot/online_quarot/online_quarot.py`     | 401         | input/output/replace/offline 四种在线模式                |
| IR 层         | `ir/quarot.py`                                        | 623         | HookIR/WrapperIR、推理导出元数据                         |
| DeepSeek 适配 | `model/deepseek_v3/quarot.py`                         | 167         | MLA 四套旋转矩阵映射                                     |
| Qwen3 适配    | `model/qwen3/model_adapter.py`（rotate 部分）         | ~70         | Dense 两套旋转矩阵映射                                   |
| 保存导出      | `core/.../save/ascendv1.py`                           | 相关 ~50 行 | `global_rotation` / heads / kronecker 导出             |
| 官方文档      | `docs/.../quarot.md`                                  | 422         | 原理、配置、适配指南                                     |

### 入口调用链

```
CLI: msmodelslim quant --config xxx.yaml
  │
  ▼
YAML spec.process[0]: type: quarot
  │
  ▼
QABCRegistry → QuaRotProcessor(model, config, adapter)
  │  adapter 必须实现 QuaRotInterface（在线旋转还需 LAOSOnlineRotationInterface）
  │
  ├── pre_run()          # Runner 逐层调度前：embedding 旋转 + 全局矩阵导出 Hook
  │     ├── adapter.get_ln_fuse_map()   → fuse_ln_linear
  │     ├── adapter.get_rotate_map()    → rotate_linear (pre_run_pairs)
  │     └── LAOSOnlineRotationProcessor.pre_run()  [online=True]
  │
  ├── preprocess(req)    # 每个 DecoderLayer 调度时：该层 Norm 融合 + 权重旋转
  │     ├── _filter_* (按 prefix 过滤本层命令)
  │     ├── _fuse_norm / _bake_mean / _rotate
  │     └── LAOSOnlineRotationProcessor.preprocess()  [online=True]
  │
  └── post_run()         # 收尾剩余命令 + HookIR → WrapperIR
        │
        ▼
后续 Processor: flex_smooth_quant → linear_quant → ascendv1_saver
  │
  ▼
Saver 遍历 IR 树：
  - QuaRotExtraInfoWrapperIR → optional/quarot.safetensors (global_rotation)
  - QuarotOnlineHeadRotationWrapper → heads_rotation 矩阵
  - QuarotOnlineKroneckerRotationWrapper → kronecker_rotation_m/n
```

---

## 1. 快速概览

| 维度      | 内容                                                                           |
| --------- | ------------------------------------------------------------------------------ |
| 语言/框架 | Python 3 + PyTorch                                                             |
| 代码规模  | QuaRot 核心 ~2660 行（processor + ir + 典型适配器）                            |
| 代码类型  | 量化前处理 Processor（data-free，无需校准数据）                                |
| 核心依赖  | PyTorch、Pydantic（配置校验）、msmodelslim IR 体系                             |
| 算法定位  | **离群值抑制（Outlier Suppression）**，为后续 W4A4/W8A8 等低比特量化铺路 |
| 本质操作  | 对权重/激活施加正交旋转 Q，数学等价前提下平滑激活分布                          |

一句话：**QuaRot 不改变模型数学语义，通过 Hadamard 正交旋转把激活里的"刺头离群值"摊平到多个通道，让后续量化器的 scale 不用为少数通道买单。**

---

## 2. 背景与动机（3 个 WHY）

### 问题本质

**要解决的问题：** LLM 激活张量存在 channel-wise 离群值（outlier），导致 per-channel / per-tensor 量化需要极大动态范围，低比特量化误差陡增。

**WHY 需要解决：** 不抑制离群值时，W4A4 等级别量化在 MLP 中间层、Attention 输出处会出现明显 perplexity 劣化；SmoothQuant 等方案通过迁移难度到权重侧缓解，但对激活分布本身的"尖峰"处理能力有限。

### 方案选择

**WHY 选择正交旋转：**

- 正交矩阵 Q 满足 Q·Qᵀ = I，保证 `Y = X·W` 在变换为 `Y = (X·Q)·(Qᵀ·W)` 后**数学严格等价**。
- Hadamard 矩阵是结构化正交矩阵，乘法可用快速 Walsh 算法，硬件友好。
- 旋转将单个通道的极端值**分散**到多个通道，使 per-channel max 更均衡。

**替代方案对比：**

| 方案                 | 思路                            | WHY 不单独使用                                        |
| -------------------- | ------------------------------- | ----------------------------------------------------- |
| SmoothQuant          | 将激活难度迁移到权重（α 混合） | 不直接重塑激活分布，对极强离群值仍吃力                |
| AWQ / GPTQ           | 基于 Hessian 的权重量化补偿     | 解决权重量化误差，不解决激活离群                      |
| FlatQuant            | 可学习仿射变换                  | 需要训练数据，计算成本更高                            |
| QuaRot + SmoothQuant | 旋转后再 smooth                 | **msModelSlim 主流组合**：先旋转再 smooth，互补 |

### 应用场景

**适用场景：** Transformer Decoder 架构（Qwen3、DeepSeek、GLM 等），配合 `flex_smooth_quant` / `linear_quant` 做 W8A8、W4A8 等。

**WHY 适用：** 标准 Transformer 的 Linear → RMSNorm → Linear 结构允许 Norm 权重折叠进 Linear，使旋转链可闭合。

**不适用场景：**

- 非 RMSNorm 架构（`fuse_ln_linear` 仅支持 1D weight 的 RMSNorm）。
- 未实现 `QuaRotInterface` 的模型。
- 在线旋转 + TP 并行时 `tp_size` 非 2 幂或超过 `max_tp_size`。

---

## 3. 核心概念网络

### 核心概念清单

**概念 1：正交旋转（Orthogonal Rotation）**

- **是什么：** 对权重左乘 Qᵀ、对激活右乘 Q（或反向），使线性层输入输出关系不变。
- **WHY 需要：** 在不改变模型输出的前提下修改数值分布。
- **WHY 这样实现：** 左乘权重、右乘激活是矩阵乘法结合律的直接推论。
- **WHY 不用其他方式：** 非正交变换会改变 logits，需要重训练。

**概念 2：Hadamard 矩阵**

- **是什么：** 元素为 ±1 的正交矩阵，维度通常为 2 的幂；非 2 幂维度用 `n = k × 2^m` 分解，k 从预置 txt 加载。
- **WHY 需要：** 全随机正交矩阵无快速乘法结构，Hadamard 兼顾正交性与 O(n log n) 运算。
- **WHY 这样实现：** `random_hadamard_matrix` = 随机对角 ±1 矩阵 × Walsh 递归。
- **WHY 不用其他方式：** 纯 QR 分解生成的随机正交矩阵推理部署无专用算子。

**概念 3：RMSNorm 融合（fuse_ln_linear）**

- **是什么：** 将 RMSNorm 的 γ 乘到下游 Linear 权重上，Norm 权重置 1。
- **WHY 需要：** RMSNorm 在旋转链路中会破坏简单的 Q 抵消关系，融合后 Norm 等价于恒等缩放。
- **WHY 这样实现：** `W' = W * γ`（广播到 input dim），bias 吸收 β（如有）。
- **WHY 不用 LayerNorm：** 代码显式检查 `ln.weight.dim() == 1`，且 bake_mean 仅用于 LayerNorm 场景。

**概念 4：RotatePair（左旋转 + 右旋转）**

- **是什么：** `left_rot: {模块名: Q}` 对权重左乘 Qᵀ；`right_rot: {模块名: Q}` 对权重右乘 Q。
- **WHY 需要：** 一次 Linear 运算 Y = X·W 需要在相邻层配对旋转，使中间的 Q·Qᵀ 抵消。
- **WHY 这样实现：** 模型适配器按 Attention/MLP 拓扑声明配对关系。
- **WHY 不用全局单矩阵：** 不同子空间（hidden、head_dim、lora_rank）维度不同，需多套矩阵。

**概念 5：离线旋转 vs 在线旋转**

- **是什么：** 离线 = 量化时直接改权重；在线 = 推理时在前向插入旋转算子（HookIR）。
- **WHY 需要：** 部分旋转链（如 down_proj 的 Kronecker 结构）离线折叠会损失精度或无法闭合。
- **WHY 这样实现：** LAOS 对 o_proj 离线融合 Kronecker 旋转，对 down_proj 保留在线 Hook。
- **WHY 不用全在线：** 全在线增加推理延迟，离线能做的尽量烘焙进权重。

**概念 6：HookIR → WrapperIR**

- **是什么：** 量化阶段用 `register_forward_pre_hook` 插入旋转；保存前转为 `WrapperIR` 持久化到导出模型。
- **WHY 需要：** Hook 是运行时临时对象，无法序列化进 AscendV1 格式。
- **WHY 这样实现：** `post_run` 遍历 HookIR，`wrapper_module()` 替换原模块。
- **WHY 不用直接改 nn.Module：** WrapperIR 携带旋转元数据供 Saver 写 safetensors。

### 概念关系矩阵

| 关系类型 | 概念 A                    | 概念 B                 | WHY 这样关联                                                    |
| -------- | ------------------------- | ---------------------- | --------------------------------------------------------------- |
| 前置依赖 | RMSNorm 融合              | 正交旋转               | 融合后 Norm 不再阻断 Q 抵消链                                   |
| 配对闭合 | left_rot (o_proj)         | right_rot (q/k/v_proj) | 中间激活上的 Q·Qᵀ 恒等                                        |
| 组合使用 | QuaRot                    | flex_smooth_quant      | 旋转改变分布后 smooth 更有效；YAML 中 QuaRot 必须在 smooth 之前 |
| 扩展     | AdaptRotation             | QuaRot                 | Stage1 优化矩阵 → Stage2 注入 QuaRotProcessor                  |
| 对比     | 离线旋转                  | 在线旋转               | 精度 vs 性能权衡；LAOS 混合策略                                 |
| 导出     | QuarotOfflineRotationInfo | ascendv1_saver         | 推理引擎加载 global_rotation 做 embedding 侧旋转                |

---

## 4. 算法与理论

### 算法：Hadamard 正交旋转

- **时间复杂度：** 单次旋转 O(n²)（朴素矩阵乘）；Walsh 快速变换 O(n log n)（`matmul_had_u` 路径）
- **空间复杂度：** O(n²) 存储旋转矩阵（块对角/ Kronecker 可降维）
- **WHY 选择：** 正交变换保等价 + 结构化矩阵硬件可实现
- **WHY 复杂度可接受：** 量化仅执行一次（离线）；在线旋转矩阵维度通常 ≤ hidden_size 且可 block 分解
- **WHY 不选 SVD 旋转：** SVD 不保证正交性，需额外正交化；且无快速推理结构
- **退化场景：** 维度 n 无法分解为 `k × 2^m`（k 来自预置表）时 `get_had_k` 抛 `UnsupportedError`
- **参考：** [QuaRot 论文](https://arxiv.org/abs/2404.00456) · [Hadamard 矩阵](https://en.wikipedia.org/wiki/Hadamard_matrix)

### 算法：Kronecker 旋转（down_proj 在线路径）

- **公式：** 对形状 `(batch, m·n)` 的激活，旋转 `R = R_m ⊗ R_n`，等价于 reshape 为 `(batch, m, n)` 后先右乘 R_nᵀ 再左乘 R_mᵀ
- **WHY 选择：** `intermediate_size` 常非 2 幂，分解为 `size_1 × size_2` 使两个子矩阵都可用 Hadamard
- **分解约束：** `get_decompose_dim(n)` 要求 `n = (a+b)(a-b)` 且 `a±b` 属于 `{1,2,4,8,...,256}`

### 算法：BLOCK_HADAMARD_SHIFTED（DeepSeek q_b_proj）

- **是什么：** 块对角 Hadamard + 循环移位矩阵 P 的多步复合：`rot @ P @ rot @ ...`
- **WHY 选择：** q_lora_rank 空间的离群值分布更不均匀，额外移位增加混合度
- **eye_step：** 部分块替换为单位矩阵，保留特定子空间不旋转

---

## 5. 设计模式

### 模式 1：策略模式（Model Adapter）

**应用位置：** `QuaRotInterface.get_rotate_map()` 各模型实现

**WHY 使用：** QuaRot 算法通用，但不同模型 Attention/MLP 拓扑不同（GQA、MLA、MoE），旋转映射必须由模型专家声明。

**WHY 不用硬编码：** DeepSeek MLA 需 4 套矩阵，Qwen3 Dense 仅需 2 套，硬编码不可维护。

**参考：** [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)

### 模式 2：模板方法（三阶段 Processor）

**应用位置：** `QuaRotProcessor.pre_run / preprocess / post_run`

**WHY 使用：** Runner 按 DecoderLayer 逐层调度，需要在全局（embedding）和层内分别执行旋转；模板固定骨架，适配器填充映射。

**WHY 不用单次遍历：** 逐层调度可与后续 flex_smooth / quant 流水线对齐，降低峰值内存。

### 模式 3：装饰器 / Wrapper（HookIR → WrapperIR）

**应用位置：** `ir/quarot.py` 中 `QuarotHeadsRotationHookIR` 等

**WHY 使用：** 量化期用 Hook 无侵入实验；导出期转 Wrapper 可序列化。

**WHY 不用 Monkey Patch：** WrapperIR 是 IR 体系一等公民，Saver 可识别并导出旋转矩阵。

### 模式 4：注册表（QABCRegistry）

**应用位置：** `@QABCRegistry.register(dispatch_key=QuaRotProcessorConfig)`

**WHY 使用：** YAML `type: quarot` 自动实例化对应 Processor，与 modelslim_v1 插件体系一致。

---

## 6. 关键代码深度解析

### 核心片段清单

| 编号 | 片段名称                       | 所在文件:行号                                                         | 优先级 | 识别理由             |
| ---- | ------------------------------ | --------------------------------------------------------------------- | ------ | -------------------- |
| #1   | QuaRotProcessor 三阶段调度     | `offline_quarot/quarot.py:92-177`                                   | ★★★ | 算法执行主入口       |
| #2   | create_rot + Hadamard 构造     | `common/quarot_utils.py:62-112` + `hadamard.py:146-151`           | ★★★ | 旋转矩阵数学核心     |
| #3   | rotate_linear + fuse_ln_linear | `common/quarot_utils.py:115-194`                                    | ★★★ | 权重变换与 Norm 折叠 |
| #4   | LAOS 在线旋转                  | `offline_quarot/laos_online.py:105-180`                             | ★★☆ | Qwen3 在线旋转路径   |
| #5   | Qwen3 vs DeepSeek 旋转映射     | `qwen3/model_adapter.py:291-331` / `deepseek_v3/quarot.py:78-166` | ★★★ | 模型差异面试必问     |
| #6   | IR 导出链                      | `ir/quarot.py:249-377` + `ascendv1.py:666-681`                    | ★★☆ | 推理部署闭环         |

**跳过说明：**

- `online_quarot/online_quarot.py`：通用在线旋转框架，DeepSeek V3.2 Indexer 路径使用，与 LAOS 并列但接口不同；面试 Deep 问 Qwen3 时优先 LAOS。
- `adapt_rotation/`：数据驱动优化，是 QuaRot 的增强版，非默认路径。

---

### 片段 #1：QuaRotProcessor 三阶段调度

> 📍 **位置：** `msmodelslim/processor/quarot/offline_quarot/quarot.py:92-177`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** Runner 调度前做全局旋转，逐层时做层内旋转，结束后清扫剩余命令并转 Hook。

#### 1.1 代码整体作用

`QuaRotProcessor` 是 QuaRot 在 msModelSlim 流水线中的**唯一主入口**。它本身不做矩阵数学，而是：

1. 从模型适配器拉取三张"施工图纸"：Norm 融合表、mean bake 列表、旋转命令表；
2. 按 Runner 生命周期分三批施工；
3. 可选地挂载全局旋转导出 Hook 和 LAOS 在线旋转。

**系统层次定位：** 量化 Processor 层（与 `flex_smooth_quant`、`linear_quant` 同级）。

**角色与依赖：** 上游依赖 `QuaRotInterface` 适配器；下游是已旋转的 fp 权重，供 smooth/quant 继续处理。

#### 1.2 核心逻辑分析

**执行流程：**

```
pre_run:
  adapter.get_ln_fuse_map()  → pre_run_fused_ln + fused_map(留存)
  adapter.get_rotate_map()   → pre_run_pairs + rotate_pairs(留存)
  _fuse_norm(pre_run) → _bake_mean → _rotate(pre_run_commands)
  rotate_commands = get_rotate_command(rotate_pairs)  # 暂存，待逐层消费
  _inject_global_rotation_export_hook()  # 可选
  online_processor.pre_run()  # 可选

preprocess(request):  # 每个 DecoderLayer 触发一次
  prefix = request.name + "."
  过滤 fused_map / bake_names / rotate_commands 中本层项
  _fuse_norm → _bake_mean → _rotate
  online_processor.preprocess(request)

post_run:
  处理三张表中剩余项（通常是最后几层或 lm_head）
  online_processor.post_run()  # HookIR → WrapperIR
```

**核心状态变量：**

| 变量名                   | 初始值       | 变化时机                 | 终态   |
| ------------------------ | ------------ | ------------------------ | ------ |
| `self.fused_map`       | adapter 返回 | 每层 preprocess 过滤消耗 | `{}` |
| `self.bake_names`      | adapter 返回 | 同上                     | `[]` |
| `self.rotate_commands` | pre_run 填充 | 每层 preprocess 过滤消耗 | `[]` |

**多执行路径：**

- **路径 A（标准离线）：** `online=False` → 仅改权重，无 Hook，推理无额外算子。
- **路径 B（LAOS 在线）：** `online=True` → 权重旋转 + o_proj/down_proj 注册 Hook，post_run 转 Wrapper，Saver 导出旋转矩阵。

#### 1.3 逐行代码解释

> **贯穿示例：** Qwen3-8B，`block_size=-1`，`online=False`，Runner 调度 `model.layers.0`

```python
def pre_run(self) -> None:
    # 步骤 1: 从适配器获取三张映射表
    pre_run_fused_ln, self.fused_map = self.adapter.get_ln_fuse_map()
    pre_run_bake_names, self.bake_names = self.adapter.get_bake_names()
    pre_run_pairs, self.rotate_pairs = self.adapter.get_rotate_map(block_size=self.config.block_size)
    # WHY: 适配器最了解模型拓扑；Processor 只负责执行
    # 此时: fused_map 含全部层的 input_layernorm → q/k/v_proj 映射

    pre_run_commands = get_rotate_command(pre_run_pairs)
    # 步骤 2: pre_run 阶段施工（通常只有 embed_tokens 右旋转）
    self._fuse_norm(pre_run_fused_ln)   # pre_run 的 Norm 融合（通常为空）
    self._bake_mean(pre_run_bake_names)
    self._rotate(pre_run_commands)     # embed: W' = W @ Q
    # 此时: embed_tokens.weight 已被右乘 Q

    # 步骤 3: 将剩余旋转对转为命令队列，留待逐层消费
    self.rotate_commands = get_rotate_command(self.rotate_pairs)
    # 此时: rotate_commands 含全部 32 层的 q/k/v/o/mlp 旋转命令

    self._inject_global_rotation_export_hook(pre_run_commands)
    # 场景: export_extra_info=True → 在 embed 上挂 Hook，保存 global_rotation=Q

def preprocess(self, request: BatchProcessRequest) -> None:
    prefix = request.name          # "model.layers.0"
    prefix = f"{prefix}."          # "model.layers.0."
    # 步骤 4: 过滤出本层相关的融合/旋转命令
    fused_map = self._filter_fused_map(prefix)
    # 场景: 提取 input_layernorm → [q_proj, k_proj, v_proj]
    rotate_commands = self._filter_commands(prefix)
    # 场景: 提取 layers.0.self_attn.* 和 layers.0.mlp.* 的旋转命令

    self._fuse_norm(fused_map)       # γ 折入 q/k/v 权重
    self._rotate(rotate_commands)    # 本层权重旋转闭环
    # 此时: 第 0 层旋转完成，后续层在各自 preprocess 中处理
```

#### 1.4 关键设计点

| 设计维度               | 分析内容                                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| **实现选择**     | 分阶段而非一次性旋转：与 Runner 逐层加载兼容（大模型省内存），且 Norm 融合只需在量化该层前完成。          |
| **性能优化**     | `is_data_free()=True`：无需校准数据；旋转一次，量化全程受益。                                           |
| **编译器相关**   | 不涉及；旋转在 PyTorch eager 模式完成。                                                                   |
| **安全与健壮性** | `_filter_*` 用 prefix 匹配防越层；旋转维度不匹配抛 `UnsupportedError` 并提示检查适配器。              |
| **可扩展性**     | 新模型只需实现`QuaRotInterface`；`_inject_global_rotation_export_hook` 注释预留 RotationTune 调整点。 |
| **潜在问题**     | `rotate_commands.remove(command)` 在循环中 O(n²)；层数上百时可优化为 deque。                           |

#### 1.5 完整示例（三组对比）

**示例 1 — Qwen3 离线 W8A8**

- **输入：** `type: quarot, online: False, block_size: -1`
- **执行：** pre_run 旋转 embed → 32 次 preprocess 各层旋转 → flex_smooth → linear_quant
- **输出：** 权重已旋转的 INT8 模型，无在线 Hook

**示例 2 — Qwen3 在线 W4A4 LAOS**

- **输入：** `online: True, down_proj_online_layers: [1,2], max_tp_size: 2`
- **关键差异：** down_proj 层 1、2 保留 Kronecker 在线 Hook；o_proj 每层有 heads 旋转 Hook
- **输出：** safetensors 含 `heads_rotation` + `kronecker_rotation_m/n`

**示例 3 — 未适配模型**

- **输入：** 适配器未实现 `QuaRotInterface`
- **处理：** `__init__` 抛 `UnsupportedError`
- **结果：** 量化中止，提示实现接口

#### 1.6 使用注意与改进建议

1. **YAML 顺序：** QuaRot 必须排在 `flex_smooth_quant` 之前。不注意会导致 smooth 在旧分布上标定，旋转后失效。
2. **export_extra_info：** 推理引擎若需 embedding 侧在线旋转，必须 True 且使用 `ascendv1_saver`；否则 global_rotation 丢失。
3. **改进：** 可将 `_filter_commands` 改为预建 prefix-trie，避免每层线性扫描命令列表。

---

### 片段 #2：create_rot + Hadamard 构造

> 📍 **位置：** `common/quarot_utils.py:62-112` · `common/hadamard.py:67-151`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 固定种子生成随机 ±1 对角阵，经 Walsh 递归构造正交 Hadamard，支持块对角和移位变体。

#### 2.1 代码整体作用

`create_rot` 是 msModelSlim 中**所有旋转矩阵的统一工厂**。它保证：

- 同一模型多次量化使用相同旋转（`seed_all(1234)`）；
- 支持全矩阵 / 块对角 / 移位复合三种模式；
- 维度不支持时尽早失败。

#### 2.2 核心逻辑分析

**两种模式：**

| 模式                       | 用途              | 生成方式                                       |
| -------------------------- | ----------------- | ---------------------------------------------- |
| `HADAMARD`               | 标准全维度旋转    | `random_hadamard_matrix(size)` 或 block_diag |
| `BLOCK_HADAMARD_SHIFTED` | DeepSeek q_b_proj | 块对角 + 循环移位 P 的多步复合                 |

**random_hadamard_matrix 流程：**

```
随机向量 r ∈ {+1,-1}^n
D = diag(r)
H = matmul_had_u(D)   # Walsh 快速变换
验证: H @ H.T ≈ I
```

#### 2.3 逐行代码解释

> **贯穿示例：** `create_rot(HADAMARD, size=4096, block_size=32)`

```python
def create_rot(mode, size, block_size=-1, rot_step=1, eye_step=(-1,), seed=1234):
    seed_all(seed)
    # WHY: 固定种子保证可复现；面试常问"旋转矩阵是否随机"——是随机构造但种子固定

    if mode == QuaRotMode.HADAMARD:
        if block_size == -1:
            transformation_dim = size      # 4096
        else:
            transformation_dim = block_size  # 32
        rot = random_hadamard_matrix(transformation_dim, dtype, device)
        # 步骤: 生成 32×32 正交矩阵

        if block_size != -1:
            rot = rot.repeat(size // block_size, 1, 1)  # 128 个块
            rot = torch.block_diag(*rot)                 # 4096×4096 块对角
        # WHY: block_size 适配 TP 切分——每个 TP rank 只持有一块，旋转不跨 rank 混合
        # 此时: rot.shape = (4096, 4096)

def random_hadamard_matrix(size, dtype, device):
    rot = torch.randint(0, 2, (size,)).float() * 2 - 1  # ±1 向量
    rot = torch.diag(rot)                                  # 随机对角阵
    return matmul_had_u(rot)                               # Walsh 正交化
    # WHY: 纯 Walsh 矩阵太"规整"，随机对角乘增加熵，不同 channel 混合更充分
```

#### 2.4 关键设计点

| 设计维度               | 分析内容                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| **实现选择**     | 预置 txt 矩阵处理 12/20/28/.../200 等非 2 幂因子 k，使 n=k·2^m 可分解。 |
| **性能优化**     | `matmul_had_u` 用蝶形运算 O(n log n)，优于朴素 O(n²)。                |
| **安全与健壮性** | 不支持的维度抛`UnsupportedError`，FAQ 指向添加 hadamard_txt。          |
| **潜在问题**     | block_diag 大矩阵占 O(n²) 内存；超大 hidden_size 时需确认 NPU 内存。    |

#### 2.5 完整示例

**示例 1：** `size=128, block_size=-1` → 单个 128×128 Hadamard，用于 Qwen3 `rot_uv`（head_dim）

**示例 2：** `size=7168, block_size=32` → 224 个 32×32 块对角，用于 DeepSeek 全维度 rot

**示例 3：** `size=1536, BLOCK_HADAMARD_SHIFTED, rot_step=2` → q_b_proj 移位复合矩阵

#### 2.6 使用注意与改进建议

1. **正交性验证：** 单元测试 `torch.allclose(rot @ rot.T, I, atol=1e-5)`，部署前勿改容差。
2. **block_size 与 TP：** `max_tp_size` 应 ≥ 实际 `tp_size`，否则块边界与 rank 切分不对齐导致精度异常。

---

### 片段 #3：rotate_linear + fuse_ln_linear

> 📍 **位置：** `common/quarot_utils.py:115-194`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 对 Linear 权重施加 W←W·Q 或 W←Qᵀ·W，融合前将 RMSNorm γ 折入 W。

#### 3.1 代码整体作用

这两个函数是 QuaRot **真正修改模型参数**的地方。`fuse_ln_linear` 消除 RMSNorm 对旋转链的阻断；`rotate_linear` 执行矩阵乘法旋转。

#### 3.2 核心逻辑分析

**旋转方向：**

| side  | 操作                            | 数学含义                  |
| ----- | ------------------------------- | ------------------------- |
| RIGHT | `W' = W @ Q`                  | 补偿激活右侧的`X @ Q`   |
| LEFT  | `W' = Qᵀ @ W`，bias 同样左乘 | 补偿激活左侧的`Qᵀ @ X` |

**维度自动 block_diag：** 若 `weight.shape[1] != rot.shape[0]` 但整除，自动将 rot 复制为 block_diag —— 支持多头拼接场景。

#### 3.3 逐行代码解释

> **贯穿示例：** `q_proj` 右旋转，`o_proj` 左旋转，Q 为 4096×4096

```python
def rotate_linear(linear, rot, right_rotate=True):
    weight_data = linear.weight.data.float()  # GLOBAL_DTYPE=float32 提高精度
    if right_rotate:
        # 场景: q_proj —— 输入侧将乘 Q，权重右乘 Q 抵消
        linear.weight.data = (weight_data @ rot).to(original_dtype)
        # WHY: (X @ Q) @ (W @ Q) 不对，应是 (X @ Q) @ (Q.T @ W)...
        # 实际配对: 右旋转的 q_proj 与左旋转的 o_proj 在 Attention 内部闭合
    else:
        # 场景: o_proj —— 输出侧左乘 Qᵀ
        linear.weight.data = (rot.T @ weight_data).to(original_dtype)
        if linear.bias is not None:
            linear.bias.data = (rot.T @ bias_data).to(original_dtype)

def fuse_ln_linear(layernorms, linear_layers):
    ln_weight = layernorm.weight.data  # RMSNorm γ, shape [hidden]
    for linear in linear_layers:
        # 步骤: W' = W * γ（逐 input channel 缩放）
        linear.weight.data = current_weight * ln_weight
    for ln in layernorms:
        ln.weight.data.fill_(1.0)  # Norm 变为恒等
    # WHY: 旋转要求 Norm 前后可交换；γ 折入 W 后 RMSNorm 等价于 1
```

#### 3.4 关键设计点

| 设计维度               | 分析内容                                                               |
| ---------------------- | ---------------------------------------------------------------------- |
| **实现选择**     | float32 中间计算防 bf16 旋转累积误差。                                 |
| **安全与健壮性** | `ln.weight.dim() != 1` 立即拒绝——LayerNorm 不走此路径。            |
| **可扩展性**     | `rotate_weight` 支持非 Linear 参数（如 `embed_tokens` 权重张量）。 |

#### 3.5 完整示例

**Attention 闭合链（Qwen3）：**

```
input_layernorm(γ) → fuse 进 q/k/v_proj
q_proj: W_q' = W_q @ Q     (right)
k_proj: W_k' = W_k @ Q     (right)
v_proj: W_v' = W_v @ Q_uv  (right, rot_uv)
o_proj: W_o' = Q_uv^T @ W_o (left, rot_uv)
等价于原始 RMSNorm → QKV → O 的计算
```

#### 3.6 使用注意与改进建议

1. **融合顺序：** 必须先 `_fuse_norm` 再 `_rotate`，否则旋转后再折 γ 会破坏等价性。
2. **bias 处理：** 仅左旋转时变换 bias；右旋转不影响 bias——面试常考点。

---

### 片段 #4：LAOS 在线旋转

> 📍 **位置：** `offline_quarot/laos_online.py:105-180`
> 🎯 **优先级：** ★★☆
> 💡 **一句话核心：** o_proj 离线融合 Kronecker 旋转，down_proj 在指定层保留在线 Kronecker Hook。

#### 4.1 代码整体作用

LAOS（Layer-wise Adaptive Online Rotation Strategy）是 Qwen3 Dense **可选的精度增强路径**。它解决的核心问题是：down_proj 的 `intermediate_size` 维旋转矩阵太大或不可闭合，需推理时执行。

#### 4.2 核心逻辑分析

```
pre_run:
  rot1, rot2 = get_decompose_dim(intermediate_size)  # 如 11008 → 96×115
  rot_online_o_proj = Hadamard(num_heads, block_size=max_tp_size)
  创建 QuarotOnlineRotationInfo 共享对象

preprocess(每层):
  online_rotate_o_proj_input: W_o ← W_o @ kron(rot_online, I_head)
  若 layer_idx ∈ down_proj_online_layers:
    online_rotate_down_proj: 离线部分旋转进 W_down
    注册 QuarotKroneckerRotationHookIR on down_proj
  注册 QuarotHeadsRotationHookIR on o_proj

post_run:
  所有 HookIR → WrapperIR
```

#### 4.3 逐行代码解释

```python
def _add_online_rotations(self, model):
    size_1, size_2 = get_decompose_dim(model.config.intermediate_size)
    # 场景: intermediate_size=11008 → (96, 115)，96 和 115 均在 Hadamard 支持集
    rot1 = create_rot(HADAMARD, size_1, block_size=max_tp_size)
    rot2 = create_rot(HADAMARD, size_2, block_size=-1)
    rot_online_o_proj = create_rot(HADAMARD, num_attn_heads, block_size=max_tp_size)
    # WHY: o_proj 按 head 维度旋转，block_size 对齐 TP

def preprocess(self, request):
    layer_idx = int(request.name.split('.')[-1])
    if layer_idx in self.config.down_proj_online_layers:
        online_rotate_down_proj(up_down_pairs, rot1, rot2)  # 部分烘焙
        hook_ir = QuarotKroneckerRotationHookIR(...)         # 剩余在线
        down_proj.register_forward_pre_hook(hook_ir)
    # 每层 o_proj 都注册 heads 旋转 Hook
```

#### 4.4 关键设计点

| 设计维度           | 分析内容                                                   |
| ------------------ | ---------------------------------------------------------- |
| **实现选择** | 混合离线/在线：能烘焙的进权重，不能闭合的留 Hook。         |
| **性能优化** | `down_proj_online_layers` 可只开关键层，权衡精度与延迟。 |
| **潜在问题** | TP 约束严格：`tp_size` 必须 2 幂且 ≤ `max_tp_size`。  |

#### 4.5 完整示例

- **基础：** `online=False` → 无 LAOS，纯离线
- **典型：** `online=True, down_proj_online_layers=[1,2], max_tp_size=2`
- **边界：** `online=True` 但适配器未实现 `LAOSOnlineRotationInterface` → 初始化报错

#### 4.6 使用注意与改进建议

1. 在线旋转依赖推理引擎支持 Wrapper 算子；MindIE 需确认 heads/kronecker 旋转 kernel。
2. 仅 Qwen3 Dense 官方实现 LAOS；MoE/DeepSeek 默认 `online: False`。

---

### 片段 #5：Qwen3 vs DeepSeek 旋转映射

> 📍 **位置：** `model/qwen3/model_adapter.py:291-331` · `model/deepseek_v3/quarot.py:78-166`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 标准 GQA 用 2 套矩阵；MLA 低秩结构需 4 套矩阵分别闭合不同维度链路。

#### 5.1 代码整体作用

模型适配器把**拓扑知识**翻译成 `RotatePair` 列表。这是面试区分"懂原理"和"懂工程"的关键。

#### 5.2 对比表（面试必背）

| 旋转名            | Qwen3 Dense | DeepSeek MLA    | 尺寸                   | 作用层                                    |
| ----------------- | ----------- | --------------- | ---------------------- | ----------------------------------------- |
| `rot`           | ✅          | ✅              | hidden_size            | embed, q/k/v 或 q_a/kv_a, o, mlp, lm_head |
| `rot_uv`        | ✅ head_dim | ✅ v_head_dim   | head 维                | v_proj↔o_proj 或 kv_b_proj V 段↔o_proj  |
| `rot_b_proj`    | ❌          | ✅ q_lora_rank  | BLOCK_HADAMARD_SHIFTED | q_a_proj↔q_b_proj                        |
| `rot_kv_b_proj` | ❌          | ✅ kv_lora_rank | kv_a↔kv_b 低秩段      |                                           |

**WHY DeepSeek 需要 4 套：** MLA 存在 hidden→lora→head 三级维度变换，单一 Q 无法同时正交化所有子空间；kv_b_proj 输出还拼接了 nope/rope 两段，需 `torch.block_diag([I, rot_uv])` 只旋转 V 段。

**WHY Qwen3 只要 2 套：** 标准 MHA/GQA，hidden 一套闭合 Attention+MLP；head_dim 一套闭合 v_proj↔o_proj。

#### 5.3 关键代码（DeepSeek rot_kv_b_proj 分割旋转）

```python
# kv_a_proj 输出分为 [kv_lora_rank | qk_rope_head_dim]
left_rot_kv_b_proj[kv_a_proj] = [
    rot_kv_b_proj,                                    # 只旋转低秩段
    torch.eye(qk_rope_head_dim),                      # RoPE 段保持恒等
]
right_rot_kv_b_proj[kv_b_proj] = rot_kv_b_proj
# WHY: RoPE 段与位置编码耦合，旋转会破坏位置信息
```

#### 5.4 关键设计点

- **列表旋转：** `left_rot_uv[kv_b_proj] = [I_nope, rot_uv]` 用 list 表示 block_diag 分段——`rotate_linear` 自动 `block_diag` 拼接。
- **MoE 扩展：** DeepSeek `rot` 覆盖每个 expert 的 gate/up/down，shared_expert 和 gate 路由层同样纳入。

#### 5.5 面试话术

> "Qwen3 是教科书式 QuaRot：hidden 维一把梭 + head 维修 V/O。DeepSeek MLA 把一条 Attention 拆成 4 条不同维度的线性链，每链一把钥匙，少一把就关不上正交性大门。"

---

### 片段 #6：IR 导出链

> 📍 **位置：** `ir/quarot.py:249-377` · `ascendv1.py:666-681`
> 🎯 **优先级：** ★★☆
> 💡 **一句话核心：** 量化期 Hook 承载旋转逻辑，Saver 期转 Wrapper 并将矩阵写入 safetensors + JSON 描述。

#### 6.1 三类导出

| 类型           | Wrapper                                  | 导出内容                                               | 用途                 |
| -------------- | ---------------------------------------- | ------------------------------------------------------ | -------------------- |
| 全局离线       | `QuaRotExtraInfoWrapperIR`             | `optional/quarot.safetensors` → `global_rotation` | embedding 侧推理旋转 |
| Heads 在线     | `QuarotOnlineHeadRotationWrapper`      | `heads_rotation`                                     | o_proj 输入旋转      |
| Kronecker 在线 | `QuarotOnlineKroneckerRotationWrapper` | `kronecker_rotation_m/n`                             | down_proj 输入旋转   |

#### 6.2 JSON 描述示例

```json
{
  "optional": {
    "quarot": {
      "rotation_map": {
        "global_rotation": "optional/quarot.safetensors"
      }
    }
  },
  "metadata": {
    "quarot": {
      "max_tp_size": 4,
      "heads_rotation": { "layers": ["model.layers.0.self_attn.o_proj", "..."] },
      "kronecker_rotation": { "layers": ["model.layers.1.mlp.down_proj"] }
    }
  }
}
```

---

## 7. 测试用例分析

### 测试文件清单

| 测试文件/目录                                            | 测试的模块             | 测试用例数量（约） |
| -------------------------------------------------------- | ---------------------- | ------------------ |
| `test/cases/processor/quarot/test_quarot_processor.py` | QuaRotProcessor 三阶段 | 10+                |
| `test/cases/processor/quarot/test_hadamard.py`         | Hadamard 正交性        | 20+                |
| `test/cases/processor/quarot/test_quarot_utils.py`     | rotate_linear / fuse   | 15+                |
| `test/cases/processor/quarot/test_online_quarot.py`    | OnlineQuaRotProcessor  | 10+                |
| `test/cases/model/deepseek_v3/test_quarot.py`          | DeepSeek 四套矩阵      | 5+                 |
| `test/smoke/test_quarot.py`                            | 端到端冒烟             | 3+                 |

### 功能覆盖矩阵

| 核心功能        | 主代码位置                | 测试覆盖            | 评估 |
| --------------- | ------------------------- | ------------------- | ---- |
| Hadamard 正交性 | `hadamard.py`           | ✅ 参数化 15 种维度 | 充分 |
| 不支持维度      | `hadamard.py`           | ✅ 11/21/87 抛错    | 充分 |
| 三阶段过滤      | `quarot.py`             | ✅ FakeQwen3        | 充分 |
| Norm 融合       | `quarot_utils.py`       | ✅                  | 基本 |
| 在线旋转        | `laos_online.py`        | ⚠️                | 部分 |
| DeepSeek 4 矩阵 | `deepseek_v3/quarot.py` | ✅                  | 基本 |

### 从测试中发现的边界条件

1. **正交性容差 1e-5：** float32 下 Hadamard 积与 I 的偏差上限，面试可提数值稳定性。
2. **不支持的 Hadamard 维度：** 11、21、87 等既非 2 幂也无法分解的数会失败——与 FAQ 文档一致。
3. **FakeQwen3 集成测试：** 验证 pre_run 后 embed 权重变化、preprocess 按层消耗命令队列。

### 测试质量建议

- 增加 MoE expert 旋转的独立用例（目前多在 DeepSeek 适配器测试中隐式覆盖）。
- 补充 `export_extra_info` 导出后 JSON/safetensors 内容断言。

---

## 8. 应用迁移场景

### 场景 1：QuaRot → 新 Transformer 模型适配

**不变的原理：** 正交旋转闭合链、RMSNorm 融合、左/右旋转配对。

**需要修改的部分：**

```python
class MyModelAdapter(..., QuaRotInterface):
    def get_ln_fuse_map(self):
        # 按模型拓扑填写 input_layernorm → [q_proj, k_proj, v_proj]
        return {}, layer_wise_fused_map

    def get_rotate_map(self, block_size):
        rot = QuaRotInterface.get_rotate_command(HADAMARD, hidden_size, block_size)
        # 声明 left_rot / right_rot 配对
        return [pre_run_pair], list(rot_pairs.values())
```

**学到的通用模式：** **拓扑声明与算法执行分离**——Processor 是通用引擎，Adapter 是模型说明书。

### 场景 2：QuaRot → AdaptRotation 精度增强

**不变的原理：** 最终仍通过 `rotate_linear` 施加旋转，仍须保持正交性。

**需要修改的部分：**

- Stage1：收集指定层激活，用 `HadamardOptimizer` 迭代优化旋转矩阵（数据驱动）。
- Stage2：将优化后矩阵写回 context，再调用 `QuaRotProcessor` 执行。

**WHY 迁移：** 固定 Hadamard 对特定模型/层可能非最优；AdaptRotation 在正交约束下搜索更平滑分布。

---

## 9. 依赖关系与使用示例

### 外部库

**PyTorch**

- **用途：** 矩阵运算、Hook 机制、nn.Module 操作
- **WHY 选择：** msModelSlim 全栈基于 PyTorch 模型对象
- **WHY 不用 NumPy：** 需直接修改 `nn.Parameter` 并保留计算图（部分在线路径）

### 内部模块依赖

```
QuaRotProcessor
  → QuaRotInterface (model adapter)
  → quarot_utils (create_rot, rotate_linear, fuse_ln_linear)
  → hadamard (random_hadamard_matrix)
  → ir/quarot (HookIR, WrapperIR, RotationInfo)
  → ascendv1_saver (导出)
```

### 完整 YAML 示例（DeepSeek V3.2 W8A8）

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "quarot"              # 必须第一步：离群值抑制
      online: false
      block_size: -1
      export_extra_info: true
    - type: "flex_smooth_quant"   # 第二步：在旋转后分布上做 smooth
      enable_subgraph_type: ['norm-linear', 'ov']
    - type: "linear_quant"        # 第三步：真正量化
      qconfig: *default_w8a8_dynamic
      include: ["*mlp*"]
  dataset: mix_calib.jsonl        # quarot 本身 data-free，但后续 smooth 需要
  save:
    - type: "ascendv1_saver"
```

**执行命令：**

```bash
msmodelslim quant \
  --model_path /path/to/DeepSeek-V3.2 \
  --save_path /path/to/output \
  --config lab_practice/deepseek_v3_2/deepseek_w8a8_quarot.yaml
```

---

## 10. 质量验证清单

### 理解深度

- [X] 每个核心概念都回答了 3 个 WHY
- [X] 能不看代码解释 pre_run/preprocess/post_run
- [X] 能画 Attention 旋转闭合链

### 技术准确性

- [X] Hadamard 构造：正交性 + 复杂度 + 不支持维度
- [X] 设计模式：Strategy（Adapter）+ Template Method（三阶段）+ Wrapper（IR）
- [X] 代码解析：6 个片段含真实代码与执行示例

### 实用性

- [X] 应用迁移：新模型适配 + AdaptRotation 增强
- [X] YAML 示例完整
- [X] 面试 Q&A 见下节

### 最终"四能"测试

1. ✅ 能否理解代码的设计思路？
2. ✅ 能否独立实现类似功能的 Processor？
3. ✅ 能否适配到新模型？
4. ✅ 能否向面试官清晰解释？

---

## 面试高频问答（速记版）

### Q1：QuaRot 解决什么问题？和 SmoothQuant 什么关系？

> LLM 激活有 channel 离群值，低比特量化动态范围不够。QuaRot 用正交旋转把离群值分散到多通道，**不改数学语义**。SmoothQuant 把量化难度从激活迁到权重。msModelSlim 里**先 QuaRot 再 Smooth**，两者互补：旋转平滑分布，smooth 进一步优化 scale。

### Q2：为什么必须融合 RMSNorm？

> 旋转推导假设线性层之间没有非线性的 scale。RMSNorm 的 γ 会打破 `Q·Qᵀ` 抵消链。把 γ 折进下游 Linear 权重后，Norm 变成恒等，旋转链才闭合。代码只支持 RMSNorm（1D weight），不支持 LayerNorm 除非 bake_mean。

### Q3：左旋转和右旋转怎么配？

> 对 `Y = X @ W`：若输入侧乘 `Q`（右），则 `W' = W @ Q`；下一层要消掉 `Q`，在合适位置左乘 `Qᵀ`。适配器用 `RotatePair` 声明：`right_rot` 的 q_proj 配 `left_rot` 的 o_proj。记忆口诀：**右旋进、左旋出**。

### Q4：Hadamard 矩阵怎么构造？

> 随机 ±1 对角阵 D，经 Walsh 蝶形变换得正交 H。维度 n 若非 2 幂，分解 n=k·2^m，k 从预置 txt 加载。支持 block_diag 适配 TP。种子固定 1234 保证可复现。

### Q5：离线旋转和在线旋转区别？

> **离线：** 量化时直接改权重，推理零开销。**在线：** 推理时插旋转算子（HookIR→WrapperIR），有延迟但精度更好。LAOS 混合：o_proj 部分离线+在线 heads 旋转；down_proj 指定层 Kronecker 在线。仅 Qwen3 Dense 完整支持 LAOS。

### Q6：为什么 DeepSeek 要 4 套旋转矩阵？

> MLA 有 hidden→lora→head 三级维度链，单一 Q 尺寸对不上。`rot` 负责 hidden 维；`rot_b_proj` 负责 q 低秩；`rot_kv_b_proj` 负责 kv 低秩；`rot_uv` 负责 V head 段。kv_b 输出还分 nope/rope 两段，用 block_diag 只旋 V 段。

### Q7：QuaRot 在 YAML 流程里排第几？为什么？

> **第一位**，在 flex_smooth_quant 和 linear_quant 之前。旋转改变权重和激活分布，后续 smooth 的 scale 标定和量化都依赖旋转后的分布。

### Q8：TP 并行有什么约束？

> 在线旋转时 `tp_size` 必须是 2 的幂，且 ≤ `max_tp_size`。block_size 控制块对角旋转对齐 TP 切分。违反会导致 rank 间旋转不一致，精度崩溃。

### Q9：量化后旋转矩阵存在哪？

> 三处：① `optional/quarot.safetensors` 的 `global_rotation`（embedding 用）；② metadata 中 heads/kronecker 层列表；③ 各 Wrapper 对应的 safetensors 张量。需 `ascendv1_saver` + `export_extra_info: true`。

### Q10：如何给新模型接入 QuaRot？

> 模型 Adapter 继承 `QuaRotInterface`，实现 `get_ln_fuse_map`、`get_bake_names`、`get_rotate_map` 三个方法。路径必须与 `named_modules()` 一致。参考 `qwen3/model_adapter.py`（简单）或 `deepseek_v3/quarot.py`（复杂）。

---

## 覆盖率摘要

| 模块                                               | 是否覆盖  | 章节      |
| -------------------------------------------------- | --------- | --------- |
| `processor/quarot/offline_quarot/quarot.py`      | ✅        | §6#1     |
| `processor/quarot/common/quarot_utils.py`        | ✅        | §6#2,#3  |
| `processor/quarot/common/hadamard.py`            | ✅        | §6#2     |
| `processor/quarot/offline_quarot/laos_online.py` | ✅        | §6#4     |
| `processor/quarot/online_quarot/`                | ⚠️ 简要 | §3 概念5 |
| `ir/quarot.py`                                   | ✅        | §6#6     |
| `model/qwen3` 旋转映射                           | ✅        | §6#5     |
| `model/deepseek_v3/quarot.py`                    | ✅        | §6#5     |
| `processor/adapt_rotation/`                      | ⚠️ 简要 | §8 场景2 |
| `save/ascendv1.py`                               | ✅        | §6#6     |
| 测试用例                                           | ✅        | §7       |
| 官方文档`quarot.md`                              | ✅        | 全文对齐  |

---

## 分析完成

**模式：** Deep

**核心发现：**

- QuaRot 通过 **Hadamard 正交旋转** 抑制激活离群值，核心数学性质是 `XQ·QᵀW = XW`
- msModelSlim 实现为 **三阶段 Processor** + **模型 Adapter 策略**，算法与拓扑解耦
- **RMSNorm 融合** 是旋转链闭合的前提；**左/右旋转配对** 由 `RotatePair` 声明
- **Qwen3：2 套矩阵**；**DeepSeek MLA：4 套矩阵**——面试最关键的工程差异
- **LAOS 在线旋转** 混合离线/在线，导出 `HookIR→WrapperIR` + safetensors 供推理引擎加载

**完整文档：** `Learning/CodeRead/msmodelslim-quarot-algorithm-interview.md`
