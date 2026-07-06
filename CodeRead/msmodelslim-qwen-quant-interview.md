# msModelSlim Qwen 系列量化方案深度理解分析

> 基于 `c:\workspace\msmodelslim` 源码，结合 `lab_practice/` 官方配置与模型适配器实现。
> 分析模式：**Deep** | 目标：面试可讲清 **WHY + 架构差异 + 量化策略选型 + 代码对应关系**
>
> 配套文档：[MXFP 量化算法详解](./msmodelslim-mxfp-quant-interview.md)（MXFP8/MXFP4 内核与 Dual Scale / SVDQuant）

---

## 理解验证状态

| 核心概念                         | 自我解释 | 理解"为什么" | 应用迁移 | 状态   |
| -------------------------------- | -------- | ------------ | -------- | ------ |
| 模型适配器分层（Default vs VLM） | ✅       | ✅           | ✅       | 已掌握 |
| 子图（Subgraph）四类拓扑         | ✅       | ✅           | ✅       | 已掌握 |
| Qwen3 q_norm/k_norm KV Smooth    | ✅       | ✅           | ✅       | 已掌握 |
| MoE 3D→Linear 解包              | ✅       | ✅           | ✅       | 已掌握 |
| 混合注意力 sparse 子图           | ✅       | ✅           | ✅       | 已掌握 |
| VLM QuaRot 双流旋转              | ✅       | ✅           | ✅       | 已掌握 |
| v0 vs v1 量化流水线              | ✅       | ✅           | ✅       | 已掌握 |
| W8A8 / W4A8 / W4A4 策略矩阵      | ✅       | ✅           | ✅       | 已掌握 |

---

## 项目完整地图

### Qwen 系列适配器目录树

```
msmodelslim/model/
├── qwen1_5/          # Qwen1.5 Dense LLM
├── qwen2/            # Qwen2 Dense LLM
├── qwen2_5/          # Qwen2.5 Dense LLM
├── qwen3/            # Qwen3 Dense LLM（功能最全）
├── qwen3_moe/        # Qwen3 MoE
├── qwen3_next/       # Qwen3-Next 混合注意力 MoE
├── qwen2_5_vl/       # Qwen2.5-VL 多模态理解
├── qwen2_5_omni_thinker/  # Qwen2.5-Omni 四模态
├── qwen3_vl/         # Qwen3-VL Dense
├── qwen3_vl_moe/     # Qwen3-VL MoE
├── qwen3_5_moe/      # Qwen3.5 VLM + MoE + MTP
├── qwen3_omni_moe/   # Qwen3-Omni MoE
└── qwen_image_edit/  # Qwen-Image-Edit 扩散 Transformer
```

### 文件清单（分类）

| 类别              | 文件路径                                                                                | 职责摘要                                                       |
| ----------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 核心 Dense 适配器 | `model/qwen3/model_adapter.py`                                                        | 全功能 LLM 适配：4 子图 + KV Smooth + QuaRot + FlatQuant + AWQ |
| MoE 适配器        | `model/qwen3_moe/model_adapter.py`                                                    | per-expert up-down + router QuaRot                             |
| 混合注意力        | `model/qwen3_next/model_adapter.py`                                                   | sparse full-attn 子图 + RMSNorm ±1 转换                       |
| VLM 适配器        | `model/qwen3_vl/model_adapter.py`                                                     | 逐层加载 + deepstack + 双流 QuaRot                             |
| VLM MoE           | `model/qwen3_vl_moe/model_adapter.py`                                                 | MoE 层/dense 层子图分流                                        |
| Qwen3.5 全家桶    | `model/qwen3_5_moe/model_adapter.py` + `moe_utils.py` + `modeling_qwen3_5_mtp.py` | MoE 解包 + MTP + mROPE                                         |
| 扩散编辑          | `model/qwen_image_edit/model_adapter.py`                                              | FA3 + Online QuaRot + Dual Scale MXFP4                         |
| 注册配置          | `config/config.ini`                                                                   | 模型名 → 适配器组 → Loader 映射                              |
| 量化配置          | `lab_practice/qwen*/`                                                                 | 各型号官方 YAML 流水线                                         |

### 继承体系与入口调用链

```
CLI: msmodelslim quant --model_type Qwen3-32B --quant_type qwen3-32b-dense-w8a8
  │
  ▼
PluginModelFactory (setup.py entry_points)
  │  config.ini: Qwen3-32B → qwen3 → Qwen3AdapterLoader
  ▼
Qwen3ModelAdapter (implements 10+ Processor Interface)
  │
  ├── get_adapter_config_for_subgraph()  → iter_smooth / flex_smooth / quarot
  ├── get_kvcache_smooth_fused_subgraph() → kv_smooth processor
  ├── get_flatquant_subgraph()           → flatquant processor
  └── generate_model_visit/forward()     → 逐层校准推理
  │
  ▼
YAML spec.process: quarot → iter_smooth → linear_quant → ascendv1_saver
```

---

## 1. 快速概览

**语言/框架：** Python + PyTorch + Transformers，面向华为 Ascend NPU（MindIE / vLLM Ascend）部署。

**Qwen 系列在 msModelSlim 中的定位：** 覆盖从 Qwen1.5 到 Qwen3.5 全代际，包括 Dense LLM、MoE、VLM、Omni 四模态、扩散编辑共 **13 个适配器**。每个适配器的核心职责不是"写量化算法"，而是 **声明模型拓扑**——告诉 anti-outlier processor 哪些层该做 norm-linear / ov / up-down 融合，以及 MoE/VLM 的特殊权重格式如何预处理。

**三代架构分化（面试必记）：**

| 代际       | 基类                    | 代表模型              | 量化路径                      |
| ---------- | ----------------------- | --------------------- | ----------------------------- |
| Gen 1–2   | `DefaultModelAdapter` | Qwen1.5/2/2.5/3       | 标准 LLM 全量加载             |
| Gen 3 MoE  | `DefaultModelAdapter` | Qwen3-MoE, Qwen3-Next | MoE expert 解包 / sparse 子图 |
| Gen 3+ VLM | `VLMBaseModelAdapter` | Qwen3-VL, Qwen3.5-MoE | 逐层 safetensors + 图文校准   |
| 扩散       | `BaseModelAdapter`    | Qwen-Image-Edit       | 独立 FA3 + MXFP4 管线         |

**典型量化方案矩阵：**

| 方案       | 权重             | 激活                  | Anti-Outlier          | 典型 Qwen 型号  |
| ---------- | ---------------- | --------------------- | --------------------- | --------------- |
| W8A8 经典  | int8 per_channel | int8 per_tensor/pdmix | m4 Smooth优化         | Qwen3-32B       |
| W8A8 现代  | int8 per_channel | int8 per_token        | quarot + iter_smooth  | Qwen3-VL        |
| W4A8 MoE   | int4 ssz(g=64)   | int8 per_token        | flex_smooth_quant     | Qwen3-30B MoE   |
| W4A4 LAOS  | int4 autoround   | int4 minmax           | adapt_rotation ×2    | Qwen3-32B       |
| W8A8S 稀疏 | W4 稀疏存储      | int8                  | m6 Flex Smooth        | Qwen3-8B        |
| MXFP8      | mxfp8 per_block  | mxfp8 per_block       | 无（纯 linear_quant） | Qwen2.5-VL-7B   |
| W4A4 MXFP4 | dualscale mxfp4  | dualscale mxfp4       | online_quarot         | Qwen-Image-Edit |

**面试一句话：** Qwen 量化的难点不在 int8 本身，而在 **模型结构差异导致的 adapter 定制**——MoE 要解包 3D expert、VLM 要逐层加载 + 双流 QuaRot、混合注意力只 smooth full-attn 层、Qwen3 的 KV smooth 从 q_proj 移到 q_norm——这些才是 adapter 存在的理由。

---

## 2. 背景与动机（3 个 WHY）

### 问题本质

**要解决的问题：** Qwen 系列模型结构迭代极快（Dense → MoE → VLM → 混合注意力 → MTP），每种结构对 outlier 分布、权重布局、校准数据要求不同。通用量化框架无法"一套配置走天下"。

**WHY 需要模型适配器：** 如果不做 adapter 定制，会出现三类灾难：

1. MoE 的 3D fused `gate_up_proj` 无法被 per-channel 量化器识别
2. VLM 全量加载 235B 模型直接 OOM
3. 混合注意力层对 norm-linear smooth 不敏感，做了反而引入噪声

### 方案选择

**WHY 选择"子图声明 + Processor 流水线"：**

- 优势：量化算法（minmax/autoround/quarot）与模型拓扑解耦，新模型只需实现 `get_adapter_config_for_subgraph()`
- 劣势：每个新 Qwen 变体都要写 adapter + 测试 + lab_practice YAML
- 权衡：华为需要覆盖 Ascend 生态内所有 Qwen 型号，定制 adapter 的成本可接受

**替代方案对比：**

- **方案 A：ONNX 图自动分析** — WHY 不选：Qwen3.5 的 mROPE / deepstack / MTP 在 ONNX 中丢失语义
- **方案 B：统一 DefaultModelAdapter** — WHY 不选：MoE/VLM/混合注意力子图拓扑根本不同
- **方案 C：仅支持 v1 Processor 流水线** — WHY 不选：大量存量 Qwen3 W8A8 用户仍走 v0 `AntiOutlier(m4) + Calibrator`

### 应用场景

**适用场景：** Ascend A2/A3/350 上部署 Qwen 系列，需要 W8A8 生产级精度或 W4A8/W4A4 极致压缩。

**不适用场景：**

- 纯 GPU（CUDA）推理 — msModelSlim 的 ascendv1 saver 格式不通用
- 不做校准的 RTN 量化 — 框架强制需要 calib dataset
- 任意自定义 Qwen fork — 需要注册 entry_point + 写 adapter

---

## 3. 核心概念网络

### 概念 1：子图（Subgraph）

- **是什么：** 由 `AdapterConfig(subgraph_type, mapping)` 声明的一组 **层间耦合关系**，供 SmoothQuant / IterSmooth / FlexSmooth / QuaRot 等 processor 做 outlier 迁移
- **WHY 需要：** 量化误差主要来自 activation outlier；outlier 集中在 LayerNorm 输出 → 下游 Linear 输入的接口处
- **WHY 四类子图：** 对应 Transformer 中四种 outlier 传播路径
- **WHY 不用全局 scale：** 不同接口的 outlier 分布差异大，必须分接口 smooth

| 子图类型          | 映射关系                   | 作用                     |
| ----------------- | -------------------------- | ------------------------ |
| `norm-linear`   | LayerNorm → QKV / gate+up | 吸收 norm 输出 outlier   |
| `ov`            | v_proj → o_proj           | GQA 下 V→O 通道 outlier |
| `up-down`       | up_proj → down_proj       | MLP 中间激活 outlier     |
| `linear-linear` | 通用 Linear → Linear      | FlatQuant 等扩展用       |

### 概念 2：Processor Interface 组合

- **是什么：** Python Mixin 接口（如 `IterSmoothInterface`、`QuaRotInterface`），adapter 通过多重继承"声明能力"
- **WHY 需要：** YAML 中 `type: iter_smooth` 的 processor 运行时会检查 adapter 是否实现了对应 interface
- **WHY 按模型裁剪接口：** Qwen3-MoE 不需要 FlatQuant/AWQ，继承太多接口会误导配置
- **WHY 不用插件注册：** Interface 继承在编译期可检查，IDE 友好

**Qwen3 Dense 接口全集 vs MoE 精简对比：**

| Interface     | Qwen3 | Qwen3-MoE | Qwen3-VL | Qwen3.5-MoE       |
| ------------- | ----- | --------- | -------- | ----------------- |
| IterSmooth    | ✅    | ✅        | ✅       | ✅                |
| FlexSmooth    | ✅    | ✅        | ✅       | ✅                |
| QuaRot        | ✅    | ✅        | ✅       | ❌                |
| KVSmooth      | ✅    | ❌        | ❌       | ❌                |
| FlatQuant     | ✅    | ❌        | ❌       | ❌                |
| AWQ           | ✅    | ❌        | ❌       | ❌                |
| AdaptRotation | ✅    | ✅        | ❌       | ❌                |
| AscendV1Save  | ❌    | ❌        | ❌       | ❌(Qwen3-Next ✅) |

### 概念 3：MoE Expert 解包

- **是什么：** 将 HF 原生的 3D fused tensor `experts.gate_up_proj[expert_idx]` 拆成标准 `nn.Linear` 的 gate_proj + up_proj
- **WHY 需要：** msModelSlim 的 `linear_quant` processor 只认识 `nn.Linear`，不认识 3D Parameter
- **WHY 在 adapter 而非 loader 做：** 解包需要 config（num_experts, intermediate_size），且只在 MoE 层按需触发
- **WHY 不用自定义 QuantLinear：** 解包后复用全部现有 processor，零算法改动

### 概念关系矩阵

| 关系类型 | 概念 A               | 概念 B               | WHY 这样关联                                                   |
| -------- | -------------------- | -------------------- | -------------------------------------------------------------- |
| 依赖     | Subgraph 声明        | IterSmooth Processor | processor 读取 adapter 的 subgraph 列表来决定 hook 哪些层      |
| 对比     | v0 AntiOutlier       | v1 iter_smooth       | 同一目标（outlier 抑制），v0 用 m4 全局算法，v1 用子图精细算法 |
| 组合     | QuaRot + iter_smooth | Qwen3-VL W8A8        | QuaRot 先做旋转使 outlier 均匀化，iter_smooth 再吸收           |
| 对比     | Qwen2 KV Smooth      | Qwen3 KV Smooth      | Qwen3 引入 q_norm/k_norm，融合点从 Linear 变为 Norm            |
| 依赖     | MoE 解包             | W4A8 ssz 量化        | 解包后 expert 才是 per-channel quantizable 的 Linear           |

---

## 4. 算法与理论分析

### 算法 1：IterSmooth（Qwen3-VL 标配）

- **时间复杂度：** O(layers × calib_samples × hidden_dim) — 每层一次校准前向
- **WHY 选择：** 比 SmoothQuant 多轮迭代，对 VLM 图文混合激活分布更鲁棒
- **WHY 复杂度可接受：** VLM 用逐层加载，每次只跑 1 层
- **WHY 不选 AWQ：** VLM 的 vision encoder 和 text decoder 激活分布差异大，AWQ 的 saliency 假设不成立
- **退化场景：** α 过大（>0.95）时权重被过度压缩 — YAML 默认 α=0.9
- **参考：** [msModelSlim Iterative Smooth 文档](https://github.com/Ascend/msmodelslim)

### 算法 2：Flex Smooth Quant（MoE 标配）

- **WHY 选择：** 自动搜索 α/β，适应 MoE 中 expert 激活稀疏性
- **WHY 不选 iter_smooth：** MoE 的 router 导致不同 token 走不同 expert，iter_smooth 的 norm-linear 假设被削弱
- **退化场景：** 全部 expert 的 gate 被 disable_names 跳过时，smooth 效果有限

### 算法 3：LAOS Adapt Rotation + AutoRound（W4A4）

- **WHY 选择：** W4A4 精度要求极高，需要两阶段：stage1 优化旋转矩阵（up_proj），stage2 全局旋转 + AutoRound 400 轮 rounding 搜索
- **WHY 不选纯 minmax：** int4 下 minmax 的 MSE 比 AutoRound 高 2-5 个点（Qwen3-32B 经验值）
- **退化场景：** 校准集 `laos_calib.jsonl` 与下游任务分布不匹配

### 算法 4：SSZ 分组量化（MoE W4A8）

- **WHY 选择：** group_size=64 的 int4 权重在 expert 上比 per-channel minmax 精度更好
- **WHY 不选 GPTQ：** MoE expert 数量多（128+），GPTQ 逐层 Hessian 计算太慢
- **退化场景：** 最后几层 expert（layers 41-47）回退 W8A8 因为 outlier 最集中

---

## 5. 设计模式分析

### 模式 1：Mixin Interface（能力声明）

**应用位置：** 所有 `*ModelAdapter` 类
**WHY 使用：** Processor 运行时通过 `isinstance(adapter, IterSmoothInterface)` 决定是否执行
**WHY 不用会怎样：** YAML 配了 iter_smooth 但 adapter 没实现接口 → 运行时崩溃
**潜在问题：** ⚠️ 接口组合爆炸，Qwen3 继承了 10+ 接口
**参考：** [Refactoring Guru — Mixin](https://refactoring.guru/design-patterns/mixin)

### 模式 2：Template Method（逐层 visit/forward）

**应用位置：** `generated_decoder_layer_visit_func` + 各 adapter 的 `generate_model_forward`
**WHY 使用：** 量化校准必须逐层跑，避免全模型 OOM
**WHY 不用会怎样：** 235B VLM 无法在校准机上加载
**潜在问题：** ⚠️ VLM adapter 的 forward 逻辑高度定制（deepstack/mROPE），不能复用默认 template

### 模式 3：Strategy（量化策略可插拔）

**应用位置：** YAML `qconfig.method`: minmax / ssz / autoround / mse_round / dualscale
**WHY 使用：** 同一模型不同层可用不同策略（W8A8 attn + W4A8 expert）
**WHY 不用会怎样：** 无法做 mixed-precision 量化

---

## 6. 关键代码深度解析

### 核心片段清单

| 编号 | 片段名称                  | 所在文件:行号                             | 优先级 | 识别理由                |
| ---- | ------------------------- | ----------------------------------------- | ------ | ----------------------- |
| #1   | Qwen3 四类子图声明        | `qwen3/model_adapter.py:183-226`        | ★★★ | Dense LLM 量化拓扑基准  |
| #2   | Qwen3 KV Smooth q_norm    | `qwen3/model_adapter.py:125-135`        | ★★★ | Qwen3 vs Qwen2 关键差异 |
| #3   | Qwen3-MoE per-expert 子图 | `qwen3_moe/model_adapter.py:88-126`     | ★★★ | MoE 量化核心            |
| #4   | MoE 3D→Linear 解包       | `qwen3_5_moe/moe_utils.py:149-181`      | ★★★ | MoE 权重量化前置条件    |
| #5   | Qwen3.5 sparse 子图       | `qwen3_5_moe/model_adapter.py:443-466`  | ★★☆ | 混合注意力适配          |
| #6   | Qwen3-VL 完整子图         | `qwen3_vl/model_adapter.py:420-483`     | ★★☆ | VLM 量化基准            |
| #7   | Qwen3-VL-MoE 层类型分流   | `qwen3_vl_moe/model_adapter.py:455-523` | ★★☆ | MoE/dense 子图分流      |
| #8   | Qwen3-Next RMSNorm ±1    | `qwen3_next/model_adapter.py:73-82`     | ★★☆ | 格式兼容 trick          |

---

### 片段 #1：Qwen3 四类子图声明

> 📍 **位置：** `msmodelslim/model/qwen3/model_adapter.py:183-226`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 每一层 decoder 声明 4 组层间耦合，驱动所有 anti-outlier processor 的 hook 安装。

#### 1.1 代码整体作用

Qwen3 Dense 是 msModelSlim 中 **子图定义最完整** 的 LLM 适配器。`get_adapter_config_for_subgraph()` 遍历全部 `num_hidden_layers`，为每层生成 4 个 `AdapterConfig`：2 个 norm-linear（attn + mlp）、1 个 ov、1 个 up-down。

**它解决了什么问题？** 没有这个方法，iter_smooth / flex_smooth_quant processor 不知道在哪些层对之间做 scale 迁移，量化精度会大幅下降。

**系统层次定位：** 模型适配层 → 被 Processor 层的 `BaseSmoothProcessor.preprocess()` 消费。

**角色与依赖：** 上游依赖 `self.config.num_hidden_layers`；下游被 SmoothQuant/IterSmooth/FlexSmooth/QuaRot 读取。

#### 1.2 核心逻辑分析

**执行流程：**

```
for layer_idx in 0..N-1:
  构建 norm-linear #1: input_layernorm → q/k/v_proj
  构建 norm-linear #2: post_attention_layernorm → gate/up_proj
  构建 ov: v_proj → o_proj (group_method=max)
  构建 up-down: up_proj → down_proj
  extend 到 adapter_config 列表
return adapter_config
```

**核心状态变量：**

| 变量名             | 初始值 | 变化时机         | 终态                   |
| ------------------ | ------ | ---------------- | ---------------------- |
| `adapter_config` | `[]` | 每层 extend 4 项 | 长度 = 4 × num_layers |
| `layer_idx`      | 0      | 每层 +1          | num_hidden_layers - 1  |

**多执行路径：**

- **路径 A（正常）：** 64 层 Qwen3-32B → 返回 256 个 AdapterConfig
- **路径 B（边界）：** num_hidden_layers=0 → 返回空列表（不会崩溃但量化无 smooth）

#### 1.3 逐行代码解释

> **贯穿示例输入：** Qwen3-32B，`num_hidden_layers=64`，分析 `layer_idx=0`

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    for layer_idx in range(self.config.num_hidden_layers):
        # 步骤 1: Attn 侧 norm-linear — input_layernorm 的输出 outlier 会被 QKV 放大
        norm_linear_mapping_config1 = MappingConfig(
            source=f"model.layers.{layer_idx}.input_layernorm",
            targets=[
                f"model.layers.{layer_idx}.self_attn.k_proj",
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.v_proj",
            ],
        )
        # WHY: Q/K/V 共享同一个 norm 输出，必须作为一组 target 做联合 smooth
        # 此时: layer_idx=0, source="model.layers.0.input_layernorm"

        # 步骤 2: MLP 侧 norm-linear
        norm_linear_mapping_config2 = MappingConfig(
            source=f"model.layers.{layer_idx}.post_attention_layernorm",
            targets=[
                f"model.layers.{layer_idx}.mlp.gate_proj",
                f"model.layers.{layer_idx}.mlp.up_proj",
            ],
        )
        # WHY: SwiGLU 的 gate 和 up 共享 post-attn norm 输出，联合 smooth 比单独做更稳定

        # 步骤 3: OV 融合 — GQA 下 V→O 是第二大 outlier 源
        ov_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.self_attn.v_proj",
            targets=[f"model.layers.{layer_idx}.self_attn.o_proj"],
        )
        # 场景 1: extra_config group_method='max'
        # WHY: GQA 中 num_kv_heads < num_heads，OV smooth 需要按 head group 取 max scale
        # WHY 不用 mean: max 更保守，避免某个 head group 的 outlier 被低估

        # 步骤 4: MLP up-down — SwiGLU 中间激活是 MLP outlier 集中区
        up_down_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.mlp.up_proj",
            targets=[f"model.layers.{layer_idx}.mlp.down_proj"],
        )

        adapter_config.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config1),
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config2),
            AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config,
                          extra_config={'group_method': 'max'}),
            AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config),
        ])
    return adapter_config
```

#### 1.4 关键设计点

| 设计维度               | 分析内容                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| **实现选择**     | 每层固定 4 子图而非动态分析图结构 — 因为 Qwen3 Dense 结构完全规整，硬编码比图分析更可靠且零开销 |
| **性能优化**     | 子图列表在 processor preprocess 阶段一次性构建，后续每层复用过滤后的子集                         |
| **编译器相关**   | 不涉及                                                                                           |
| **安全与健壮性** | 无显式边界检查，依赖 config.num_hidden_layers 正确                                               |
| **可扩展性**     | 新增子图类型只需加 AdapterConfig，processor 侧 SubgraphRegistry 注册 handler 即可                |
| **潜在问题**     | ⚠️ 所有层统一处理，无法跳过敏感层 — 敏感层回退靠 YAML exclude 而非 adapter                    |

#### 1.5 完整示例（三组对比）

**示例 1 — Qwen3-8B（28 层）**

- **输入：** num_hidden_layers=28 → **输出：** 112 个 AdapterConfig（28×4）

**示例 2 — Qwen3-32B（64 层）**

- **输入：** num_hidden_layers=64 → **关键差异：** YAML 中 11 个 down_proj 在 disable_names 跳过 anti，但子图仍声明（processor 层过滤）

**示例 3 — 边界：config 缺失 num_hidden_layers**

- **输入：** AttributeError → **处理：** `_load_model` 阶段已校验，adapter 不会被创建

#### 1.6 使用注意与改进建议

1. **OV 的 group_method='max' 必须与 GQA 配置一致** — 如果 num_key_value_groups 算错，OV smooth 会错位到错误的 head group，表现是 perplexity 异常升高而非崩溃。
2. **子图声明不等于实际执行** — YAML 的 `enable_subgraph_type` 和 `include/exclude` 会进一步过滤；面试时要区分"adapter 声明了啥"和"YAML 实际跑啥"。

**改进方向：** 可以考虑让 adapter 接受 `sensitive_layers` 参数，直接在子图声明阶段跳过已知敏感层，避免 processor 层重复过滤。

---

### 片段 #2：Qwen3 KV Smooth — q_norm/k_norm 融合点

> 📍 **位置：** `msmodelslim/model/qwen3/model_adapter.py:125-135`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** Qwen3 把 QK Norm 放在 RoPE 之前，KV Cache 量化需要在 Norm 而非 Linear 处做 smooth。

#### 1.1 代码整体作用

Qwen3 引入了 **QK-Norm**（在 attention 内部对 Q/K 做 RMSNorm），这与 Qwen2 直接在 q_proj/k_proj 输出上做 KV smooth 完全不同。`get_kvcache_smooth_fused_subgraph()` 声明融合点为 `q_norm` 和 `k_norm`，类型为 `StateViaRopeToNorm`。

**不用它会有什么后果？** KV Cache INT8 量化后，RoPE 旋转前的 norm 输出 outlier 无法被吸收，长序列 perplexity 显著劣化。

#### 1.3 逐行代码解释

```python
def get_kvcache_smooth_fused_subgraph(self) -> List[KVSmoothFusedUnit]:
    return [
        KVSmoothFusedUnit(
            attention_name=f"model.layers.{i}.self_attn",
            layer_idx=i,
            fused_from_query_states_name="q_norm",    # Qwen3 特有：不是 q_proj
            fused_from_key_states_name="k_norm",      # Qwen3 特有：不是 k_proj
            fused_type=KVSmoothFusedType.StateViaRopeToNorm,
        )
        for i in range(self.config.num_hidden_layers)
    ]
```

**对比 Qwen2（`qwen2/model_adapter.py`）：**

```python
fused_from_query_states_name="q_proj",   # 直接在投影层
fused_from_key_states_name="k_proj",
fused_type=KVSmoothFusedType.StateViaRopeToLinear,
```

**面试要点：** 这不是实现细节，而是 **架构演进导致的量化接口变更**。Qwen3 的 QK-Norm 改变了 outlier 在计算图中的位置，adapter 必须跟着改。

---

### 片段 #3：Qwen3-MoE per-expert 子图

> 📍 **位置：** `msmodelslim/model/qwen3_moe/model_adapter.py:88-126`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** MoE 层不做 post-attn norm-linear，改为每个 expert 独立的 up-down 子图。

#### 1.2 核心逻辑分析

**与 Dense 的关键差异：**

| 子图               | Qwen3 Dense             | Qwen3-MoE                     |
| ------------------ | ----------------------- | ----------------------------- |
| norm-linear (attn) | ✅ 全层                 | ✅ 全层                       |
| norm-linear (mlp)  | ✅ post_attn → gate/up | ❌**跳过**              |
| ov                 | ✅ 全层                 | ✅ 全层                       |
| up-down            | ✅ 1 个/层              | ✅**num_experts 个/层** |

**WHY 跳过 MLP norm-linear：** MoE 的 post_attention_layernorm 下游是 router + 128 个 expert，无法用一个 scale 统一 smooth 所有 expert 的 gate/up。

**WHY 每 expert 独立 up-down：** 每个 expert 的 up_proj → down_proj 是独立的 MLP 路径，outlier 分布因 expert 而异。

#### 1.3 逐行代码解释

```python
def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
    adapter_config = []
    expert_num = getattr(self.config, 'num_experts', None)
    if expert_num is None:
        return adapter_config  # 场景 1: 非 MoE config → 空列表，安全降级

    for layer_idx in range(self.config.num_hidden_layers):
        # 步骤 1: 仅 attn 侧 norm-linear（与 Dense 相同）
        norm_linear_mapping_config = MappingConfig(
            source=f"model.layers.{layer_idx}.input_layernorm",
            targets=[...q/k/v_proj...],
        )
        # 步骤 2: OV（与 Dense 相同）
        ov_mapping_config = MappingConfig(...)

        adapter_config.extend([
            AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
            AdapterConfig(subgraph_type="ov", mapping=ov_mapping_config,
                          extra_config={'group_method': 'max'}),
        ])
        # 步骤 3: 每个 expert 一个 up-down
        for expert in range(expert_num):
            up_down_mapping_config = MappingConfig(
                source=f"model.layers.{layer_idx}.mlp.experts.{expert}.up_proj",
                targets=[f"model.layers.{layer_idx}.mlp.experts.{expert}.down_proj"],
            )
            adapter_config.extend([
                AdapterConfig(subgraph_type="up-down", mapping=up_down_mapping_config)
            ])
    return adapter_config
```

**贯穿示例：** Qwen3-30B-A3B，48 层 × 128 experts → 子图数 = 48 × (2 + 128) = **6240**

#### 1.5 完整示例

**示例 1 — Qwen3-30B-A3B：** 48 层, 128 experts → 6240 子图
**示例 2 — Qwen3-235B：** 94 层, 128 experts → 12220 子图 — flex_smooth 只处理 norm-linear + ov 子集
**示例 3 — num_experts=None：** 返回 `[]` — 不会 crash，但 smooth 完全跳过

---

### 片段 #4：MoE 3D→Linear 解包

> 📍 **位置：** `msmodelslim/model/qwen3_5_moe/moe_utils.py:149-181`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 把 HF 的 fused 3D expert 权重拆成标准 Linear，使通用 quantizer 能处理。

#### 1.3 逐行代码解释

```python
def convert_experts_to_mlp(original_moe_block, config) -> Qwen3_5MoeSparseMoeBlockWithMLP:
    new_moe_block = Qwen3_5MoeSparseMoeBlockWithMLP(config)
    with torch.no_grad():
        # 步骤 1: Router 直接 copy
        new_moe_block.gate.weight.copy_(original_moe_block.gate.weight)

        # 步骤 2: 每个 expert — 核心解包逻辑
        for expert_idx in range(config.num_experts):
            gate_up_weight = original_moe_block.experts.gate_up_proj[expert_idx]
            # gate_up_proj shape: [2*intermediate, hidden] — 3D tensor 的一个 slice
            gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
            # WHY chunk(2): SwiGLU 的 gate 和 up 在 HF 实现中拼接在一起
            new_moe_block.experts[expert_idx].gate_proj.weight.copy_(gate_weight)
            new_moe_block.experts[expert_idx].up_proj.weight.copy_(up_weight)
            new_moe_block.experts[expert_idx].down_proj.weight.copy_(
                original_moe_block.experts.down_proj[expert_idx]
            )

        # 步骤 3: Shared expert（Qwen3.5 特有）
        new_moe_block.shared_expert.gate_proj.weight.copy_(...)
        new_moe_block.shared_expert_gate.weight.copy_(...)
    return new_moe_block
```

**面试要点：** 这个函数是 **量化正确性的前置条件**。如果不解包，`linear_quant` 的 `named_modules()` 遍历找不到 `nn.Linear` 类型的 expert 权重，W4A8 量化会直接跳过 expert 层。

---

### 片段 #5：Qwen3.5 混合注意力 sparse 子图

> 📍 **位置：** `msmodelslim/model/qwen3_5_moe/model_adapter.py:443-466`
> 🎯 **优先级：** ★★☆
> 💡 **一句话核心：** 只对 full_attention_interval 间隔层做 norm-linear smooth，linear attention 层不参与。

#### 1.2 核心逻辑分析

Qwen3.5 和 Qwen3-Next 都采用 **混合注意力**：部分层用 full attention，部分层用 linear attention（如 gated delta net）。Linear attention 层的激活分布与 full attention 完全不同，做 norm-linear smooth 反而有害。

```python
for layer_idx in range(
    self.config.text_config.full_attention_interval - 1,  # 从 interval-1 开始
    self.config.text_config.num_hidden_layers,
    self.config.text_config.full_attention_interval,     # 步长 = interval
):
    # 仅 norm-linear，无 ov / up-down
    adapter_config.extend([
        AdapterConfig(subgraph_type="norm-linear", mapping=norm_linear_mapping_config),
    ])
```

**贯穿示例：** full_attention_interval=4, num_hidden_layers=48 → smooth 层 = {3, 7, 11, ..., 47}，共 12 层

---

### 片段 #6：Qwen3-VL-MoE 层类型分流

> 📍 **位置：** `msmodelslim/model/qwen3_vl_moe/model_adapter.py:455-523`
> 🎯 **优先级：** ★★☆
> 💡 **一句话核心：** 根据层索引判断 MoE 层 vs dense 层，分别声明不同的 MLP 子图。

```python
# 所有层都做: norm-linear(attn) + ov
adapter_config.extend([norm-linear, ov])

# 仅 dense MLP 层额外做: norm-linear(mlp) + up-down
if layer_idx not in self.config.text_config.mlp_only_layers:
    if (layer_idx + 1) % self.config.text_config.decoder_sparse_step != 0:
        adapter_config.extend([norm-linear(mlp), up-down])
    # MoE 层 (step 匹配): 跳过 MLP 子图，expert 在 QuaRot rotate_map 中单独处理
```

**面试要点：** MoE 层的Expert 不在 subgraph 中做 up-down smooth，是因为 expert 在 QuaRot 阶段已被旋转；量化阶段用 per-expert 动态 W8A8（`qwen3_vl_moe_w8a8.yaml`）。

---

## 7. 测试用例分析

### 测试文件清单

| 测试文件                              | 测试模块    | 核心验证                                                          |
| ------------------------------------- | ----------- | ----------------------------------------------------------------- |
| `test_model_adapter_qwen3.py`       | Qwen3 Dense | GQA 参数（head_dim, num_kv_groups）                               |
| `test_model_adapter_qwen3_moe.py`   | Qwen3-MoE   | 子图数 = layers×(2+experts)；QuaRot ln_fuse 含 router            |
| `test_model_adapter_qwen3_5_moe.py` | Qwen3.5     | sparse 子图；VLM 校准强制 image+text；MTP 检测                    |
| `test_model_adapter_qwen3_vl.py`    | Qwen3-VL    | 4 子图/层；ov group_method=max；tie_word_embeddings 不支持 QuaRot |
| `test_model_adapter_qwen2_5_vl.py`  | Qwen2.5-VL  | 3 子图/层（**无 ov**）                                      |
| `test_model_adapter_qwen3_next.py`  | Qwen3-Next  | sparse 子图；RMSNorm weight±1                                    |

### 功能覆盖矩阵

| 核心功能        | 主代码位置      | 测试覆盖          | 评估                       |
| --------------- | --------------- | ----------------- | -------------------------- |
| 四类子图声明    | qwen3 adapter   | ⚠️ 无直接测试   | 靠 MoE/VL 测试间接验证     |
| MoE expert 解包 | moe_utils.py    | ⚠️ 无单测       | 靠集成测试                 |
| MTP 加载        | qwen3_5 adapter | ✅`_has_mtp()`  | 良好                       |
| QuaRot 映射     | 各 adapter      | ✅ ln_fuse/rotate | 良好                       |
| 端到端量化数值  | —              | ❌ 无             | 依赖 lab_practice 人工验证 |

### 从测试中发现的边界条件

1. **Qwen2.5-VL 无 ov 子图** — 测试确认只有 3 子图/层，说明该模型上 OV smooth 会导致精度下降
2. **Qwen3-VL tie_word_embeddings=True 抛 UnsupportedError** — QuaRot 不支持 tied embedding
3. **Qwen3.5 必须 image+text 校准** — `handle_dataset` 强制检查，纯文本校准会被拒绝
4. **Qwen3-Next RMSNorm ±1** — 测试验证 save 预处理时 weight-1，load 时 weight+1

---

## 8. 应用迁移场景

### 场景 1：Qwen3 Dense 方案 → 新 Dense LLM（如 Llama 3）

**不变的原理：** 4 子图拓扑（norm-linear×2 + ov + up-down）适用于所有标准 Pre-LN Transformer

**需要修改的部分：**

- KV Smooth 融合点：Llama 3 无 q_norm → 用 `StateViaRopeToLinear` 在 q_proj/k_proj
- 模块命名前缀：`model.layers` → `model.model.layers`（Llama 嵌套更深）
- QuaRot rotate_map：head_dim 计算方式可能不同

**通用模式：** "声明式子图 + YAML 流水线" 可复用到任何 Transformer LLM

### 场景 2：Qwen3-MoE 方案 → 其他 MoE LLM（如 Mixtral / DeepSeek-MoE）

**不变的原理：** MoE 层跳过 MLP norm-linear + per-expert up-down

**需要修改的部分：**

- Expert 权重格式：Mixtral 可能已是独立 Linear，无需解包
- Router 命名：`mlp.gate` vs `block_sparse_moe.gate`
- Shared expert：Qwen3.5 有 shared expert + gate，Mixtral 无

---

## 9. 依赖关系与使用示例

### 量化配置选型决策树

```
Qwen 型号？
├── Dense LLM
│   ├── W8A8 生产级 → v0: anti(m4) + calib(pdmix)     [qwen3-32b-dense-w8a8.yaml]
│   ├── W4A4 极致压缩 → v1: adapt_rotation + autoround  [qwen3-32b-dense-w4a4.yaml]
│   └── W8A8S 稀疏   → v0: anti(m6) + 低比特稀疏       [qwen3-8b-w8a8s.yaml]
├── MoE LLM
│   ├── W8A8 → v0: anti(m4) + mix_cfg(MLP动态)         [qwen3-30b-a3b-w8a8.yaml]
│   └── W4A8 → v1: flex_smooth + ssz(group=64)         [qwen3-30b-w4a8-v1.yaml]
├── VLM
│   ├── W8A8 → v1: quarot + iter_smooth + linear_quant [qwen3_vl_w8a8.yaml]
│   ├── MoE W8A8 → 上述 + 专家动态量化                  [qwen3_vl_moe_w8a8.yaml]
│   └── MXFP8 → 纯 linear_quant per_block              [qwen2.5-vl-7b-w8a8-mxfp.yaml]
├── VLM+MoE (Qwen3.5)
│   ├── W8A8 → 无 anti，直接 linear_quant×2             [qwen3_5_moe_w8a8.yaml]
│   └── W4A8 → attn/visual W8A8 + expert W4A8           [qwen3.5_moe_w4a8.yaml]
└── 扩散编辑
    └── W4A4 MXFP4 → dualscale + online_quarot + fa3    [qwen-image-edit-w4a4f4-mxfp.yaml]
```

### 完整配置对比表

| 配置文件                    | API    | 格式        | 流水线                        | Anti-Outlier  | 敏感层策略                 |
| --------------------------- | ------ | ----------- | ----------------------------- | ------------- | -------------------------- |
| qwen3-32b-dense-w8a8        | v0     | W8A8        | m4→calib                     | m4            | 11层 down/gate disable     |
| qwen3-32b-dense-w4a4        | v1     | W4A4/W8A8混 | adapt_rot×2→autoround       | LAOS          | attn+down W8A8, MLP W4A4   |
| qwen3-8b-w8a8s              | v0     | W8A8S       | m6→calib                     | m6 Flex       | 多层 gate/up/down float    |
| qwen3-30b-a3b-w8a8          | v0     | W8A8混      | m4→calib                     | m4            | 全部 gate disable; MLP动态 |
| qwen3-30b-w4a8-v1           | v1     | W4A8        | flex_smooth→group            | flex_smooth   | 专家 W4A8; L41-47 W8A8     |
| qwen3_5_moe_w8a8            | VLM v1 | W8A8        | linear_quant×2               | 无            | 排除 o_proj, mtp           |
| qwen3.5_moe_w4a8            | VLM v1 | W4A8        | linear_quant×2               | 无            | visual+attn W8A8           |
| qwen3_vl_w8a8               | VLM v1 | W8A8        | quarot→iter_smooth→quant    | quarot+iter   | 排除 down_proj/merger      |
| qwen3_vl_moe_w8a8           | VLM v1 | W8A8混      | quarot→iter_smooth→quant×2 | quarot+iter   | 专家动态                   |
| qwen2.5-vl-7b-w8a8-mxfp     | VLM v1 | MXFP8       | linear_quant                  | 无            | 排除 merger/visual down    |
| qwen3-next-80b-a3b-w8a8     | v1     | W8A8        | flex_smooth→group            | flex_smooth   | +linear_attn 层            |
| qwen-image-edit-w4a4f4-mxfp | SD v1  | W4A4 MXFP4  | quant→online_quarot→fa3     | online_quarot | 排除 mod 层                |

### 一键量化命令示例

```bash
# Qwen3-32B W8A8（最常用生产配置）
msmodelslim quant \
  --model_path /path/to/Qwen3-32B \
  --save_path /path/to/Qwen3-32B-W8A8 \
  --device npu \
  --model_type Qwen3-32B \
  --quant_type qwen3-32b-dense-w8a8 \
  --trust_remote_code True

# Qwen3-VL W8A8（多模态，需要图文校准集）
msmodelslim quant \
  --model_path /path/to/Qwen3-VL-32B-Instruct \
  --save_path /path/to/Qwen3-VL-32B-W8A8 \
  --device npu \
  --model_type Qwen3-VL-32B-Instruct \
  --quant_type qwen3_vl_w8a8 \
  --trust_remote_code True
```

---

## 10. 质量验证清单

### 理解深度

- [X] 每个核心概念都回答了 3 个 WHY
- [X] 自我解释测试：能不看代码解释 Qwen3 vs Qwen3-MoE 子图差异
- [X] 概念连接：子图 → Processor → YAML 三层关系清晰

### 技术准确性

- [X] 算法选择有 WHY + 退化场景
- [X] 设计模式有应用位置和不用后果
- [X] 代码解析有真实代码 + 贯穿示例

### 实用性

- [X] 应用迁移：Dense → 新 LLM、MoE → 新 MoE
- [X] 配置选型决策树
- [X] 一键量化命令

### 最终"四能"测试

1. ✅ 能否理解 Qwen 量化的设计思路？— 子图声明 + 流水线解耦
2. ✅ 能否独立为新 Qwen 变体写 adapter？— 继承正确基类 + 实现 subgraph
3. ✅ 能否为不同 Qwen 型号选量化方案？— 查决策树
4. ✅ 能否向面试官清晰解释 MoE 解包和 sparse 子图？— 见片段 #3/#4/#5

---

## 附录 A：面试高频问答

### Q1: Qwen3 和 Qwen2 的量化适配有什么区别？

| 维度             | Qwen2           | Qwen3                                             |
| ---------------- | --------------- | ------------------------------------------------- |
| KV Smooth 融合点 | q_proj / k_proj | **q_norm / k_norm**                         |
| 子图             | 3 类（无 ov）   | **4 类（含 ov, group_method=max）**         |
| QuaRot rot_uv    | 无              | **有（head_dim 级 Hadamard）**              |
| Processor 接口   | KVSmooth + AWQ  | KVSmooth + AWQ + FlatQuant + AdaptRotation + LAOS |

### Q2: 为什么 MoE 模型要做 expert 解包？

HF Transformers 为内存效率把 expert 权重存成 3D tensor（`[num_experts, 2*intermediate, hidden]`）。msModelSlim 的 `linear_quant` 通过 `isinstance(module, nn.Linear)` 识别可量化层。3D Parameter 不是 Linear，会被跳过。解包后每个 expert 是标准 SwiGLU（gate_proj + up_proj + down_proj），W4A8 ssz 量化才能生效。

### Q3: Qwen3-VL 的 QuaRot 为什么需要"双流旋转"？

VLM 中 text embedding 和 visual feature 在同一 residual stream 汇合。QuaRot 用全局旋转矩阵 R 统一处理：

- Text 侧：右旋转 `W_new = W_old @ R`（embedding、QKV、MLP）
- Visual 侧：左旋转 `W_new = R^T @ W_old`（merger、deepstack_merger）

这样 visual feature 进入 text decoder 时已处于同一旋转空间。如果不做双流，图文特征在旋转空间不对齐，量化误差会集中在 cross-modal 接口。

### Q4: v0 和 v1 量化路径怎么选？

- **v0**（`modelslim_v0`）：老路径，`AntiOutlier(m1-m6) + Calibrator`，适合 W8A8 生产部署，配置简单
- **v1**（`modelslim_v1`）：新路径，Processor 流水线（quarot/iter_smooth/linear_quant），适合 W4A4/W4A8 低比特和精细控制
- **VLM v1**（`multimodal_vlm_modelslim_v1`）：v1 + 多模态校准数据支持

选择原则：W8A8 用 v0 足够；W4A4 必须 v1；VLM 必须 VLM v1。

### Q5: 敏感层回退策略有哪些？

1. **YAML disable_names**（v0）：跳过 anti-outlier，如 Qwen3-32B 的 11 个 down_proj
2. **YAML exclude**（v1）：跳过量化，如 Qwen3-VL 的 down_proj / merger
3. **YAML 分层 qconfig**（v1）：敏感层 W8A8 + 其余 W4A4，如 qwen3-32b-dense-w4a4
4. **mix_cfg float 回退**（v0）：特定层保持 float，如 qwen3-8b-w8a8s
5. **Adapter 层类型分流**：Qwen3.5 的 linear attention 层不做 smooth

### Q6: Qwen3.5 的 MTP 怎么量化？

MTP（Multi-Token Prediction）层在 YAML 中通过 `exclude: "mtp*"` **整体排除**。原因是 MTP 是训练辅助头，推理时可选；且 MTP 的 packed expert 权重格式与 base model 不同，需要独立的 `_load_mtp_predictor()` 加载逻辑。当前策略是 **不量化 MTP**，只量化 backbone。

---

## 附录 B：子图拓扑速查图

```
Qwen3 Dense (每层 4 子图):
  input_layernorm ──→ q/k/v_proj     [norm-linear]
  post_attn_layernorm ──→ gate/up_proj [norm-linear]
  v_proj ──→ o_proj                   [ov, group=max]
  up_proj ──→ down_proj               [up-down]

Qwen3-MoE (每层 2+E 子图):
  input_layernorm ──→ q/k/v_proj     [norm-linear]
  v_proj ──→ o_proj                   [ov]
  experts.{i}.up ──→ experts.{i}.down [up-down] × E

Qwen3-VL (text 每层 4 子图, 同 Dense):
  model.language_model.layers.{i}.*   [同上, 前缀不同]

Qwen3-VL-MoE (MoE 层 2 子图, dense 层 4 子图):
  所有层: norm-linear(attn) + ov
  dense 层额外: norm-linear(mlp) + up-down

Qwen3.5/Next (sparse, 仅 full-attn 层 1 子图):
  layers {interval-1, 2*interval-1, ...}: norm-linear(attn only)
```

---

## 附录 C：相关文档索引

| 文档               | 路径                                                                                    | 内容                       |
| ------------------ | --------------------------------------------------------------------------------------- | -------------------------- |
| MXFP 量化算法      | [msmodelslim-mxfp-quant-interview.md](./msmodelslim-mxfp-quant-interview.md)             | MXFP8/MXFP4 内核           |
| Qwen3-32B 精度调优 | `msmodelslim/docs/zh/best_practices/qwen3_32b_w8a8_precision_tuning_case.md`          | iter_smooth vs flex_smooth |
| 大模型支持矩阵     | `msmodelslim/docs/zh/user_guide/model_support/foundation_model_support_matrix.md`     | 型号支持一览               |
| LAOS W4A4          | `msmodelslim/docs/zh/user_guide/quantization_algorithms/.../laos.md`                  | adapt_rotation 原理        |
| 多模态接入指南     | `msmodelslim/docs/zh/development_guide/integrating_multimodal_understanding_model.md` | VLM adapter 设计           |

---

*文档生成时间：2026-07-06 | 基于 msmodelslim master 分支源码分析*
