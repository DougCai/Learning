# DeepSpec Ascend NPU 适配（RFC #6 + PR #9）深度理解分析

> 基于本地仓库 `/home/caishengcheng/DeepSpec`（分支 `pr-9-npu-support`），结合 [RFC Issue #6](https://github.com/deepseek-ai/DeepSpec/issues/6) 与 [PR #9](https://github.com/deepseek-ai/DeepSpec/pull/9) 原文。
> 分析模式：**Deep（面试导向）** | 目标：能讲清 **WHY + 设计演进 + 代码落点 + 面试追问应答**
>
> 本地代码分支：`pr-9-npu-support`（5 commits，19 文件，+254/-69）

---

## 理解验证状态

| 核心概念                            | 自我解释 | 理解"为什么" | 应用迁移 | 状态   |
| ----------------------------------- | -------- | ------------ | -------- | ------ |
| DeepSpec 三阶段流水线               | ✅       | ✅           | ✅       | 已掌握 |
| 运行时设备检测 vs 硬编码 CUDA       | ✅       | ✅           | ✅       | 已掌握 |
| `deepspec/utils/device.py` 抽象层 | ✅       | ✅           | ✅       | 已掌握 |
| HCCL / NCCL 分布式初始化差异        | ✅       | ✅           | ✅       | 已掌握 |
| DSpark Attention Mask 双路径        | ✅       | ✅           | ✅       | 已掌握 |
| flex_attention → SDPA 降级         | ✅       | ✅           | ✅       | 已掌握 |
| NPU float64 all_reduce 限制         | ✅       | ✅           | ✅       | 已掌握 |
| FSDP DeviceMesh 设备类型            | ✅       | ✅           | ✅       | 已掌握 |
| Checkpoint RNG 跨后端兼容           | ✅       | ✅           | ✅       | 已掌握 |
| GPU 零回归设计                      | ✅       | ✅           | ✅       | 已掌握 |

---

## 项目完整地图

### DeepSpec 是什么

DeepSpec 是 DeepSeek 开源的 **推测解码（Speculative Decoding）全栈框架**，覆盖：

```
数据准备 → Draft 模型训练 → 推测解码评估
   ↓              ↓                ↓
target cache   DSpark/Eagle3    accept_len / verify_rate
```

支持的 Draft 算法：**DSpark**、**DFlash**、**Eagle3**。PR #9 主要适配 **DSpark/Qwen3 + Gemma4** 路径；Eagle3 通过延迟 import 避免 NPU 环境下不必要的 CUDA 依赖加载。

### PR #9 改动文件地图

| 类别                       | 文件                                            | 职责 / 改动摘要                                   |
| -------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| **设备抽象（新增）** | `deepspec/utils/device.py`                    | NPU/CUDA 运行时检测、`DEEPSPEC_DEVICE` 覆盖     |
| **分布式**           | `deepspec/utils/distributed.py`               | `init_dist` 选 HCCL/NCCL                        |
| **工具导出**         | `deepspec/utils/__init__.py`                  | 统一 re-export device 层 API                      |
| **Metrics**          | `deepspec/utils/metrics.py`                   | `hccl` backend 纳入 CPU→设备搬运判断           |
| **数据加载**         | `deepspec/data/cuda_prefetcher.py`            | Stream 模块动态选择 npu/cuda                      |
| **Attention Mask**   | `deepspec/modeling/dspark/common.py`          | `flex_attention` BlockMask vs SDPA 4D bool mask |
| **Draft Config**     | `deepspec/modeling/dspark/qwen3/config.py`    | NPU 用`sdpa`，GPU 用 `flex_attention`         |
|                            | `deepspec/modeling/dspark/gemma4/config.py`   | 同上                                              |
| **Modeling**         | `deepspec/modeling/dspark/qwen3/modeling.py`  | mask 类型与 attn impl 对齐                        |
|                            | `deepspec/modeling/dspark/gemma4/modeling.py` | 同上                                              |
| **Trainer**          | `deepspec/trainer/base_trainer.py`            | FSDP`device_mesh` 用 `device_type()`          |
| **Checkpoint**       | `deepspec/trainer/ckpt_manager.py`            | RNG state 跨后端 save/load                        |
| **Trainer 导入**     | `deepspec/trainer/__init__.py`                | Eagle3 trainer 延迟 import                        |
| **Eval 指标**        | `deepspec/eval/dspark/confidence_head.py`     | NPU 用 float32 累加 histogram                     |
| **入口**             | `train.py` / `eval.py`                      | spawn 进程数走`device_count()`                  |
| **Cache 生成**       | `scripts/data/prepare_target_cache.py`        | 设备无关 init +`empty_cache()`                  |
| **脚本**             | `scripts/train/train.sh`                      | NPU/CUDA 环境变量分支                             |
| **依赖**             | `requirements.txt`                            | 无强制`torch_npu`（可选安装）                   |

### 核心调用链（NPU 训练）

```
train.sh (ASCEND_RT_VISIBLE_DEVICES)
  └─ train.py: spawn(nprocs=device_count())
       └─ DSparkTrainer.__init__
            └─ init_dist(local_rank)          # HCCL + torch.npu
                 └─ init_device_mesh("npu")   # FSDP hybrid sharding
                      └─ Qwen3DSparkModel.forward
                           └─ create_dspark_attention_mask(attn_implementation="sdpa")
                                └─ 4D bool mask → SDPA kernel
```

---

## 1. 快速概览

| 维度               | 内容                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------- |
| **项目**     | deepseek-ai/DeepSpec — 推测解码训练与评估框架                                          |
| **PR 标题**  | Add Ascend NPU Support While Preserving GPU Compatibility                               |
| **作者**     | sunny-infra（架构/端到端验证）、hazelduan（重构/Review 修复）、wangmhua（GPU 回归验证） |
| **目标硬件** | Ascend 910B / 910C，单节点 2–16 卡                                                     |
| **验证模型** | Qwen3-8B（主）、Gemma4（补充）                                                          |
| **设计原则** | 运行时 detect-and-fallback，GPU 路径 structurally no-op                                 |
| **公开 API** | 不变；新增可选 env`DEEPSPEC_DEVICE`                                                   |
| **PR 状态**  | Open（2026-06-28 提交，rajpratham1 Approved，huangxinV587 多条 Review 已 Revised）      |

---

## 2. 背景与动机（3 个 WHY）

### 2.1 问题本质

**要解决的问题：** DeepSpec 原版所有设备路径硬编码 `torch.cuda`（NCCL、FSDP mesh、Stream、RNG、flex_attention），无法在 Ascend NPU 集群上直接运行。

**WHY 需要解决：**

- README 写明默认假设 8 GPU 单节点，国内大量数据中心以 910B/910C 为主
- 推测解码研究社区需要在 NPU 上复现 Qwen3 等 target 模型的 Draft 训练
- 若每个团队 fork 一份 patch，维护成本极高且与 upstream 脱节

### 2.2 方案选择

**WHY 选择「运行时检测 + 设备抽象层」而非 `#ifdef NPU` 编译分支：**

| 方案                            | 优势                                     | 劣势                          | 结论                         |
| ------------------------------- | ---------------------------------------- | ----------------------------- | ---------------------------- |
| **运行时检测（PR 采用）** | 单份源码、GPU 零依赖 torch_npu、易 merge | 每处需 gating；检测逻辑需集中 | ✅ 选中                      |
| **独立 NPU fork**         | 改动激进、可 NPU 特化                    | 与 main 分叉、难同步          | ❌                           |
| **显式 `--device` CLI** | 行为可预测                               | 用户负担、易与 env 冲突       | ⚠️ RFC 开放问题，PR 用 env |

**替代方案对比：**

- **方案 A：torch.device 全局注入** — 改动面小，但 Stream/RNG/FSDP 等 API 仍散落各处，不够彻底
- **方案 B：完全重写为 Accelerate/DeepSpeed 抽象** — 过度工程，与现有 FSDP 训练栈冲突

### 2.3 应用场景

**适用：** 单节点 Ascend 910B+，Qwen3/Gemma4 DSpark 全流程（cache → train → eval）

**不适用 / 未验证：**

- 多节点 NPU（HCCL 支持但未测）
- DFlash / Eagle3 完整 NPU 建模（仅 DSpark 主路径 + Eagle3 延迟 import）
- NPU CI（无自动化回归，靠人工 smoke test）

---

## 3. 核心概念网络

### 概念 1：Detect-and-Fallback 设备抽象

- **是什么：** 通过 `deepspec/utils/device.py` 在运行时选择 `torch.npu` 或 `torch.cuda`
- **WHY 需要：** 消除 19 个文件中的硬编码 CUDA 调用
- **WHY 这样实现：** `getattr(torch, "npu")` + `is_available()` 懒检测；`torch_npu` 未安装时 `_npu_module()` 返回 None，自动走 CUDA
- **WHY 不用编译宏：** Python 项目需要单 wheel/单 repo 分发

**`DEEPSPEC_DEVICE` 环境变量：**

```python
# deepspec/utils/device.py
def is_npu_available() -> bool:
    requested = os.environ.get("DEEPSPEC_DEVICE", "").strip().lower()
    if requested == "cuda":
        return False          # 强制 GPU，即使有 NPU
    if requested == "npu":
        return _npu_module() is not None
    return _npu_module() is not None  # 默认：有 NPU 就用 NPU
```

### 概念 2：分布式 Backend 映射

| Platform | Backend  | Device Module  | Process Group                                 |
| -------- | -------- | -------------- | --------------------------------------------- |
| NPU      | `hccl` | `torch.npu`  | `init_process_group("hccl", ...)`           |
| CUDA     | `nccl` | `torch.cuda` | `init_process_group("nccl", device_id=...)` |

Rank 计算公式（单节点多卡）：

```
global_rank = node_rank * local_world_size + local_rank
world_size  = node_world_size * local_world_size
```

其中 `RANK`/`WORLD_SIZE` 表示 **节点级** rank（单节点恒为 0/1），卡级 rank 由 `local_rank` 决定。

### 概念 3：DSpark Attention 双路径

DSpark 的 attention mask 不是普通 causal mask，而是 **anchor-block 结构化 mask**：

- **Context 区**（`kv_idx < seq_len`）：只能 attend 到 anchor 位置之前
- **Draft 区**（`kv_idx >= seq_len`）：同 block 内 token 互相可见

**GPU 快路径：** `create_block_mask` → `flex_attention`（PyTorch 2.x 稀疏 block API）

**NPU 降级路径：** 物化为 4D bool mask `[bsz, 1, q_len, kv_len]` → `sdpa` / `eager`

**WHY NPU 不能 flex_attention：** `torch.nn.attention.flex_attention` 仅支持 CUDA/CPU/HPU，NPU 直接报错。

### 概念 4：HCCL float64 限制

`confidence_head.py` 校准 histogram 在 GPU 上用 float64 累加以保证 AUROC/ECE 数值 fidelity。Ascend HCCL 不支持 float64 `all_reduce`（`ERR02007`）。

**PR 最终方案：** 在构造 metrics 时直接选 dtype——NPU 用 float32，GPU 用 float64：

```python
def confidence_metric_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "npu" else torch.float64
```

比 RFC 原方案的「reduce 前 cast」更干净，GPU 路径完全不变。

### 概念关系矩阵

| 关系 | 概念 A              | 概念 B                       | WHY 关联                         |
| ---- | ------------------- | ---------------------------- | -------------------------------- |
| 依赖 | device.py           | distributed.py               | init_dist 通过抽象层选 backend   |
| 依赖 | device.py           | cuda_prefetcher.py           | Stream 必须跟 device 一致        |
| 对比 | flex_attention      | SDPA dense mask              | 同一 mask 语义，不同物化方式     |
| 组合 | attn_implementation | create_dspark_attention_mask | impl 决定 mask 类型，必须对齐    |
| 约束 | HCCL                | confidence_head dtype        | 分布式归约 dtype 受 backend 限制 |

---

## 4. RFC → PR 设计演进对照

PR 在 RFC 基础上做了几处重要迭代，**面试时讲「演进」比只背 RFC 更加分**：

| 主题            | RFC 初版（Issue#6）                                          | PR 最终版（#9, 6/30 更新）                                                                                                                                    |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 设备 helper     | 各 entry 文件重复`_is_npu_available()`                     | 集中到`deepspec/utils/device.py`                                                                                                                            |
| 强制后端        | 仅运行时 auto-detect                                         | 新增`DEEPSPEC_DEVICE=cuda\|npu`                                                                                                                              |
| NPU Attention   | `eager` + 4D additive float mask（-inf）                   | **`sdpa` + 4D bool mask**（更高效、副作用更小）                                                                                                       |
| visible device  | `_visible_device_count()` 读 `ASCEND_RT_VISIBLE_DEVICES` | **最终代码 `device_count()` 直接调 backend API**；靠 `train.sh` 预设 env。RFC 中的 NPU spawn 死锁 fix 在最终版是否完全覆盖需结合 torch_npu 版本验证 |
| eval`--tasks` | RFC 新增 CLI flag                                            | **PR 最终 trim 掉了**（review 反馈「review-only eval changes」）                                                                                        |
| per-epoch ckpt  | RFC 提到                                                     | 当前`ckpt_manager.py` 未见 epoch alias 逻辑，可能未合入或在其他 commit                                                                                      |
| Eagle3          | 未详述                                                       | `trainer/__init__.py` 延迟 import，避免 NPU 环境加载 CUDA-only 依赖                                                                                         |

---

## 5. 设计模式分析

### 模式 1：Strategy（设备后端策略）

**应用位置：** `deepspec/utils/device.py`

**WHY 使用：** 训练/评估/数据加载/Checkpoint 等 10+ 模块都需要同一套「当前加速器是谁」的答案，Strategy 集中决策、调用方只问 `accelerator_module()`。

**WHY 不用单例全局 `DEVICE` 变量：** 环境变量可能在 import 后变化；函数调用保证每次读取最新状态。

### 模式 2：Adapter（Attention Mask 适配）

**应用位置：** `create_dspark_attention_mask(..., attn_implementation=...)`

**WHY 使用：** 同一 `dspark_mask_mod` 语义，GPU 用 BlockMask 适配 flex_attention，NPU 用 dense bool 适配 SDPA——经典 Adapter，不修改上层 DSpark forward 逻辑。

**不用会怎样：** 在 NPU 上直接 crash，或被迫 fork 整个 modeling 文件。

### 模式 3：Lazy Import（Eagle3 Trainer）

**应用位置：** `deepspec/trainer/__init__.py` 的 `__getattr__`

**WHY 使用：** 用户只训 DSpark 时不应 import Eagle3 路径里可能的 CUDA-only 代码；延迟到真正引用 `Qwen3Eagle3Trainer` 时才加载。

---

## 6. 关键代码深度解析

### 核心片段清单

| 编号 | 片段名称                     | 所在文件:行号                                      | 优先级 | 识别理由                   |
| ---- | ---------------------------- | -------------------------------------------------- | ------ | -------------------------- |
| #1   | 设备抽象层                   | `deepspec/utils/device.py:1-76`                  | ★★★ | 整个 PR 的架构锚点         |
| #2   | 分布式初始化                 | `deepspec/utils/distributed.py:20-41`            | ★★★ | HCCL/NCCL 分叉 + rank 计算 |
| #3   | DSpark Attention Mask 双路径 | `deepspec/modeling/dspark/common.py:78-133`      | ★★★ | 唯一 modeling 级 NPU 特化  |
| #4   | Draft Config Attention 选择  | `deepspec/modeling/dspark/qwen3/config.py:7-45`  | ★★☆ | 运行时 impl 绑定           |
| #5   | Confidence Head dtype        | `deepspec/eval/dspark/confidence_head.py:30-109` | ★★☆ | HCCL 数值限制 workaround   |
| #6   | Checkpoint RNG               | `deepspec/trainer/ckpt_manager.py:116-220`       | ★★☆ | 跨设备 resume 兼容         |
| #7   | Prefetcher Stream            | `deepspec/data/cuda_prefetcher.py:16-73`         | ★☆☆ | 设备无关 H2D 流水线        |

---

### 片段 #1：设备抽象层

> 📍 **位置：** `deepspec/utils/device.py:1-76`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 全项目唯一的「加速器是谁」决策中心。

#### 1.1 代码整体作用

这是 PR #9 在 review 后最重要的重构成果。RFC 初版在 `train.py`、`eval.py`、`prepare_target_cache.py` 等入口各写一份 `_is_npu_available()`，huangxinV587 指出「重复太多」后，hazelduan 将其收敛到单一模块。

**它解决了什么问题？** 没有它，每个模块各自检测 NPU，极易出现「训练用 NPU、prefetcher 用 CUDA Stream」这种隐蔽 bug。

**系统层次定位：** Infrastructure / Platform 层，位于所有 Trainer、Evaluator、DataLoader 之下。

#### 1.2 核心逻辑分析

**执行流程：**

```
import torch
  → _npu_module(): getattr(torch,"npu") + is_available()
  → is_npu_available(): 读 DEEPSPEC_DEVICE → 决定 bool
  → device_type(): "npu" | "cuda"
  → accelerator_module(): torch.npu | torch.cuda
  → 上层调用 device_count/set_device/make_device/...
```

**多执行路径：**

- **路径 A（纯 GPU 机器）：** `torch.npu` 不存在 → `_npu_module()=None` → 全程 cuda
- **路径 B（NPU 机器，默认）：** NPU available → 全程 npu + hccl
- **路径 C（NPU 机器，强制 CUDA）：** `DEEPSPEC_DEVICE=cuda` → 即使有 NPU 也走 cuda（调试/对比实验）

#### 1.3 逐行代码解释

> **贯穿示例：** 910B 服务器，已安装 torch_npu，未设置 `DEEPSPEC_DEVICE`

```python
def _npu_module():
    module = getattr(torch, "npu", None)   # 步骤 1: 懒探测 npu 后端是否注册
    if module is None:
        return None                         # GPU 机：torch 无 npu 属性，直接 None
    try:
        return module if module.is_available() else None
        # WHY: is_available 可能抛异常（驱动未加载）；try/except 保证 import 不 crash
    except Exception:
        return None

def device_type() -> str:
    requested = os.environ.get("DEEPSPEC_DEVICE", "").strip().lower()
    if requested in {"cuda", "npu"}:
        return requested                    # 场景 1: 用户显式指定，跳过 auto-detect
    if _npu_module() is not None:
        return "npu"                        # 场景 2: 910B 默认路径 → "npu"
    return "cuda"                           # 场景 3: 纯 GPU → "cuda"

def accelerator_backend() -> str:
    return "hccl" if device_type() == "npu" else "nccl"
    # WHY: PyTorch distributed 后端与硬件绑定；不能 NPU 上用 nccl
```

#### 1.4 关键设计点

| 设计维度           | 分析                                                                                                                                           |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **实现选择** | 函数式 API 而非 class，与 DeepSpec 原有 utils 风格一致；零状态，多进程 spawn 安全                                                              |
| **性能**     | 每次调用读 env + is_available，开销可忽略；不在 hot path 内                                                                                    |
| **可扩展性** | 未来加 MPS/XPU 只需扩展`device_type()` 分支                                                                                                  |
| **潜在问题** | `device_count()` 未显式解析 `ASCEND_RT_VISIBLE_DEVICES`（RFC 曾强调）；若 torch_npu 的 `device_count()` 仍返回物理卡数，spawn 数可能偏多 |

#### 1.5 完整示例（三组对比）

| 场景                | 环境                     | `device_type()` | `accelerator_backend()` |
| ------------------- | ------------------------ | ----------------- | ------------------------- |
| 纯 GPU 开发机       | 无 torch_npu             | `cuda`          | `nccl`                  |
| 910B 生产           | torch_npu OK             | `npu`           | `hccl`                  |
| NPU 上调试 GPU 路径 | `DEEPSPEC_DEVICE=cuda` | `cuda`          | `nccl`                  |

#### 1.6 使用注意与改进建议

1. **spawn 前设置好 env**：`DEEPSPEC_DEVICE` 和 `ASCEND_RT_VISIBLE_DEVICES` 必须在 `train.py` spawn 之前 export，子进程继承父进程 env。
2. **可改进：** 在 `device_count()` 中实现 RFC 的 `_visible_device_count()`，显式解析 visible devices env，避免 NPU 上 hccl barrier 死锁。

---

### 片段 #2：分布式初始化

> 📍 **位置：** `deepspec/utils/distributed.py:20-41`

#### 2.1 代码整体作用

DeepSpec 用 `torch.multiprocessing.spawn` 每卡一进程。`init_dist` 负责把 local_rank 映射到 global_rank，并初始化 process group。

#### 2.3 逐行代码解释

```python
def init_dist(local_rank: int, timeout_minutes: int = 60):
    local_world_size = device_count()       # 步骤 1: 可见卡数
    node_rank = int(os.environ["RANK"])     # 步骤 2: 节点 rank（单节点=0）
    node_world_size = int(os.environ["WORLD_SIZE"])  # 单节点=1
    rank = node_rank * local_world_size + local_rank
    world_size = node_world_size * local_world_size
    set_device(local_rank)                  # 步骤 3: 绑定本进程 NPU/GPU
    device = make_device(local_rank)

    init_kwargs = dict(
        backend=accelerator_backend(),      # hccl 或 nccl
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        rank=rank, world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )
    if device.type == "cuda":
        init_kwargs["device_id"] = device   # PyTorch 2.x NCCL 推荐绑 device
    dist.init_process_group(**init_kwargs)
    return device, rank, world_size
```

**面试追问：** RFC 写 NPU 需要 `device_id`，最终代码只在 CUDA 分支设置——可能因为 HCCL 通过 `set_device(local_rank)` 已足够，或 torch_npu 版本差异。合码前应确认 NPU 上 `init_process_group` 是否还需 `device_id`。

---

### 片段 #3：DSpark Attention Mask 双路径

> 📍 **位置：** `deepspec/modeling/dspark/common.py:78-133`
> 🎯 **优先级：** ★★★
> 💡 **一句话核心：** 同一套 DSpark 可见性规则，两种物化方式服务不同 attention kernel。

#### 3.1 代码整体作用

DSpark draft 模型在 forward 时把序列分为 **context**（原始 token hidden）和 **noise/draft**（mask token 区块）。Attention 必须保证：

1. Draft token 不能看到未来的 context
2. Draft token 只能看到同一 anchor block 内的其他 draft token
3. 无效 anchor（padding block）应被 mask 掉

GPU 用 `flex_attention` 的 `BlockMask` 稀疏表示；NPU 用 dense bool mask 喂给 SDPA。

#### 3.2 核心逻辑分析

**NPU/SDPA 路径（`attn_implementation != "flex_attention"`）：**

```
q_idx, kv_idx 网格
  → q_block_ids = q_idx // block_size
  → anchor_pos = gather(anchor_positions, q_block_ids)   # ⚠️ bsz 维正确 expand
  → mask_context = (kv < seq_len) & (kv < anchor_pos)
  → mask_draft   = (kv >= seq_len) & (same block)
  → dense_mask = (mask_context | mask_draft) & block_keep_mask
  → empty row fallback: 至少保留 self-attention 避免全 -inf
```

**GPU/flex_attention 路径：** 同样的规则写成 `dspark_mask_mod(b,h,q_idx,kv_idx)` 闭包，交给 `create_block_mask`。

**Review 要点：** huangxinV587 指出 early RFC 版本 `q_idx` shape 为 `(1,1,q_len,1)` 在 `bsz>1` 时有 bug；最终版用 `q_block_ids.expand(bsz, -1)` 修复。

#### 3.3 逐行代码解释（NPU 路径核心）

```python
if attn_implementation != "flex_attention":
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len
    q_idx = torch.arange(q_len, device=device)
    q_block_ids = (q_idx // block_size).unsqueeze(0).expand(bsz, -1)
    # WHY expand(bsz): 每个 batch 样本 anchor 位置不同，必须 per-sample gather
    anchor_pos = anchor_positions.gather(1, q_block_ids).unsqueeze(-1)

    is_context = kv_idx < seq_len
    mask_context = is_context & (kv_idx < anchor_pos)
    # 场景: draft 位置 100 的 anchor=50 → 只能看 context[0:50)

    is_draft = kv_idx >= seq_len
    kv_block_ids = (kv_idx - seq_len) // block_size
    mask_draft = is_draft & (q_block_ids.unsqueeze(-1) == kv_block_ids)
    # 场景: 同 block 内 draft token 互相可见

    dense_mask = (mask_context | mask_draft) & is_valid_block
    # empty_rows fallback 防止某 query 行全 False 导致 SDPA NaN
    empty_rows = ~dense_mask.any(dim=-1, keepdim=True)
    self_kv_idx = int(seq_len) + q_idx.view(1, -1, 1)
    dense_mask = dense_mask | (empty_rows & (kv_idx == self_kv_idx))
    return dense_mask.unsqueeze(1)   # [bsz, 1, q_len, kv_len] bool
```

#### 3.4 关键设计点

| 设计维度           | 分析                                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **实现选择** | bool mask 而非 float -inf mask（RFC 初版），因 SDPA 原生支持 bool/additive mask，且 bool 更省内存带宽                            |
| **性能**     | dense mask O(q_len × kv_len)；DSpark block 结构下 q_len 相对可控，但比 flex_attention 稀疏路径慢——NPU 上 acceptable trade-off |
| **正确性**   | mask**语义**与 flex 路径共享同一逻辑，仅物化不同；GPU 用户可通过 `TRAIN_ATTN_IMPLEMENTATION=sdpa` 做等价测试             |

#### 3.5 完整示例

| 场景     | `attn_implementation` | 返回类型            | 消费者                 |
| -------- | ----------------------- | ------------------- | ---------------------- |
| GPU 训练 | `flex_attention`      | `BlockMask`       | flex_attention kernel  |
| NPU 训练 | `sdpa`                | `bool [B,1,Q,KV]` | SDPA                   |
| GPU 调试 | `sdpa`                | `bool [B,1,Q,KV]` | SDPA（验证 mask 等价） |

---

### 片段 #4：Config 层 Attention 绑定

```python
# deepspec/modeling/dspark/qwen3/config.py
TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"

draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
```

```python
# deepspec/modeling/dspark/qwen3/modeling.py (call site)
dspark_attn_mask = create_dspark_attention_mask(
    ...
    attn_implementation=self.config._attn_implementation,
)
```

**WHY 在 config 而非 forward 内 if：** 保证 `from_pretrained` 加载的 config 自描述 attention 类型；FSDP checkpoint resume 时 `attn_implementation` 与权重一致。

**PR 相对 RFC 的改进：** RFC 写 `eager`，PR 改为 `sdpa`——SDPA 在 NPU 上有 fused kernel，比纯 eager matmul+softmax 快。

---

### 片段 #5：Confidence Head — HCCL dtype

```python
def confidence_metric_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "npu" else torch.float64
```

Histogram 累加后用 `dist.all_reduce(SUM)` 跨卡聚合。float32 对 AUROC/ECE 影响可忽略，因为 bin count 本质是整数计数，概率输入已是 float32。

---

### 片段 #6：Checkpoint RNG 跨后端

```python
# save
"torch_accelerator_rng": get_rng_state(),   # 新 key
"torch_rng": torch.get_rng_state(),         # CPU generator

# load（向后兼容）
rng_state = checkpoint.get("torch_accelerator_rng", checkpoint.get("torch_cuda_rng"))
if rng_state is not None:
    set_rng_state(rng_state)
```

**WHY 保留 `torch_cuda_rng` fallback：** 旧 checkpoint 只有 CUDA RNG key，NPU resume 时不 crash。

---

## 7. 端到端验证数据（面试可引用）

PR 在 **8× Ascend 910B2（65GB HBM）** 上验证 Qwen3-8B：

| 阶段         | 配置                                                                | 结果                                              |
| ------------ | ------------------------------------------------------------------- | ------------------------------------------------- |
| Target Cache | 10k samples, metamath395k                                           | ~217 GB cache；4,729,878 tokens；mean seq_len 473 |
| DSpark 训练  | 3 epochs, 19 steps/epoch, num_anchors=**256**, bf16, no_shard | loss 5.55 → 2.52                                 |
| Eval         | gsm8k 10 samples, 4 NPU, temp=1.0                                   | accept_len=1.03, verify_rate=0.1289, AUC=0.7422   |

**⚠️ 面试 honesty：** 这是 smoke test 指标，不是 SOTA 质量声明；910C、Gemma4、GPU 回归由 wangmhua / hazelduan 补充验证。

**OOM 经验：** Qwen3-8B DSpark 在 65GB NPU 上需 `num_anchors` 512→256（`aligned_target_hidden` gather 峰值显存）。这与 NPU 特化无关，40GB GPU 同样会遇到。

---

## 8. Review 意见与修复（体现工程成熟度）

| Reviewer     | 意见                                                          | 修复                  |
| ------------ | ------------------------------------------------------------- | --------------------- |
| huangxinV587 | eval.py 调试残留`("gsm8k", 10)`                             | Revised               |
| huangxinV587 | prepare_target_cache GPU 被强制 eager                         | Revised，改回设备感知 |
| huangxinV587 | debug 日志与原实现不对齐                                      | Revised               |
| huangxinV587 | `_is_npu_available` 重复太多                                | →`device.py` 集中  |
| huangxinV587 | common.py bsz>1 mask bug                                      | Revised expand        |
| rajpratham1  | Approved；建议验证 cross-device ckpt、benchmark SDPA fallback | 待 follow-up          |

---

## 9. 面试高频问答

### Q1：为什么不用 `#ifdef` 或分支仓库，而要 runtime detection？

**答：** DeepSpec 是开源 research codebase，维护方和用户同时有 GPU 和 NPU 集群。Runtime detection 保证 **单分支 merge**，GPU 用户无需安装 torch_npu，NPU 用户无需维护 fork。代价是代码里有 device gating，但通过 `device.py` 集中后可控。

### Q2：NPU 上最大的技术 blocker 是什么？

**答：** 两个：**flex_attention 不可用**（需 SDPA + dense/bool mask 降级）；**HCCL 不支持 float64 all_reduce**（confidence head 用 float32）。其余 FSDP、SDPA、hccl 在 torch_npu 上已可用。

### Q3：如何保证 GPU 零回归？

**答：** 三层保障：

1. 所有分支由 `is_npu_available()` / `device_type()` gating，GPU 机 `_npu_module()` 为 None
2. GPU 仍用 `flex_attention` + `nccl` + float64 metrics
3. wangmhua 在 GPU 上做了功能验证；structurally 每个 diff 在 CUDA 路径是 no-op

### Q4：DSpark attention mask 为什么复杂？能不能用标准 causal mask？

**答：** 不能。DSpark 在 **多个 anchor 位置** 并行 draft block，每个 block 看不同长度的 context + 自己的 draft 区。这是 DSpark 算法本身的 structured attention，不是标准 decoder causal。

### Q5：`DEEPSPEC_DEVICE` 和 `ASCEND_RT_VISIBLE_DEVICES` 区别？

**答：** 前者选 **backend 类型**（cuda vs npu）；后者选 **哪些物理卡可见**（类比 CUDA_VISIBLE_DEVICES）。两者正交，通常在 `train.sh` 一起设置。

### Q6：如果让你继续改进这个 PR，你会做什么？

**答（参考 rajpratham1 + RFC Future Work）：**

1. `device_count()` 实现 visible devices 解析，彻底修复 NPU spawn 死锁
2. 加 `tests/npu/` smoke test（init、FSDP one-step、mask 等价性）
3. NPU CI 或 periodic manual benchmark
4. README 增加 NPU Hardware 章节
5. 评估 SDPA dense mask 相对 flex_attention 的 NPU 性能差距

### Q7：推测解码训练与推理的关系？DeepSpec 在栈中的位置？

**答：** DeepSpec 训练 **draft model**（小模型），学习预测 target model 的 token/hidden；评估阶段用 draft 提出 candidate tokens，target 批量 verify。NPU 适配让整个 **research loop** 不绑定 NVIDIA，不影响推测解码算法本身。

---

## 10. 应用迁移场景

### 场景 1：DeepSpec NPU 适配 → msModelSlim 量化工具 NPU 化

**不变原理：** Platform abstraction layer（device/backend/RNG/Stream 统一入口）

**需修改：**

- 量化校准里的 `torch.cuda.empty_cache` → `empty_cache()`
- 分布式量化 save/load 的 RNG key 命名兼容
- 算子不支持时（如 flex_attention、float64 reduce）找 **语义等价 fallback**

### 场景 2：DeepSpec → 其他训练框架（Megatron / MindSpeed）

**不变原理：** Detect-and-fallback 不改变算法，只换 comm backend 和 attention kernel

**需修改：** FSDP → 对方并行策略；DeviceMesh 维度可能不同

---

## 11. 依赖关系

| 依赖             | 用途              | NPU 说明                                    |
| ---------------- | ----------------- | ------------------------------------------- |
| `torch`        | 核心              | GPU/NPU 共用                                |
| `torch_npu`    | Ascend 后端       | **可选**；lazy import，GPU 环境不需要 |
| `transformers` | Target/Draft 模型 | 不变                                        |
| `hccl`         | NPU 分布式        | PyTorch 内置 backend 名                     |

**环境变量速查：**

```bash
# NPU 训练示例
export DEEPSPEC_DEVICE=npu                              # 可选，默认 auto
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
bash scripts/train/train.sh
```

---

## 12. 质量验证清单

### 理解深度

- [X] 每个核心概念 3 WHY
- [X] RFC vs PR 演进可讲清
- [X] 能指到具体文件行号

### 技术准确性

- [X] HCCL/NCCL、flex/SDPA、float32/float64 差异准确
- [X] DSpark mask 语义与代码一致
- [X] 验证数据来自 PR/RFC 原文

### 最终「四能」测试

1. ✅ 能否理解设计思路？— 运行时抽象 + 算法层最小分叉
2. ✅ 能否独立实现类似功能？— 可复刻 device.py + mask 双路径
3. ✅ 能否应用到不同场景？— 见 §10
4. ✅ 能否向他人清晰解释？— 本文档 + §9 问答

---

## 附录 A：PR Commit 历史

```
30629cf Add Ascend NPU support while preserving GPU compatibility
8506a3e Merge branch 'deepseek-ai:main' into npu-support
53167f5 fix: align Ascend DSpark compatibility
00a7ab8 fix: defer Eagle3 trainer imports
72fe322 fix: trim review-only eval changes
```

## 附录 B：关键链接

- RFC：[deepseek-ai/DeepSpec#6](https://github.com/deepseek-ai/DeepSpec/issues/6)
- PR：[deepseek-ai/DeepSpec#9](https://github.com/deepseek-ai/DeepSpec/pull/9)
- Fork 实现分支：[sunny-infra/DeepSpec-Ascend:npu-support](https://github.com/sunny-infra/DeepSpec-Ascend/tree/npu-support)
- 本地分支：`/home/caishengcheng/DeepSpec` @ `pr-9-npu-support`
