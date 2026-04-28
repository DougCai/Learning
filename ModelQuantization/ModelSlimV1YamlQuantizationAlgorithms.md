# modelslim_v1 YAML 量化模式与算法梳理

来源目录：`C:\workspace\msmodelslim\lab_practice`

筛选范围：统计 v1 体系 YAML，包括 `modelslim_v1`、`multimodal_sd_modelslim_v1` 和 `multimodal_vlm_modelslim_v1`；`modelslim_v0` 以及未标记为 v1 体系的配置不纳入。

| 模型 | 量化模式 | 主要流程算法 | 底层量化算法 / 数据类型 | 备注 |
| --- | --- | --- | --- | --- |
| DeepSeek-R1-0528 | W4A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | W4A8 per-channel，未启用 KV Cache |
| DeepSeek-R1-0528 | W4A8C8 | `quarot` + `flex_smooth_quant` + `linear_quant` + `fa3_quant` | Act: INT8 `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | `kv_cache: True`，含 FA3 Quant / C8 相关量化 |
| DeepSeek-V3.1 | W4A8C8 | `quarot` + `flex_smooth_quant` + `linear_quant` + `fa3_quant` | Act: INT8 `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | `kv_cache: True`，含 FA3 Quant / C8 相关量化 |
| DeepSeek-V3.2-Exp | W4A8 | `quarot` + `flex_awq_ssz` + `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | W4A8 低比特权重量化，叠加 AWQ/SSZ 搜索型离群值抑制 |
| DeepSeek-V3.2-Exp | W8A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token/per-tensor `minmax`; Weight: INT8 per-channel `minmax` | 常规 W8A8 |
| DeepSeek-V3.2 | W8A8 | `quarot` + `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token/per-tensor `minmax`; Weight: INT8 per-channel `minmax` | W8A8，额外使用 QuaRot |
| GLM-4.7 | W8A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | MoE W8A8 |
| GLM-5 | W8A8 | `quarot` + `flex_smooth_quant` + `linear_quant` | Attention: INT8 per-tensor/per-channel `minmax`; MLP: INT8 dynamic per-token/per-channel `minmax` | W8A8，Attention 与 MLP 使用不同 qconfig |
| GLM-5 | W4A8 | `quarot` + `flex_awq_ssz` + `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | W4A8，专家层走 INT4 SSZ |
| Kimi-K2-Instruct-0905 / Kimi-K2-Thinking | W8A8 | `quarot` + `flex_smooth_quant` + `linear_quant` | Attention: INT8 per-tensor/per-channel `minmax`; MLP: INT8 dynamic per-token/per-channel `minmax` | W8A8，额外使用 QuaRot |
| Qwen3-32B | W8A8C8 | `iter_smooth` + `linear_quant` + `dynamic_cache` | Linear: INT8 `pd_mix`/per-channel `minmax`; KV Cache: INT8 per-channel `minmax` | `kv_cache: True`，PDMIX + DynamicCache |
| Qwen3-32B | W4A4 | LAOS 组合：`adapt_rotation` + `autoround_quant` | Weight: INT4/INT8 per-channel `autoround`; Act: INT4/INT8 per-token `minmax` | 两阶段 Adapt Rotation，AutoRound 训练型量化 |
| Qwen3-32B | W4A4 MXFP | LAOS 组合：`adapt_rotation` + `autoround_quant` | Weight: MXFP4/MXFP8 per-block `autoround`; Act: MXFP4/MXFP8 per-block `minmax` | W4A4 的 MXFP 版本 |
| Qwen3-32B | W16A16 Sparse | `float_sparse` | FP16/BF16 权重稀疏化，`sparse_ratio: 0.4` | 不是整数低比特量化，属于浮点稀疏压缩 |
| Qwen3-8B | W4A8 MXFP | `linear_quant` | Act: MXFP8 per-block `minmax`; Weight: MXFP4 per-block `minmax` | YAML 无 metadata/label，模型与模式按文件名和 qconfig 推断 |
| Qwen3-235B | W4A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | MoE W4A8 |
| Qwen3-30B | W4A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4/INT8 per-channel `ssz`/`minmax` | MoE W4A8，按层分组量化 |
| Qwen3-Coder-480B-A35B | W4A8 | `quarot` + `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token/per-tensor `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | Coder MoE W4A8，额外使用 QuaRot |
| Qwen3-Next-80B-A3B-Instruct | W8A8 | `flex_smooth_quant` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | W8A8，按 attention / MLP / router 等分组 |
| Qwen-QwQ-32B / QwQ-32B | W8A8 | `quarot` + `iter_smooth` + `linear_quant` | Act: INT8 per-tensor `minmax`; Weight: INT8 per-channel `minmax` | W8A8，QuaRot + IterativeSmooth |

## 多模态生成：`multimodal_sd_modelslim_v1`

| 模型 | 量化模式 | 主要流程算法 | 底层量化算法 / 数据类型 | 备注 |
| --- | --- | --- | --- | --- |
| FLUX.1 | W8A8F8 MXFP | `online_quarot` + `fa3_quant` + `linear_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | 多模态生成模型，含在线 QuaRot 与 FA3 FP8 |
| HunyuanVideo | W8A8F8/C8 MXFP | `linear_quant` + `online_quarot` + `fa3_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | `kv_cache: True`，文件名为 `w8a8f8_mxfp8`，metadata config_id 写作 `w8a8c8_mxfp8` |
| Qwen-Image-Edit-2509 | W8A8F8 MXFP | `linear_quant` + `online_quarot` + `fa3_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | 图像编辑生成模型 |
| Wan2.1 | W8A8 Dynamic | `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | 动态 W8A8 |
| Wan2.2 I2V | W8A8F8 MXFP | `linear_quant` + `online_quarot` + `fa3_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | Image-to-Video 版本 |
| Wan2.2 T2V | W8A8F8 MXFP | `linear_quant` + `online_quarot` + `fa3_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | Text-to-Video 版本 |
| Wan2.2 TI2V | W8A8F8 MXFP | `linear_quant` + `online_quarot` + `fa3_quant` | Linear: Act/Weight MXFP8 per-block `minmax`; FA3: FP8 E4M3 per-token `minmax` | Text/Image-to-Video 版本 |

## 多模态理解：`multimodal_vlm_modelslim_v1`

| 模型 | 量化模式 | 主要流程算法 | 底层量化算法 / 数据类型 | 备注 |
| --- | --- | --- | --- | --- |
| GLM-4.6V | W8A8 | `iter_smooth` + `linear_quant` | Act: INT8 per-token/per-tensor `minmax`; Weight: INT8 per-channel `minmax` | 多模态理解 W8A8 |
| Qwen2.5-Omni-7B | W8A8 | `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | 校准数据为视频类 `calibVideos` |
| Qwen2.5-VL-7B-Instruct | W8A8 MXFP | `linear_quant` | Act/Weight: MXFP8 per-block `minmax` | MXFP8 W8A8 |
| Qwen2.5-VL-7B-Instruct | W8A8 MXFP + C8/FP8 | `linear_quant` + `dynamic_cache` | Linear: MXFP8 per-block `minmax`; KV Cache: FP8 E4M3 per-channel `minmax` | 文件名标识 C8/FP8，使用 `dynamic_cache` |
| Qwen2.5-VL-32B-Instruct | W8A8 MXFP | `linear_quant` | Act/Weight: MXFP8 per-block `minmax` | MXFP8 W8A8 |
| Qwen2.5-VL-72B-Instruct | W8A8 MXFP | `linear_quant` | Act/Weight: MXFP8 per-block `minmax` | MXFP8 W8A8 |
| Qwen3.5-27B | W8A8 | `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | Dense W8A8 |
| Qwen3.5-397B-A17B | W4A8 | `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT4 per-channel `ssz`; 部分层 INT8 `minmax` | MoE W4A8 |
| Qwen3.5-397B-A17B / Qwen3.5-122B-A10B / Qwen3.5-35B-A3B | W8A8 | `group` + `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | MoE W8A8，分组量化 |
| Qwen3-Omni-30B-A3B-Thinking / Instruct | W8A8 | `linear_quant` | Act: INT8 per-token `minmax`; Weight: INT8 per-channel `minmax` | 校准数据为视频类 `calibVideos` |
| Qwen3-VL-4B-Instruct | W8A8 | `iter_smooth` + `linear_quant` | Act: INT8 per-tensor `minmax`; Weight: INT8 per-channel `minmax` | VLM W8A8 |
| Qwen3-VL-4B-Instruct | W8A8C8 MXFP/FP8 | `linear_quant` + `dynamic_cache` | Linear: MXFP8 per-block `minmax`; KV Cache: FP8 E4M3 per-channel `minmax` | YAML 无 metadata，模型按文件名推断 |
| Qwen3-VL-32B-Instruct | W8A8 | `quarot` + `iter_smooth` + `linear_quant` | Act: INT8 per-tensor `minmax`; Weight: INT8 per-channel `minmax` | VLM W8A8，额外使用 QuaRot |
| Qwen3-VL-235B-A22B / Qwen3-VL-30B-A3B | W8A8 | `quarot` + `iter_smooth` + `linear_quant` | Act: INT8 per-token/per-tensor `minmax`; Weight: INT8 per-channel `minmax` | VLM MoE W8A8 |

## 算法字段速查

| YAML 字段 / processor | 含义 |
|---|---|
| `linear_quant` | 线性层量化处理器，真正的统计/舍入方法由 qconfig 中的 `method` 指定 |
| `minmax` | 基于最小值/最大值统计 scale 的基础量化算法 |
| `ssz` | 权重量化参数优化算法，常用于 INT4 per-channel 权重量化 |
| `autoround_quant` / `autoround` | AutoRound 训练型舍入优化，常用于 W4A4/低比特权重量化 |
| `iter_smooth` | IterativeSmooth 离群值抑制 |
| `flex_smooth_quant` | FlexSmoothQuant，搜索式/自适应 SmoothQuant 离群值抑制 |
| `flex_awq_ssz` | AWQ + SSZ 组合离群值抑制/权重量化辅助 |
| `quarot` | QuaRot 旋转变换，用于扩散离群值 |
| `online_quarot` | 在线 QuaRot，常见于多模态生成的 FP8/MXFP 配置 |
| `adapt_rotation` | 基于校准数据优化旋转矩阵，和 AutoRound 组合构成 LAOS 类 W4A4 方案 |
| `fa3_quant` | Attention FA3 相关量化处理 |
| `dynamic_cache` | KV Cache 动态量化 |
| `float_sparse` | 浮点稀疏化压缩，不属于传统 INT 低比特量化 |
