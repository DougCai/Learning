# msModelSlim 领域驱动架构面试话术

## 核心主线

> 我们不是为了套用 DDD 概念，而是希望把散落在一次性量化脚本里的模型知识、算法知识和工程依赖，沉淀成可组合、可复用、可扩展的产品能力。

## 90 秒面试话术

msModelSlim 早期的量化方式比较接近传统脚本：模型加载、模型结构识别、量化算法、校准流程和权重保存都耦合在一个脚本里。这样做单模型交付很快，但随着模型和量化方案增多，会出现两个问题：第一，新模型接入需要复制和修改大量脚本；第二，同一个算法很难在 Qwen、DeepSeek 等不同模型之间复用，社区贡献者也很难判断修改边界。

因此，我在开源易用性建设中引入了领域驱动的设计思想。这里主要采用的是 DDD 的战略设计和分层思想，没有机械照搬实体、聚合根等战术概念。我们围绕量化业务划分了接口层、应用层、领域层和基础设施层。

接口层负责 CLI 参数解析和依赖装配；应用层负责一键量化、敏感层分析、自动调优等用例编排；领域层沉淀 IR 量化模式、Processor 量化算法、Runner 调度和最佳实践等核心知识；基础设施层负责适配 Transformers 模型、数据集、文件系统、评测工具以及推理框架。

其中最关键的设计是依赖反转：算法和应用不直接依赖某个具体模型或者文件系统，而是由内部领域提出接口协议，外部模型适配器和基础设施去实现。例如 SmoothQuant 只声明自己需要哪些 Norm-Linear 结构信息，Qwen 和 DeepSeek 的适配器分别提供这些信息，算法本身不需要感知模型差异。

最终，一次量化请求由 CLI 创建应用和基础设施，应用选择模型适配器与 YAML 最佳实践，量化服务根据配置组合 Runner、Processor 和保存格式完成执行。对用户来说，由编写 Python 量化脚本变成一条命令或一份 YAML；对开发者来说，新增模型主要增加模型适配器，新增算法主要增加 Processor 和 IR，核心流程一般不用修改，从而降低了使用和开源贡献门槛。

## 四层架构

| 层次       | 面试表述                                           | 代码中的典型内容                            |
| ---------- | -------------------------------------------------- | ------------------------------------------- |
| 接口层     | 接收用户意图，并作为组合根装配依赖，不承载量化算法 | `cli`、`msmodelslim quant/analyze/tune` |
| 应用层     | 编排用例，决定先做什么、后做什么，但不实现具体算法 | 一键量化、自动调优、敏感层分析              |
| 领域层     | 沉淀稳定的量化知识，是架构核心                     | IR、Processor、Runner、量化服务、最佳实践   |
| 基础设施层 | 处理变化频繁的外部依赖，按照内部协议提供实现       | 模型适配、数据集、YAML、评测服务、文件存储  |

一次量化请求的典型调用链如下：

```text
CLI
  -> NaiveQuantizationApplication
  -> 创建模型适配器、匹配 YAML 最佳实践
  -> QuantServiceProxy 按 apiversion 选择量化服务
  -> Runner 按配置组合多个 Processor
  -> 校准、量化、保存量化权重
```

对应的代码证据：

- CLI 负责装配应用、模型工厂、数据集和配置管理器：[__main__.py](../../msmodelslim/msmodelslim/cli/naive_quantization/__main__.py#L53)
- 应用层编排“模型适配 -> 最佳实践 -> 量化服务”：[application.py](../../msmodelslim/msmodelslim/app/naive_quantization/application.py#L462)
- 量化服务按照 YAML 组合 Runner 和 Processor：[quant_service.py](../../msmodelslim/msmodelslim/core/quant_service/modelslim_v1/quant_service.py#L184)
- 模型通过插件工厂加载：[plugin_model_factory.py](../../msmodelslim/msmodelslim/model/plugin_factory/plugin_model_factory.py#L37)
- 项目的正式设计说明：[architecture.md](../../msmodelslim/docs/zh/development_guide/architecture.md#L54)

## DDD 在项目中的具体映射

### 1. 统一领域语言

我们把团队日常讨论的量化概念直接映射到代码和 YAML：

- 量化模式：W8A8、W4A8、MXFP8、MXFP4。
- Processor：SmoothQuant、GPTQ、QuaRot 等算法处理单元。
- Model Adapter：模型结构知识以及模型推理流程适配。
- Practice：特定模型和场景对应的最佳量化方案。
- Runner：模型级、逐层、DP 多卡等执行策略。
- Format：AscendV1、MindIE、compressed-tensors 等权重交付格式。

这样算法、模型、推理和算子团队交流时使用的是同一套语言，减少了“同一个词在不同团队含义不同”的问题。

### 2. 按知识领域划分边界

核心代码不是简单按照“工具类、公共类、服务类”划分，而是按照量化能力划分：

- `ir`：定义什么是某种量化结构，对量化公式和参数进行形式化描述。
- `processor`：定义如何识别、校准和替换量化结构。
- `format`：定义如何把量化结果交付给推理框架。
- `model`：适配具体模型，满足算法和调度对模型结构的诉求。
- `app`：把上述知识编排成一键量化、分析和调优等用户场景。

这里的目标不是让目录看起来整齐，而是让不同维度的变化被限制在各自边界中。

### 3. 依赖反转

接口协议由提出需求的一方定义，而不是由外部实现决定内部逻辑。

例如，一键量化应用需要读取最佳实践，就由应用层声明 `PracticeManagerInfra`。基于 YAML 文件的管理器只是它的一个实现，应用层不会直接依赖 YAML 文件的读取细节。未来如果最佳实践改为数据库或远端服务，应用流程可以保持不变。

模型适配也采用相同思路。SmoothQuant、QuaRot、FA3 等领域组件分别声明自己需要的模型结构信息，具体模型适配器按需实现这些接口。一个模型如果没有使用 SmoothQuant，就不需要实现 SmoothQuant 的适配协议，避免形成不断膨胀的万能模型接口。

### 4. 配置驱动与组件组合

YAML 描述的是一套量化方案，不是简单地把 Python 函数参数搬到配置文件中。Runner 会按照配置动态组合多个 Processor，因此：

- W8A8 切换为 W4A8，主要是切换量化组件和配置。
- 增加 SmoothQuant 或 QuaRot，主要是增加一个处理阶段。
- 切换保存格式，主要是替换 Format 或 Saver 组件。
- 单卡逐层执行切换为 DP 多卡执行，主要是切换 Runner。

这将“模型 × 算法 × 执行策略 × 权重格式”的组合问题，从复制脚本转化成了组件组合问题。

### 5. 插件化扩展

模型适配器和量化服务可以通过插件机制注册。核心框架依赖统一接口和配置类型，不需要预先知道所有具体实现。

这样做有两个直接价值：

1. 新模型或新服务可以在明确边界内扩展，降低修改核心代码的概率。
2. 第三方贡献者能够通过适配器、Processor、配置或插件提交独立能力，减少多人修改同一个主流程造成的冲突。

## 为什么采用这套架构

### 传统量化脚本的问题

传统脚本通常把以下内容混合在一起：

- 模型加载和 Transformers 版本处理。
- 模型结构名称以及模块遍历逻辑。
- 校准数据预处理。
- 离群值抑制和量化参数计算。
- 单卡或多卡调度。
- 权重格式转换和保存。

这种方式在第一个模型上开发速度快，但模型和方案规模扩大后，会出现：

- 新模型接入需要复制已有脚本，再修改大量模型结构细节。
- 同一量化算法在不同模型脚本中产生多个变体，难以统一修复和演进。
- 算法、调度、存储和外部依赖相互影响，改动范围不可控。
- 用户需要理解大量 Python API 和算法实现细节。
- 社区贡献者难以确定新能力应放在哪里、需要修改哪些模块。

### 选择领域驱动思想的原因

msModelSlim 的核心复杂度不在参数校验或者 CRUD，而在不同知识维度会独立变化：

- 模型结构持续变化。
- 量化算法持续增加。
- 低精度数据格式和硬件能力持续演进。
- 推理框架和权重格式持续变化。
- 一键量化、敏感层分析和自动调优等应用场景持续增加。

因此，需要按照知识边界管理变化，而不能继续围绕单次交付脚本组织代码。

## 面试追问与回答

### 追问 1：为什么不用普通分层架构，非要说领域驱动？

普通分层主要回答代码放在哪里，DDD 更重要的是回答知识边界如何划分。msModelSlim 的主要复杂度不是数据库增删改查，而是模型结构、量化算法、执行调度和权重格式会独立演进。我们按照这些知识领域建模，让变化尽可能被限制在各自边界内。

四层只是外在结构，真正体现领域驱动的是统一领域语言、能力边界、接口协议以及领域知识的持续沉淀。

### 追问 2：这是完整的 DDD 吗？

不是教科书式的全套战术 DDD。这个项目不是典型的交易系统，没有强行设计实体、聚合根和领域事件。我们主要采用了统一语言、领域边界、应用服务、依赖反转和基础设施适配。

因此，更准确的说法是：采用 DDD 的领域建模思想，并结合分层架构、端口适配器和插件机制完成工程落地。

### 追问 3：为什么模型适配属于基础设施，而不是领域层？

量化算法真正关心的是 Norm-Linear 结构对、Attention 结构、模型层遍历方式和校准推理过程，而不应该关心 Qwen 或 DeepSeek 具体由哪个 Transformers 类实现。

具体模型、Transformers 版本和远程模型代码都属于外部变化源。因此，由领域层声明自己需要的模型能力，模型适配器负责把外部模型转换成领域可以理解的接口。这能够把算法知识和模型实现解耦。

### 追问 4：新增一个模型需要修改什么？

通常需要完成以下工作：

1. 实现基础模型协议，提供模型类型、路径等基本信息。
2. 实现量化调度 Pipeline，描述数据处理、模型加载、逐层访问和前向过程。
3. 根据目标量化方案，按需实现 SmoothQuant、QuaRot、FA3 等算法适配接口。
4. 注册模型插件。
5. 增加经过验证的 YAML 最佳实践配置。

例如只支持 W8A8 动态量化时，通常不需要实现所有离群值抑制接口。正常情况下也不需要修改一键量化应用和通用算法主流程。

### 追问 5：新增一个量化算法需要修改什么？

主要新增 Processor 配置和算法实现；如果引入了新的量化结构，再补充对应 IR；如果算法依赖某种模型结构，则声明一个最小模型适配协议。

完成后，算法可以通过 YAML 组合进已有 Runner 和量化服务，不需要为每个模型重新开发一套端到端脚本。

### 追问 6：基础设施如何替换？

应用或领域首先声明自己需要的协议，然后由基础设施提供实现。例如最佳实践目前可以由 YAML 文件管理，未来也可以实现数据库或远端服务版本。只要新实现满足同一个接口，应用层的业务编排不需要变化。

这也是依赖反转带来的核心价值：内部业务规则决定接口，外部技术实现依赖这些接口。

### 追问 7：这套架构最大的代价是什么？

代价是抽象、接口和配置类型明显增加，初期开发成本高于直接编写一个量化脚本。因此，它不适合一次性的单模型验证。

对于开源产品，模型、算法、格式和执行策略会长期并行增长，这些抽象成本能够被后续复用摊薄。实际演进中还需要持续控制三个风险：万能接口、循环依赖和 YAML 配置复杂度。

### 追问 8：如何证明它降低了开发门槛？

可以从改动边界而不是只从主观感受说明：

- 用户不再必须编写和维护完整 Python 量化脚本，可以通过 CLI 和 YAML 使用最佳实践。
- 新模型的差异主要收敛到 Model Adapter。
- 新算法的差异主要收敛到 Processor 和 IR。
- 新执行策略主要收敛到 Runner。
- 新权重格式主要收敛到 Format 和 Saver。
- 一键量化、自动调优和敏感层分析可以复用相同的模型、算法和执行能力。

如果面试官要求量化数据，应只使用自己实际统计和能够解释的数据，例如接入工时、重复代码量、跨模型算法复用数量或社区 PR 修改范围，不要临时编造比例。

### 追问 9：你个人在其中承担了什么工作？

可以结合实际经历回答：

> 我主要负责总体架构和关键边界设计，包括识别传统脚本的耦合问题、划分应用与领域、定义模型适配和基础设施接口、统一 YAML 量化语言，以及推动一键量化场景落地。具体算法和模型适配由不同成员协作完成，我重点保证新能力能够沿着明确扩展点接入，而不是继续修改和复制主流程。

回答个人贡献时，应区分“我主导设计”“我负责实现”和“我推动团队落地”，避免把团队全部代码描述成个人完成。

## 可进一步展开的技术设计

### 量化模式为什么抽象为 IR

量化模式是量化工具、量化算子和推理框架协作的共同锚点。例如 W8A8 静态量化一旦确定，其量化公式、参数集合和输入输出映射也基本确定。

IR 只描述这种形式化映射，不绑定具体 NPU 算子或推理框架。领域层可以通过高精度计算模拟低精度过程，验证量化精度；Format 层再负责把相同的量化知识转换成具体推理框架需要的权重格式。

### Processor 为什么适合表示算法

很多量化方案不是单一算法，而是多个步骤的组合，例如：

```text
加载模型
  -> 离群值抑制
  -> 权重/激活量化
  -> KV Cache 或 Attention 量化
  -> 保存权重
```

把每个步骤抽象为 Processor 后，Runner 可以统一调度，YAML 可以调整组合顺序，算法组件也可以复用于一键量化、敏感层分析和自动调优等应用。

### 为什么需要 Model Adapter

如果没有 Model Adapter，算法中会出现大量具体模型结构名称，例如某个模型的 `self_attn.q_proj`、另一个模型的融合 QKV 层或者 MoE 专家结构。这会导致算法代码同时承载数学机制和模型结构知识。

Model Adapter 将模型结构知识集中起来，使算法主要处理“Norm-Linear 对”“待量化 Linear”“Attention 输入输出”等领域概念。模型变化时优先修改适配器，算法变化时优先修改 Processor，从而降低交叉影响。

## 回答时容易踩的坑

### 不要只背四层目录

只说“我们有接口层、应用层、领域层和基础设施层”无法证明真正理解架构。需要进一步说明：

- 为什么这样划分。
- 哪些变化被隔离了。
- 依赖方向是什么。
- 新增模型或算法时改哪些地方。
- 这套设计付出了什么成本。

### 不要把它说成严格的战术 DDD

项目中没有必要强行寻找聚合根、领域事件或仓储模式。应明确说明这是 DDD 领域建模思想与分层、端口适配器、依赖反转、插件机制的结合。

### 不要只讲技术，不讲业务问题

架构的出发点是开源易用性和长期演进：用户不应理解所有算法实现，贡献者应能找到清晰的修改边界，模型和算法知识应当能够跨场景复用。

### 不要虚构量化收益

如果没有经过统计，不要直接说“接入效率提升 80%”或者“代码量减少 50%”。可以先描述可验证的结构性结果：从修改完整脚本变成新增适配器、Processor 或 YAML 配置。

## 面试现场的代码设计框架

如果面试官要求现场写一个大概的代码框架，不需要还原 msModelSlim 的完整实现。重点是通过少量伪代码证明以下几点：

1. 应用层只负责编排，不直接实现量化算法。
2. 领域层定义量化概念、Processor 和执行规则。
3. 外部模型、数据集和存储通过接口接入。
4. 新模型、新算法和新格式能够独立扩展。

### 1. 先画整体依赖关系

```text
┌───────────────────────────────────────────────┐
│ Interface: CLI / Python API                   │
│ 解析参数、装配依赖、调用 Application          │
└──────────────────────┬────────────────────────┘
                       │
┌──────────────────────▼────────────────────────┐
│ Application: QuantizationApplication          │
│ 创建模型适配器、选择最佳实践、调用量化服务    │
└───────────────┬──────────────────┬────────────┘
                │                  │ 依赖抽象接口
┌───────────────▼──────────────────▼────────────┐
│ Domain: QuantService / Runner / Processor / IR│
│ 量化模式、算法处理、模型调度、上下文           │
└───────────────▲──────────────────▲────────────┘
                │ 实现领域提出的接口             │
┌───────────────┴──────────────────┴────────────┐
│ Infrastructure: Model Adapter / YAML / Dataset│
│ Transformers、文件系统、数据库、评测服务      │
└───────────────────────────────────────────────┘
```

现场可以边画边解释：依赖方向不是简单地从上到下直接依赖具体类，而是内部层定义协议，外部实现协议；CLI 作为组合根，在最外层把具体实现装配起来。

### 2. 定义任务和量化方案

```python
from dataclasses import dataclass
from enum import Enum


class RunnerType(Enum):
    MODEL_WISE = "model_wise"
    LAYER_WISE = "layer_wise"
    DP_LAYER_WISE = "dp_layer_wise"


@dataclass
class QuantizeCommand:
    model_type: str
    model_path: str
    save_path: str
    quant_type: str
    device_ids: list[int]


@dataclass
class ProcessorSpec:
    # type 是领域语言，例如 smooth_quant、linear_quant、quarot
    type: str
    params: dict


@dataclass
class QuantizationPlan:
    # 对应经过校验的 YAML 量化方案
    api_version: str
    runner: RunnerType
    dataset_name: str
    processors: list[ProcessorSpec]
    output_format: str
```

这里要强调：`QuantizationPlan` 表达的是业务方案，而不是 CLI 参数的简单复制。它描述使用什么执行策略、哪些量化算法以及输出什么权重格式。

### 3. 定义内部需要的端口协议

```python
from typing import Any, Iterable, Protocol, runtime_checkable


class ModelPort(Protocol):
    """所有模型适配器都需要提供的最小能力。"""

    def get_model_type(self) -> str:
        ...

    def load_model(self, device: str) -> Any:
        ...


class PipelinePort(ModelPort, Protocol):
    """Runner 对模型调度能力的诉求。"""

    def prepare_dataset(self, raw_dataset: Any) -> list[Any]:
        ...

    def iter_layers(self, model: Any) -> Iterable[tuple[str, Any]]:
        ...

    def forward_layer(self, layer: Any, inputs: Any) -> Any:
        ...


@runtime_checkable
class SmoothQuantModelPort(Protocol):
    """SmoothQuant 单独提出的模型结构诉求。"""

    def get_norm_linear_pairs(self, model: Any) -> list[tuple[Any, Any]]:
        ...


class PracticeRepository(Protocol):
    """应用层对最佳实践存储的诉求。"""

    def find_best(self, model_type: str, quant_type: str) -> QuantizationPlan:
        ...


class DatasetLoaderPort(Protocol):
    def load(self, dataset_name: str) -> Any:
        ...


class ModelFactoryPort(Protocol):
    def create(self, model_type: str, model_path: str) -> PipelinePort:
        ...
```

这一段是面试中的重点。不要定义一个包含几十个方法的 `UniversalModelAdapter`。不同算法分别提出最小接口，模型适配器只实现当前量化方案需要的能力，这体现了接口隔离和依赖反转。

### 4. 领域层：量化 IR

```python
from abc import ABC, abstractmethod


class QuantIR(ABC):
    """对量化结构的形式化描述，不绑定具体硬件算子。"""

    @abstractmethod
    def fake_quant(self, tensor):
        """用高精度计算模拟低精度量化过程。"""
        ...


class W8A8StaticIR(QuantIR):
    def __init__(self, weight_scale, activation_scale):
        self.weight_scale = weight_scale
        self.activation_scale = activation_scale

    def fake_quant(self, tensor):
        integer = round(tensor / self.activation_scale)
        integer = clamp(integer, -128, 127)
        return integer * self.activation_scale
```

这里不需要现场展开真实量化公式。面试官主要观察是否理解：IR 描述的是“量化后的结构和数值映射”，Processor 负责计算参数并把 IR 应用到模型。

### 5. 领域层：Processor 抽象

```python
class QuantContext:
    """在多个 Processor 之间共享校准统计量和量化状态。"""

    def __init__(self):
        self.values: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get(self, key: str, default=None) -> Any:
        return self.values.get(key, default)


class Processor(ABC):
    @abstractmethod
    def process(
        self,
        model: Any,
        adapter: ModelPort,
        calibration_data: Any,
        context: QuantContext,
    ) -> None:
        ...


class SmoothQuantProcessor(Processor):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def process(self, model, adapter, calibration_data, context):
        if not isinstance(adapter, SmoothQuantModelPort):
            raise UnsupportedModel("model does not support SmoothQuant")

        pairs = adapter.get_norm_linear_pairs(model)
        for norm, linear in pairs:
            activation_stats = collect_activation(linear, calibration_data)
            scale = search_smooth_scale(
                norm=norm,
                linear=linear,
                activation_stats=activation_stats,
                alpha=self.alpha,
            )
            apply_smooth_scale(norm, linear, scale)

        context.put("smooth_quant.finished", True)


class LinearQuantProcessor(Processor):
    def __init__(self, weight_bits: int, activation_bits: int):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    def process(self, model, adapter, calibration_data, context):
        for name, module in find_quantizable_linears(model):
            qparams = calculate_qparams(
                module=module,
                calibration_data=calibration_data,
                weight_bits=self.weight_bits,
                activation_bits=self.activation_bits,
            )
            quant_ir = W8A8StaticIR(
                weight_scale=qparams.weight_scale,
                activation_scale=qparams.activation_scale,
            )
            replace_module_with_quant_ir(model, name, quant_ir)
```

这两个 Processor 展示了两类典型知识：

- SmoothQuant 依赖模型结构信息，因此通过 `SmoothQuantModelPort` 获取结构对。
- LinearQuant 主要依赖通用线性层和 IR，不需要知道当前模型是 Qwen 还是 DeepSeek。

### 6. Processor 工厂与注册机制

```python
PROCESSOR_REGISTRY: dict[str, type[Processor]] = {}


def register_processor(name: str):
    def decorator(processor_cls):
        PROCESSOR_REGISTRY[name] = processor_cls
        return processor_cls
    return decorator


register_processor("smooth_quant")(SmoothQuantProcessor)
register_processor("linear_quant")(LinearQuantProcessor)


class ProcessorFactory:
    def create(self, spec: ProcessorSpec) -> Processor:
        processor_cls = PROCESSOR_REGISTRY.get(spec.type)
        if processor_cls is None:
            raise UnsupportedProcessor(spec.type)
        return processor_cls(**spec.params)
```

注册机制解决的是开放扩展问题。新增算法时注册新的 Processor，不需要在主流程中不断增加 `if algorithm == ...` 分支。

### 7. 领域层：Runner 负责统一调度

```python
class Runner(ABC):
    def __init__(self, processor_factory: ProcessorFactory):
        self.processor_factory = processor_factory

    @abstractmethod
    def run(
        self,
        adapter: PipelinePort,
        plan: QuantizationPlan,
        raw_dataset: Any,
    ) -> Any:
        ...


class LayerWiseRunner(Runner):
    def run(self, adapter, plan, raw_dataset):
        model = adapter.load_model(device="npu")
        calibration_data = adapter.prepare_dataset(raw_dataset)
        context = QuantContext()

        processors = [
            self.processor_factory.create(spec)
            for spec in plan.processors
        ]

        for layer_name, layer in adapter.iter_layers(model):
            layer_inputs = build_layer_inputs(layer_name, calibration_data, context)

            for processor in processors:
                processor.process(
                    model=layer,
                    adapter=adapter,
                    calibration_data=layer_inputs,
                    context=context,
                )

            layer_outputs = adapter.forward_layer(layer, layer_inputs)
            context.put(f"output.{layer_name}", layer_outputs)
            release_layer_memory(layer)

        return model


class DistributedLayerWiseRunner(LayerWiseRunner):
    def run(self, adapter, plan, raw_dataset):
        # 伪代码：切分层或算法子任务，由多个 rank 并行执行并同步结果。
        tasks = partition_tasks(adapter, plan, world_size=get_world_size())
        local_result = execute_local_tasks(tasks.local, adapter, raw_dataset)
        return merge_results(all_gather(local_result))
```

Runner 负责“如何执行”，Processor 负责“执行什么算法”。因此单卡逐层切换到 DP 多卡时，算法实现通常不需要重写。

### 8. 领域服务：根据方案组织执行

```python
class QuantizationService:
    def __init__(
        self,
        dataset_loader: DatasetLoaderPort,
        runner_factory,
        saver_factory,
    ):
        self.dataset_loader = dataset_loader
        self.runner_factory = runner_factory
        self.saver_factory = saver_factory

    def quantize(
        self,
        plan: QuantizationPlan,
        adapter: PipelinePort,
        save_path: str,
    ) -> None:
        dataset = self.dataset_loader.load(plan.dataset_name)
        runner = self.runner_factory.create(plan.runner)

        quantized_model = runner.run(
            adapter=adapter,
            plan=plan,
            raw_dataset=dataset,
        )

        saver = self.saver_factory.create(plan.output_format)
        saver.save(quantized_model, save_path)
```

如果需要支持多个版本的量化服务，还可以增加一个代理，根据 `api_version` 路由到不同服务：

```python
class QuantizationServiceProxy:
    def __init__(self, service_plugins):
        self.service_plugins = service_plugins

    def quantize(self, plan, adapter, save_path):
        service = self.service_plugins.create(plan.api_version)
        service.quantize(plan, adapter, save_path)
```

### 9. 应用层：编排一键量化用例

```python
class QuantizationApplication:
    def __init__(
        self,
        model_factory: ModelFactoryPort,
        practice_repository: PracticeRepository,
        quantization_service: QuantizationService,
    ):
        self.model_factory = model_factory
        self.practice_repository = practice_repository
        self.quantization_service = quantization_service

    def execute(self, command: QuantizeCommand) -> None:
        # 1. 将外部模型转换成内部可识别的模型适配器。
        adapter = self.model_factory.create(
            model_type=command.model_type,
            model_path=command.model_path,
        )

        # 2. 根据模型和用户目标选择经过验证的量化方案。
        plan = self.practice_repository.find_best(
            model_type=adapter.get_model_type(),
            quant_type=command.quant_type,
        )

        # 3. 委托领域服务执行量化，应用层不实现具体算法。
        self.quantization_service.quantize(
            plan=plan,
            adapter=adapter,
            save_path=command.save_path,
        )
```

这段代码最能体现应用层的职责：它知道业务步骤，但不知道 SmoothQuant 的公式、Qwen 的层结构或者 YAML 的读取方式。

### 10. 基础设施层：模型适配器

```python
class QwenModelAdapter(PipelinePort, SmoothQuantModelPort):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def get_model_type(self) -> str:
        return "qwen"

    def load_model(self, device: str):
        # 具体依赖 Transformers，属于外部实现细节。
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        ).to(device)

    def prepare_dataset(self, raw_dataset):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return [tokenizer(item["text"]) for item in raw_dataset]

    def iter_layers(self, model):
        for index, layer in enumerate(model.model.layers):
            yield f"model.layers.{index}", layer

    def forward_layer(self, layer, inputs):
        return layer(**inputs)

    def get_norm_linear_pairs(self, model):
        pairs = []
        for layer in model.model.layers:
            pairs.extend([
                (layer.input_layernorm, layer.self_attn.q_proj),
                (layer.input_layernorm, layer.self_attn.k_proj),
                (layer.input_layernorm, layer.self_attn.v_proj),
                (layer.post_attention_layernorm, layer.mlp.up_proj),
            ])
        return pairs
```

Qwen 的具体字段只出现在适配器中，不进入 SmoothQuant 的通用实现。DeepSeek 可以用不同的结构实现相同协议。

### 11. 基础设施层：最佳实践仓储

```python
class YamlPracticeRepository(PracticeRepository):
    def __init__(self, config_directory: str):
        self.config_directory = config_directory

    def find_best(self, model_type: str, quant_type: str) -> QuantizationPlan:
        candidates = load_yaml_candidates(
            directory=self.config_directory,
            model_type=model_type,
        )
        matched = [
            item for item in candidates
            if item.matches(model_type=model_type, quant_type=quant_type)
        ]
        best = max(matched, key=lambda item: item.score)
        return validate_and_convert_to_plan(best)
```

如果将来改为数据库，只需要新增另一个实现：

```python
class DatabasePracticeRepository(PracticeRepository):
    def find_best(self, model_type, quant_type):
        row = database.query_best_practice(model_type, quant_type)
        return convert_row_to_plan(row)
```

应用层和领域层不需要随存储方案变化而修改。

### 12. 接口层：CLI 作为组合根

```python
def main(args):
    command = QuantizeCommand(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        quant_type=args.quant_type,
        device_ids=parse_device_ids(args.device),
    )

    # 所有具体基础设施都在系统最外层装配。
    application = QuantizationApplication(
        model_factory=PluginModelFactory(),
        practice_repository=YamlPracticeRepository("./practices"),
        quantization_service=QuantizationService(
            dataset_loader=FileDatasetLoader("./datasets"),
            runner_factory=RunnerFactory(ProcessorFactory()),
            saver_factory=SaverFactory(),
        ),
    )

    application.execute(command)
```

CLI 可以直接依赖具体基础设施，因为它是组合根；领域层和应用层不应主动创建 `YamlPracticeRepository`、`AutoModel` 等具体对象。

## 两个扩展示例

### 示例一：新增一个 DeepSeek 模型

```python
class DeepSeekModelAdapter(PipelinePort, SmoothQuantModelPort):
    def load_model(self, device):
        return load_deepseek(self.model_path, device)

    def iter_layers(self, model):
        # DeepSeek 的层结构和 Qwen 不同，差异封装在这里。
        yield from visit_deepseek_layers(model)

    def get_norm_linear_pairs(self, model):
        # MoE、共享专家等差异同样留在适配器中。
        return build_deepseek_norm_linear_pairs(model)


model_plugins.register("deepseek", DeepSeekModelAdapter)
```

扩展过程中不需要修改 `SmoothQuantProcessor`、`LinearQuantProcessor` 和 `QuantizationApplication`。

### 示例二：新增 GPTQ 算法

```python
@register_processor("gptq")
class GPTQProcessor(Processor):
    def __init__(self, bits: int, group_size: int):
        self.bits = bits
        self.group_size = group_size

    def process(self, model, adapter, calibration_data, context):
        for name, linear in find_quantizable_linears(model):
            hessian = collect_hessian(linear, calibration_data)
            qweight, qparams = gptq_quantize(
                weight=linear.weight,
                hessian=hessian,
                bits=self.bits,
                group_size=self.group_size,
            )
            replace_with_gptq_ir(model, name, qweight, qparams)
```

YAML 中增加相应 Processor 即可启用：

```yaml
api_version: modelslim_v1
runner: layer_wise
dataset: calibration_set
processors:
  - type: smooth_quant
    params:
      alpha: 0.5
  - type: gptq
    params:
      bits: 4
      group_size: 128
output_format: ascend_v1
```

## 现场书写顺序

如果只有 15～20 分钟，建议按照下面的顺序写，不要一开始陷入量化公式：

1. 画四层图和依赖箭头。
2. 写 `QuantizationPlan`，说明统一领域语言和配置驱动。
3. 写 `PipelinePort`、`PracticeRepository`、`DatasetLoaderPort`。
4. 写 `Processor` 和一个 `LinearQuantProcessor` 示例。
5. 写 `Runner`，展示 Processor 的组合执行。
6. 写 `QuantizationApplication`，展示应用编排。
7. 写 `QwenModelAdapter`，展示外部模型适配。
8. 最后说明新增模型、算法和格式分别改哪里。

可以主动告诉面试官：

> 为了在有限时间内突出架构，我会省略张量公式、异常处理和分布式通信细节，重点展示领域边界、依赖方向和扩展方式。

## 代码中体现的设计模式

| 设计问题                   | 使用的思想或模式                         | 代码体现                                       |
| -------------------------- | ---------------------------------------- | ---------------------------------------------- |
| 隔离具体模型差异           | Adapter                                  | `QwenModelAdapter`、`DeepSeekModelAdapter` |
| 避免依赖具体基础设施       | Dependency Inversion、Ports and Adapters | `PracticeRepository`、`DatasetLoaderPort`  |
| 切换模型级、逐层、多卡执行 | Strategy                                 | 不同`Runner` 实现                            |
| 按配置创建算法组件         | Factory、Registry                        | `ProcessorFactory`、`PROCESSOR_REGISTRY`   |
| 组合多个量化步骤           | Pipeline、Chain of Responsibility        | `Runner` 顺序调度 Processor                  |
| 按版本选择量化后端         | Proxy、Plugin                            | `QuantizationServiceProxy`                   |
| 共享校准统计和中间状态     | Context                                  | `QuantContext`                               |

面试时不需要刻意罗列模式名称。优先说明问题和权衡，面试官追问时再指出对应模式。

## 最后收口

> 这套架构的核心成果，不只是把目录拆成了四层，而是把“模型 × 算法 × 执行策略 × 权重格式”的组合问题，从复制脚本变成了组件组合问题。用户获得统一入口，开发者获得明确扩展边界，量化知识也能够持续沉淀和复用。
