# msModelSlim 逐层量化与 DP 并行多卡量化面试话术

## 核心主线

> 全模型量化首先遇到的是显存容量问题，所以设计逐层量化；逐层量化解决了“能不能跑”，但串行耗时较长，因此进一步设计 DP 多卡量化解决“跑得够不够快”。

## 90 秒面试话术

随着大模型参数量增长，传统全模型量化需要把模型权重、校准激活和算法临时张量同时放到设备上，单卡很容易 OOM。针对这个问题，我提出了逐层量化方案。

这里的逐层量化是一种执行和内存调度方式，不特指 GPTQ 的逐层算法。模型主体先保留在 CPU 或卸载设备上，模型适配器把完整前向过程拆成 Decoder Layer 级的生成式流水线。调度器每次只把当前层加载到 NPU，依次完成校准前向、离群值抑制、量化参数计算、结构替换和权重保存，然后立即卸载当前层并释放显存，再处理下一层。

因此，设备侧显存复杂度从与整个模型参数量相关，下降为与最大单层权重、当前层激活和算法临时空间相关，使几十 B 甚至更大模型可以只用一张量化设备完成处理。

逐层量化解决了容量问题，但所有层和校准数据仍然串行处理，耗时比较长。因此我们进一步设计了 DP 并行多卡量化框架。主进程根据设备列表使用 `spawn` 拉起一进程一卡的 Worker，每个进程绑定独立 NPU，并通过 HCCL 建立进程组。校准数据由 `DistributedSampler` 切分，各 Rank 对相同层处理不同的数据分片，然后通过 `all-reduce` 或 `all-gather` 汇总激活统计量，保证最终计算出的量化参数与单卡使用完整校准集语义一致。

我主要参与了分布式执行框架设计，并负责进程生命周期、设备与 Rank 映射、端口管理、异常清理和跨卡同步。实际测试中，多卡相对单卡逐层量化获得了大约 `0.5 × 卡数` 的有效加速，例如四卡约两倍。没有达到理想线性，主要是模型加载卸载、部分重复算法、集合通信和权重保存仍然存在串行或固定开销。

## 两层优化分别解决什么问题

| 方案        | 主要问题                       | 核心方法                                       | 主要收益                                   |
| ----------- | ------------------------------ | ---------------------------------------------- | ------------------------------------------ |
| 逐层量化    | 整个大模型无法同时放入单卡显存 | 一次只加载并处理当前 Decoder Layer             | 降低设备侧峰值显存，使超大模型可以单卡量化 |
| DP 多卡量化 | 单卡逐层量化耗时较长           | 多进程一进程一卡，切分校准数据并同步全局统计量 | 提高校准和量化吞吐，缩短整体耗时           |

需要先向面试官说明：这里的“逐层量化”主要指框架的执行调度和内存调度粒度，不是只指 GPTQ 等算法在数学意义上的逐层优化。

## 逐层量化的技术流程

```text
模型权重保留在 CPU/卸载设备
              │
              ▼
处理 Decoder Layer 0
  加载当前层到 NPU
  → 校准前向
  → 收集激活统计量
  → 执行 Smooth/Quant 等 Processor
  → 生成量化参数和量化结构
  → 保存当前层权重
  → 卸载并释放显存
              │
              ▼
将当前层输出作为下一层校准输入
              │
              ▼
处理 Decoder Layer 1 ... Layer N
```

### 逐层调度如何落地

模型适配器提供两个相互对齐的生成器：

- `generate_model_visit`：在无校准数据算法中，按照模型结构逐层访问模块。
- `generate_model_forward`：在有校准数据算法中，按照真实前向顺序逐层产生模块和输入。

每次生成一个 `ProcessRequest`，其中包含当前层名称、模块对象和前向输入。Runner 将同一层交给 Load、Smooth、Quant、Save、Offload 等 Processor 依次处理。

```python
def layer_wise_quantize(model, adapter, processors, calibration_data):
    layer_generator = adapter.generate_model_forward(model, calibration_data)
    previous_output = None

    while True:
        try:
            request = layer_generator.send(previous_output)
        except StopIteration:
            break

        layer = request.module

        # 当前时刻只有该层进入 NPU。
        layer.to("npu:0")

        for processor in processors:
            processor.preprocess(request)
            processor.process(request)
            processor.postprocess(request)

        previous_output = run_current_layer(layer, request.args, request.kwargs)

        save_quantized_layer(request.name, layer)
        layer.to("meta")
        release_device_cache()
```

实际实现中，加载和卸载同样被抽象成 Processor，Runner 会在处理器列表前后自动插入：

```python
processor_list.insert(
    0,
    LoadProcessorConfig(
        device=current_npu,
        mode="load",
        post_offload=True,
    ),
)

processor_list.append(
    LoadProcessorConfig(
        device="meta",
        mode="offload",
        cleanup=True,
    )
)
```

### 显存收益

可以用一个简化公式说明：

```text
全模型量化峰值显存
≈ 全模型权重 + 全模型运行激活 + 算法临时空间

逐层量化峰值显存
≈ 最大单层权重 + 当前层校准激活 + 当前算法临时空间
```

如果总模型权重为 `P`，最大单层权重为 `Pmax`，那么设备侧权重占用由近似 `O(P)` 下降为 `O(Pmax)`。

这并不意味着整体资源完全与模型大小无关，以下内容仍可能成为瓶颈：

- CPU 侧模型权重和模型初始化内存。
- 校准数据及中间激活缓存。
- 最大单层的权重规模。
- GPTQ Hessian、搜索类算法等额外临时空间。
- CPU 与 NPU 之间的逐层搬运耗时。

### 对应代码

- 当前层加载和卸载策略：[layer_wise_runner.py](../../msmodelslim/msmodelslim/core/runner/layer_wise_runner.py#L49)
- 模块加载、卸载和显存清理：[load.py](../../msmodelslim/msmodelslim/processor/memory/load.py#L72)
- 模型逐层访问协议：[pipeline_interface.py](../../msmodelslim/msmodelslim/core/runner/pipeline_interface.py#L67)
- 多个 Processor 的生成式交错调度：[generated_runner.py](../../msmodelslim/msmodelslim/core/runner/generated_runner.py#L203)
- Qwen3 模型对逐层访问协议的实现：[model_adapter.py](../../msmodelslim/msmodelslim/model/qwen3/model_adapter.py#L109)

## DP 多卡量化的技术流程

```text
主进程
  │
  ├─ 解析 device_indices，确定 world_size
  ├─ 查找通信端口
  ├─ 创建跨进程共享上下文
  └─ mp.spawn 拉起 N 个 Worker
          │
          ├─ Rank 0 → NPU 0
          ├─ Rank 1 → NPU 1
          └─ Rank N → NPU N
                    │
                    ▼
          HCCL 初始化进程组
                    │
                    ▼
       DistributedSampler 切分校准数据
                    │
                    ▼
       各 Rank 对相同层处理不同数据分片
                    │
                    ▼
        聚合统计量并计算一致的量化参数
                    │
                    ▼
       保存分片/汇总权重，销毁进程组
```

这里的 DP 不是训练中的梯度 DDP，而是量化校准的数据并行：

- 每个 Rank 处理不同的校准样本。
- 不需要反向传播和梯度同步。
- 需要同步的是激活统计量、量化参数和模型修改结果。
- 各 Rank 最终模型状态必须与完整校准集的单卡结果等价。

### 主进程管理

```python
def run_distributed(device_indices, model, calibration_data):
    world_size = len(device_indices)

    # NPU/CUDA 运行时不适合继承父进程上下文，因此使用 spawn。
    multiprocessing.set_start_method("spawn", force=True)

    master_port = find_free_port()
    os.environ["MASTER_PORT"] = str(master_port)

    try:
        multiprocessing.spawn(
            distributed_worker,
            args=(
                world_size,
                device_indices,
                model,
                calibration_data,
                master_port,
            ),
            nprocs=world_size,
            join=True,
        )
    finally:
        # 由当前流程设置的环境变量由当前流程负责清理。
        os.environ.pop("MASTER_PORT", None)
```

这里体现了一个重要的工程原则：端口、环境变量、共享队列等资源都要有明确的所有者，避免一次量化污染后续多阶段量化任务。

### Worker 初始化

```python
def distributed_worker(
    rank,
    world_size,
    device_indices,
    model,
    calibration_data,
    master_port,
):
    device_index = device_indices[rank]

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)

        torch.npu.set_device(f"npu:{device_index}")

        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
        )

        check_all_processors_support_distributed()
        run_layer_wise_quantization(model, calibration_data)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
```

`rank` 是通信进程组编号，`device_index` 是实际设备编号，两者不能默认永远相同。例如用户可能指定 `npu:2,4,6,7`，此时 Rank 0 应绑定 NPU 2。

### 校准数据切分

进程组初始化后，DataLoader 使用 `DistributedSampler` 对校准集进行切分：

```python
sampler = DistributedSampler(
    calibration_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
)

local_loader = DataLoader(
    calibration_dataset,
    sampler=sampler,
)
```

例如 128 条校准数据在四卡场景下，每张卡原则上处理约 32 条，然后同步全局统计量。

### 对应代码

- Worker 初始化和执行：[dp_layer_wise_runner.py](../../msmodelslim/msmodelslim/core/runner/dp_layer_wise_runner.py#L99)
- 多进程启动和资源清理：[dp_layer_wise_runner.py](../../msmodelslim/msmodelslim/core/runner/dp_layer_wise_runner.py#L185)
- Rank、设备和 HCCL 进程组初始化：[dist_setup.py](../../msmodelslim/msmodelslim/utils/distributed/dist_setup.py#L73)
- 数据集分片：[generated_runner.py](../../msmodelslim/msmodelslim/core/runner/generated_runner.py#L271)
- 用户侧 DP Runner 配置：[usage.md](../../msmodelslim/docs/zh/user_guide/feature_guide/quick_quantization_v1/usage.md#L176)

## 跨卡同步设计

跨卡同步的目标不是“所有东西都同步”，而是先判断量化数学语义要求哪些状态保持全局一致，再选择合适的集合通信。

### 1. 可规约统计量：all-reduce

激活最小值、最大值、求和和均值等统计量可以直接规约：

```python
# 每张卡根据自己的校准数据分片计算局部统计量。
local_min = activation.min()
local_max = activation.max()

# 规约完成后，各 Rank 都得到完整校准集对应的全局统计量。
dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

qparams = calculate_qparams(local_min, local_max)
```

对应关系为：

```text
global_min = min(local_min_rank0, local_min_rank1, ...)
global_max = max(local_max_rank0, local_max_rank1, ...)
global_sum = sum(local_sum_rank0, local_sum_rank1, ...)
global_mean = global_sum / global_sample_count
```

这种同步的数据量较小，是优先选择。

### 2. 需要保留样本分布：all-gather

部分算法需要保留原始激活数据做网格搜索或误差比较，单纯的 min/max 不能满足要求，此时需要收集各 Rank 的激活：

```python
local_activations = collect_local_activations()
global_activations = all_gather_variable_shape(local_activations)
scale = search_best_scale(global_activations)
```

`all-gather` 的通信和内存开销更大，尤其是各 Rank 张量形状不一致时，还需要先同步 Shape，再对齐缓冲区并收集数据。因此，能转换成充分统计量时，优先采用 `all-reduce`。

### 3. 模型修改结果：broadcast

如果算法子任务只在某一个 Rank 执行，例如某个 Rank 完成权重量化或 IR 替换，则需要把修改结果广播到其他 Rank：

```python
executor_rank = task.owner

if rank == executor_rank:
    run_algorithm_task(module)

broadcast_module_parameters(module, src=executor_rank)
broadcast_module_buffers(module, src=executor_rank)
```

广播完成后，各 Rank 的模型状态应等价于都执行过该任务。

### 4. 只同步共享模块

普通 Dense 模型中，各 Rank 通常具有相同模块；但在 EP 或部分 MoE 场景中，某些 Expert 只存在于部分 Rank。

框架通过 `DistHelper` 汇总各 Rank 的模块名称，并分类为：

- `shared module`：所有 Rank 都存在，可以参与同步。
- `local-only module`：只在部分 Rank 存在，不能直接进入全局集合通信。

```python
class DistHelper:
    def __init__(self, model):
        local_module_names = set(name for name, _ in model.named_modules())

        all_rank_names = all_gather_object(local_module_names)
        self.shared_modules = set.intersection(*all_rank_names)
        self.local_only_modules = local_module_names - self.shared_modules

    def is_shared(self, module_name):
        return module_name in self.shared_modules
```

如果某个 Rank 对本地 Expert 执行 `all-reduce`，而其他 Rank 没有进入对应通信点，程序就可能永久阻塞。因此，同步前必须先判断模块作用域。

### 同步工具代码

- 共享模块和本地模块分类：[dist_helper.py](../../msmodelslim/msmodelslim/utils/distributed/dist_helper.py#L65)
- min/max/sum/mean 等统计量规约：[dist_ops.py](../../msmodelslim/msmodelslim/utils/distributed/dist_ops.py#L52)
- 张量和张量列表收集：[dist_ops.py](../../msmodelslim/msmodelslim/utils/distributed/dist_ops.py#L101)
- 模型参数和 Buffer 广播：[sync.py](../../msmodelslim/msmodelslim/utils/distributed/task_scheduler/sync.py#L16)

## 多卡量化的正确性要求

### 基本目标

多卡量化不是只要求程序能够运行，而是要求：

```text
多卡数据分片 + 跨卡同步后的量化结果
≈ 单卡使用完整校准集得到的量化结果
```

严格逐元素完全一致可能受到浮点规约顺序影响，但量化参数、权重和最终精度应在可接受误差范围内等价。

### 验证方法

1. 固定随机种子、校准数据、数据顺序和量化配置。
2. 分别执行单卡逐层和多卡逐层量化。
3. 比较每层的激活 min/max、Scale、Zero Point 等量化参数。
4. 比较保存后的量化权重和量化配置。
5. 对量化模型执行统一精度评测。
6. 通过日志确认各 Rank 数据量和通信顺序一致。

### Processor 必须显式声明支持

不同算法的同步语义不同，不能认为所有 Processor 天然支持分布式。因此 Processor 基类默认返回不支持，多卡 Runner 启动前检查全部组件：

```python
class Processor:
    def support_distributed(self) -> bool:
        return False


class DistributedMinMaxProcessor(Processor):
    def support_distributed(self) -> bool:
        return True
```

如果量化方案中包含不支持多卡的算法，应在真正执行前快速失败，而不是运行到一半后产生错误权重。

## 为什么不是理想线性加速

可以使用 Amdahl 定律解释：

```text
S(N) ≈ 1 / [(1 - P) + P/N + C(N)]
```

其中：

- `P`：能够通过数据切分并行的校准计算比例。
- `1-P`：模型初始化、逐层加载卸载、部分算法和保存等串行或重复开销。
- `C(N)`：`all-reduce`、`all-gather`、进程启动和同步等待等通信开销。

主要损耗来源包括：

1. 每个 Rank 都需要初始化模型并运行逐层框架。
2. CPU 到 NPU 的逐层权重搬运不能被完全隐藏。
3. 部分权重量化属于 Data-free 任务，简单 DP 下各 Rank 会重复执行。
4. 激活统计量需要跨卡集合通信。
5. 不同层和不同算法子任务的耗时不均匀，会产生尾部等待。
6. 最终权重保存和元数据合并存在串行部分。
7. 校准数据规模较小时，多进程和通信开销可能超过并行收益。

因此，简历中的性能数据建议表述为：

> 在测试卡数范围内，有效加速比约为 `0.5N`，例如四卡相对单卡约两倍。这是具体模型、算法和校准规模下的实测经验值，而不是理论线性保证。

不要说“时间减少了 `0.5N` 倍”，容易产生歧义。使用“加速比约为 `0.5N`”更准确。

## 分布式任务调度的进一步优化

基础 DP 主要加速有校准数据的前向和统计过程。对于只依赖权重的 Data-free 子任务，如果各 Rank 都执行一遍，就会产生重复计算。

可以进一步将 Processor 内部可拆分的子图或权重量化任务放入共享任务队列：

```text
优化前
  Rank 0: T1 → T2 → T3 → T4
  Rank 1: T1 → T2 → T3 → T4

优化后
  Rank 0: T1 ─────→ T3
  Rank 1:   T2 ─────→ T4
                    │
                    ▼
              同步模型状态
```

伪代码如下：

```python
with DistributedTaskScheduler(model) as scheduler:
    for layer_name in quantizable_layers:
        scheduler.submit(
            fn=quantize_one_layer,
            args=(layer_name,),
            dependencies=[layer_name],
            parallel=True,
        )

    scheduler.run()
```

每个任务通常只由一个 Rank 执行，完成后将对应模块参数、Buffer 或 IR 状态同步给其他 Rank。这样能减少简单 DP 中各 Rank 重复执行权重量化任务的问题。

但是需要遵守一个硬约束：

> 一个只由单 Rank 执行的任务内部不能再调用要求所有 Rank 同时参与的 `all-reduce` 或 `all-gather`，否则其他 Rank 没有进入相同通信点，会造成死锁。

## 常见面试追问

### 追问 1：为什么不直接用 TP 把模型切到多卡上？

TP 主要解决单次模型前向的模型容量和计算问题，但需要模型结构相关的切分规则与通信逻辑，量化算法也需要理解分片权重。我们的第一目标是让不同模型共用统一量化框架，因此先使用逐层 Offload 解决容量，再使用 DP 切分校准数据，模型适配成本更低。

它的代价是各 Rank 可能持有完整的 CPU 侧逻辑模型，主机内存开销会增加。对于极大模型，可以进一步采用共享内存、内存映射、懒加载或 Model-free 等方案优化。

### 追问 2：DP 多卡量化和训练 DDP 有什么区别？

训练 DDP 的主要目标是并行计算梯度，并通过 `all-reduce` 保持各 Rank 参数更新一致。量化 DP 没有反向传播，主要并行校准前向，并同步激活统计量、量化参数和模型结构修改结果。

### 追问 3：怎么保证多卡和单卡精度一致？

核心是保证统计语义等价：

- 校准集被完整切分且不遗漏关键样本。
- min/max/sum 等统计量使用正确的规约算子。
- 需要完整分布的数据使用 gather。
- 所有 Rank 按一致顺序处理共享模块并进入集合通信。
- 最终对比量化参数、量化权重和精度评测结果。

### 追问 4：为什么使用 spawn，而不是 fork？

NPU/CUDA 运行时、通信上下文和后台线程状态通常不适合被子进程直接继承。`spawn` 会为每个 Worker 创建独立解释器和设备上下文，隔离性更好。

代价是启动成本更高，并且传入 Worker 的对象必须支持序列化，因此被装饰函数、模型适配器和上下文对象都要注意 Pickle 兼容性。

### 追问 5：如何避免分布式死锁？

- 所有 Rank 必须以相同顺序进入集合通信。
- 只有各 Rank 都存在的共享模块才能同步。
- 单 Rank 执行的子任务内部不能包含全 Rank collective。
- 启动前检查全部 Processor 是否支持分布式。
- 不在不同 Rank 上执行不一致的条件分支。
- Worker 异常时及时退出，并在 `finally` 中销毁进程组。
- 对端口、环境变量和共享队列明确所有权并负责清理。

### 追问 6：逐层量化一定不会 OOM 吗？

不一定。它主要去掉了全模型权重同时驻留 NPU 的压力，但以下场景仍可能 OOM：

- 单层本身很大，例如超大 MoE Expert 或融合矩阵。
- 校准 Batch 或序列长度过大。
- 算法需要保存大量激活或 Hessian 矩阵。
- 多个 Processor 的中间状态没有及时释放。
- Offload 后仍有 Hook、Tensor 引用或缓存持有设备内存。

因此还需要控制校准规模、及时释放引用、清理缓存，并针对算法设计分块计算。

### 追问 7：DP 多卡还有什么优化空间？

- 使用双缓冲，让下一层 H2D 搬运与当前层计算重叠。
- 根据历史耗时进行动态任务负载均衡。
- 尽量把原始激活 gather 转换成统计量 reduce。
- 对小统计量进行通信分桶，减少调用次数。
- 将 Data-free 算法拆成跨 Rank 子任务，避免重复计算。
- 异步保存权重，降低保存对主计算链路的阻塞。
- 使用内存映射或懒加载降低每个 Rank 的 CPU 内存占用。
- 对多阶段量化复用已初始化的进程池和模型状态。

### 追问 8：逐层输出如何传给下一层？

模型适配器的前向生成器会在当前层暂停，Runner 执行当前层后，将输出写入 `DataUnit`，下一轮再通过 `generator.send(output)` 传回生成器，继续产生下一层请求。

因此它既保留了真实模型前向的数据依赖，又允许框架在层与层之间插入量化、保存和 Offload 操作。

### 追问 9：算法如何接入 DP 多卡框架？

通常需要四步：

1. `support_distributed()` 显式返回 `True`。
2. 明确算法中哪些状态需要保持全局一致。
3. 根据状态类型选择 `all-reduce`、`all-gather` 或 `broadcast`。
4. 使用 `DistHelper` 判断模块是否为所有 Rank 共享，避免同步局部模块。

如果算法存在多个可以独立执行的子任务，还可以接入分布式任务调度器进一步减少重复计算。

## 个人贡献表述

结合简历，可以这样回答：

> 我提出了逐层量化的整体方案，核心是把模型前向拆成 Decoder Layer 级流水线，通过当前层按需加载、处理完成后及时保存和卸载，解决大模型单卡量化的显存问题。在此基础上，我参与了 DP 多卡量化执行框架设计，主要负责多进程启动与回收、Rank 和设备映射、动态端口管理、异常清理以及跨卡统计量同步。最终既保留了单卡逐层的低显存特性，又通过多卡切分校准计算获得了约 `0.5 × 卡数` 的有效加速。

如果需要区分个人工作和团队工作，可以进一步说：

> 逐层量化方案由我提出并推动落地；DP 多卡框架是团队协作完成，我重点参与总体设计，并负责进程管理与跨卡同步相关部分。不同量化算法的分布式适配由算法负责人协同完成。

## 回答时容易踩的坑

### 不要把逐层量化说成“每一层独立量化，互不相关”

真实前向中，上一层输出仍然是下一层输入。逐层调度只是限制设备同时驻留的模型范围，并没有切断模型层之间的数据依赖。

### 不要说 DP 多卡不需要模型副本

基本 DP 语义下，各 Rank 通常具有相同的逻辑模型状态，只是校准数据不同。逐层 Offload 降低的是每张设备的 NPU 显存，不一定降低所有 Rank 的 CPU 内存总量。

### 不要把所有同步都描述成 all-reduce

- min/max/sum 等统计量适合 `all-reduce`。
- 原始激活或变长张量可能需要 `all-gather`。
- 单 Rank 产生的模型状态适合 `broadcast`。
- EP 局部模块可能不应该同步。

### 不要把 `0.5N` 描述成理论保证

它是特定模型、算法、校准集和卡数范围内的实测有效加速比。面试时应主动解释非并行部分和通信成本。

## 最后收口

> 逐层量化解决的是资源上限，让超大模型能够在单卡上完成量化；DP 多卡量化解决的是吞吐和耗时，并通过严格的统计量同步保证多卡结果与单卡语义一致。
