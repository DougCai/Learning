# Linear Quant 端到端流程梳理

本文基于 msmodelslim 代码，梳理 `linear_quant` 从命令行启动、加载权重、前向校准、替换模块、计算量化参数、deploy 到保存导出的完整流程。

## 1. 总览

`linear_quant` 不是一个单独的数学量化算法，而是线性层量化的统一处理器。它负责扫描模型里的 `torch.nn.Linear`，按配置把命中的线性层替换成 `LinearQuantizer`，在校准前向中收集激活统计并完成权重量化，最后把 `LinearQuantizer` deploy 成可保存的 FakeQuant Linear IR。

核心路径：

| 阶段 | 主要代码 | 作用 |
|---|---|---|
| CLI 入口 | `msmodelslim/cli/__main__.py` | 解析 `msmodelslim quant ...` 参数 |
| 量化应用 | `msmodelslim/app/naive_quantization/application.py` | 校验参数、创建模型 adapter、选择最佳实践配置 |
| 量化服务 | `msmodelslim/core/quant_service/modelslim_v1/quant_service.py` | 准备 dataset / runner / processor / saver |
| 调度器 | `msmodelslim/core/runner/generated_runner.py`、`layer_wise_runner.py` | 按模型访问顺序调度 processor |
| Linear Processor | `msmodelslim/processor/quant/linear.py` | 查找并替换 `nn.Linear` |
| Linear Quantizer | `msmodelslim/core/quantizer/linear.py` | 调用激活/权重量化器，执行 fake quant 前向 |
| 底层量化器 | `msmodelslim/core/quantizer/impl/*.py` | `minmax`、`histogram`、`ssz`、`gptq` 等 |
| FakeQuant IR | `msmodelslim/ir/*.py` | deploy 后的部署态量化模块 |
| 保存器 | `msmodelslim/core/quant_service/modelslim_v1/save/ascendv1.py` | 写 `safetensors` 和 `quant_model_description.json` |

## 2. 从命令行开始

用户一般从 CLI 进入：

```bash
msmodelslim quant \
  --model_type <model_type> \
  --model_path <origin_model_dir> \
  --save_path <quant_model_dir> \
  --device npu \
  --quant_type <quant_type>
```

入口在 `msmodelslim/cli/__main__.py`：

1. 创建 `argparse` 子命令 `quant`。
2. 解析 `model_type`、`model_path`、`save_path`、`device`、`quant_type`、`config_path` 等参数。
3. 当 `args.command == "quant"` 时，调用 `msmodelslim.cli.naive_quantization.__main__.main(args)`。

`naive_quantization.__main__.py` 继续完成基础组件组装：

1. 创建 `YamlPracticeManager`，用于从 `lab_practice` 或自定义配置目录查找量化 recipe。
2. 创建 `FileDatasetLoader` / `VLMDatasetLoader`，用于读取校准数据。
3. 创建 `QuantServiceProxy`，它会根据配置选择实际 backend，例如 `modelslim_v1`。
4. 创建 `PluginModelFactory`，用于根据 `model_type` 和 `model_path` 构造模型 adapter。
5. 调用 `NaiveQuantizationApplication.quant(...)`。

## 3. 选择配置与创建模型 Adapter

`NaiveQuantizationApplication.quant()` 先做参数校验和路径规范化，然后进入 `_quant()`。

`_quant()` 的关键动作：

1. 调用 `model_factory.create(model_type, model_path, trust_remote_code)` 创建模型 adapter。
2. 调用 `get_best_practice()` 根据 `model_type`、`quant_type`、`config_path`、`tag` 选择最佳实践配置。
3. 将最终量化配置导出到 `save_path`，便于复现。
4. 调用：

```python
self.quant_service.quantize(
    quant_config=practice_config.extract_quant_config(),
    model_adapter=model_adapter,
    save_path=save_path,
    device=device_type,
    device_indices=device_index
)
```

这里的 `model_adapter` 很关键。后续加载权重、处理 dataset、按层生成前向调度，都由 adapter 提供统一接口。

## 4. ModelSlim V1 量化服务

`ModelslimV1QuantService.quantize()` 会把通用 `BaseQuantConfig` 转成 `ModelslimV1QuantConfig`，再进入 `quant_process()`。

`quant_process()` 的主要流程：

1. 如果 `save_path` 已存在，先清理目录下旧的 `.safetensors` 文件。
2. 固定随机种子。
3. 如果设备是 NPU，设置 `torch.npu.set_compile_mode(jit_compile=False)`。
4. 给每个 saver 配置 `save_directory`。
5. 根据配置选择 runner：
   - `model_wise`：使用 `PPRunner`。
   - `layer_wise`：使用 `LayerWiseRunner`。
   - `dp_layer_wise`：使用 `DPLayerWiseRunner`。
   - `auto` 默认偏向 layer-wise，多卡时走 DP layer-wise。
6. 执行 prior stage。
7. 执行主量化阶段：
   - 读取 `spec.dataset` 指定的校准数据。
   - 创建 runner。
   - 把 `spec.process` 里的 processor 加入 runner。
   - 如果有 `save_path`，把 `spec.save` 里的 saver 也加入 runner。
   - 调用 `runner.run(calib_data=dataset, device=device)`。

`linear_quant` 就是在 `spec.process` 中出现的一种 processor 配置。

## 5. Runner 如何驱动前向和 Processor

v1 的核心调度在 `GeneratedRunner`。

### 5.1 处理校准数据

runner 调用：

```python
get_input_datas(adapter, calib_data, device)
```

内部会调用：

```python
adapter.handle_dataset(calib_data, device)
```

adapter 把原始校准集转换成模型可直接消费的输入形式，例如：

```python
model(*args, **kwargs)
```

### 5.2 加载模型和权重

如果 runner 没有传入已加载模型，就调用：

```python
model = adapter.init_model(device=device)
```

在 layer-wise runner 中，模型通常先初始化到 CPU，然后 runner 自动插入两个 `LoadProcessor`：

- 前置 `LoadProcessor(mode="load")`：逐层把模块加载到执行设备。
- 后置 `LoadProcessor(mode="offload")`：处理完后卸载，降低显存占用。

因此，权重加载入口不是 `linear_quant` 本身，而是模型 adapter 和 runner 的加载策略。

### 5.3 生成调度单元

runner 会把每个 processor config 实例化成 processor：

```python
processor = AutoSessionProcessor.from_config(model, processor_config, adapter)
```

然后为每个 processor 创建 `GeneratedProcessUnit`。

每个 unit 会初始化 generator：

- 如果 processor 不是 data-free，需要校准数据，使用 `adapter.generate_model_forward(model, inputs)`。
- 如果 processor 是 data-free，使用 `adapter.generate_model_visit(model)`。

`LinearQuantProcessor.is_data_free()` 取决于内部 `LinearQuantizer` 的激活和权重量化器是否都 data-free。常见静态激活量化需要数据；纯权重量化或动态激活量化可能不需要额外校准数据。

### 5.4 每一步调度

调度循环每次从 generator 取出一个 `ProcessRequest`，合并成 `BatchProcessRequest`：

```python
BatchProcessRequest(
    name=requests[0].name,
    module=requests[0].module,
    datas=[(request.args, request.kwargs) for request in requests],
)
```

然后依次调用：

```python
processor.preprocess(batch_request)
processor.process(batch_request)
processor.postprocess(batch_request)
```

对 `LinearQuantProcessor` 来说，这三个阶段分别对应：

- `preprocess`：把当前模块内部命中的 `nn.Linear` 替换为 `LinearQuantizer`。
- `process`：如果需要数据，执行当前模块前向。
- `postprocess`：把 `LinearQuantizer` deploy 成 FakeQuant Linear IR。

## 6. LinearQuantProcessor 的模块替换

`LinearQuantProcessor` 位于 `msmodelslim/processor/quant/linear.py`。

配置结构是：

```python
class LinearProcessorConfig(AutoProcessorConfig):
    type: Literal["linear_quant"] = "linear_quant"
    qconfig: LinearQConfig
    include: List[str] = ["*"]
    exclude: List[str] = []
```

其中：

- `include`：控制哪些模块名参与量化。
- `exclude`：控制哪些模块名跳过量化。
- `qconfig`：分别描述激活和权重的量化配置。

### 6.1 preprocess：查找并替换 Linear

`preprocess()` 调用 `_install_quantizer()`：

```python
for name, submodule in module.named_modules(prefix=prefix):
    if not isinstance(submodule, nn.Linear):
        continue
    if name not in self.include:
        continue
    if name in self.exclude:
        continue
    self._process_linear(name, submodule)
```

命中的 `nn.Linear` 会进入 `_process_linear()`：

```python
quantizer = LinearQuantizer(self.config.qconfig)
quantizer.setup(module)
self.model.set_submodule(full_name, quantizer)
```

如果分布式已初始化，并且当前层是共享层，还会调用 `quantizer.enable_sync()`，使激活/权重量化器启用同步逻辑。

### 6.2 setup：接管权重、bias 和已有 hook

`LinearQuantizer.setup(linear)` 做三件事：

1. 保存原始 `linear.weight` 和 `linear.bias`。
2. 调用权重量化器：

```python
self.weight_quantizer.init_weight(
    QStorage(QDType.FLOAT, value=linear.weight.detach()),
    self.bias
)
```

3. 把原始 `linear` 上已有的 forward pre-hook 迁移到 `LinearQuantizer`：

```python
for hook_id, hook in linear._forward_pre_hooks.items():
    with_kwargs = hook_id in linear._forward_pre_hooks_with_kwargs
    self.register_forward_pre_hook(hook, with_kwargs=with_kwargs)
```

这里的“替换 hook”更准确地说是：`linear_quant` 替换了模块本身，同时保留原模块上的 pre-hook。后续 deploy 时，如果 pre-hook 是 `HookIR`，还会把它转换成 wrapper，包在 FakeQuant Linear 外面。

## 7. LinearQuantizer 前向计算

`LinearQuantizer.forward(x)` 是校准和 fake quant 感知的核心：

```python
with QStorage.set_value_float_type(x.dtype):
    x = self.input_quantizer(x)
    weight = self.weight_quantizer(x)
return F.linear(x, weight, self.bias)
```

可以拆成三步：

1. 激活量化器处理输入 `x`。
2. 权重量化器处理原始权重。
3. 使用量化感知后的 `x` 和 `weight` 执行 `F.linear()`。

### 7.1 激活侧

激活量化器由：

```python
AutoActQuantizer.from_config(config.act)
```

根据 `(QScheme, method)` 从 registry 创建。常见方法：

- `none`：不量化激活。
- `minmax`：用 min/max 统计 scale / offset。
- `histogram`：用直方图搜索更优截断范围。
- `pdmix`：prefill 和 decode 使用不同激活量化策略。

静态激活量化通常需要前向校准数据；动态激活量化则可能在每次 forward 中按 token 或当前输入即时计算量化参数。

### 7.2 权重侧

权重量化器由：

```python
AutoWeightQuantizer.from_config(config.weight)
```

创建。常见方法：

- `minmax`：按权重范围计算量化参数。
- `ssz`：在 MinMax 初值基础上迭代优化 scale / offset。
- `gptq`：利用激活输入构造 Hessian 近似，做二阶误差补偿。

权重量化器在 `setup()` 阶段拿到原始权重；真正计算通常发生在第一次 forward 或 `get_q_storage()` / `get_q_param()` 被调用时。

### 7.3 前向获取数据的意义

校准前向的主要作用是让底层量化器获得必要统计：

- 静态激活量化：收集输入激活的 min/max 或 histogram。
- GPTQ 类权重量化：可能需要当前层输入来统计 Hessian 近似。
- data-free 权重量化：可能不依赖校准数据，只用权重本身即可完成。

runner 会把当前模块的输入 `args/kwargs` 传入 `request.module(*args, **kwargs)`，因此 `LinearQuantizer.forward()` 会自然拿到真实校准输入。

## 8. postprocess：deploy 成 FakeQuant IR

当前模块前向处理完成后，`LinearQuantProcessor.postprocess()` 调用 `_deploy()`：

```python
for name, submodule in module.named_modules(prefix=prefix):
    if hasattr(submodule, "deploy"):
        self.model.set_submodule(name, submodule.deploy())
```

对 `LinearQuantizer` 来说，`deploy()` 会调用：

```python
qir.AutoFakeQuantLinear.create(
    self.input_quantizer.get_q_param(),
    self.weight_quantizer.get_q_param(),
    self.weight_quantizer.get_q_storage(),
    self.bias
)
```

这一步会根据激活和权重的 `QScheme` 自动选择具体 FakeQuant Linear 类型。例如：

- 静态 W8A8：`W8A8StaticFakeQuantLinear`
- 动态 W8A8 per-channel：`W8A8DynamicPerChannelFakeQuantLinear`
- 动态 W8A8 per-group：`W8A8DynamicPerGroupFakeQuantLinear`
- W4A8 / W4A4 / MXFP 等其他注册过的 IR

deploy 后，模型中的对应子模块不再是 `LinearQuantizer`，而是部署态 FakeQuant IR。这个 IR 持有最终量化参数和量化权重。

如果 `LinearQuantizer` 上存在 `HookIR` 类型的 pre-hook，deploy 时还会执行：

```python
fake_quantizer = hook.wrapper_module(fake_quantizer)
```

也就是把 hook 表达成一个 wrapper module，避免保存/导出时丢失这类图变换语义。

## 9. 保存阶段

在主量化阶段，如果提供了 `save_path`，runner 会把 saver processor 追加到 processor 列表中。常见保存器是 `AscendV1Saver`。

保存器是 data-free processor，不需要重新跑校准数据。它在 `postprocess()` 或 `post_run()` 中遍历模块：

```python
for name, sub_module in module.named_modules(...):
    self._process_module_maybe_wrapper_ir(name, sub_module)
```

`AutoSaverProcessor` 内部有类型到处理函数的映射：

```python
qir.W8A8StaticFakeQuantLinear: self.on_w8a8_static
qir.W8A8DynamicPerChannelFakeQuantLinear: self.on_w8a8_dynamic_per_channel
qir.W8A8PDMixFakeQuantLinear: self.on_w8a8_pd_mix
nn.Linear: self.on_float_linear
...
```

### 9.1 W8A8 静态量化保存内容

以 `on_w8a8_static()` 为例，会写出：

- `<prefix>.weight`：int8 量化权重。
- `<prefix>.quant_bias`：融合量化尺度后的 int32 bias。
- `<prefix>.input_scale`：激活 scale。
- `<prefix>.input_offset`：激活 offset。
- `<prefix>.deq_scale`：输入 scale 与权重 scale 组合出的反量化 scale。
- `<prefix>.bias`：必要时保留 float bias。

这些 tensor 会写入 safetensors，同时在 `quant_model_description.json` 中写入对应量化类型描述。

### 9.2 动态量化保存内容

以 `W8A8DynamicPerChannelFakeQuantLinear` 为例，保存器写出：

- `<prefix>.weight`
- `<prefix>.weight_scale`
- `<prefix>.weight_offset`
- `<prefix>.bias`

动态激活的 scale 通常不作为静态参数保存，因为它在推理 forward 时根据输入动态计算。

### 9.3 最终导出文件

`AscendV1Saver.post_run()` 最终会：

1. 写入所有量化 tensor 到 safetensors。
2. 写入 `quant_model_description.json`：
   - `version`
   - `model_quant_type`
   - `metadata`
   - `group_size`
   - 每个 tensor 的量化类型描述
3. 复制原模型目录下的 `.json`、`.py`、`.txt`、`.jinja` 等配置文件。
4. 移除 `config.json` 中已有的 `quantization_config` 字段。
5. 如果模型 adapter 实现了 `AscendV1SaveInterface`，调用 adapter 的保存后处理。

## 10. Hook 和 Wrapper 的关系

`linear_quant` 里有两类容易混淆的“hook”：

1. 原始 `nn.Linear` 上已有的 `forward_pre_hook`。
2. `HookIR` 这类用于表达图变换的特殊 hook。

替换 `nn.Linear` 为 `LinearQuantizer` 时，代码会把原始 pre-hook 注册到新模块上，保证原有前处理逻辑不丢。

deploy 或保存时，如果发现 hook 是 `qir.HookIR`，会把它转换为 wrapper module：

```python
wrapper = hook.wrapper_module(sub_module)
module.set_submodule(name, wrapper)
```

这样导出的模型结构中会显式保留 wrapper，而不是依赖 Python runtime hook。

## 11. v0 路径补充

仓内还保留了 `modelslim_v0` 量化服务。v0 路径不走 `LinearQuantProcessor` / `GeneratedRunner` 这套细粒度 processor 调度，而是更接近旧版 PTQ：

1. 读取校准集。
2. `model_adapter.load_model(device=device)` 加载完整模型。
3. 可选执行 `AntiOutlier`。
4. 创建 `QuantConfig` 和 `Calibrator`。
5. `calibrator.run()` 完成校准。
6. `calibrator.save(..., save_type=["ascendV1"])` 保存。

因此，如果讨论当前 `linear_quant` 的 processor / quantizer / saver 分层，主要对应的是 `modelslim_v1` 路径。

## 12. 一条典型 Linear Quant 流程串起来

可以把整个流程压缩成下面这条链：

```text
msmodelslim quant
  -> CLI 解析参数
  -> NaiveQuantizationApplication 校验路径和参数
  -> PluginModelFactory 创建 model_adapter
  -> YamlPracticeManager 选择量化配置
  -> QuantServiceProxy 选择 modelslim_v1 backend
  -> ModelslimV1QuantService 准备 dataset / runner / saver
  -> runner 调用 adapter.init_model() 加载模型和权重
  -> runner 调用 adapter.handle_dataset() 准备校准输入
  -> runner 根据 adapter.generate_model_forward() 按层产生 ProcessRequest
  -> LinearQuantProcessor.preprocess()
       扫描 nn.Linear
       按 include/exclude 过滤
       创建 LinearQuantizer
       setup 原始 weight/bias
       迁移原始 pre-hook
       set_submodule 替换原模块
  -> LinearQuantProcessor.process()
       当前模块执行前向
       LinearQuantizer.forward()
       input_quantizer 收集/计算激活量化参数
       weight_quantizer 计算量化权重和权重量化参数
       F.linear() 输出校准结果
  -> LinearQuantProcessor.postprocess()
       LinearQuantizer.deploy()
       根据 q_param 创建 FakeQuantLinear IR
       HookIR 转 wrapper
       set_submodule 替换为部署态模块
  -> AscendV1Saver 遍历 FakeQuantLinear
       写量化权重、scale、offset、deq_scale、bias
       写 safetensors
       写 quant_model_description.json
       复制模型配置文件
```

## 13. 关键结论

- `LinearQuantProcessor` 负责流程：找层、替换、前向校准、deploy。
- `LinearQuantizer` 负责把激活量化器和权重量化器组合到一次 `F.linear()` 前向中。
- `AutoActQuantizer` / `AutoWeightQuantizer` 才是真正选择 `minmax`、`histogram`、`ssz`、`gptq` 等底层算法的位置。
- runner 提供真实校准输入，使静态激活量化和 GPTQ 这类依赖数据的算法能够收集统计。
- deploy 后的 FakeQuant IR 是保存器识别和导出的对象。
- 保存阶段输出的不只是权重，还包括描述文件、scale / offset / deq_scale / bias 等推理所需参数。
