介绍：
我是来自昇腾计算产品线的蔡圣诚，目前在推理使能组工作，组内业务主要包括模型量化、投机推理和测评三个方向。本人现在主要担任下一代际芯片950的模型量化负责人，近期完成的工作主要是浮点低精度量化。去年因为昇腾全面开源开放，作为负责人也对模型量化从产品力、易用性的角度做了不同的优化。简单介绍一下两个项目点，一个是从人、主观的角度，让用户更易用，降低开发者门槛，二是从物料环境的客观角度，让产品的使用限制放开，从多卡到单卡再到并行量化，最后cpu量化。在低精浮点量化方面，主要是完成了mxfp量化的落地，在二十多个开源模型上线，并且对于低精度的量化完成了多种算法优化，最终达成客户目标。早期还参与过算子开发工具的研发，主要是涉及到算子的调试调优，目的是为了提升算子开发效率。

从人+主观分析： 使用和开发

领域驱动：
传统脚本把权重加载、离群值抑制、量化算法、保存等等都耦合在一个或者个别脚本内，如果有新模型接入，需要重复很多代码量，霰弹式修改，算法接入后也不能复用
提出领域驱动设计，也不是完全照搬，量化知识复杂，但是量化流程不复杂，所以也没有什么实体、聚合根这些固定的范式，主要还是参考一下思想
提出接口-应用-领域-基础设施四层架构
接口：和用户交互的cli，获取参数和命令
应用：选择领域功能，一键量化、敏感层分析、自动调优等，做流程编排
领域：量化模式、量化算法、调度器等
基础设施：模型适配器

依赖反转，上层应用不直接依赖底层模块，而是依赖接口。每个算法提出自己的interface，告诉别人自己需要什么，比如子图，前向要求等等，作为基础设施的模型适配器，实现这些接口，使得量化算法可以正常运行。每个模型适配不同的接口，保证自己在领域边界内

做了这些之后从用户的角度来看有直接的cli可以交互，易用性大大提升。作为贡献开发者来说，有了清晰的领域边界，如果要新增算法就在算法领域内完成，要接入模型，就在模型适配器中开发，不用感知内部流程编排以及接口调用，都以注入的方式完成了。

从环境+客观分析： 能不能量化->加快量化->不用npu量化

逐层量化：
是单卡执行每一个decode layer，包括像load、smooth、量化、save、offload。解决的是超大模型无法量化的问题

DP：
切分数据，每个rank执行一份数据，是数据并行
spawn启动多进程，每个进程绑定一个npu，distributed 创建 hccl
根据算法不同，选择all reduce还是all gather，如果是min max就是all reduce得到全局的min max就行。如果要全量激活就all gather
但是后续计算每个卡都在做同样的计算，不过相对来说开销很小，为了保持一致性，主要聚焦的还是节省下来的前向数据统计耗时
保存也是并行的

DTS：
DTS是程序启动DP后的自动选择
任务并行，比如data-free的情况下，dp没有用，可以把每个rank处理不同的linear，这时候每个卡做各自做计算

同步机制：
先同步owner map，也就是每个卡把自己处理的linear gather起来
一级同步：普通的linear任务，只有weight这样的parameter计算
二级同步：特殊的量化算法任务，有自定义的数据结构，比如awq有ratio，autoround有偏移值
三级同步：跨模块的任务，比如smooth，要做融合的，这个是最顶层的

权重转换：
多进程open权重，这时候还没加加载到内存，就是读了个head
在线程中根据key，去get tensor读取每个weight到内存中进行计算，这时候线程就有用了，因为有IO开销
不然会有GIL，python规定进程内线程必须CPU串行
除非有些torch算子的内部c++实现有多线程功能，OPENMP参数，set_num_thread()函数
竞品llm-compressor：cpu是纯串行，很慢。有多线程机制，但是结合GPU运行

hadamard矩阵是怎么计算的？
先确定size，和要旋转的那一维一致，比如hidden_size、head_dim
先用随机种子seed生成+1和-1的对角矩阵
再用walsh推导hadamard矩阵
[1, 1]
[1,-1]
只能2次幂的
然后用随机对角矩阵乘hadmard矩阵，最后除根号size
非2次幂的，拆分成预先生成的非二次幂hadamard和二次幂的相乘（butterfly计算，比walsh推导性能更好）

AWQ：
先计算每个通道的平均值=每个通道的scale
枚举ratio，0.05一个步长，和scale逐元素相乘，计算mse

SSZ：
1.原始scale、offset，计算得到q
2.固定q，最小二乘计算scale、offset
3.得到新q，执行2
4.不断循环，直到收敛

GPTQ：
残差，计算误差后，补偿到未量化元素中，很耗时

DeepSeek V4：
wq_a：把q压缩到latent，4096->1024
wq_b：latent的q打成多头，相当于扩了，1024->64*512
wkv：B,S,4096直接变成B,S,512，都没有多头，所有q对应一组kv
Compressor：根据compress_ratio，将连续多个kv加权求和成1个，变成B,S/compress_ratio,512
windows：上面的基础上加上windows的长度也就是B,win+S/compress_ratio,512
indexer：会在S/compress_ratio里选topk个，默认512个，所以最后kv结果是B,win+topk,512
wo_a、wo_b：先降维再升维，比o proj省参数

msAgent：
先做一次敏感层分析，上界：敏感层全回退，下界：不回退
二分搜索最小达标的点，然后继续摸高，减少回退or换抑制策略

量化收益在推理时怎么拿到？
首先比较直观的是数据的显存占用减少了

计算量的层面分析（prefill）、访存带宽层面分析（decode）、并行层面分析（ep并行）
1.是否能低精度load权重，不然显存都没法省
2.硬件是否支持原生低精度计算，不然要反量化成bf16，此处有开销，且bf16还会反存回显存，显存收益也没有了，如果做了融合，可能还会有点显存收益或者带宽收益（decode阶段）
3.是否有反量化矩阵乘融合算子，减少反量化的写回开销
4.动态量化会有一定开销，因为要在线计算激活scale等
5.通信收益：首先权重量化了之后，每个卡上的EP可以放的更多了，并行度更大了，然后通信交互的是低比特的数据，也能提高带宽
6.prefill阶段是计算密集型，更看重硬件是否支持低精度计算。decode阶段是memory-bound，读取带宽减少可以提升性能
7.为什么达不到理论倍数：不是所有层都做量化，有scale等数据，需要反量化，硬件没有原生支持低精度计算
8.是否attention成为瓶颈，可以做fa3，或者kv cache量化

TP：
开始（gate/up、qkv）列切，后面（down、o）行切，做all-reduce（因为是累加）
首先可以做gate_up的pack，这样直接拼成一个大向量，更有利于tp并行
比如gate+up的weight左右拼接，假如tp=4，每个rank上保存0.25个gate+up，每个rank上都有全量的x，x和gate+up计算，然后在rank内部做silu和乘法，每个rank上得到0.25份的结果，这个结果是按照列切分的，然后把down proj做行切，每个rank上加载0.25个weight，刚好可以和列切的做计算，最后做一个累加allreduce

EP：
每个卡放n个experts，router计算完之后all-to-all，分到不同的experts。专家计算完mlp，all-to-all做合并

FA3和KV CACHE量化
kv cache量化：长上下文，decode阶段读取量太大，又不想影响到q，只对k、v引入误差
MHA、GQA，有dynamic cache接口
fa3量化：对计算效率要求高，优化attention计算效率
mla、多模态生成场景（没有kv cache)

MTP原理：
输入是 主模型的最后一个hidden + 主模型输出的token id，然后预测出一个token，并把hidden和token id传给下一个mtp
和eagle的区别在于，eagle是单独训练的，mtp是和主模型一起训练的。eagle是单层网络循环外推hidden，且head是直接用主模型的。mtp有多层结构，也有自己的head
mtp和eagle都是串行的，dflash和dspark都是并行多token一起输出的。然后dflash有双向注意力

DSpark量化：
DSpark原理：并行输出多个token，然后markov head会做轻量的顺序依赖，confidence sceduling会预测置信度，低的就砍掉

首先确认背景，V4主模型是做了旋转的，且dspark mtp中的embed和head是共用主模型的权重
DSpark需要接收两个输入，一个是用来prefill构造kv cache的main hidden，一个是普通的经过主模型head输出的output
假设有两种方案，一个是mtp就不旋转
1.main hidden需要经过main proj和main norm，然后进入attn，计算wkv和kv norm。由于策略是不旋转，我需要把带旋转的main_hidden反旋回来，那就需要main proj做一个右旋来抵消
2.但是output（不带旋转的）正常经过embed之后，由于embed是共享主模型的，所以是旋转过的，这时候就有矛盾了，所以策略是复制主模型没旋转前的embed和head，保证output这条decode分支也不带旋转
还有一个是如果mtp精度不行，需要QuaRot
1.难点在于如何处理wkv，因为在prefill阶段，wkv接收的是不带旋转的，但是在decode阶段接收的x是带旋转的，所以这里只能复制一份wkv。其他保持一致

vllm-ascend基本推理服务步骤：
1.服务初始化
2.模型加载、切分、预热（分配显存）
3.接收请求、调度
4.prefill，保存kv cache，生成首token。会分配kv block，按照pagedattention
prefill阶段优化手段：chunked preill（长序列切小，一般和Continuous Batching一起使用，组合不同的prefill和decode成为一个batch进行推理）
Prefix Caching（前缀和匹配，复用kv cache）
5.decode，Continuous Batching动态处理多个请求
6.sample采样输出
pd分离：避免prefill处理长序列占用大量资源影响decode，就是需要传输kv cache

量化权重精度排查：
1.环境、配置排查
2.首token不对，prefill，逐层输出中间激活，计算余弦相似度，找到误差突然放大的层，然后再具体看里面每个阶段的输出，检查该层权重参数、算子等
3.decode乱码，排查decode是否有不一样的算子，kv cache是否有问题
4.量化异常也不一定是权重异常，可能是量化触发了框架或者算子某些不同的执行路径
