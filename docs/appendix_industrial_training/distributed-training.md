# B.2 分布式训练架构

> 当模型从 7B 涨到 70B，从单机训练扩展到多机多卡，你需要理解分布式训练的核心策略。

## 为什么需要分布式？

- **PPO 训练需要同时跑 4 个模型**：Actor、Critic、Reference、Reward Model
- 以 7B 参数模型为例，每个模型 FP16 需要 ~14GB 显存，4 个模型就是 ~56GB
- 加上优化器状态和梯度，一张 A100（80GB）都装不下
- 更大的模型（70B+）需要多机多卡并行

## 三大分布式策略

| 策略                                   | 原理                                       | 适用场景               |
| -------------------------------------- | ------------------------------------------ | ---------------------- |
| **FSDP** (Fully Sharded Data Parallel) | 把模型参数、梯度、优化器状态切分到所有 GPU | PyTorch 原生，通用性强 |
| **DeepSpeed ZeRO**                     | 类似 FSDP 但有 ZeRO-1/2/3 三档             | 与 DeepSpeed 生态集成  |
| **Megatron-LM TP**                     | 张量并行，按层内切分                       | 超大模型（70B+）       |

## 四大并行策略详解

实际工业级训练中，通常需要组合使用多种并行策略：

**数据并行（DP）**：

- 每个 GPU 持有模型的完整副本，但处理不同的数据 batch
- 梯度通过 AllReduce 同步
- 最简单的并行方式，但单卡必须能装下整个模型

**张量并行（TP）**：

- 将单个矩阵运算拆分到多个 GPU 上并行计算
- 例如：一个 4096×4096 的矩阵乘法拆成 4 个 GPU 各算 1024×4096
- 需要 NVLink 等高带宽互联（通常在节点内使用）
- Megatron-LM 的核心实现

**流水线并行（PP）**：

- 将模型按层切分，不同 GPU 负责不同的层
- 例如：GPU 0 负责第 1-10 层，GPU 1 负责第 11-20 层
- 有"气泡"问题：前面的 GPU 算完后，后面的 GPU 才能开始
- 微批次调度（Micro-batching）可以缓解气泡

**专家并行（EP）**：

- 专为 MoE（混合专家）模型设计
- 不同的专家网络分布在不同 GPU 上
- Router 决定每个 token 发给哪个专家

**典型组合**：70B 模型常用 DP+TP+PP 三维混合并行；MoE 模型（如 Mixtral）额外使用 EP。

## 混合精度训练

| 精度格式  | 显存占用   | 计算速度 | 数值稳定性 | 适用场景                |
| --------- | ---------- | -------- | ---------- | ----------------------- |
| FP32      | 基准       | 慢       | 最好       | 关键精度场景            |
| FP16      | 减半       | 快       | 有溢出风险 | 训练（需 loss scaling） |
| BF16      | 减半       | 快       | 较好       | 训练（Ampere+ GPU）     |
| FP8       | 1/4        | 最快     | 较差       | 推理、实验性训练        |
| INT8/INT4 | 1/4 到 1/8 | 极快     | 有损       | 推理部署                |

实践建议：训练用 BF16（A100/H100 原生支持），推理用 INT4/INT8 量化。

## 实际框架使用指南

**veRL（火山引擎）**：

- 专为 RL 训练设计，核心特性是 Rollout 和 Training 的分离式架构
- 支持 GRPO/DAPO/PPO 等多种 RL 算法
- 适合需要大规模采样的场景（如 GRPO 的 k=16 组内采样）
- 典型用法：Rollout 阶段用 vLLM 加速推理，Training 阶段用 Megatron/FSDP 做反向传播

**LLaMA-Factory**：

- 一站式微调框架，支持 SFT/DPO/PPO/GRPO 全流程
- 低门槛上手：配置文件驱动，不需要写分布式训练代码
- 适合中小规模实验（单机到少量节点）

**DeepSpeed**：

- 通过 ZeRO 优化器实现显存优化，ZeRO-3 可以训练任意大小的模型
- 支持卸载到 CPU/NVMe 进一步节省显存
- 与 HuggingFace Transformers 深度集成

**Megatron-LM**：

- NVIDIA 出品，张量并行的工业标准实现
- 适合超大规模模型（70B+）的预训练和微调
- 需要较多的工程配置，学习曲线较陡

## RL 特有的分布式挑战

- **Rollout 阶段是推理密集型**：需要大量 GPU 做批量生成（vLLM 加速）
- **Training 阶段是计算密集型**：需要大量 GPU 做反向传播
- **两个阶段对 GPU 的需求是波动的**：Rollout 时 Training 闲置，反之亦然
- **解法**：异步 Rollout + 分离式架构（Rollout 和 Training 用不同的 GPU 组）

## 主流框架对比

| 框架                  | 特点                             | 适用场景            |
| --------------------- | -------------------------------- | ------------------- |
| **TRL** (HuggingFace) | API 友好，上手快                 | 研究、小规模实验    |
| **OpenRLHF**          | Ray 分布式，支持大集群           | 工业级 PPO/DPO 训练 |
| **veRL** (Volcengine) | 灵活的 Rollout-Training 分离     | 大规模 GRPO/DAPO    |
| **Sample Factory**    | 极高吞吐，专为 Atari/3D 环境设计 | 游戏、连续控制      |

## 前沿挑战

以下问题正在推动分布式 RL 训练架构的演进。相关基础设施讨论见 [B.1 RL 基础设施纵览](./rl-infrastructure)。

### 挑战一：MoE 模型的训练-推理不一致

MoE（混合专家）模型（DeepSeek-V3、Qwen3-MoE、Mixtral）正在成为后训练的默认起点。但 MoE 引入了传统密集模型不存在的结构性一致性问题。

**Expert Routing 不一致**：推理框架（vLLM/SGLang）和训练框架（Megatron/FSDP）各自实现 MoE Router，浮点舍入差异可能对相同输入激活不同专家。当 expert routing 分歧时，梯度步骤假设 Expert A 激活，但实际权重在 Expert B 下激活——优化方向直接错误。

DeepSeek-V3.2 的解决方案 **Keep Routing**：推理框架记录 routing decision，训练时强制执行相同路径。这要求推理服务器返回额外的 `expert_routing` 元数据，并修改训练 forward pass 接受这些路由决策。

**Sampling Truncation Mask 不匹配**：Top-p/Top-k 采样在生成时截断词汇表，训练时 π_θ 看到完整词汇表。这使得 IS 比率对被 mask 的 token 数学上无定义。**Keep Sampling Mask** 要求推理服务器返回采样 mask 并在训练时应用。

> 当前没有开源异步 RL 框架原生支持 Keep Routing 或 Keep Sampling Mask。这对训练 MoE 模型的团队是正确性前提，详见 [B.1 前沿趋势](./rl-infrastructure#趋势四moe-模型的训练-推理不一致)。

### 挑战二：Expert Parallelism (EP) 的权重同步代价

密集模型的权重同步是一次 NCCL Broadcast。MoE 模型需要先 AllGather 所有 EP rank 上的 expert 参数，再 Broadcast 到推理服务器——对于一个 256 experts 的 235B 模型，这是一笔不可忽视的通信开销。

目前只有 Megatron 系框架（veRL、SLIME、MILES、ROLL、NeMo-RL）和 PRIME-RL 的 FSDP2+EP 路径正确支持 EP。ZeRO 系框架（PipelineRL、verifiers-rl、OAT）可以加载 MoE 模型，但每个 expert 被分片到所有 ZeRO-3 rank，失去了稀疏性优势。

### 挑战三：PRM 的分布式基础设施影响

Process Reward Model 对推理链的每一步打分，计算量可能接近 rollout 本身。这在分布式架构中引入了一个新的 GPU 池——**Reward Scoring Pool**，位于推理池和训练池之间：

```
推理池 (vLLM) → Rollout Buffer → Reward Scoring Pool (PRM) → Training Buffer → 训练池 (FSDP)
```

PRM scoring 必须异步流水线化，否则将成为新的训练瓶颈。PRIME-RL 的 Orchestrator 已实现了这种流水线。

---

## 参考文献

- [^1] HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026. 分布式训练后端的完整对比（Axis 7），包含 MoE EP 支持、训练并行策略等详细分析。
- [^2] PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025. 面向 infra 工程师的后训练全流程入门，包含 PPO 的逐行代码推导和分布式视角。
- [^3] OpenRLHF, [OpenRLHF: A High-Performance RLHF Framework](https://arxiv.org/html/2501.03262v4), EMNLP 2025 Demo. 基于 Ray + vLLM + DeepSpeed，工业级 RLHF 训练框架。
- [^4] veRL Project, [HybridFlow: A Flexible and Efficient RL Training Framework](https://github.com/verl-project/verl), ByteDance. 分离式 RL 训练架构的工业实现。
- [^5] DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024. Keep Routing 和 Keep Sampling Mask 的原始出处。
