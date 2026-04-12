# B.1.1 异步 RL 训练架构深潜

> 本节是 [B.1 RL 基础设施纵览](./rl-infrastructure) 阶段三的延伸阅读。如果你正在设计或选型大规模 LLM RL 训练系统，这里的内容帮助你做出正确的架构决策。

## 2026 年异步 RL 生态全景

异步训练已成为 LLM RL 的主导范式。HuggingFace 调研了 **16 个开源异步 RL 库**[^1]，涵盖从单机到万卡的全部场景：

| 库 | 组织 | 编排方式 | 推理引擎 | 训练后端 | GitHub ⭐ |
|---|---|---|---|---|---|
| **veRL** | ByteDance | Ray | vLLM / SGLang | FSDP / Megatron | 19K+ |
| **OpenRLHF** | 开源社区 | Ray | vLLM | DeepSpeed | — |
| **AReaL** | Ant Group | asyncio + HTTP | vLLM / SGLang | FSDP2 / Megatron | 4.3K |
| **SLIME** | THUDM | Ray | SGLang | Megatron | 4.6K |
| **PipelineRL** | ServiceNow | asyncio + Redis | vLLM | DeepSpeed | — |
| **NeMo-RL** | NVIDIA | Ray | vLLM / SGLang | FSDP2 / Megatron | — |
| **ART** | CoreWeave / OpenPipe | asyncio | vLLM | Unsloth / Megatron | 9K |
| **PRIME-RL** | PrimeIntellect | asyncio + ZMQ | vLLM | FSDP2 | — |
| **TorchForge** | Meta | Monarch | vLLM | FSDP2 (TorchTitan) | — |
| **Tunix** | Google | JAX native | vLLM / SGLang / JAX | JAX/XLA | — |

> 完整的 16 库 × 7 轴对比表见 HuggingFace 原文[^1]。

## 部署模式：同步 vs Colocated vs Disaggregated

这是 LLM RL 基础设施中**最核心的架构决策**，直接决定 GPU 利用率的上限。

### 同步模式（Synchronous）

生成→训练→生成→训练，严格交替。TRL 默认模式。实现最简单，但**推理和训练不能重叠**，GPU 大量空闲。

```
GPU A: [生成 batch 0] [等待] [训练 batch 0] [等待] [生成 batch 1] [等待] ...
```

### Colocated 模式（共置）

推理和训练在**同一组 GPU** 上轮替使用。省 GPU 但不能重叠。权重同步本质上是 in-place reshard，几乎零成本。

```
GPU A: [生成 batch 0] [Reshard] [训练 batch 0] [Reshard] [生成 batch 1] ...
```

### Disaggregated 模式（分离式/异步）

推理集群和训练集群**物理分离**，通过 buffer 连接，两边并发运行。所有大规模 RL 训练的标配。

```
GPU Group A (推理): [生成 b0] [生成 b1] [生成 b2] [生成 b3] ...  ← 持续运行
                      ↓ buffer  ↓ buffer  ↓ buffer
GPU Group B (训练):    [训练 b0] [训练 b1] [训练 b2] ...          ← 持续运行
                      ↑ 权重同步 ↑ 权重同步
```

| 模式 | GPU 需求 | 推理/训练重叠 | 适用场景 |
|---|---|---|---|
| 同步 | 少（单组 GPU） | 不重叠 | 实验、小规模 |
| Colocated | 中（单组 GPU） | 不重叠 | 中等规模、预算有限 |
| Disaggregated | 多（两组 GPU） | **完全重叠** | 大规模生产 |

> **Colocated vs Disaggregated 的关键区分**：Colocated 可以做快速 reshard（Sleep/Wake）来加速推理，但本质上还是轮替，不是并发。真正的异步重叠只在 Disaggregated 模式下实现。

## Rollout 瓶颈：为什么异步如此重要

**生成阶段消耗 >90% 的训练时间**[^1]。

以 vLLM 在单张 H100 上的吞吐为基准（bf16，离线模式）：

- 7B 模型（DeepSeek-R1-Distill-Qwen-7B）：~6,300 tokens/s
- 32B 模型（DeepSeek-R1-Distill-Qwen-32B）：~1,200 tokens/s

一个典型的 GRPO 训练步：**G=8 completions/prompt × 64 prompts = 512 rollouts**

| 输出长度 | 总 token 数 | 7B 模型耗时 (1×H100) | 32B 模型耗时 (1×H100) |
|---|---|---|---|
| 2K tokens（短 CoT） | ~1M | ~3 分钟 | ~14 分钟 |
| 8K tokens（中 CoT） | ~4M | ~11 分钟 | **~56 分钟** |
| 32K tokens（长 CoT） | ~16M | ~45 分钟 | **~3.7 小时** |

即使用 8 张推理 GPU 分摊，32B 模型 32K-token rollout 仍需 ~28 分钟/步。这期间训练 GPU 完全空闲。

**Straggler 问题**进一步恶化：GRPO 的 G=8 组内采样，batch 必须等最慢的 completion 完成。CoT 长度高度可变（1K 到 32K），batch 被**最长的那个 completion**卡住。Continuous batching 只能部分缓解。

## 权重同步：推理集群如何获取最新参数

训练集群更新完参数后，需要把新权重推送到推理集群。这是异步架构中最复杂的工程问题。

### 传输方式

| 方式 | 延迟 | 使用框架 |
|---|---|---|
| NCCL Broadcast | ~100-500ms | 多数框架默认 |
| NCCL + Bucketing（打包传输） | ~20ms | veRL |
| CUDA IPC（零拷贝） | 极低 | NeMo-RL、MILES |
| LoRA Adapter-only | 亚毫秒（~50MB） | veRL、AReaL、ART |
| Filesystem + HTTP | 中-高 | PRIME-RL、Atropos |

### 中断粒度——推理在什么时候停下来换权重？

这是架构设计的分水岭：

| 粒度 | 行为 | 框架 |
|---|---|---|
| **Never**（不中断） | 在两次 forward pass 之间换权重（~1-10ms），生成不停止 | PipelineRL |
| Per HTTP Request | 中断当前 HTTP 请求，abort + prefix-resume | SkyRL、SLIME |
| Soft Pause（排空） | 停止接受新请求，等正在飞的完成后再 sync | PRIME-RL、AReaL、veRL |
| Per Batch（阻塞） | 整个 batch 完成后才 sync | NeMo-RL、Tunix |

PipelineRL 的"永不停止"方案最为激进：hook 进推理引擎，在每个 token 的 forward pass 之间获取锁、交换参数、释放锁。其他所有框架都在更粗的边界上暂停。

### LoRA 的特殊优势

如果只训练 LoRA adapter（rank-32 ≈ 50MB），权重同步问题几乎消失。即使是 HTTP 热替换也变成亚毫秒级。这使得 **LoRA + 异步训练** 成为有限 GPU 预算下的最佳实践。

8/13 个支持 LoRA 的框架可以只推送 adapter delta，无需传输完整模型权重。

## Staleness 管理：旧策略的数据怎么处理

异步训练必然引入 **off-policy 问题**：推理集群生成的数据可能来自若干步之前的旧策略。三个**正交**的解决策略：

### 策略一：版本拒绝（Version Rejection）

每个样本标记生成时的 policy version，训练时丢弃太旧的样本。简单正确，但浪费了生成这些样本的算力。

### 策略二：深度限制（Depth Bounding）

限制 buffer 的队列深度，架构上约束最大延迟。不需要逐样本版本追踪。Double-buffer（depth=1）是最简单的情况。

### 策略三：重要性采样校正（IS Correction）

不丢弃样本，用重要性采样比率 $\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$ 重新加权，通常截断（Truncated IS）。保留吞吐量但增加梯度方差。

### 框架对比

| 框架 | 版本拒绝 | 深度限制 | IS 校正 | 备注 |
|---|---|---|---|---|
| **veRL** | - | - | 截断 IS | 可选 OPSM |
| **AReaL** | - | 容量公式 | 可选 | `max_head_offpolicyness` |
| **PipelineRL** | `max_lag` | - | - | 逐样本版本标签 |
| **PRIME-RL** | `max_async_level` | `max_off_policy_steps` | IPO 信任域 | **三策略全用** |
| **ROLL** | - | - | 6 种变体 | TIS/TOPR/CISPO 等 |
| **TorchForge** | `max_policy_age` | - | - | 硬丢弃 |

工业趋势（PRIME-RL、AReaL、open-instruct）：**混合策略**——深度限制兜底 + 可选 IS 校正做安全网。

## 编排层：如何协调分布式组件

16 个库在编排方式上分为四类：

| 编排类型 | 原理 | 框架 |
|---|---|---|
| **Ray Actor Model** | 组件是带邮箱的独立 actor，Ray 管理调度、容错、对象传输 | veRL、SkyRL、NeMo-RL 等（**8/16**） |
| **Native Python** | asyncio / threading / multiprocessing，无外部依赖 | ART、AReaL、PipelineRL |
| **Pub/Sub 消息总线** | Redis Streams / JSONL，解耦生产者-消费者 | PipelineRL（跨节点） |
| **HTTP 微服务** | REST API，语言无关 | Atropos |

Ray 占据半壁江山（8/16），因为 RL 训练天然需要编排异构组件（推理引擎、训练引擎、奖励模型、环境池），Ray 的 actor 模型直接映射这些组件。

> **Monarch**（Meta/TorchForge）：PyTorch 原生的 GPU actor 框架，通过 RDMA 做 weight sync，不中断生成。架构上与 Ray 同源但专为 CUDA 生态设计。
>
> **Tunix**（Google）：JAX/XLA 原生 mesh 模型，`ThreadPoolExecutor` 做异步重叠，`jax.device_put` 做跨 mesh 权重传输，完全在 TPU 生态内。

## 前沿趋势：下一代 RL Infra 的挑战

### Critic-Free 算法减轻显存但加重同步压力

GRPO、REINFORCE++ 等 critic-free 方法消除了 Value Network，释放 ~50% 显存。但需要更大的组大小（G=8-32）获得低方差优势估计，这意味着**每步更多 rollout → 策略漂移更快 → 权重同步压力增大**。

### PRM 成为新的同步瓶颈

Process Reward Model 对推理链的每一步打分，计算量可能接近 rollout 本身。这要求：

- **异步 Reward Pipeline** 成为必需（PRIME-RL 已实现）
- **独立的 Preprocessor GPU Pool** 可能成为标配
- DEEP-GRPO 的 pivot resampling 需要保存 KV Cache 状态，当前**没有框架原生支持**

### 多 Agent 协同训练的 Straggler 倍增

多 Agent 链式调用（Proposer + Solver），straggler 是**多个长度分布的乘积**——如果每个 Agent 的 P90 是中位数的 5×，联合 P90 约为 25×。原子工作单元需要从 (prompt, completion, reward) 三元组变为整个 **episode（有向图）**。

### MoE 模型的训练-推理不一致

DeepSeek-V3.2 暴露了两个 IS 校正无法修复的结构性问题。详见 [B.2 分布式训练架构 → 前沿挑战](./distributed-training)。

---

## 参考文献

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026. 本节核心数据来源。七轴对比框架：编排原语、Buffer 设计、权重同步协议、Staleness 管理、Partial Rollout 处理、LoRA 支持、训练后端。

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025. 原文写给 Meta 基础设施团队，包含 PPO 逐行代码推导和分布式视角。

[^3]: OpenRLHF, [OpenRLHF: A High-Performance RLHF Framework](https://arxiv.org/html/2501.03262v4), EMNLP 2025 Demo. 基于 Ray + vLLM + DeepSpeed 的工业级 RLHF 框架。

[^4]: veRL Project, [HybridFlow: A Flexible and Efficient RL Training Framework](https://github.com/verl-project/verl), ByteDance / Volcano Engine. 分离式 RL 训练架构，支持 GRPO/DAPO/PPO，GitHub ⭐ 19K+。

[^5]: Netflix Tech Blog, [Scaling LLM Post-Training at Netflix](https://netflixtechblog.com/scaling-llm-post-training-at-netflix-0046f8790194), 2025. 工业规模后训练的工程实践。

[^6]: Google Cloud Blog, [Run High-Scale RL for LLMs on GKE](https://cloud.google.com/blog/products/compute/run-high-scale-rl-for-llms-on-gke/), 2025. Kubernetes 上运行大规模 RL 训练的实践指南。

[^7]: LLM Stats Blog, [Post-Training Techniques 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026), 2026. GRPO/DAPO/RLVR 取代传统 RLHF 的技术综述。
