# B.1 RL 基础设施纵览：从单机到 Agentic

> RL 训练的瓶颈往往不在算法，而在基础设施。本节按规模递进，梳理不同阶段 RL 系统的环境与采样架构。LLM RL 的异步训练架构深潜见 [B.1.1 异步 RL 训练架构](./rl-infrastructure-async)。

## 为什么 RL 的基础设施比监督学习复杂？

监督学习的训练循环是**静态**的：

```
数据集 → DataLoader → 前向 → 反向 → 更新
```

RL 的训练循环是**动态**的：

```
策略采样 → 环境交互 → 收集轨迹 → 前向 → 反向 → 更新 → 用新策略重新采样
```

关键区别：**训练数据是策略自己生成的**。这意味着：

1. 采样和训练之间存在数据依赖——不能提前准备好数据集
2. 采样速度直接决定训练速度——环境交互往往是瓶颈
3. 策略更新后，数据分布发生变化——经验回放有 on-policy 的过期问题

## 四个规模阶段

### 阶段一：单机 RL（Gymnasium）

**典型场景**：CartPole、LunarLander、学术实验

**核心组件**：

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()
action = policy(obs)           # 策略决定动作
obs, reward, terminated, truncated, info = env.step(action)
```

**采样加速——向量化环境**：

单环境太慢，Gymnasium 提供 `SyncVectorEnv` 和 `AsyncVectorEnv` 并行采样：

```python
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

# 同步向量化：N 个环境串行 step，但批量返回
envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
obs, info = envs.reset()            # shape: (8, obs_dim)
actions = policy(obs)               # 一次推理得到 8 个动作
obs, rewards, ... = envs.step(actions)  # 批量交互
```

| 方式 | 原理 | 适用场景 |
|---|---|---|
| `SyncVectorEnv` | 在主进程中顺序 step | 轻量环境（CartPole、Atari） |
| `AsyncVectorEnv` | 多进程并行 step | 计算密集环境（物理仿真） |

**要点**：这一阶段全部在单机完成，CPU 做环境交互，GPU 做策略推理和训练。对于 CartPole 这种轻量环境，瓶颈通常在 GPU 侧。

### 阶段二：游戏/仿真 RL（并行化环境）

**典型场景**：Atari、MuJoCo、Isaac Gym 机器人仿真

这一阶段的瓶颈转移到**环境交互**。一帧 Atari 画面需要 CPU 模拟，一个 MuJoCo 步需要物理引擎计算。

**Atari 并行化**：

```python
# 典型配置：32-64 个并行环境
envs = AsyncVectorEnv([make_atari_env() for _ in range(64)])
# 吞吐量：单环境 ~500 fps → 64 并行 ~30000 fps
```

**Isaac Gym——GPU 并行仿真**：

NVIDIA Isaac Gym 把物理仿真直接搬到 GPU 上，实现数万个环境并行：

```
传统方式：  CPU 物理引擎 × 64 环境 →  GPU 策略推理
Isaac Gym： GPU 物理仿真 × 4096 环境 + GPU 策略推理（全在 GPU 上）
```

| 对比 | CPU 并行 (MuJoCo × 64) | GPU 并行 (Isaac Gym × 4096) |
|---|---|---|
| 采样速度 | ~10K fps | ~1M fps |
| 数据传输 | CPU→GPU 每步 | 零拷贝 |
| 适用场景 | 少关节机器人 | 人形机器人、灵巧手 |

**Sample Factory 极致吞吐**：

Sample Factory 专为高吞吐设计，在 Atari 上可达 100K+ fps：

- 异步 Actor-Learner 架构：采样和训练完全解耦
- Actor 进程持续收集轨迹，Learner 进程持续更新策略
- 共享内存缓冲区，零拷贝数据传输

```
Actor 1 ──→ ┐
Actor 2 ──→ ├─→ 共享内存 Buffer ──→ Learner (GPU)
Actor 3 ──→ ┘
```

### 阶段三：LLM RL 采样基础设施

**典型场景**：PPO/GRPO 训练 LLM（7B-70B）

LLM RL 的环境不再是 Gym，而是**文本生成**本身。核心挑战：

1. **Rollout 推理密集**：GRPO 的 k=16 意味着每个 prompt 要生成 16 个回答
2. **模型本身就很大**：7B 模型的推理需要一整张 GPU
3. **生成和训练的 GPU 需求波动**：Rollout 时训练闲置，反之亦然

**关键架构：Rollout-Training 分离**

```
┌─────────────────────┐     ┌─────────────────────┐
│   Rollout Workers    │     │   Training Workers   │
│   (vLLM 推理集群)    │     │   (FSDP/Megatron)    │
│                     │     │                     │
│  GPU Group A        │────→│  GPU Group B         │
│  批量生成轨迹        │ 数据 │  梯度更新策略         │
│  高吞吐推理          │ 传输 │  显存密集反向传播      │
└─────────────────────┘     └─────────────────────┘
```

**vLLM 加速 Rollout**：PagedAttention 把 KV Cache 管理起来，使批量推理吞吐量提升 2-4x。

**框架对比**（详见 [B.2 分布式训练架构](./distributed-training)）：

| 框架 | 架构 | Rollout 方案 | 适用规模 |
|---|---|---|---|
| **TRL** | 单进程 | HuggingFace generate | 单卡-小集群 |
| **OpenRLHF** | Ray 分布式 | vLLM + Ray Actors | 工业级集群 |
| **veRL** | 分离式 | vLLM / SGLang | 大规模 GRPO/DAPO |

**显存优化技巧**：

| 技巧 | 原理 | 节省 |
|---|---|---|
| Reference 模型共享 | Ref 不参与训练，可和 Actor 共享权重副本 | ~25% 显存 |
| LoRA Rollout | 推理时只加载 LoRA adapter | ~50% 显存 |
| Offload Reference | Ref 模型卸载到 CPU | ~25% GPU 显存 |
| Gradient Checkpointing | 重计算代替存储中间激活 | ~40% 显存（换时间） |

**异步训练——核心架构决策**：生成阶段消耗 >90% 的训练时间，推理和训练必须并发运行。这涉及三种部署模式（同步 / Colocated / Disaggregated）、权重同步协议、Staleness 管理等关键设计。这些内容在 **[B.1.1 异步 RL 训练架构](./rl-infrastructure-async)** 中展开讨论。

### 阶段四：Agentic RL 基础设施

**典型场景**：Web Agent、Code Agent、Tool-Use Agent 的多轮 RL 训练

Agentic RL 的环境不再是单步交互，而是**多轮、有状态、需要工具执行**的复杂环境。

**新增的基础设施需求**：

```
┌──────────────────────────────────────────────┐
│               Agentic RL 训练集群              │
│                                              │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Policy   │  │ 沙箱集群  │  │ 轨迹存储     │ │
│  │ (LLM)    │  │ (Docker)  │  │ (Redis/S3)  │ │
│  └────┬────┘  └─────┬────┘  └──────┬──────┘ │
│       │             │              │         │
│       ▼             ▼              ▼         │
│  ┌──────────────────────────────────────┐    │
│  │         Orchestrator（编排器）         │    │
│  │                                      │    │
│  │  多轮对话管理 → 工具调用 → 结果收集    │    │
│  │  信用分配 → 轨迹切分 → 奖励计算       │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

**1. 沙箱隔离**：Agent 执行代码、浏览网页、调用 API，每个 episode 需要隔离环境：

```python
# 每次交互启动一个隔离容器
sandbox = DockerSandbox(
    image="agent-env:latest",
    timeout=30,           # 单步超时
    memory_limit="512m",  # 内存限制
    network="none"        # 默认断网
)
result = sandbox.execute(agent_action)
```

**2. 轨迹存储与回放**：多轮交互产生的轨迹很长且不规则，需要专门存储：

```
Trajectory {
    id: "traj_001"
    task: "修复这个 Python bug"
    steps: [
        {role: "user", content: "报错信息..."},
        {role: "agent", content: "我看到问题了", tool_call: "edit_file(...)"},
        {role: "tool", content: "文件已修改"},
        {role: "agent", content: "让我测试一下", tool_call: "run_python(...)"},
        {role: "tool", content: "测试通过 ✓"},
        {role: "agent", content: "修复完成"}
    ]
    reward: 1.0  # 任务完成
    token_count: 1523
}
```

**3. 工具执行管理**：Agent 调用的工具需要并发管理和结果缓存：

- 工具执行超时控制（避免 Agent 卡在无限循环）
- 并发执行多个独立工具调用
- 结果缓存（相同输入不重复计算）

**4. 多轮信用分配**：这是 Agentic RL 独有的基础设施问题：

```
Agent 做了 5 步才完成任务，奖励 = 1.0
→ 哪一步贡献最大？
→ 第 3 步的 edit_file 是关键动作，应该获得更多信用
→ 需要轨迹级的奖励分解（credit assignment）基础设施
```

**5. 评测集群**：Agentic 任务的评测本身就是多轮交互，成本很高：

| 评测方式 | 成本 | 覆盖度 |
|---|---|---|
| 静态匹配（SWE-bench gold patch） | 低 | 只看结果 |
| 自动化执行（跑测试用例） | 中 | 验证功能正确性 |
| LLM-as-Judge | 中高 | 评估过程质量 |
| 人工评测 | 高 | 最可靠 |

## 规模化路径总结

```
单机 Gym                    游戏/仿真并行              LLM RL 集群              Agentic RL
─────────────────────────────────────────────────────────────────────────────────────────
gym.make()               AsyncVectorEnv           vLLM + FSDP             沙箱集群
SyncVectorEnv            Isaac Gym (GPU)          Rollout-Training 分离    轨迹存储
CPU 采样 + GPU 训练      Sample Factory           Ray 分布式编排            工具执行管理
~1K steps/s              ~100K steps/s            ~10K prompts/hour        多轮信用分配

瓶颈：GPU 训练            瓶颈：环境交互            瓶颈：Rollout 推理        瓶颈：评测成本
```

**选型决策树**：

- 环境轻量 + 单卡能装下模型 → 阶段一（TRL / CleanRL）
- 环境重 + 需要高吞吐采样 → 阶段二（Sample Factory / Isaac Gym）
- LLM + 需要大规模 PPO/GRPO → 阶段三（OpenRLHF / veRL），异步架构详见 [B.1.1](./rl-infrastructure-async)
- Agent + 多轮交互 + 工具调用 → 阶段四（TRL + 自建沙箱 / Agent framework）

## 参考文献

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026. 调研 veRL、OpenRLHF、AReaL、SLIME 等 16 个异步 RL 框架的完整对比。

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025. 面向 infra 工程师的后训练全流程入门，包含 PPO 逐行推导。

[^3]: OpenRLHF, [OpenRLHF: A High-Performance RLHF Framework](https://arxiv.org/html/2501.03262v4), EMNLP 2025 Demo. 基于 Ray + vLLM + DeepSpeed 的工业级 RLHF 框架。

[^4]: veRL Project, [HybridFlow: A Flexible and Efficient RL Training Framework](https://github.com/verl-project/verl), ByteDance / Volcano Engine. 分离式 RL 训练架构，GitHub ⭐ 19K+。
