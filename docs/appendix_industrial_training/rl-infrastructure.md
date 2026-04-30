# B.1 RL 采样基础设施

> RL 训练不是在一个固定数据集上反复做 SGD。它的训练数据由当前策略在线生成，策略每更新一次，后面的数据分布也会随之改变。
>
> 所以 RL 基础设施的核心问题不是“怎么读数据”，而是：**谁负责生产新数据、数据以什么单位流动、训练端能不能及时消费、旧策略生成的数据还能不能用**。

## 一条主线：谁在生产训练数据

看 RL 基础设施时，最容易混乱的地方是把工具按名字平铺：Gymnasium、IMPALA、vLLM、OpenRLHF、veRL、slime。它们确实都出现在 RL 训练链路里，但不是同一层东西。

更清楚的看法是按**数据生产链路**分层：

| 层级                  | 数据从哪里来                                        | 数据单位                          | 主要瓶颈                                    | 代表系统                          |
| --------------------- | --------------------------------------------------- | --------------------------------- | ------------------------------------------- | --------------------------------- |
| 环境接口与单机采样    | 策略和环境交互                                      | transition / episode              | 单环境 `step()` 慢，CPU/GPU 来回等          | Gymnasium VectorEnv               |
| 经典 RL 采样/仿真系统 | 大量 Actor 或仿真环境持续产轨迹                     | trajectory / rollout fragment     | Actor、Learner、物理仿真互等                | IMPALA、Sample Factory、Isaac Gym |
| LLM rollout engine    | 当前语言模型对 prompt 生成回答                      | token / completion / conversation | 自回归 decode、KV cache、长尾样本、权重更新 | vLLM、SGLang                      |
| LLM RL 后训练框架     | 编排 rollout、reward、training、buffer、weight sync | rollout batch / sample stream     | 推理和训练如何并发，旧数据如何控制          | OpenRLHF、veRL、slime             |

这张表是本文的骨架。Gymnasium 不是 veRL 的竞品；vLLM 也不是 OpenRLHF 的替代品。Gymnasium 解决“环境 API 和批量环境”问题，vLLM/SGLang 解决“模型怎么高吞吐生成 token”问题，OpenRLHF/veRL/slime 才是“把 rollout、reward、训练和权重同步组织起来”的后训练框架。

## 为什么采样端决定 RL 系统上限

监督学习的训练循环是静态的：

```
数据集 → DataLoader → 前向 → 反向 → 更新
```

RL 的训练循环是动态的：

```
策略采样 → 环境/生成器产生反馈 → 收集轨迹 → 计算奖励 → 更新策略 → 用新策略重新采样
```

关键区别是：**RL 的 DataLoader 本身就是一个在线系统**。它不仅要读数据，还要运行策略、推进环境、生成文本、算奖励、记录轨迹、处理 episode 结束，再把这些数据交给 learner。

因此 RL 系统吞吐由三类速率共同决定：

- 采样端产出数据的速度：`steps/s`、`tokens/s`、`samples/s`
- 训练端消化数据的速度：batch size、反向传播、并行策略
- 反馈端返回奖励的速度：环境 step、规则判题、Reward Model、LLM-as-Judge、代码执行

只要有一块跟不上，整条链路就会被卡住。早期 RL 常卡在 Python 环境 step；游戏和机器人 RL 常卡在仿真吞吐；LLM RL 则把“环境交互”变成了长文本自回归生成，瓶颈自然转移到 rollout engine。

## 第一层：环境接口与单机采样

**代表**：Gymnasium

Gymnasium 首先是一个**环境接口**，不是分布式训练框架。它定义了 `reset()`、`step(action)`、observation、reward、terminated/truncated 等基本交互方式。你用 CartPole、LunarLander、Atari、MuJoCo 做算法实验时，通常先从这里开始。

这一层的瓶颈很朴素：单个环境太慢，GPU 大部分时间在等 CPU 做 `env.step()`。所以 Gymnasium 提供同步和异步向量环境，把多个环境实例包装成一个批量环境 [^gym_vec]。

**解法：向量化环境（vectorized env）**

```python
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
obs, info = envs.reset()                 # shape: (8, obs_dim)
actions = policy(obs)                    # 一次推理得到 8 个动作
obs, rewards, terms, truncs, infos = envs.step(actions)
```

| 方式             | 原理              | 适用场景                               |
| ---------------- | ----------------- | -------------------------------------- |
| `SyncVectorEnv`  | 主进程中顺序 step | 轻量环境，如 CartPole、部分 Atari 实验 |
| `AsyncVectorEnv` | 多进程并行 step   | step 本身较重的环境，如物理仿真        |

这一层的工程目标是让策略网络一次处理一批 observation，而不是一个环境一个环境地等。此时训练、采样、奖励都在一台机器上，最重要的是 batch 形状、episode reset、终止条件和日志统计处理干净。

## 第二层：经典 RL 采样/仿真系统

**代表**：IMPALA、Sample Factory、Isaac Gym

当任务从小环境变成 Atari、DeepMind Lab、ViZDoom、MuJoCo 或机器人仿真时，瓶颈从“单环境太慢”变成“很多环境怎么持续不断地产生轨迹”。这时简单地增加 learner GPU 没有用，因为 learner 等不到足够多的新数据。

### 2.1 Actor-Learner 解耦：IMPALA

```python
# 典型配置：32-64 个并行环境
envs = AsyncVectorEnv([make_atari_env() for _ in range(64)])
# 吞吐量：单环境 ~500 fps → 64 并行 ~30000 fps
```

CPU 并行先解决单环境慢的问题。但继续扩大规模时，Python 调度、进程通信和数据拷贝会成为新瓶颈。分布式 RL 系统通常把角色拆成 Actor 和 Learner：Actor 只负责采样，Learner 只负责训练。

IMPALA 是这条路线的代表。大量 Actor 并行生成 trajectory，把数据发送给中心 Learner；Actor 不再把梯度发回参数服务器，而是发送完整轨迹，让 Learner 在 GPU 上连续消费 batch。由于 Actor 采样时可能使用稍旧的策略，IMPALA 用 V-trace 做 off-policy 修正 [^impala]。这奠定了很多后续系统的基本形状：**采样和训练解耦，吞吐优先，再用算法处理数据过期**。

![IMPALA actor-learner 架构与同步/异步时间线](./images/impala-actor-learner.png)

_图 1：IMPALA 论文中的 Actor-Learner 架构和时间线。左边说明 Actor 只负责生成轨迹并从 Learner 拉取参数；右边说明 IMPALA 不再等待所有 Actor 同步完成，而是让 acting 和 learning 解耦。（来源：IMPALA 论文 [^impala]）_

### 2.2 单机高吞吐采样：Sample Factory

Sample Factory 把这个方向推到单机工程极限：异步 Actor-Learner、共享内存、批量推理和更少的 Python 开销，使 Atari/3D 控制任务可以达到 100K+ fps 量级 [^sf]。它不是简单地“多开几个环境”，而是把 workload 拆成专门组件：

- Rollout worker：CPU 侧只跑环境，自己不持有策略副本，因此可以大量并行
- Policy worker：GPU 侧做批量 action generation，把 observation 合并成更大的 forward batch
- Learner：消费完整轨迹做反向传播，并把新参数写入共享 CUDA memory

![Sample Factory 架构图](./images/sample-factory-architecture.png)

_图 2：Sample Factory 论文中的系统架构。它把环境模拟、策略前向、反向训练拆成独立组件，用 FIFO queue 和共享内存降低通信成本。（来源：Sample Factory 论文 [^sf]）_

这张图要看的不是组件名字，而是数据走向：observation 从 rollout worker 经 shared memory 到 policy worker，action 再回到 rollout worker；完整 trajectory 进入 learner；更新后的参数进入 GPU memory，再被 policy worker 取走。后面 LLM RL 的 rollout engine、trainer、buffer，本质上是在更重的模型和更长的序列上复用这套思路。

### 2.3 GPU 物理仿真：Isaac Gym

NVIDIA Isaac Gym 把物理仿真直接搬到 GPU 上，数万个环境并行，核心收益是减少 CPU 物理引擎和 GPU 策略网络之间的逐步数据搬运 [^isaac]。

![Isaac Gym Tensor API 和 GPU 物理仿真管线](./images/isaac-gym-pipeline.png)

_图 3：Isaac Gym 论文中的 GPU pipeline。Learning Framework、Environment Logic、IsaacGym Tensor API 和 PhysX 都围绕 GPU tensor 交换状态、动作和配置，避免每一步都跨 CPU/GPU 拷贝。（来源：Isaac Gym 论文 [^isaac]）_

```
传统方式：  CPU 物理引擎 × 64 环境 → GPU 策略推理
Isaac Gym： GPU 物理仿真 × 4096 环境 + GPU 策略推理
```

| 对比     | CPU 并行 (MuJoCo × 64) | GPU 并行 (Isaac Gym × 4096) |
| -------- | ---------------------- | --------------------------- |
| 采样速度 | ~10K fps               | ~1M fps                     |
| 数据传输 | CPU→GPU 每步           | 零拷贝                      |
| 适用场景 | 少关节机器人           | 人形机器人、灵巧手          |

把前两层放在一起看，它们解决的是传统 RL 里的“环境数据怎么来”：

| 系统                | 定位                      | 数据单位             | 主要瓶颈                                     |
| ------------------- | ------------------------- | -------------------- | -------------------------------------------- |
| Gymnasium VectorEnv | 环境接口/单机批量环境     | transition / episode | Python `env.step()`                          |
| IMPALA              | 分布式 Actor-Learner 架构 | trajectory           | Actor 与 Learner 不同步、policy lag          |
| Sample Factory      | 单机高吞吐异步采样系统    | trajectory buffer    | CPU rollout、GPU policy worker、learner 互等 |
| Isaac Gym           | GPU 物理仿真平台          | GPU tensor state     | CPU/GPU 数据搬运和物理仿真吞吐               |

## 第三层：LLM rollout engine

**代表**：vLLM、SGLang

进入 LLM RL 以后，“环境 step”的含义变了。PPO/GRPO 训练 LLM 时，采样不是让智能体推进一个 Gym 环境，而是让当前语言模型对 prompt 生成回答。也就是说，LLM RL 的 rollout 主要是**自回归文本生成**。

这里的 rollout engine 不是普通意义上的“推理服务”。在在线服务里，模型面对的是用户请求；在 RL 后训练里，模型面对的是训练循环。它不仅要生成文本，还要服从采样参数、记录策略版本、配合奖励计算、等待或接收新权重，并把可训练的数据交给后面的 buffer 和 trainer。

### 3.1 rollout engine 在训练循环里做什么

一个 LLM RL step 可以简化成：

```
Prompt batch
  → Rollout engine 生成 N 条回答
  → Reward / verifier / judge 计算分数
  → Buffer 组织 token、mask、reward、版本号
  → Trainer 计算 PPO/GRPO/RLOO/REINFORCE++ loss
  → 新权重同步回 Rollout engine
```

rollout engine 至少要产出这些信息：

- token ids：训练 loss 需要逐 token 对齐
- attention mask / response mask：区分 prompt、response、padding、截断位置
- finish reason：正常结束、长度截断、stop token、工具调用中断
- sampling metadata：temperature、top-p、top-k、seed、每个 prompt 采样多少条回答
- policy version：这批样本由哪个版本的 actor 生成
- 可选的 logprob：有些系统直接从推理端取 old logprob，有些系统在训练端重新计算，以减少推理/训练 kernel 差异带来的不一致

这就是它和在线推理服务的第一个差别：**在线推理服务交付答案，rollout engine 交付可训练的轨迹样本**。

### 3.2 为什么在线推理服务的思路不够

LLM serving 和 LLM RL rollout 都用推理引擎，但优化目标不一样：

| 维度       | 在线 serving     | RL rollout engine                                     |
| ---------- | ---------------- | ----------------------------------------------------- |
| 第一目标   | 用户延迟和 SLA   | 单位时间产出可训练样本                                |
| 请求形态   | 用户请求随机到达 | trainer 批量下发 prompt，常常每个 prompt 生成多条回答 |
| 输出长度   | 受产品交互约束   | 常有长推理、长代码、长 CoT、长尾样本                  |
| 状态管理   | 通常固定权重服务 | 权重会周期性更新，需要版本管理                        |
| 正确性要求 | 文本结果正确即可 | token、mask、logprob、版本号都要和训练对齐            |
| 调度问题   | p50/p99 latency  | tokens/s、samples/s、长尾拖批、GPU 利用率             |

GRPO 里常见的 `num_generations=8` 或 `16` 会让同一个 prompt 生成多条回答。数学题、代码题、Agent 任务的回答长度差异很大：短样本很快结束，长样本还在 decode。一个 batch 的训练数据通常要等最慢的那批 completion 回来，长尾就会直接拖慢训练。

### 3.3 prefill、decode、KV cache 和长尾

LLM 生成可以拆成两段：

- **Prefill**：处理 prompt，计算初始 KV cache。它更偏计算密集，prompt 长时成本高。
- **Decode**：逐 token 生成 response。它更偏 memory bandwidth 和调度，输出越长越容易被长尾拖慢。

RL rollout 会把这两个问题放大：

1. **共享前缀很多**。同一批 prompt 可能共享 system prompt、few-shot 示例、题面模板，甚至同一个题目会采样多条回答。Prefix cache 命中率直接影响 prefill 成本。
2. **输出长度分布重尾**。多数回答可能几百 token，少数回答会生成几千 token。batch 内最长样本决定何时能交付完整 rollout batch。
3. **KV cache 占用随并发和上下文增长**。KV cache 大小和层数、head 数、序列长度、并发请求数相关。显存不足时，吞吐会突然掉下来，甚至触发抢占或重算。
4. **权重会更新**。serving 可以长期固定一个 checkpoint，RL rollout 侧却需要频繁接收 trainer 的新权重。更新太慢，rollout GPU 空等；更新太快，正在生成的样本可能跨策略版本。

vLLM 的 PagedAttention 把 KV cache 按 block 管理，避免为每条请求预留连续的大块显存，从而提升动态批处理时的显存利用率和吞吐 [^vllm][^vllm_blog]。

![vLLM PagedAttention 将 KV cache 按 block 管理](./images/vllm-pagedattention.gif)

_图 4：vLLM 官方博客中的 PagedAttention 动图。对 LLM RL 来说，rollout 吞吐很大程度取决于 KV cache 管理、连续批处理和长输出调度。（来源：vLLM 官方博客 [^vllm_blog]）_

SGLang 也把这类问题作为核心能力来做：RadixAttention 用于复用共享前缀，router/gateway 负责多实例调度，PD disaggregation 把 prefill 和 decode 拆到不同执行资源上，RL 系统接口则直接关注权重更新、pause generation、deterministic inference 等训练场景需求 [^sglang_rl][^sglang_pd][^sglang_router]。

### 3.4 rollout engine 真正负责的五件事

把 rollout engine 放到 LLM RL 系统里看，它通常要负责五件事。

**第一，批量生成。** 它要把大量 prompt 组织成高吞吐请求，同时支持每个 prompt 生成多条回答。这里的关键不是“能不能 generate”，而是如何把 prefill、decode、padding、stop condition 和 batch 调度组织好。

**第二，KV cache 管理。** PagedAttention、prefix caching、RadixAttention、chunked prefill、KV eviction 这些能力都会直接影响 `tokens/s` 和显存占用。对 RL 来说，prompt 模板和多样本采样会带来很多可复用前缀，cache 命中率不是小优化。

**第三，长尾控制。** RL rollout 常常不是单条请求结束就返回，而是等一个训练 batch 凑齐。少数超长回答会拖住整批数据。工程上常用最大长度、early stop、分桶调度、partial batch return、异步队列来降低长尾影响。

**第四，权重生命周期。** Trainer 更新 actor 后，rollout engine 要接收新权重。这个过程可能涉及张量并行格式、FSDP/Megatron 分片格式、LoRA adapter、GPU 间通信、sleep/wake、pause/resume generation。vLLM 文档专门把 RLHF 场景和 sleep mode、权重同步放在一起讨论 [^vllm_rlhf][^vllm_sleep]。

**第五，版本和一致性。** rollout 侧生成样本时用的是旧策略还是新策略，必须被记录下来。严格 on-policy 时，旧数据要丢弃；异步训练时，旧数据可以保留，但要通过 staleness、importance sampling、KL 或截断权重控制风险。这一部分会在 **[B.2 异步训练架构](./async-training)** 继续展开。

### 3.5 vLLM 和 SGLang 在这一层的差别

vLLM 和 SGLang 都可以作为 rollout engine，但工程气质不同：

| 系统   | 更突出的能力                                                                          | 在 RL rollout 中的意义                                        |
| ------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| vLLM   | PagedAttention、continuous batching、并行采样、prefix caching、sleep mode、RLHF 集成  | 作为通用高吞吐 rollout engine，容易接入 OpenRLHF、veRL 等框架 |
| SGLang | RadixAttention、structured generation、router/gateway、PD disaggregation、RL 系统接口 | 适合长上下文、多轮交互、MoE、SGLang-native 后训练系统         |

OpenRLHF 常见组合是 Ray + vLLM + DeepSpeed；veRL 同时支持 vLLM、SGLang、HF Transformers 等 rollout 后端；slime 则更明确地把 SGLang 当成原生 rollout 层。这样看，vLLM/SGLang 不该和 OpenRLHF/veRL/slime 横向比较：前者是**生成引擎**，后者是**训练编排框架**。

## 第四层：LLM RL 后训练框架

**代表**：OpenRLHF、veRL、slime

OpenRLHF、veRL、slime 才是同一层的东西。它们通常会调用 vLLM 或 SGLang 做 rollout，但自己负责更完整的 RL 训练系统：

- Rollout workers：批量生成回答，接 vLLM、SGLang 或其他推理后端
- Reward/Judge workers：规则判题、奖励模型推理、LLM-as-Judge、代码执行
- Training workers：PPO/GRPO/RLOO/REINFORCE++ loss、优势估计、反向传播、参数更新
- Buffer/Queue：缓存样本，记录策略版本，控制旧数据比例
- Weight sync：把 trainer 的新权重同步到 rollout 侧

如果只看论文里的算法，PPO/GRPO 像几行 loss；但在真实系统里，框架差异主要体现在四个平面：

| 平面                | 要解决的问题                                                     |
| ------------------- | ---------------------------------------------------------------- |
| Rollout plane       | 用哪个推理引擎，如何生成、截断、重试、并发、处理长尾             |
| Reward plane        | 奖励来自规则、RM、Judge、verifier 还是工具环境，是否会成为新瓶颈 |
| Training plane      | 用 DeepSpeed、FSDP、Megatron-LM 还是自研训练栈                   |
| Data / Weight plane | 样本如何入队、是否流式、权重如何同步、旧样本如何处理             |

HybridFlow 论文里的框架对比表很适合做这几个维度的索引。它把 DeepSpeed-Chat、OpenRLHF、NeMo-Aligner 和 HybridFlow 放在同一张表里，比较 parallelism、actor weights、model placement 和 execution pattern [^hybridflow]。

![HybridFlow 论文中的 RLHF 框架对比表](./images/hybridflow-framework-comparison.png)

_图 5：HybridFlow 论文对 RLHF 框架执行模式的比较。OpenRLHF 用分离设备和两份 actor weights 换取生成/训练并行；HybridFlow 进一步强调 zero-redundancy model resharding 和 flexible placement。（来源：HybridFlow 论文 [^hybridflow]）_

下面只在这一层里比较 OpenRLHF、veRL 和 slime。

### OpenRLHF：Ray + vLLM + DeepSpeed

OpenRLHF 的技术报告和 README 把它描述为 Ray + vLLM 分布式架构：Ray 做调度和控制，vLLM 做 rollout 推理，DeepSpeed 做 Actor/Critic/Reward/Reference 等模型训练和推理，Transformers 负责模型格式和状态对接，底层通过 NCCL / CUDA IPC 做高速通信 [^openrlhf][^openrlhf_readme]。

![OpenRLHF Ray + vLLM 分布式架构](./images/openrlhf-architecture.png)

_图 6：OpenRLHF README 中的 Ray + vLLM 架构图。它体现了 LLM RL 的常见拆法：调度层、推理引擎、训练引擎、模型权重格式、GPU 间通信。（来源：OpenRLHF README [^openrlhf_readme]）_

这张图里的关键点是角色边界：

- Ray 负责把 Actor、Critic、Reward、Reference、vLLM engine 等组件调度到不同 GPU 上
- vLLM 负责高吞吐生成，是 rollout 侧核心
- DeepSpeed 负责训练侧的显存优化和分布式反向传播
- Transformers 作为权重格式和模型状态的桥
- NCCL / CUDA IPC 负责权重同步和 GPU 间传输

OpenRLHF 的实用价值在于它把几种常见部署方式做成了显式参数：

| 模式                      | 典型参数                                           | 工程含义                                         | 风险                              |
| ------------------------- | -------------------------------------------------- | ------------------------------------------------ | --------------------------------- |
| Hybrid Engine / colocated | `--train.colocate_all`、`--vllm.enable_sleep`      | 同一组 GPU 在生成和训练之间切换，尽量省卡        | 严格串行，吞吐受 rollout 长尾影响 |
| Async Training            | `--train.async_enable`、`--train.async_queue_size` | rollout 和 training 并发跑，队列越大吞吐越高     | 队列越深，样本越 off-policy       |
| Async + Partial Rollout   | `--train.partial_rollout_enable`                   | 利用 vLLM pause/resume，让权重同步不完全阻塞生成 | in-flight 样本可能混合新旧权重    |

这三个模式正好对应 B.2 中的核心矛盾：省 GPU、严格 on-policy、高吞吐三者很难同时满足。OpenRLHF 倾向于把这些旋钮暴露给用户。研究阶段可以用 colocated 保稳定；生产吞吐阶段再打开 async；如果能接受更复杂的 off-policy 修正，再尝试 partial rollout 和重要性采样校正 [^openrlhf_async]。

### veRL：HybridFlow 与更细的执行流

veRL 是 HybridFlow 论文的开源实现。它强调 single-controller 编排、可组合的 model engine / rollout engine，以及用队列把 rollout 和 training 解耦 [^hybridflow][^verl_readme]。

![veRL 项目架构图](./images/verl-architecture.png)

_图 7：veRL README 中的架构图。图中的 TransferQueue、Rollout Engine、Model Engine 和 CheckpointEngine 对应了 LLM RL 系统里的数据流、推理流、训练流和权重同步。（来源：veRL README [^verl_readme]）_

这张图对理解 LLM RL 很有帮助：训练系统不再只是“一个 trainer 调一个 generate 函数”。Rollout engine 可能接 vLLM、SGLang 或 TensorRT-LLM；Model engine 可能接 FSDP、Megatron-Core 或其他训练后端；TransferQueue 负责把生成样本流式送到训练侧；CheckpointEngine 则负责保存和广播新权重。

veRL 的重点是把 RL 训练抽象成一组可组合 worker。README 中强调 hybrid-controller programming model、flexible device mapping，以及与 FSDP/FSDP2、Megatron-LM、vLLM、SGLang、HF Transformers 等已有 LLM infra 的模块化集成 [^verl_readme]。这意味着：

- 训练侧可以根据模型规模选择 FSDP 或 Megatron 风格的切分
- 推理侧可以根据场景选择 vLLM、SGLang 或 HF Transformers
- rollout、reference logprob、actor update、critic update 等步骤可以在统一控制器下组合
- 异步、off-policy、多模态/机器人等实验性方向可以继续接入同一套执行流

如果把 OpenRLHF 看成“Ray + vLLM + DeepSpeed 的工程化 RLHF 框架”，veRL 更像是“把 RL 训练流抽象出来，允许不同 backend 拼装”的系统。它适合需要改训练流程、换 rollout engine、插入自定义 reward、支持 VLM/multi-turn/tool calling，或研究新算法的团队。

### slime：Megatron + SGLang + Data Buffer

slime 的定位更偏向大规模 RL scaling。它的 README 把核心能力概括为两点：用 Megatron + SGLang 支持高性能训练，以及通过自定义数据生成接口和 server-based engine 支持灵活 rollout [^slime_readme]。

![slime 官方架构图](./images/slime-architecture.png)

_图 8：slime README 中的架构图。训练侧是 Megatron，推理侧是 SGLang server/router，中间用 data buffer 管理 prompt、rollout 数据和自定义生成逻辑。（来源：slime README [^slime_readme]）_

slime 的结构非常直接：

- **training (Megatron)**：从 Data Buffer 读取训练数据，完成训练后把新参数同步到 rollout 模块
- **rollout (SGLang + router)**：生成新数据，包含 reward/verifier 输出，并写回 Data Buffer
- **data buffer**：管理 prompt 初始化、自定义数据和 rollout 生成方法

它和 OpenRLHF / veRL 最大的差异，是把 SGLang 当成原生推理层，而不是一个普通可替换插件。slime 文档明确强调：内部以 server 模式启动 SGLang，SGLang 参数可以通过 `--sglang-*` 直接传递，并提供 `--debug-rollout-only` 用来单独调 rollout 性能 [^slime_intro]。训练侧也类似：Megatron 参数可以透传，支持 TP/PP/EP/CP 等并行策略，并提供 `--debug-train-only` 调试训练部分 [^slime_intro]。

slime README 里列出的下游项目也能说明它的定位：Relax 把 Actor、Rollout、ActorFwd、Reference、Advantage 等组件拆到独立 GPU 集群，并用 TransferQueue 和异步 DCS 做权重同步；APRIL 专门优化 rollout 长尾；TritonForge、RLVE、P1 等则把 slime 用到代码生成、可验证环境和物理推理等任务上 [^slime_readme]。这说明 slime 的 Data Buffer 不只是经验回放表，而是把 prompt 初始化、自定义生成、verifier/reward 输出和训练数据组织到同一个后训练数据平面里。

slime 的 release note 还讨论了非常工程化的问题：RL 推理延迟不能简单靠加 GPU 解决，因为还要等最长样本 decode 完；过大的 inference batch 又会带来 off-policy 问题 [^slime_release]。因此它关注 KV cache 空间、MoE fp8 rollout、DeepEP、Megatron offload、NCCL group 重建等底层优化。这已经超出“写一个 PPO loop”很多，是真正工业 RL 训练系统会碰到的问题。

## 怎么选：先定位瓶颈，再选系统层

| 你在做什么                                | 第一优先问题                                            | 对应层级           | 典型选择                |
| ----------------------------------------- | ------------------------------------------------------- | ------------------ | ----------------------- |
| CartPole / LunarLander / 小型控制实验     | 环境接口和批量环境                                      | 环境接口与单机采样 | Gymnasium VectorEnv     |
| Atari / ViZDoom / DeepMind Lab 高吞吐训练 | CPU 环境、policy forward、learner 怎么不互等            | 经典 RL 采样系统   | IMPALA / Sample Factory |
| 机器人仿真、灵巧手、人形控制              | 物理仿真和策略网络之间如何减少拷贝                      | GPU 仿真系统       | Isaac Gym               |
| LLM RL 原型                               | 生成回答的推理吞吐                                      | LLM rollout engine | vLLM / SGLang           |
| 7B-70B LLM PPO/GRPO/RLOO                  | rollout、reward、training、buffer、weight sync 如何编排 | LLM RL 后训练框架  | OpenRLHF / veRL / slime |

如果只记一个判断标准：**瓶颈在哪里，系统边界就应该切在哪里**。瓶颈是环境 step，就并行环境；瓶颈是物理仿真，就把仿真搬到 GPU；瓶颈是文本生成，就独立 rollout engine；瓶颈是奖励判题，就把 reward/judge 独立成服务；瓶颈是权重同步，就优化通信和策略版本管理。

也就是说，正确比较方式不是：

> Gymnasium vs veRL vs slime

而是：

> Gymnasium 属于环境接口；IMPALA/Sample Factory/Isaac Gym 属于经典 RL 采样/仿真；vLLM/SGLang 属于 LLM rollout engine；OpenRLHF/veRL/slime 属于 LLM RL 后训练框架。

OpenRLHF、veRL、slime 的选择也可以这样看：

| 框架     | 训练后端                     | Rollout 后端                    | 调度/数据通道                                 | 更适合的场景                                               |
| -------- | ---------------------------- | ------------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| OpenRLHF | DeepSpeed / Transformers     | vLLM                            | Ray + Hybrid Engine / Async Queue             | 快速跑 PPO/GRPO/RLOO，接受框架给定的 RLHF pipeline         |
| veRL     | FSDP / Megatron 等可组合后端 | vLLM / SGLang / HF Transformers | single-controller + TransferQueue/worker 组合 | 需要改训练流、做多 backend 实验、多轮 rollout 或研究新算法 |
| slime    | Megatron                     | SGLang + router                 | Data Buffer + 自定义 data generation          | 已有 Megatron 体系，重度依赖 SGLang/MoE/大规模 rollout     |

模型如何分布式地切到多张卡上，详见 **[B.3 分布式并行策略](./parallelism)**。Agentic RL（多轮交互 + 工具调用）的基础设施需求更高，详见 **[B.4 Agentic RL 基础设施](./agentic-rl-infra)**。

## 参考文献

[^gym_vec]: Gymnasium Documentation, [Vector Environments (SyncVectorEnv / AsyncVectorEnv)](https://gymnasium.farama.org/api/vector/).

[^impala]: Espeholt L, Soyer H, Munos R, et al. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://proceedings.mlr.press/v80/espeholt18a.html), ICML 2018.

[^sf]: Petrenko A, Huang Z, Kumar T, Sukhatme G S, Koltun V. [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning](https://arxiv.org/abs/2006.11751), ICML 2020.

[^isaac]: Makoviychuk V, Wawrzyniak L, Guo Y, et al. [Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning](https://research.nvidia.com/labs/srl/publication/makoviychuk-2021-isaac/), NeurIPS 2021 (Datasets and Benchmarks).

[^vllm]: Kwon W, Li Z, Zhuang S, et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023. (vLLM / PagedAttention)

[^vllm_blog]: vLLM Team, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm), 2023.

[^vllm_rlhf]: vLLM Documentation, [Reinforcement Learning from Human Feedback](https://docs.vllm.ai/en/stable/training/rlhf/), 2026.

[^vllm_sleep]: vLLM Documentation, [Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/), 2026.

[^sglang_rl]: SGLang Documentation, [SGLang for RL Systems](https://docs.sglang.io/advanced_features/sglang_for_rl.html), 2026.

[^sglang_pd]: SGLang Documentation, [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation), 2026.

[^sglang_router]: SGLang Documentation, [SGLang Router](https://docs.sglang.io/advanced_features/router.html), 2026.

[^hf_async]: Hugging Face Blog, [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^pt_primer]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

[^openrlhf]: OpenRLHF Team, [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), 2024. [GitHub](https://github.com/OpenRLHF/OpenRLHF).

[^hybridflow]: Sheng G, Zhang C, Ye Z, et al. [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256), 2024. [veRL GitHub](https://github.com/verl-project/verl).

[^openrlhf_readme]: OpenRLHF Project, [Architecture Foundation: Ray + vLLM Distribution](https://github.com/OpenRLHF/OpenRLHF#architecture-foundation-ray--vllm-distribution), README.

[^openrlhf_async]: OpenRLHF Documentation, [Async Training & Partial Rollout](https://openrlhf.readthedocs.io/en/latest/async_training.html), 2026.

[^verl_readme]: veRL Project, [README and architecture diagram](https://github.com/verl-project/verl), 2026.

[^slime_readme]: THUDM slime Project, [slime: An LLM post-training framework for RL Scaling](https://github.com/THUDM/slime), README.

[^slime_intro]: slime Documentation, [slime：为 RL Scaling 设计的 SGLang-Native 后训练框架](https://thudm.github.io/slime/zh/blogs/introducing_slime.html), 2025.

[^slime_release]: slime Documentation, [v0.1.0: Redefining High-Performance RL Training Frameworks](https://thudm.github.io/slime/blogs/release_v0.1.0.html), 2025.
