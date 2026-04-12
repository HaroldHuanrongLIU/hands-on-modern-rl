# Part 4: 进阶与前沿 — 知识总结

## 这一 Part 我们学了什么？

最后五章覆盖了现代 RL 的前沿方向，每一章都代表了 RL 从实验室走向真实世界的关键一步。学完后，你应该掌握以下核心知识点：

- **连续控制**：DDPG 用确定性策略处理连续动作，TD3 用截断双 Q + 延迟更新 + 目标平滑解决高估问题，SAC 用最大熵目标同时保证探索和利用，扩散策略用生成模型支持多模态动作分布。
- **RLHF 工程流水线**：混合奖励函数 $R = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}}$，奖励黑客的防御，RLAIF 用 AI 替代人类标注。
- **VLM RL**：差异化学习率（视觉编码器 lr 更小），奖励归因难题（视觉 token vs 文本 token 谁的错），视觉幻觉惩罚。
- **Agentic RL**：ORM（只在最后给奖励）vs PRM（每步给奖励），工具调用 RL 的"SFT 教格式 + RL 教策略"流程。
- **未来趋势**：测试时计算（Tree of Thought / MCTS），自博弈（Generator-Judge 对抗），离线 RL（CQL/IQL/DT 从固定数据集学习），多智能体 RL。

下面让我们逐章复习这些内容。

## 第 9 章：连续动作控制——当动作不再是"左或右"

### 从离散到连续：DQN 的局限

DQN 能玩转 CartPole 和 Atari，因为动作是有限的：左/右，或者手柄的几个方向。但机械臂的每个关节能施加 $[-10, 10]$ N·m 之间任意力矩，6 个关节联合起来有无限种组合。对每种组合都算一个 Q 值是不可能的。

DDPG（Deep Deterministic Policy Gradient）给出了第一条出路。它借鉴了 DQN 的经验回放和目标网络，但策略从"选最优动作"变成了"直接输出动作值"。具体来说，Actor 网络 $\mu_\theta(s)$ 直接输出连续动作，Critic 网络 $Q_\phi(s, a)$ 告诉它这个动作有多好，梯度从 Critic 反传到 Actor：

$$\nabla_\theta J \approx \mathbb{E}\left[\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

直觉上：Critic 画出一张"地形图"（Q 值关于动作的梯度），Actor 顺着上坡方向走。因为是确定性策略（每次对同一个状态输出同一个动作），需要外加噪声来探索。

### TD3：三个技巧让 DDPG 更稳定

DDPG 在实践中容易高估 Q 值，导致策略崩溃。TD3 用了三个精巧的改进。

**截断双 Q 学习**训练两个 Critic，取它们的较小值作为 TD Target，从根源上抑制高估。就像买东西时货比两家，取便宜的报价——虽然可能低估，但至少不会花冤枉钱：

```python
# TD3 的 TD Target 计算
with torch.no_grad():
    noise = torch.clamp(torch.randn_like(action) * 0.2, -0.5, 0.5)
    smoothed_action = torch.clamp(target_actor(next_state) + noise, -1, 1)
    target_q1 = target_critic1(next_state, smoothed_action)
    target_q2 = target_critic2(next_state, smoothed_action)
    td_target = reward + gamma * torch.min(target_q1, target_q2) * (1 - done)
```

**延迟策略更新**让 Actor 每 $d$ 步才更新一次（通常 $d=2$），先让 Critic 学准确了再更新策略。**目标策略平滑**在目标动作上加一点噪声，防止 Critic 对窄尖峰过拟合。

### SAC：用熵鼓励探索

SAC（Soft Actor-Critic）在 TD3 的基础上做了一个优雅的改变：把策略熵直接加入目标函数。

$$J_{\text{max-ent}} = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

这里的 $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ 是策略的熵，$\alpha$ 控制熵的重要性。熵越大，策略越"随机"，探索越充分。SAC 会自动调节 $\alpha$：如果策略太确定了，就增大 $\alpha$ 鼓励探索；如果太随机了，就减小 $\alpha$ 让策略更聚焦。

SAC 使用**重参数化技巧** $a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon$（$\epsilon \sim \mathcal{N}(0, I)$）来让采样过程可微，从而能端到端地优化策略参数。这比 DDPG 的确定性策略多了"随机性"，反而让训练更稳定。

### 扩散策略：用生成模型做控制

DDPG、TD3、SAC 的策略都输出一个动作（或一个高斯分布），本质上是单峰的。但有些任务需要"在多种截然不同的动作之间切换"——比如机器人抓取，从上方抓和从侧面抓可能同样好。扩散策略用扩散模型来生成动作序列，天然支持多模态分布，适合这种需要多样化行为的场景。

## 第 10 章：RLHF 完整流水线——从理论到工程

### 奖励函数设计

在真实的 RLHF 系统中，奖励函数远不止"一个 RM 模型打分"这么简单。通常是一个混合公式：

$$R_{\text{total}} = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}} + \gamma R_{\text{correctness}}$$

其中 $R_{\text{RM}}$ 是奖励模型的主观评分，$R_{\text{format}}$ 是格式合规性（如 JSON 是否合法），$R_{\text{length}}$ 控制回答长度，$R_{\text{correctness}}$ 是可验证的正确性（如代码能否通过测试）。

奖励函数的粒度也很重要。最简单的是 **Sequence-level**：整个回答一个分数。更精细的是 **Step-level**：对推理的每一步打分（这就是过程奖励模型 PRM）。最精细的是 **Token-level**：每个 token 都有一个奖励信号。

### 奖励黑客：模型学会骗分

RL 训练中的一个经典问题是**奖励黑客**（Reward Hacking）。模型很聪明，它会找到奖励函数的漏洞来骗取高分，而不是真正提升回答质量。常见表现包括：

- **长度膨胀**：模型发现写长回答更容易得高分，于是写一堆废话
- **重复刷分**：反复输出某些奖励模型偏好的高频短语
- **格式作弊**：用 markdown 标题和列表把格式弄得很漂亮，但内容空洞

防御手段包括：分析长度与奖励的相关性（如果模型越长分越高，可能出了问题）、统计高频短语、定期做人工评估对比。在 PPO 训练中，KL 散度惩罚 $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 也是一道防线——如果模型的输出分布偏离参考模型太远，就会被惩罚。

### RLAIF：用 AI 替代人类

标注偏好数据成本很高。RLAIF（RL from AI Feedback）的思路是用一个更强大的模型（如 GPT-4）来替代人类做偏好标注。Constitutional AI 更进一步：让模型自己对生成的回答做批评和修订（Self-Critique），然后用这些修订前后的对比作为偏好数据。这形成了一个数据闭环：部署模型 → 收集用户反馈 → 识别薄弱环节 → 用 AI 构造偏好数据 → 重新训练。

## 第 11 章：VLM 强化学习——让视觉模型学会推理

### 视觉带来的新挑战

给多模态模型做 RL，问题比纯文本复杂得多。第一个挑战是**奖励归因**：如果模型回答"图中有三只猫"但实际只有两只，错误出在视觉理解还是文本推理？视觉 token 和文本 token 混在一起，很难区分责任。这就像考试时，你不知道是题目没看清还是计算错了——两种错误需要的补救方式完全不同。

第二个挑战是**视觉幻觉**（Hallucination）。模型可能"看到"图片中不存在的东西——这在纯文本 RL 中不是问题，但在 VLM 中很严重。解决方法之一是加一个幻觉惩罚：

$$R_{\text{hallucination}} = -\lambda \cdot \mathbb{1}(\text{描述与图像内容不符})$$

第三个挑战是**安全性与探索的矛盾**。自动驾驶场景中，你不能让模型"探索"闯红灯会怎样。

### VLM GRPO 训练

VLM 的 RL 训练核心仍然是 GRPO，但有一个关键差异：**差异化学习率**。视觉编码器的学习率通常是文本解码器的 1/10。为什么？因为视觉编码器已经在海量图像上预训练好了（比如在 billions 张图片上），改动太大会破坏已经学到的视觉特征；而文本解码器需要学习如何根据视觉信息来调整输出，需要更大的学习率。

```python
optimizer = AdamW([
    {"params": model.vision_encoder.parameters(), "lr": 1e-6},   # 视觉：小 lr
    {"params": model.text_decoder.parameters(), "lr": 1e-5},     # 文本：大 lr
    {"params": model.projector.parameters(), "lr": 1e-5},        # 对齐层：大 lr
])
```

## 第 12 章：Agentic RL——让模型学会使用工具

### 多轮交互与信用分配

传统的 RL 对齐是"单轮"的：给一个 prompt，模型生成一个回答，给一个奖励。但真实的 Agent 是多轮交互的——模型可能需要搜索网页、运行代码、查看结果，然后再决定下一步。这种多轮场景带来了**信用分配**的难题：如果最终任务失败了，是哪一步出的问题？

**ORM**（Outcome Reward Model）最简单：只在最后给一个 0 或 1 的奖励，前面所有步骤的奖励都是 0。缺点是信号太稀疏——就像考试只告诉你是"及格"还是"不及格"，不告诉你哪道题做错了。**PRM**（Process Reward Model）更精细：给每一步都打分，提供密集的学习信号。但 PRM 的训练数据标注成本更高——你需要对每一步的好坏做出判断。

### 工具调用的 RL 训练

Web Agent 和 Code Agent 是 Agentic RL 的两个典型场景。Web Agent 需要学会在网页上点击、输入、导航；Code Agent 需要学会编写代码、运行测试、根据错误信息修改。奖励函数通常设计为：

$$R_{\text{total}} = R_{\text{task}} - \lambda_{\text{efficiency}} \cdot T - \lambda_{\text{format}} \cdot \mathbb{1}(\text{格式错误})$$

其中 $R_{\text{task}}$ 是任务完成度（如测试通过率），$T$ 是交互轮数（鼓励高效），格式错误项惩罚模型生成无效的工具调用。

训练流程通常是"SFT 教格式 + RL 教策略"：先用监督学习教会模型如何调用工具（格式），再用 RL 教会模型何时、如何使用工具来完成任务（策略）。这就像先学开车的基本操作（方向盘、油门、刹车），再学在实际道路上怎么开。

## 第 13 章：未来趋势——从 CartPole 到自进化系统

### 测试时计算：让模型"想一会儿再回答"

传统的 RL 训练提升的是模型的"能力"，但模型的"推理时间"是固定的。测试时计算（Test-Time Compute）的核心思想是：给模型更多推理时间，它就能给出更好的回答。就像考试时，想得越久往往答得越好。

Tree of Thought 让模型同时探索多条推理路径，保留最有希望的分支。MCTS（蒙特卡洛树搜索）把棋类 AI 的搜索策略搬到语言空间中。OpenAI o1/o3 和 DeepSeek-R1 则展示了另一种可能：通过 RL 训练，模型学会了在内部隐式地搜索和验证，不需要显式的搜索树——它学会了"在脑子里想"。

### 自博弈与自进化

DeepSeek-R1-Zero 的实验揭示了一个惊人的现象：一个基础模型，不加任何 SFT，只用 RLVR 训练，就能涌现出 chain-of-thought 推理能力。模型会自己学会"想一会儿再回答"——在输出中自然出现反思、回溯、验证等行为。

这指向了一个更宏大的方向——**自博弈**（Self-Play）。Generator-Judge 对抗训练：Generator 试图生成让 Judge 给高分的回答，Judge 试图准确评估回答质量。两者相互竞争，共同提升——就像围棋中的自我对弈，通过不断和过去的自己下棋来变强。配合在线学习循环——部署 → 收集反馈 → 识别薄弱环节 → 构造数据 → 重新训练——模型可以实现持续的**自进化**。

### 离线 RL 与多智能体

不是所有场景都能在线与环境交互（你不能让自动驾驶在真实道路上"探索"）。**离线强化学习**（Offline RL）从固定的历史数据集中学习策略，不能主动尝试新动作。CQL（Conservative Q-Learning）通过惩罚对未见过状态的 Q 值来避免过估计；IQL（Implicit Q-Learning）用 expectile regression 直接从数据中提取最优策略；Decision Transformer 把 RL 问题转化为序列建模——给定"期望回报"，生成达成该回报的动作序列。

多智能体 RL（MARL）则研究多个智能体在共享环境中同时学习和交互。每个智能体 $i$ 的目标函数为 $J_i(\pi_1, \ldots, \pi_n) = \mathbb{E}[\sum_t \gamma^t r_i^t]$，不仅取决于自己的策略，还取决于其他智能体的策略。这引入了非平稳性（其他智能体也在变）、信用分配（团队胜利了谁的功劳）等全新的挑战。

## 小结

全书从 CartPole 出发，经过 MDP、DQN、策略梯度、PPO、DPO、GRPO，最终到达 Agentic RL 和自进化系统。这条路径上的每一个概念都环环相扣：贝尔曼方程的递归思想贯穿 Q-Learning 和 DQN；策略梯度定理是 REINFORCE、Actor-Critic 和 PPO 的共同基础；PPO 的裁剪机制被 GRPO 继承；DPO 的隐式奖励启发了 RLVR 的规则验证。强化学习不是一个孤立的学科，而是一套解决"从经验中学习决策"的统一方法论。

> 回到 [前言](/preface/intro) 或前往 [附录](/appendix_common_pitfalls/intro) 继续深入学习。
