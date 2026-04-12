# Part 3: LLM 时代 — 知识总结

## 这一 Part 我们学了什么？

这两章是全书的核心转折点。我们把前六章积累的 RL 理论，应用到了大语言模型对齐的真实场景中。学完后，你应该掌握以下核心知识点：

- **DPO 隐式奖励**：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。奖励函数藏在策略模型的概率比值里，不需要单独训练奖励模型。
- **DPO 损失**：让好回答的隐式奖励高于坏回答，把 RL 问题转化为分类问题。两个模型搞定，不需要 Critic 和 RM。
- **DPO 家族**：KTO 只需好/坏标签（不需配对），SimPO 去掉参考模型，ORPO 合并 SFT 和对齐。
- **GRPO 组内归一化**：$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$。对同一个 prompt 生成 $k$ 个回答，用组内 $z$-score 替代 Critic。只需要 1 个模型。
- **DAPO 四大改进**：Clip-Higher（给低概率动作更多上升空间）、Token 级损失（按 token 计算梯度）、动态采样（过滤已掌握的 prompt）、Overlong 过滤。
- **RLVR**：在数学、代码等客观任务中用规则验证器（答案匹配、单元测试）取代人工标注的奖励模型。DeepSeek-R1-Zero 证明纯 RLVR 训练就能涌现推理能力。

下面让我们逐章复习这些内容。

## 第 7 章：DPO——绕过奖励模型的魔法

### 从 RLHF 到 DPO：一个关键的数学等价

传统 RLHF 的流程很复杂：先用人类偏好数据训练一个奖励模型（Reward Model），再用 PPO 训练语言模型来最大化这个奖励，同时加一个 KL 散度惩罚防止模型跑偏。整个过程需要四个模型同时跑，显存和工程复杂度都很高。

DPO 的突破在于一个优美的数学发现：在 KL 约束下的 RL 目标

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

的闭式最优解恰好是

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

把两边取对数再整理，奖励函数可以用策略的概率比来表达：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

由于 $Z(x)$ 只依赖提示词 $x$ 不依赖回答 $y$，在 Bradley-Terry 偏好模型 $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$ 中，$Z(x)$ 会自动消掉。于是我们得到了 DPO 损失：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

这个推导可能看起来有点复杂，但它的直觉很清晰：$\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$ 衡量的是"模型现在比训练前更喜欢这个好回答多少"，减去对应的坏回答项，得到的差值越大，模型区分好坏的能力越强，损失越小。

### DPO 训练实战

和传统 RLHF 需要 PPO 的复杂训练循环不同，DPO 的训练就是一个标准的监督学习过程：

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    beta=0.1,                      # KL 惩罚系数
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    num_train_epochs=3,
)

trainer = DPOTrainer(
    model=model,                   # 正在训练的策略模型
    ref_model=ref_model,           # 冻结的参考模型
    args=training_args,
    train_dataset=preference_data, # 包含 prompt, chosen, rejected 三列
    processing_class=tokenizer,
)
trainer.train()
```

训练过程中，你可以监控 **Reward Margin**——好回答的隐式奖励与坏回答的隐式奖励之差。这个值越大，说明模型越能区分好坏。如果 Reward Margin 停滞不前甚至下降，可能意味着偏好数据的质量有问题，或者 $\beta$ 设置不合理。

### DPO 家族

DPO 打开了"绕过 RM 直接对齐"的大门，之后涌现了一系列变体。**KTO** 只需要"好"或"坏"的单个标签（不需要配对的好/坏回答），降低了数据标注成本——比如你只需要告诉模型"这个回答好"，不需要同时提供一个坏回答。**SimPO** 去掉了参考模型 $\pi_{\text{ref}}$，改用序列长度的对数来归一化，进一步简化了训练流程。**ORPO** 更激进——把 SFT 和对齐合并为一步，用损失函数中的 odds ratio 同时完成"学会回答格式"和"学会区分好坏"。

## 第 8 章：GRPO、DAPO 与 RLVR——从策略优化到可验证奖励

### GRPO：用组内统计替代 Critic

PPO 需要一个 Critic 网络来估计优势函数 $A(s,a)$。在 LLM 场景中，这个 Critic 本身就是一个大模型，显存开销巨大。GRPO（Group Relative Policy Optimization）提出了一个巧妙的替代方案：

对同一个 prompt $x$，让模型生成 $k$ 个回答 $y_1, y_2, \ldots, y_k$，用奖励函数（可以是 RM 也可以是规则）给每个回答打分 $r_1, r_2, \ldots, r_k$。然后计算组内归一化的优势：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$$

这个归一化为什么有效？关键在于"相对比较"。同一个 prompt 下的 $k$ 个回答共享同一个参考水平（组内均值），优势的正负直接反映"比平均好"还是"比平均差"，大小反映好了多少。这和 PPO 用 Critic 计算 $A = Q - V$ 的逻辑完全一致——都是"比基准好多少"——只是 GRPO 用组内统计代替了 Critic 网络。

GRPO 的更新继承了 PPO 的裁剪机制，只是用组内 $z$-score 替换了 GAE 计算的优势：

```python
# GRPO 核心循环（简化版）
for prompt in prompts:
    # 组采样：对同一个 prompt 生成 k 个回答
    responses = [generate(policy, prompt) for _ in range(k)]
    scores = [reward_fn(prompt, resp) for resp in responses]

    # 组内归一化（替代 Critic）
    mean_r, std_r = np.mean(scores), np.std(scores)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in scores]

    # PPO 裁剪更新
    for resp, adv in zip(responses, advantages):
        ratio = torch.exp(new_log_prob(resp) - old_log_prob(resp))
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        loss = -torch.min(ratio * adv, clipped * adv)
        update(loss)
```

### DAPO：让 GRPO 更强的四个改进

DeepSeek 团队在 GRPO 基础上提出了 DAPO，包含四个关键改进。

**Clip-Higher** 把上下裁剪范围解耦。标准 PPO 的裁剪是对称的 $[1-\varepsilon, 1+\varepsilon]$，但在 GRPO 中，一个低概率的好动作需要大幅提升概率才能被选中，对称裁剪会限制它的上升空间。DAPO 用更大的上界裁剪 $\varepsilon_{\text{high}} > \varepsilon_{\text{low}}$，给探索更多呼吸空间。

**Token 级损失** 不再以整个序列为单位计算梯度，而是按 token 求和：$\nabla_\theta \mathcal{L} = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t$。想象一个 500 token 的回答，如果只有最后一步算错了，用序列级损失的话整个回答的梯度都会被拉低；用 token 级损失，只有出错的那个 token 附近会受到惩罚。

**动态采样** 自动检测并过滤掉模型已经"掌握"的 prompt（所有 $k$ 个回答都答对）。这保证了训练数据始终有一定的难度梯度，避免在简单题上浪费算力。

**Overlong 过滤** 移除超出长度限制的样本，防止它们带来的奖励偏差干扰训练。

### RLVR：用规则验证取代人工标注

GRPO/DAPO 最大的威力在于它不再依赖训练一个奖励模型——只要有一个能打分的东西就行。在数学推理、代码生成等客观任务中，这个打分者可以是一个**规则验证器**（verifier）：数学题直接比对最终答案，代码题跑一遍单元测试。这就是 RLVR（Reinforcement Learning with Verifiable Rewards）。

RLVR 的意义是革命性的。传统的 RLHF 需要大量人工标注偏好数据（"这个回答比那个好"），成本高且难以扩展。RLVR 用客观规则替代了主观标注，让 RL 训练可以几乎零成本地扩展到海量题目上。

DeepSeek-R1-Zero 的实验更是证明了一个惊人的结论：一个基础模型，不加任何 SFT，只用 RLVR 训练，就能涌现出 chain-of-thought 推理能力——模型会自己学会"想一会儿再回答"，在输出中自然出现反思、回溯、验证等行为。这说明 RL 不仅能让模型"听话"，还能让模型"变聪明"。

## 小结

Part 3 展现了一条清晰的演进路线：RLHF 需要 4 个模型 → DPO 用隐式奖励砍到 2 个 → GRPO 用组内归一化砍到 1 个 → RLVR 用规则验证彻底去掉 RM。每一步都是"用更简单的机制替代更复杂的组件"，但数学上保持等价甚至更强。

同时，我们也看到了 RL 在 LLM 时代的新角色：不再只是"对齐人类偏好"的工具，而是"激发模型推理能力"的关键技术。从 PPO 到 GRPO，从人工标注到规则验证，RL 正在大模型的后训练阶段扮演越来越核心的角色。

> **下一站**：[Part 4: 进阶与前沿](/chapter09_continuous_control/intro)
