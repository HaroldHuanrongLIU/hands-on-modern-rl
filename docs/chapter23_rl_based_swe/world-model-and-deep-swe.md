# 23.2 Code World Model 与 DeepSWE

上一节我们看到 Meta SWE-RL 的核心瓶颈——**长 horizon 训练不稳定**。16 步以上的 trajectory，RL 的 credit assignment 难以学习。

更深层的问题是：**每次 rollout 都要跑真实测试，速度慢、成本高**。一个 trajectory 涉及多次 `pytest` 调用，每次几秒到几分钟。如果一次 RL 训练需要 100 万次 rollout，总时间可能几周。

2025 年下半年出现了两个突破方向：

- **Code World Model（CWM）**：训练一个模型"模拟"代码执行，避免真实测试
- **DeepSWE**：用 world model + 长序列 RL 训练 deep agent

这一节我们详细讨论这两个方向。

## 12.3.1 Code World Model（CWM）

[Code World Model](https://arxiv.org/abs/2503.02561)（CWM，2025.03）的核心思想：**把代码执行建模为 MDP，训练一个 world model 预测代码的状态变化**。

### CWM 的 MDP 定义

把 SWE 任务建模为 MDP：

| MDP 元素                      | SWE 对应                                        |
| ----------------------------- | ----------------------------------------------- |
| 状态 $s_t$                    | 仓库代码 + 当前修改历史 + 测试结果              |
| 动作 $a_t$                    | 模型的下一步（读文件、改代码、跑测试）          |
| 转移 $T(s_{t+1} \| s_t, a_t)$ | 代码执行——文件改动后状态如何变化                |
| 奖励 $r_t$                    | 每步的反馈（中间状态）+ 最终 reward（测试通过） |

### World Model 的训练

CWM 训练一个独立的 **world model** $\hat{T}$：

$$\hat{T}(s_{t+1} | s_t, a_t) \approx T(s_{t+1} | s_t, a_t)$$

这个 world model 是一个 LLM，输入 $(s_t, a_t)$，输出 $s_{t+1}$。

训练数据：

- 从真实 SWE 任务收集 trajectory
- (s*t, a_t, s*{t+1}) 三元组作为训练样本
- 让 world model 学会"给定当前代码状态和动作，预测下一步状态"

### CWM 的训练流程

```text
┌────────────────────────────────────────────────────────────┐
│ Phase 1: World Model 预训练                                │
│   - 从真实 SWE 任务收集 trajectory                         │
│   - 训练 world model 预测代码状态变化                       │
├────────────────────────────────────────────────────────────┤
│ Phase 2: RL with World Model                              │
│   - Policy 与 world model 交互                            │
│   - World model 快速模拟"代码执行"                          │
│   - 不需要真实测试，速度快 100 倍                          │
├────────────────────────────────────────────────────────────┤
│ Phase 3: 真实测试 fine-tune                                │
│   - 用 world model 训练后的 policy 在真实环境做最后 RL      │
│   - 修正 world model 与真实环境的偏差                      │
└────────────────────────────────────────────────────────────┘
```

### CWM 的优势

**优势一：速度快**

World model 是一个 LLM forward——几毫秒。真实测试是几秒到几分钟。**CWM 让训练速度提升 100-1000 倍**。

**优势二：可以模拟失败**

World model 可以模拟"如果这样改，会发生什么"——policy 可以在 world model 里大量探索失败模式，学习避免。

**优势三：数据效率高**

World model 学到代码执行的"规律"——这些规律可以泛化到新任务。

### CWM 的局限

**局限一：World model 的准确性**

World model 是个 LLM，会错。如果它预测错了"代码执行结果"，policy 学到错误的策略。

工业实践中的缓解：**定期用真实测试校正 world model**——每 N 步 rollout 用真实测试 ground truth 校正。

**局限二：复杂依赖**

代码执行涉及复杂依赖（库版本、环境变量、外部服务）。World model 难以完全模拟这些。

**局限三：训练成本**

训练 world model 本身需要大量 trajectory 数据和算力——比直接训练 policy 复杂。

### CWM 与 model-based RL 的关系

CWM 是 model-based RL 在 SWE 领域的应用。经典 model-based RL（如 MuZero、Dreamer）已经在游戏、控制任务上证明了价值。CWM 把这个思想带到 LLM + SWE 领域。

参考：[第 7 章 model-based RL](../chapter10_ppo/rl-long-horizon-planning) 和 [第 14 章 future trends / model-based RL](../chapter28_vla/embodied-intelligence/model-based-rl/)。

## 12.3.2 DeepSWE 与 长 horizon agent 的 RL

[DeepSWE](https://arxiv.org/abs/2508.19298)（字节 Seed，2025.08）是另一个 SWE-RL 突破。它的核心贡献是：**用 verifiable reward 训练长 horizon agent（32 步以上 trajectory）**。

### DeepSWE 的核心思路

DeepSWE 的关键洞察：**长 horizon RL 不稳定的根本原因是 credit assignment 难**。一个 32 步 trajectory 只有最终测试 reward，怎么把这个 reward 反向传播到 32 步？

DeepSWE 用三个技巧解决：

**技巧一：Step-level Reward Shaping**

不是只有最终 reward，而是给每步一个 shaping reward：

```python
def deep_swe_reward(trajectory, final_test_result):
    # 基础 reward：最终测试结果
    base_reward = 1.0 if final_test_result else 0.0

    # Shaping reward：每步的"贡献度"
    step_rewards = []
    for step in trajectory:
        # 用 LLM judge 评估这一步是否"有意义"
        step_quality = llm_judge(step)
        step_rewards.append(step_quality)

    # 总 reward = base + sum(step rewards)
    return base_reward + sum(step_rewards) * 0.1
```

这种 shaping 让模型在每步都能得到反馈，避免 credit assignment 的困难。

**技巧二：Value Model**

DeepSWE 重新引入 value model（与 VAPO 思路一致）——[参考第 9 章 VAPO](../chapter18_grpo/grpo-family)。

Value model $V_\phi(s_t)$ 估计当前状态的"未来 reward 期望"。这让 RL 可以用 GAE 做 credit assignment：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots$$

其中 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$。

**技巧三：Hierarchical RL**

把长 trajectory 分层：

- **高层 policy**：决定"接下来要修哪个文件"（粗粒度）
- **低层 policy**：决定"具体怎么改这个文件"（细粒度）

高层用稀疏 reward（最终测试），低层用密集 reward（每步 shaping）。

### DeepSWE 的训练流程

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 数据收集                                        │
│   - 用 SFT 模型在 SWE-bench 上 rollout                  │
│   - 收集 32-64 步 trajectory                            │
├──────────────────────────────────────────────────────────┤
│ Phase 2: World Model 训练（与 CWM 类似）                 │
│   - 加速后续 RL                                          │
├──────────────────────────────────────────────────────────┤
│ Phase 3: Value Model 训练                                │
│   - 用收集的 trajectory 训练 V_phi                      │
├──────────────────────────────────────────────────────────┤
│ Phase 4: 分层 RL                                        │
│   - 高层 policy: PPO + sparse reward                    │
│   - 低层 policy: GRPO + dense reward                    │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Test-time search                                │
│   - 推理时用 MCTS 或 Beam Search                        │
│   - 借助 value model 评估中间状态                       │
└──────────────────────────────────────────────────────────┘
```

### DeepSWE 的成绩

DeepSWE 在 SWE-bench Verified 上的成绩：

| 模型                   | SWE-bench Verified |
| ---------------------- | ------------------ |
| Meta SWE-RL            | 41.0%              |
| **DeepSWE（字节）**    | **50.0%**          |
| SWE-Lancer（OpenAI）   | 45.0%              |
| Claude Opus 4.5 + 工具 | 60%+               |

DeepSWE 在开源模型中达到 50%——证明了长 horizon RL 训练的可行性。

### DeepSWE 与 VAPO 的关系

DeepSWE 的设计与 [字节 VAPO](../chapter18_grpo/grpo-family) 高度相似——都用 value model 替代 GRPO 的"无 critic"路线。这反映了字节 Seed 内部对 **"长 horizon 任务需要 critic"** 的共识。

这也印证了 [第 9 章 GRPO 改进家族](../chapter18_grpo/grpo-family) 的结论——**critic-free 是工程妥协，不是算法必然**。在长 horizon 任务（长 CoT 推理、长 SWE trajectory）上，value model 重新证明了自己的价值。

## 12.3.3 Test-time Search 集成

CWM 和 DeepSWE 都集成了 **test-time search**——推理时用 MCTS 或 Beam Search 提升性能。

### CWM 上的 MCTS

CWM 的 world model 让 MCTS 变得高效：

```python
def cwm_mcts(issue, model, world_model, depth=10):
    # 在 world model 上做 MCTS
    root_state = initialize_state(issue)

    for _ in range(N_iter):
        # Selection: 用 UCB 选最优子节点
        node = select(root_state)

        # Expansion: 用 policy 生成动作，用 world model 模拟下一状态
        action = model.policy(node.state)
        next_state = world_model.predict(node.state, action)

        # Simulation: 快速 rollout 到终止
        rollout_reward = quick_rollout(next_state, world_model)

        # Backprop: 更新节点统计
        backpropagate(node, rollout_reward)

    # 返回 root 的最优动作
    return best_action(root_state)
```

整个 MCTS 在 world model 上完成——**不需要真实测试**，速度极快。

### DeepSWE 上的 Beam Search

DeepSWE 在推理时用 Beam Search：

```python
def deep_swe_beam_search(issue, model, value_model, K=4):
    beams = [{"state": init_state(issue), "score": 0}]

    for step in range(MAX_STEPS):
        candidates = []
        for beam in beams:
            # 生成 K 个候选动作
            actions = model.generate_actions(beam["state"], n=K)

            for action in actions:
                next_state = apply_action(beam["state"], action)
                # 用 value model 评估
                value = value_model.estimate(next_state)
                candidates.append({
                    "state": next_state,
                    "score": beam["score"] + value
                })

        # 选 top-K
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

    return beams[0]["state"]
```

Beam Search 让 DeepSWE 在推理时多花算力换准确率——与 [第 19 章 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) 一致。

## 12.3.4 工业实践的对比

到 2026 年中，主流 SWE-RL 工业方案：

| 方案             | 代表             | 特点       | SWE-bench Verified |
| ---------------- | ---------------- | ---------- | ------------------ |
| Simple GRPO      | Meta SWE-RL      | 开源、简单 | 41.0%              |
| + World Model    | Code World Model | 训练快     | ~45%               |
| + Value + Search | DeepSWE          | 长 horizon | 50.0%              |
| + 多 agent 协作  | Claude Opus 4.7  | 闭源、复杂 | 65%+               |

可以看到，**算法复杂度与性能正相关**——从简单 GRPO 到多 agent 协作，每个改进都带来几个百分点的提升。

## 小结

Code World Model 和 DeepSWE 是 SWE-RL 的两个重要突破：

- **CWM**：用 world model 加速训练，避免真实测试的高成本
- **DeepSWE**：用 value model + 分层 RL + test-time search 处理长 horizon

两者都反映了 SWE-RL 的一个共性：**长 horizon 任务需要更精细的算法**。简单 GRPO 适合短任务（< 8 步），但 SWE 任务的 16-64 步 trajectory 需要更强的工具。

下一节我们看 Self-play SWE-RL——**让模型自己生成训练数据**，进一步降低对人工数据的依赖。
