# 23.3 Self-play SWE-RL 与工业落地

至此我们讨论了 SWE-RL 的三个支柱：

- **数据**：[SWE-bench](./swe-bench-and-rlvr)（真实 PR）+ SWE-smith（合成 bug）
- **算法**：[Meta SWE-RL](./meta-swe-rl) 的 GRPO + binary reward
- **加速**：[CWM](./world-model-and-deep-swe) + DeepSWE 的 world model + value model

但这些方法都依赖**预先收集的训练数据**——SWE-bench 或 SWE-smith。数据收集本身是昂贵且受限的。

这一节讨论一个新方向：**Self-play SWE-RL（SSR）**——让模型自己生成训练数据，形成"数据 flywheel"。

## 12.4.1 Self-play 的核心思想

Self-play 的灵感来自 AlphaGo Zero——**模型与自己下棋，从对弈结果中学习**。SSR 把这个思想用到 SWE：

```text
┌──────────────────────────────────────────────────────────┐
│ Player A (Bug Generator):                                │
│   - 在仓库中找一个地方注入 bug                           │
│   - 生成对应的测试（验证 bug 存在）                       │
│   - 生成对应的 issue 描述                                │
├──────────────────────────────────────────────────────────┤
│ Player B (Bug Fixer):                                    │
│   - 看到 issue 描述                                      │
│   - 尝试修复                                              │
│   - 跑测试验证                                           │
├──────────────────────────────────────────────────────────┤
│ RL Update:                                               │
│   - Player A 学会"生成更难的 bug"（Player B 修不好）     │
│   - Player B 学会"修更复杂的 bug"                        │
│   - 形成对抗性提升                                       │
└──────────────────────────────────────────────────────────┘
```

### 与 SWE-smith 的区别

[12.1 节 SWE-smith](./swe-bench-and-rlvr) 是**离线合成数据**——一次性生成 50K 数据，然后训练。

SSR 是**在线合成数据**——模型在训练过程中持续生成数据，数据质量随着模型能力提升而提升。

| 维度     | SWE-smith（离线） | SSR（在线）    |
| -------- | ----------------- | -------------- |
| 数据生成 | 一次性            | 训练中持续     |
| 数据难度 | 固定              | 随模型能力调整 |
| 数据质量 | 与生成器能力无关  | 随模型能力提升 |
| 适用阶段 | 训练初期          | 训练全过程     |

### SSR 的数据 flywheel

SSR 的核心价值是**数据 flywheel**——模型越强，生成数据越好；数据越好，模型越强。

```text
强模型 → 生成难 bug + 优秀修复 → 高质量训练数据 → 模型更强 → ...
```

这种正反馈让 SSR 在训练后期效果显著——模型自己探索的"难题"比人工设计的更能突破能力上限。

## 12.4.2 SSR 的算法细节

清华 [SSR](https://arxiv.org/abs/2507.17492)（Self-play SWE-RL）的具体设计：

### Bug Generator（Player A）

Bug Generator 是一个 LLM，输入仓库代码，输出"注入 bug 后的代码 + 测试 + issue 描述"。

```python
def generate_bug(generator_model, repo, file_path):
    # 1. 选择一个文件
    original_code = repo.read(file_path)

    # 2. 让 generator 注入 bug
    prompt = f"""
    Here is the code in {file_path}:
    {original_code}

    Please:
    1. Choose a function to modify
    2. Inject a subtle bug (logic error, not syntax error)
    3. Generate a test that would fail with the bug
    4. Generate an issue description (without revealing the bug)
    """

    response = generator_model.generate(prompt)
    bug_code, test, issue = parse_response(response)

    # 3. 验证 bug 是否有效（测试在 bug 代码上失败，在原代码上通过）
    if not validate_bug(original_code, bug_code, test):
        return None  # 无效 bug，丢弃

    return {
        "original_code": original_code,
        "bug_code": bug_code,
        "test": test,
        "issue": issue
    }
```

### Bug Fixer（Player B）

Bug Fixer 是被训练的 policy model，输入 issue + bug 代码，输出修复 patch。

```python
def fix_bug(fixer_model, task):
    # 1. 给 fixer 看 issue 和 bug 代码（不给看原代码）
    prompt = f"""
    Issue: {task['issue']}

    Current code: {task['bug_code']}

    Please fix the bug.
    """

    # 2. Fixer 用 agentic 方式修复
    trajectory = []
    while not done:
        action = fixer_model.act(prompt)
        trajectory.append(action)

        if action.type == "edit":
            apply_edit(action)
        elif action.type == "test":
            result = run_tests()
            if result.all_passed:
                done = True

    # 3. 计算 reward
    reward = 1.0 if tests_passed else 0.0

    return trajectory, reward
```

### 对抗性训练

```python
def ssr_training(generator_model, fixer_model, repo):
    for epoch in range(N_EPOCHS):
        # 1. Generator 生成 bug
        task = generate_bug(generator_model, repo, random_file())

        # 2. Fixer 尝试修复
        trajectory, reward = fix_bug(fixer_model, task)

        # 3. 对抗性 reward
        generator_reward = -reward  # Fixer 修不好 → Generator 赢
        fixer_reward = reward       # Fixer 修好 → Fixer 赢

        # 4. 更新两个模型
        update_generator(generator_model, task, generator_reward)
        update_fixer(fixer_model, trajectory, fixer_reward)
```

### Curriculum Learning

SSR 自然产生 curriculum——Generator 早期生成简单 bug，Fixer 容易修；随着 Fixer 变强，Generator 必须生成更难的 bug 才能赢。

```text
Epoch 0-100:  Generator 生成简单 typo / 一行 bug
Epoch 100-500: Generator 生成多文件、跨函数 bug
Epoch 500-2000: Generator 生成微妙逻辑错误、跨模块影响
```

这种 curriculum 是**自适应的**——不需要人工设计难度阶梯。

## 12.4.3 SSR 的实验结果

SSR 在 SWE-bench Verified 上的实验：

| 训练方法      | 数据来源                          | SWE-bench Verified |
| ------------- | --------------------------------- | ------------------ |
| Meta SWE-RL   | 真实 PR + SWE-smith               | 41.0%              |
| DeepSWE       | 真实 PR + SWE-smith + world model | 50.0%              |
| **SSR**       | 真实 PR + self-play 生成          | **47.5%**          |
| SSR + DeepSWE | 全部                              | **53.2%**          |

SSR 单独训练（不依赖 SWE-smith）就能达到 47.5%——证明了 self-play 数据的有效性。结合 DeepSWE 的 world model，达到 53.2%。

### 数据效率对比

| 方法                | 训练数据量               | 达到的准确率 |
| ------------------- | ------------------------ | ------------ |
| SWE-smith（一次性） | 50K                      | 41%          |
| SSR（self-play）    | 5K 种子 + 50K self-play  | 47%          |
| SSR + curriculum    | 5K 种子 + 100K self-play | 53%          |

**Self-play 让数据效率提升**——同样的训练数据量，self-play 比 static data 多 6 个百分点。

## 12.4.4 SSR 的局限与未来

### Generator 可能产生无效 bug

如果 Generator 学会"生成语法错误"的 bug（Fixer 很难修），这其实是无效训练——语法错误在真实 SWE 任务中很少见。

缓解：在 Generator 的 reward 中加入"bug 真实性"奖励——用 LLM judge 判断 bug 是否像真实 bug。

### Generator 和 Fixer 不平衡

如果 Generator 远强于 Fixer，Fixer 永远修不好——训练无信号。如果 Fixer 远强于 Generator，Generator 无法产生有效挑战——curriculum 停滞。

缓解：动态调整两者的训练频率——保持平衡。

### 领域漂移

Self-play 生成的 bug 可能与真实 bug 分布不同——比如 Generator 可能集中在某类 bug（typo），而真实世界 bug 类型多样。

缓解：用真实 PR 作为种子，让 Generator 在真实 bug 模式基础上变异。

## 12.4.5 RL-based SWE 的工业落地

到 2026 年中，RL-based SWE 已经在多个产品中落地：

### Cursor

[Cursor](https://cursor.sh) 是最受欢迎的 AI 代码编辑器之一。它的核心能力：

- **多文件理解**：用 RAG 让模型看到整个项目
- **Agentic 修复**：模型可以自主 read、edit、test
- **基于 Claude Opus + 工具调用**

Cursor 不公开训练方法，但推测它使用了类似 SWE-RL 的训练数据（GitHub PR + 内部代码）。

### Cognition Devin

[Devin](https://devin.ai) 是 Cognition 推出的"AI 软件工程师"——可以独立完成完整的开发任务（规划、写代码、测试、部署）。

Devin 的训练细节不公开，但 Cognition 在博客中提到："我们的 RL 训练让 Devin 学会了从规划到实现的全流程。"

### 字节 Trae

[Trae](https://www.trae.ai) 是字节的 AI IDE，基于 DeepSWE 的研究成果。在国内市场活跃。

### OpenAI Codex（2025+）

OpenAI 重新推出了 Codex——一个基于 o3 的代码 agent。它的特点：

- 用 o3 的推理能力做复杂规划
- 与 ChatGPT 集成，可以并行处理多个任务
- 在 SWE-bench Verified 上达到约 53%

### Anthropic Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) 是 Anthropic 的 CLI 工具，基于 Claude Opus 4.6/4.7。它的特点：

- 推理模型 + agentic 工具
- 长上下文（200K-1M）
- 在 SWE-bench Verified 上达到 65%+

## 12.4.6 多语言、多仓库扩展

当前 SWE-RL 主要集中在 Python。未来的扩展方向：

### 多语言

- **JavaScript/TypeScript**：Jest、Mocha 测试框架成熟，可以类似 Python 处理
- **Java**：JUnit 测试成熟，但代码风格严格，需要更强的 KL 约束
- **C/C++**：编译型，测试运行慢，对 world model 需求更大
- **Go/Rust**：现代语言，测试覆盖率普遍高，适合 SWE-RL

### 多仓库

- **企业内部代码**：每个公司有自己的代码风格、依赖、测试规范
- **微服务架构**：跨仓库修改、API 兼容性
- **遗留系统**：旧代码、缺测试、文档不全

多仓库扩展需要：

- **快速环境搭建**：每个仓库的依赖管理
- **领域特化 reward**：不同仓库的"好代码"标准不同
- **跨仓库 reasoning**：理解仓库间的依赖关系

## 12.4.7 多 agent 协作

复杂 SWE 任务可能需要多个 agent 协作：

```text
Planner Agent: 分析 issue，制定修复计划
  ↓
Explorer Agent: 在仓库中定位相关文件
  ↓
Editor Agent: 实施修改
  ↓
Tester Agent: 运行测试，反馈结果
  ↓
Reviewer Agent: 检查代码质量
```

这种多 agent 协作在 Claude Opus 4.7、Cursor、Devin 中已经出现。训练这种系统需要：

- **多 agent RL**：联合训练多个 policy
- **通信协议**：agent 间如何传递信息
- **共享 value model**：评估整体 trajectory 质量

这是 [第 22 章 Agentic RL 多智能体部分](../chapter22_agentic/build-agentic-training-system) 在 SWE 领域的具体应用。

## 本章总结

这一章我们梳理了 RL-based SWE 的全貌：

- **12.1 节**：SWE-bench 与 RLVR 范式——为什么 SWE 是 RLVR 的理想战场
- **12.2 节**：Meta SWE-RL——开源 SOTA，GRPO + 简单 reward
- **12.3 节**：Code World Model + DeepSWE——加速训练 + 长 horizon 处理
- **12.4 节**：Self-play SSR——数据 flywheel，工业落地

**核心收获**：

1. **SWE 是 RLVR 最成功的工业应用之一**——明确答案、自动化验证、海量数据
2. **简单 reward > 复杂 shaping**——binary 测试通过比连续 reward 效果好
3. **长 horizon 需要更强算法**——value model、world model、test-time search
4. **Self-play 是数据扩展的关键**——模型自己生成数据，质量随能力提升
5. **工业落地已经成熟**——Cursor、Devin、Claude Code 都用 RL-based SWE

**接下来的章节**：

- [第 13 章 PRM 与搜索](../chapter20_prm_search/intro)——SWE-RL 中的 step-level reward
- [第 14 章奖励黑客](../chapter15_rlhf/evaluation)——SWE 任务的 hacking（如"删除测试让 reward 变高"）
- [第 12.8 节 Agentic RL 训练系统](../chapter22_agentic/build-agentic-training-system)——SWE-RL 的工程实现
