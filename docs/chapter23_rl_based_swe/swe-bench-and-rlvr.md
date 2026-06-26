# 23.1 SWE-bench 与 RL-based SWE 范式

这一节我们建立 SWE-RL 的基础概念——什么是 SWE-bench、为什么 SWE 是 RLVR 的理想战场、SWE-RL 与传统 code generation 的本质区别。

## 12.1.1 SWE-bench 任务定义

[SWE-bench](https://arxiv.org/abs/2310.06770)（Jimenez et al. 2023）是 SWE-RL 的核心 benchmark。它的任务定义是：

```text
输入：
  - GitHub 仓库（包含完整代码）
  - 一个 Issue 描述（自然语言，描述 bug 或 feature 需求）
  - 测试用例（用来验证修复是否正确）

输出：
  - 一个代码 patch（修改后的代码）

验证：
  - 应用 patch 到仓库
  - 运行测试用例
  - 全部通过 → 任务成功
  - 任何测试失败 → 任务失败
```

### 一个具体例子

```text
仓库：django/django（Django Web 框架）

Issue:
  "在 Django 4.2 中，使用 `Model.objects.filter(field__in=[])`
   返回空 queryset，但 SQL 查询仍然执行。
   应该短路返回空结果，避免不必要的数据库调用。"

测试用例：
  def test_empty_in_lookup_short_circuits(self):
      # 期望：filter(field__in=[]) 不触发 SQL
      with self.assertNumQueries(0):
          list(Model.objects.filter(field__in=[]))

模型输出：
  - 修改 django/db/models/sql/query.py
  - 在 as_sql 方法中加入：if not self.bloom_metadata and not value: return '', []

验证：
  - 应用 patch
  - 跑测试：✓ 通过
  - 任务成功
```

### SWE-bench 的难度

SWE-bench 的难度远超传统的代码生成：

| 维度     | 普通 code generation | SWE-bench             |
| -------- | -------------------- | --------------------- |
| 上下文   | 单个函数 / 短描述    | 整个仓库（10K-1M 行） |
| 输出     | 完整代码片段         | 精确的 patch（diff）  |
| 验证     | 人工或测试           | 自动化测试套件        |
| 多文件   | 很少                 | 经常需要跨文件修改    |
| 推理深度 | 1-10 步              | 10-100+ 步            |

SWE-bench Verified（高质量子集，500 题）的 SOTA 表现：

- 2024 年初：约 12%（OpenAI SWE-agent）
- 2024 年中：约 25%（Cognition Devin）
- 2025 年初：约 40%（开源 SWE-RL 系列）
- 2025 年底：约 53%（NVIDIA 等）
- 2026 年初：约 65%（Claude Opus 4.7 + 工具调用）

## 12.1.2 为什么 SWE 是 RLVR 的理想战场

回顾 [第 9 章 RLVR](../chapter18_grpo/rlvr)——RLVR 的核心思想是**用规则验证替代 RM**。RLVR 需要三个条件：

1. **任务有明确答案**：对就是对，错就是错
2. **验证可以自动化**：不需要人工判断
3. **训练数据足够多**：能支撑大规模 RL

SWE 完美满足这三个条件：

### 明确答案

代码要么通过测试，要么不通过——没有"半对"或"主观判断"。这是数学之外最纯粹的"对错分明"领域。

### 自动化验证

`pytest`、`unittest` 等测试框架自动运行测试，输出 PASS/FAIL。整个验证过程不需要人工干预。

### 海量数据

- GitHub 有超过 4 亿个仓库
- 每个 PR 都是一个天然的 SWE 任务（issue + patch + tests）
- 工业公司内部的 commit 历史更是海量训练数据

这三个条件让 SWE-RL 成为 RLVR 在工业上**最成功的应用**之一。Meta、字节、Cognition、阿里、清华都在这个方向投入了大量资源。

## 12.1.3 SWE-RL vs 传统 Code Generation

传统 code generation（如 HumanEval、MBPP）的任务是：

```text
输入：函数签名 + docstring
输出：完整函数实现
```

这是**短上下文、单文件、无测试反馈**的设置。RL 在这种任务上效果有限——因为生成空间小，SFT 就能达到 SOTA。

SWE-RL 的任务是：

```text
输入：完整仓库 + Issue + 测试用例
输出：精确 patch
允许：多步交互（read file、edit、run test、edit again）
```

这是**长上下文、多文件、有测试反馈**的设置。RL 在这种任务上效果显著——因为：

- **探索空间巨大**：可能的 patch 数量天文数字，RL 可以高效探索
- **延迟反馈**：测试结果是延迟 reward，与 RL 的优势估计天然匹配
- **多步决策**：read → think → edit → test → fix → submit 是典型 agent trajectory

## 12.1.4 SWE-bench 的数据制造

SWE-RL 训练需要大量 (Issue, patch, tests) 三元组。来源有三个：

### 真实 PR（SWE-bench 方法）

从 GitHub 抓取 PR，提取：

- Issue 文本（PR 关联的 issue）
- 代码 diff（PR 的修改）
- 测试用例（PR 新增或修改的测试）

规模：约 2300 条（SWE-bench 原版）

局限：

- **数据少**：2300 条不足以训练大模型
- **依赖 PR 质量**：低质量 PR 也会被收集
- **测试可能缺失**：很多 PR 没有完整测试

### 合成数据（SWE-smith 方法）

[SWE-smith](../chapter22_agentic/agent-data-swe-smith)（[arXiv:2504.21798](https://arxiv.org/abs/2504.21798)）——**故意往好代码里注入 bug，跑测试看哪些 bug 被检测到**。

规模：50,000+ 条（覆盖 128 个 Python 仓库）

优势：

- **数据量大**：是 SWE-bench 的 20 倍
- **可控**：bug 的类型和难度可以调整
- **测试完备**：每个 bug 都有对应测试

### 模型自生成（Self-play SSR 方法）

让模型自己：

1. 在仓库中找一个"看起来像 bug"的地方
2. 写一个"修复"
3. 跑测试看是否通过
4. 通过的（issue, patch, test）三元组作为训练数据

这是 [12.5 节 SSR](./self-play-ssr) 的核心思想——**模型自己生成训练数据**。

## 12.1.5 SWE-RL 的奖励函数

SWE-RL 的 reward 通常极其简单：

```python
def swe_reward(test_results):
    """测试结果作为 reward"""
    passed = sum(test_results)
    total = len(test_results)
    return passed / total  # 或者 binary: 1.0 if passed == total else 0.0
```

这个 reward 函数与 R1-Zero 的数学奖励完全一致——**0/1 binary reward**。

### Reward shaping 的细节

但工业实践中会加入几个 shaping term：

**Term 1：测试通过比例**

```python
reward = passed / total
```

不是 binary，而是连续值。这让模型在"修了一半"时也能得到部分 reward。

**Term 2：长度惩罚**

```python
reward -= 0.01 * len(trajectory)
```

鼓励模型用更少的步骤完成任务——避免"先随便改，跑测试失败再改"的浪费。

**Term 3：编辑质量**

```python
patch_quality = score_patch(model_output)  # 用 LLM judge
reward += 0.1 * patch_quality
```

鼓励模型生成更优雅的 patch（如不重复代码、不破坏现有逻辑）。

**Term 4：上下文使用**

```python
context_efficiency = relevant_files_read / total_files_read
reward += 0.05 * context_efficiency
```

鼓励模型只读相关文件，避免"读所有文件"的浪费。

但 [Meta SWE-RL](https://arxiv.org/abs/2502.18452) 报告了一个重要发现：**最简单的 reward（测试通过 binary）效果最好**。复杂的 shaping 容易引入 reward hacking——模型学会"优化 shaping term"而不是真正修 bug。

这与 [R1-Zero 的发现](../chapter18_grpo/deepseek-dapo) 一致：**简单 reward + 大规模 RL > 复杂 reward + 小规模 RL**。

## 12.1.6 SWE-RL 的训练流程

一个完整的 SWE-RL 训练流程：

```text
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Base model 选择                                     │
│   - 通常是 code-tuned LLM（如 Qwen-Coder、DeepSeek-Coder）  │
│   - 已经在大量代码上预训练                                  │
├─────────────────────────────────────────────────────────────┤
│ Step 2: SFT 冷启动（可选）                                  │
│   - 用 SWE-bench / SWE-smith 数据做 SFT                    │
│   - 让模型学会基本的 trajectory 格式                        │
├─────────────────────────────────────────────────────────────┤
│ Step 3: RL 训练                                             │
│   - GRPO / PPO                                              │
│   - Reward: 测试通过 binary                                │
│   - 长 horizon：每个 trajectory 可能 16-100+ 步             │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Rejection sampling + 二次 SFT                      │
│   - 从 RL 训练后的模型生成多个候选                          │
│   - 选最好的做 SFT                                          │
├─────────────────────────────────────────────────────────────┤
│ Step 5: 评测                                                │
│   - SWE-bench Verified                                     │
│   - 内部 evaluation set                                    │
└─────────────────────────────────────────────────────────────┘
```

这个流程与 [DeepSeek-R1 的训练流程](../chapter18_grpo/deepseek-dapo) 高度相似——都是 SFT + RL + 二次 SFT 的组合。差别只在于：

- R1 的 reward 是数学答案对错
- SWE-RL 的 reward 是测试通过与否

这种相似性说明：**RLVR 的训练范式是通用的**——只要找到合适的 verifier，同样的算法可以应用到不同领域。

## 小结

SWE-bench 是 SWE-RL 的核心 benchmark，定义了 (issue, patch, tests) 的任务格式。SWE 是 RLVR 的理想战场——明确答案、自动化验证、海量数据。

SWE-RL 与传统 code generation 有本质区别——长上下文、多文件、有测试反馈、多步决策。这让它与 Agentic RL 高度一致，是 RL 在工业上最有价值的应用之一。

下一节我们看 Meta SWE-RL——开源 SWE-RL 的代表作。
