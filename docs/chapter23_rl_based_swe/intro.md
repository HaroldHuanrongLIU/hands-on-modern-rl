# 第 23 章 RL-based SWE 与 让模型自己学会修 Bug

[第 22 章 Agentic RL](../chapter22_agentic/intro) 介绍了智能体在工具调用、多轮交互上的 RL 训练。这一章我们聚焦一个最有工业价值的细分领域：**RL-based SWE（Software Engineering）**——用 RL 训练模型自动修复 bug、实现 feature、写测试。

为什么单独成章？三个原因：

1. **SWE 是 RLVR 的天然战场**——单元测试是完美的"零噪声 verifier"，与 [第 11 章 PRM 的形式化路线](../chapter20_prm_search/formal-prm) 同源
2. **2025 年这个领域出现了多个工业级突破**——Meta SWE-RL、字节 DeepSWE、清华 SSR、阿里 CWM，每个都把 SWE-bench 准确率推向新高度
3. **SWE-RL 是 Agentic RL 的"算法实验室"**——它的很多发现（长 horizon credit assignment、self-play、world model）可以推广到其他领域

## 这一章要回答的问题

- **SWE-bench 是什么**？为什么它是 SWE-RL 的核心 benchmark？
- **Meta SWE-RL** 怎么用开源数据 + 简单 GRPO 达到 SOTA？
- **Code World Model（CWM）** 怎么把代码执行建模为 MDP？
- **DeepSWE** 怎么用 verifiable reward 训练长 horizon agent？
- **Self-play SWE-RL（SSR）** 怎么让模型自己生成训练数据？
- **SWE-RL 的未来**——多语言、多仓库、多 agent 的扩展方向

## 章节地图

```text
12.1 SWE-bench 与 RL-based SWE 范式
     ├── SWE-bench 任务定义
     ├── 为什么 SWE 是 RLVR 的理想战场
     └── 数据制造：SWE-smith 与 SWE-gym
12.2 Meta SWE-RL：开源 SOTA 的代表
     ├── 数据规模与构成
     ├── 算法选择：GRPO + rule-based reward
     ├── 工程细节：context 管理、test sampling
     └── SWE-bench Verified 41.0%
12.3 Code World Model（CWM）
     ├── 把代码执行建模为 MDP
     ├── World model 训练
     ├── 基于 CWM 的 RL
     └── 与 model-based RL 的关系
12.4 DeepSWE：长 horizon agent 的 RL
     ├── 16 步以上 trajectory 的挑战
     ├── Step-level reward shaping
     ├── Test-time search 集成
     └── 字节 Seed 的工业实践
12.5 Self-play SWE-RL（SSR）
     ├── 模型自己生成 bug + 修复
     ├── Curriculum learning
     ├── 清华 SSR 工作
     └── 数据 flywheel 的形成
12.6 RL-based SWE 的工业落地
     ├── Cursor、Cognition Devin、字节 Trae
     ├── 商业模式与成本结构
     └── 多语言、多仓库扩展
```

## 与其他章的关系

这一章假定你已经读过：

- [第 9 章 GRPO 改进家族](../chapter18_grpo/grpo-family)——基础 RL 算法
- [第 22 章 Agentic RL](../chapter22_agentic/intro)——agent 的多轮交互基础
- [第 11 章 PRM](../chapter20_prm_search/intro)——形式化 verifier 思想

本章后续会指向：

- [第 12.4 节 agent 训练系统](../chapter22_agentic/build-agentic-training-system)——SWE-RL 的工程实现
- [第 14 章奖励黑客](../chapter15_rlhf/evaluation)——SWE-RL 特有的 hacking

## 一个直觉性的开场

**直觉：SWE-RL 是把 PRM 形式化思想用到代码领域**。Lean4 是数学的形式化 verifier；单元测试是代码的形式化 verifier。两者的核心思想一致：**用零误判的外部验证替代人工或 LLM 的主观判断**。

但 SWE 有 Lean4 没有的优势——**工业实践极其丰富**。GitHub 上有数亿行代码、数百万 PR、数千万 commit——这是数学界无法比拟的训练数据规模。

带着这个直觉，我们进入 12.1。
