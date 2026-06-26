# 24.2 评测基准与开源项目

> [24.2](./browser-rl-harness) 搭建了训练 harness。但训出来的 Deep Research Agent 到底好不好？这需要**评测基准**。本节覆盖两件事：(1) 主流 Deep Research 评测基准（BrowseComp、xbench-DeepSearch、GAIA）的设计哲学与陷阱；(2) 可复现的开源项目（GPT-Researcher、STORM、OpenResearcher），让你不用从零造轮子。

## 为什么 Deep Research 评测特别难

传统 LLM 评测（MMLU、GSM8K）有两个特点：(1) **答案唯一**（一道数学题只有一个正确答案）；(2) **不需要工具**（模型直接作答）。Deep Research 把这两点都打破了：

- **答案不唯一**：问"比较 React 和 Vue 的状态管理"，正确答案有无数种表述
- **必须用工具**：模型不能凭记忆回答"2026 年 6 月的比特币价格"
- **过程重要**：模型答对了，但用了 50 步还是 5 步？是否引用了可靠来源？
- **数据污染**：互联网内容随时变化，今天的答案明天可能失效

因此 Deep Research 评测需要专门的基准设计。

## 主流评测基准

### BrowseComp（Meta，2025）

**BrowseComp** 是 Meta 2025 年发布的浏览器智能体基准，专门测试 agent 在开放网络上找信息的能力。

**设计哲学**：

- **困难到必须用浏览器**：每个问题都设计成"凭模型参数记忆无法回答"
- **答案唯一且可验证**：每个问题有一个明确答案，可字符串匹配
- **反 Google**：直接 Google 搜不到，必须多步导航

**示例**：

> Q: "1998 年法国世界杯 1/4决赛中，为阿根廷打进唯一进球的球员，他退役后在哪支球队担任青训教练？"
>
> A: "Argentinos Juniors"（精确字符串）

模型要解决这个问题，必须：(1) 查 1998 年世界杯 1/4 决赛阿根廷的进球者 → 巴蒂斯图塔；(2) 查巴蒂斯图塔退役后的职业 → 青训教练；(3) 查他在哪支球队。至少 3-5 步浏览器导航。

**指标**：精确匹配准确率（Exact Match Accuracy）。

**SOTA 表现**（截至 2026.06）：

| 系统                | BrowseComp | 备注                   |
| ------------------- | ---------- | ---------------------- |
| GPT-5 + 浏览器      | 38.2%      | OpenAI Operator 升级版 |
| Claude Opus 4.6     | 35.7%      | Anthropic 内部         |
| Kimi K2.5 Swarm     | 72.1%      | 多 agent 协作          |
| Tongyi DeepResearch | 51.4%      | 阿里 2026.03           |
| 人类专家            | 87.5%      | 单人 30 分钟限制       |

注意 Kimi K2.5 Swarm 比单 agent 高出 30+ 个百分点——这是 [22.6 多智能体协作](../chapter22_agentic/multi-agent-swarm) 的实战证据。

### xbench-DeepSearch（清华大学，2025）

**xbench-DeepSearch** 是清华 + 香港大学 2025 年发布的中文 Deep Research 基准，针对 BrowseComp 的几个缺陷：

- **中文为主**：BrowseComp 是英文，xbench-DeepSearch 覆盖中英文双语
- **任务类型多样**：BrowseComp 都是单实体问答，xbench-DeepSearch 包含多文档综合、对比分析、时间推理
- **难度可控**：每个问题标注难度（1-5 星），可按模型能力选子集

**任务类型**：

| 类型       | 占比 | 示例                                                      |
| ---------- | ---- | --------------------------------------------------------- |
| 单实体问答 | 30%  | "2025 年图灵奖得主本科毕业于哪所大学？"                   |
| 多文档综合 | 25%  | "对比 DeepSeek V3 和 Llama 4 的训练成本"                  |
| 对比分析   | 20%  | "React 19 和 Vue 3.5 在 SSR 性能上的差异"                 |
| 时间推理   | 15%  | "2024 年 Apple WWDC 上发布的 Vision Pro 中国大陆发售日期" |
| 隐含推理   | 10%  | "如果按 X 论文的方法，Y 数据集的预期准确率是多少？"       |

**指标**：除了 EM，xbench-DeepSearch 还报告：

- **过程评分**（Process Score）：中间步骤的正确率
- **效率**（Efficiency）：平均步数 / 最少步数
- **引用质量**（Citation Quality）：是否引用了可靠来源

### GAIA（Meta + HuggingFace，2024）

**GAIA**（General AI Assistants）是更早的基准，但仍是 Deep Research 的标准测试集之一。GAIA 设计了三个难度等级：

| Level   | 任务复杂度 | 平均步数 | 示例                         |
| ------- | ---------- | -------- | ---------------------------- |
| Level 1 | 简单       | 5-10     | "找一张特定条件下的图片"     |
| Level 2 | 中等       | 10-30    | "整理一份 PDF 中的表格"      |
| Level 3 | 困难       | 30-100   | "规划一次跨欧洲的多城市旅行" |

**指标**：准确率 + 平均步数（越少越好）。

GAIA 与 BrowseComp 的关键区别：GAIA 任务更接近"个人助理"，BrowseComp 更接近"研究任务"。

## 评测的四大陷阱

Deep Research 评测有几个独特陷阱，如果不小心，数字会虚高：

### 数据污染

LLM 预训练数据可能已经包含答案。例如问"2024 年诺贝尔物理学奖得主"，模型凭记忆就能回答（不需要浏览器）。

**对策**：

- 用**时间敏感问题**（答案发布在训练 cutoff 之后）
- 用**反事实问题**（"如果 X 事件没发生，Y 会怎样？"——模型必须查 X 的真实情况）
- BrowseComp 通过"必须多步导航"的设计缓解这点

### 答案表述多样性

问"对比 React 和 Vue"，agent 回答"React 用 JSX，Vue 用 template"和"Vue 用 template，React 用 JSX"都对，但 EM 会判错。

**对策**：

- 用 **LLM-as-Judge**（GPT-4 / Claude）评判语义等价
- 用**结构化答案**（如 JSON、Markdown 表格）减少表述差异
- xbench-DeepSearch 用 LLM Judge 校准

### 过程作弊

agent 可能不实际浏览，而是直接生成看似合理的答案（hallucinate citation）。

**对策**：

- **引用必须可点击**：评估时检查 agent 提供的 URL 是否真实存在
- **网页快照**：评测时保存 agent 访问的页面快照，事后审核
- BrowseComp 设计了"反向验证"：故意问一些答案是随机字符串的问题，agent 不可能猜中

### 成本污染

不同 agent 的 token 成本差 10-30 倍（[22.6](../chapter22_agentic/multi-agent-swarm) 提到 Kimi K2.5 Swarm 是单 agent 的 15×）。简单比较准确率会偏向昂贵系统。

**对策**：

- 报告**准确率 / token 成本** 效率指标
- 在固定 budget 下比较（如"每个问题最多 100K token"）

## 开源项目复现

不用从零搭 harness——以下开源项目提供了完整的 Deep Research 训练 / 推理 pipeline。

### GPT-Researcher（assafelovic-gpt-researcher）

**最流行的开源 Deep Research 框架**。GitHub star 18K+，活跃维护。

**特点**：

- **Python**，基于 Playwright
- 内置 Planner / Researcher / Writer 三层架构（典型 Orchestrator-Worker）
- 支持多种搜索后端（Tavily、SerpAPI、Google CSE、Bing）
- 输出 Markdown 报告 + 引用

**适合场景**：快速搭建产品级 Deep Research 服务。**不适合**：RL 训练（设计为推理时使用）。

```bash
pip install gpt-researcher
```

```python
from gpt_researcher import GPTResearcher

async def research():
    researcher = GPTResearcher("Compare React 19 vs Vue 3.5 SSR performance")
    report = await researcher.conduct_research()
    print(report)
```

### Stanford STORM（stanford-omp-storm）

**斯坦福 Oval 组的开源研究框架**，专门做"长篇结构化文章生成"。

**特点**：

- 基于 Wikipedia 写作流程：先做"模拟对话"（多个 persona 互相提问），再写大纲，最后写正文
- 内置 Wikipedia 检索 + 引用管理
- 输出 Wikipedia 风格长文（5K-20K 字）

**适合场景**：学术综述、深度报告。**优势**：引用质量高（Wikipedia 标准）。

```bash
pip install knowledge-storm
```

```python
from storm import STORMWikiRunner

runner = STORMWikiRunner(...)
runner.run("History of reinforcement learning")
```

### OpenResearcher（tjuloonkopen-researcher）

**完全开源的 Deep Research 训练 pipeline**，含 RL 训练代码。

**特点**：

- **可复现训练**：包含 100K 轨迹数据集 + GRPO 训练脚本
- 基于 Search-R1 架构
- 7B 模型达到 BrowseComp 31.2%
- 完整文档（英文）

**适合场景**：从零训一个 Deep Research Agent。**优势**：完整的 vLLM + veRL pipeline，可扩展。

```bash
git clone https://github.com/tjuloonk/open-researcher
cd open-researcher
bash train.sh --model qwen2.5-7b --algo grpo
```

### 其他值得关注的项目

| 项目                         | 机构           | 特点                                 |
| ---------------------------- | -------------- | ------------------------------------ |
| **Search-R1**                | UIUC           | 最早的开源 Deep Research RL 训练代码 |
| **R1-Searcher**              | Renmin Univ    | 多阶段训练（SFT → RL）               |
| **Tongyi DeepResearch 复现** | 阿里达摩院官方 | 工业级规模，需 H100 集群             |
| **PokeeResearch**            | 北大           | 7B 小模型实现 70B 级表现             |
| **DeepResearcher**           | 人大           | 端到端 RL 训练开源                   |

## 端到端实验 与 从 0 训一个 Deep Research Agent

把本节内容串起来，一个完整的实验流程：

### Step 1 与 选基座模型

- 入门：Qwen2.5-7B-Instruct（容易跑通）
- 进阶：Llama-3.1-8B-Instruct
- 高级：Qwen3-14B / DeepSeek-V2-Lite

### Step 2 与 选动作空间

- 简单（API）：用 Search-R1 的 3-action 空间
- 真实（Playwright）：用 OpenResearcher 的 7-action 空间

### Step 3 与 选训练数据

- xbench-DeepSearch 训练集（10K）
- HotpotQA + 自然 Questions（需改造）
- 自己合成：用 GPT-5 / Claude 生成问题 + 答案

### Step 4 与 训练

```bash
# 用 OpenResearcher 的训练脚本
bash train.sh \
    --model qwen2.5-7b \
    --algo grpo \
    --env api \
    --data xbench-train.jsonl \
    --batch-size 256 \
    --lr 5e-7 \
    --epochs 3
```

### Step 5 与 评测

```bash
# BrowseComp 评测
python eval.py \
    --model checkpoint-final \
    --benchmark browsecomp \
    --max-steps 30
```

预期结果：Qwen2.5-7B 训 3 epoch 后，BrowseComp 准确率从 8%（SFT baseline）提升到 25-30%。这已接近 GPT-4 + 浏览器水平（35%）。

## 本节总结

Deep Research 评测的四大基准（BrowseComp / xbench-DeepSearch / GAIA / 自建）各有侧重——BrowseComp 测"必须多步导航"，xbench-DeepSearch 测中文 + 任务类型多样，GAIA 测个人助理场景。**没有银弹**，建议至少跑两个基准。

开源复现上，**GPT-Researcher 适合做产品，OpenResearcher 适合做研究**。前者工程成熟，后者训练透明。如果你是研究 / 学习目的，从 OpenResearcher 入手；如果是产品落地，从 GPT-Researcher 入手。

下一章 [第 25 章 Computer Use 与 GUI Agent](../chapter25_computer_use/intro) 从浏览器转移到整个桌面——agent 不再只是查网页，而是操作任意 GUI 应用（Excel、PS、内部 OA）。
