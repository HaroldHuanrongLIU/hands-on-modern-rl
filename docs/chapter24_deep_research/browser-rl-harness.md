# 24.1 浏览器 RL 动作空间与 harness 工程

> [24.1](./intro) 介绍了 Deep Research 的任务定义和主流模型。但当你真正动手训练一个 Deep Research Agent 时，会立刻撞上两个工程问题：(1) **动作空间怎么设计**——浏览器有成百上千种操作，哪些该暴露给 agent？(2) **harness 怎么搭**——agent 生成的动作要落到真实浏览器上，需要一套完整的执行、监控、奖励计算环境。本节解决这两件事，给出可复现的工程模板。

## 浏览器作为 RL 环境

Deep Research 的"环境"是浏览器（或搜索引擎 API）。从 RL 视角看，这是一个**部分可观 + 长程 + 稀疏奖励**的 MDP：

$$\mathcal{M}_{\text{browser}} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$$

- $\mathcal{S}$：浏览器状态空间，包括当前 URL、DOM 树、可见文本、滚动位置、Cookie/Session 等
- $\mathcal{A}$：动作空间（见下文）
- $P$：环境转移函数（由真实浏览器决定，对 agent 未知）
- $R$：稀疏二值奖励，通常 $r_T = \mathbb{1}[\text{答案正确}]$，中间步 $r_{t<T} = 0$
- $\gamma$：折扣因子，Deep Research 任务 $T = 20-100$ 步，$\gamma = 1$（无折扣）
- $T$：最大步数（budget），通常 30-50

与 [第 25 章 Computer Use](../chapter25_computer_use/intro) 的 GUI MDP 相比，Deep Research 的关键差异：

| 维度             | Deep Research                         | Computer Use              |
| ---------------- | ------------------------------------- | ------------------------- |
| 观察空间         | DOM 文本 / 截图                       | 截图为主                  |
| 动作粒度         | 抽象（search / click_link / extract） | 原子（pixel click / key） |
| 状态转移可预测性 | 较高（搜索结果相对稳定）              | 低（GUI 动画、弹窗）      |
| 奖励稀疏度       | 极稀疏（最后一步）                    | 极稀疏（最后一步）        |
| 典型步数         | 20-50                                 | 50-500                    |

## 动作空间设计 与 三种主流方案

### 搜索 API 抽象

最简单的方案——不暴露真实浏览器，只给 agent 一个**搜索 API**：

```python
ACTIONS = {
    "search":   {"query": str},          # 调用搜索引擎，返回 top-K 结果
    "visit":    {"url": str},            # 抓取指定 URL 的纯文本
    "answer":   {"text": str},           # 提交最终答案
}
```

这是 Search-R1、R1-Searcher 用的方案。优点：

- 动作空间只有 3 个原子操作，易学
- 每步观察是干净的 Markdown，不需要视觉模型
- 工程简单，一个 `requests.get()` 搞定

缺点：

- 无法处理需要 JavaScript 的页面（SPA、动态加载）
- 无法点击/滚动/翻页（只能取第一屏）
- 不接近真实"上网研究"体验

适合：开放域 QA、学术论文检索等"文本为主"的任务。

### Playwright 真实浏览器

用 Playwright / Puppeteer 暴露完整浏览器能力：

```python
ACTIONS = {
    "goto":         {"url": str},
    "click":        {"selector": str},        # CSS selector 或文本匹配
    "fill":         {"selector": str, "value": str},
    "scroll":       {"dx": int, "dy": int},
    "back":         {},
    "extract_text": {"selector": str},        # 提取指定元素文本
    "screenshot":   {},
    "answer":       {"text": str},
}
```

这是 DeepResearcher、Tongyi DeepResearch 用的方案。优点：

- 真实浏览器能力，能处理任意网页
- 可截图作为视觉观察（用于 VLM agent）
- 接近人类研究行为

缺点：

- 动作空间大（7-10 个），需要更多训练数据
- 真实浏览器慢（每步 1-3 秒），训练成本高
- CSS selector 失败率高（页面变化导致 selector 失效）

适合：金融调研、产品对比、需要交互式翻页的任务。

### Set-of-Mark 混合

借鉴 [第 25 章 GUI Grounding](../chapter25_computer_use/intro) 的 SoM 思路：每步把页面所有可交互元素编号，agent 只需输出编号：

```
Agent observes:
[页面截图 + 编号]
  [1] 搜索框
  [2] "下一页" 按钮
  [3] 第一个搜索结果的链接
  [4] 第二个搜索结果的链接
  ...

Agent action: click(3)  # 点击第一个搜索结果
```

这是 BrowseComp 评测里多数 SOTA 系统用的方案。优点：

- 动作空间退化为"选编号"，极简
- 不依赖 CSS selector 的脆弱性
- 兼容 VLM（看截图）和 LLM（看编号列表）

缺点：

- 需要 OCR / DOM 解析做编号（额外组件）
- 编号错误的代价高（点错链接）

## Harness 工程的五个核心模块

无论选哪种动作空间，Deep Research 的训练 harness 都需要以下五个模块：

### 环境封装（Environment Wrapper）

```python
class BrowserEnv:
    def __init__(self, mode='api' | 'playwright' | 'som'):
        self.mode = mode
        self.browser = None  # Playwright instance
        self.history = []    # 轨迹历史

    def reset(self, query: str) -> Observation:
        """开始新 trajectory，返回初始观察"""
        self.history = [{'role': 'user', 'content': query}]
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """执行动作，返回 (next_obs, reward, done, info)"""
        # 1. 解析 action
        # 2. 调用浏览器 / API
        # 3. 抓取新观察
        # 4. 判断是否 done（agent 主动 answer 或超出 budget）
        # 5. 计算 reward（done 时才有，否则 0）
        ...
```

**关键工程点**：

- **超时处理**：真实网页可能 hang，必须有 timeout（通常 10 秒）
- **错误恢复**：CSS selector 失败、网络断开、JS 报错——都要捕获并返回友好的 error obs
- **状态持久化**：Cookie / Session 跨步骤保留（否则登录态丢失）

### 动作解析与验证（Action Parser）

LLM 输出的是文本，需要解析成结构化 action：

````python
def parse_action(output: str, mode: str) -> Action:
    """从 LLM 输出解析 action，失败时返回 NoOp"""
    try:
        if mode == 'api':
            # 期望格式: <action>search</action><query>...</query>
            return ApiAction.from_xml(output)
        elif mode == 'playwright':
            # 期望格式: ```python\nAction(...)\n```
            return PlaywrightAction.from_code(output)
        elif mode == 'som':
            # 期望格式: click(3)
            return SomAction.from_text(output)
    except ParseError as e:
        # 解析失败：返回错误观察，让 agent 重试
        return ErrorAction(f"Parse failed: {e}")
````

**关键工程点**：

- **格式容错**：LLM 输出经常有格式错误，parser 要 robust
- **重试机制**：解析失败时返回 error obs，让 agent 自纠错（这是 emergent behavior 的重要来源）
- **动作白名单**：禁止危险动作（如 `format_disk`、`rm -rf`），即使 agent 想做

### 奖励计算器（Reward Verifier）

Deep Research 的奖励是**任务完成度**，需要分任务类型设计：

```python
class RewardVerifier:
    def __call__(self, query: str, answer: str, task_type: str) -> float:
        if task_type == 'qa':
            # 答案匹配（EM / F1 / LLM-as-Judge）
            return self.qa_score(query, answer)
        elif task_type == 'citation':
            # 引用准确性（CaRR 指标）
            return self.citation_score(query, answer)
        elif task_type == 'multi_doc':
            # 多文档综合（需要 LLM 评判）
            return self.multi_doc_score(query, answer)
        elif task_type == 'browse_comp':
            # BrowseComp 基准：精确字符串匹配
            return self.browse_comp_score(query, answer)
```

**关键工程点**：

- **奖励稀疏化缓解**：可加过程奖励（PRM）作为辅助，但主奖励仍是端到端
- **LLM-as-Judge 偏置**：用 GPT-4 / Claude 做 judge 时有已知偏置（长答案偏好、自身风格偏好），需要校准
- **反作弊**：检测 agent 是否"复述问题"或"拼接搜索摘要"等作弊策略

### 进度跟踪（Progress Tracker）

长程任务（30+ 步）必须可视化进度，否则训练时无法 debug：

```python
# claude-progress.txt 风格的进度文件
[2026-06-25 10:23:15] Step 1: search("2024 US GDP")
[2026-06-25 10:23:18] → Got 10 results, top: bea.gov
[2026-06-25 10:23:22] Step 2: visit("https://bea.gov/...")
[2026-06-25 10:23:25] → Page loaded, 15KB text
[2026-06-25 10:23:29] Step 3: extract("main table")
[2026-06-25 10:23:32] → Extracted table: 4 rows × 3 cols
[2026-06-25 10:23:36] Step 4: answer("2024 US GDP was $28.5T")
[2026-06-25 10:23:38] → Reward: 1.0 (correct)
```

这个文件有两个用途：

1. **训练 debug**：失败 trajectory 一眼看出哪步错
2. **数据合成**：成功 trajectory 可作为 SFT 数据

### 并行 Rollout（Parallel Rollout Engine）

Deep Research 单条 trajectory 30-50 步 × 每步 1-3 秒 = 60-150 秒。训练 batch size 1024 时，串行需 25 小时/step。必须并行：

```python
async def parallel_rollout(
    agent, prompts: list[str], num_parallel: int = 256
) -> list[Trajectory]:
    semaphore = asyncio.Semaphore(num_parallel)

    async def rollout_one(prompt):
        async with semaphore:
            env = BrowserEnv(mode='playwright')
            obs = await env.reset(prompt)
            trajectory = []
            for t in range(MAX_STEPS):
                action = await agent.act(obs)
                next_obs, r, done, info = await env.step(action)
                trajectory.append((obs, action, r))
                if done:
                    break
                obs = next_obs
            return trajectory

    return await asyncio.gather(*[rollout_one(p) for p in prompts])
```

**关键工程点**：

- **浏览器池**：复用浏览器实例（启动开销大）
- **网络代理**：避免被目标网站封 IP（用住宅代理）
- **失败隔离**：单条 trajectory 崩溃不影响其他

## 完整训练流水线

把这五个模块拼起来：

```
┌─────────────────────────────────────────────────┐
│ 1. Prompt Batch (1024 questions)                │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. Parallel Rollout (256 concurrent browsers)   │
│    ├─ Environment Wrapper (Playwright)          │
│    ├─ Action Parser (XML / code / SoM)          │
│    ├─ Progress Tracker (claude-progress.txt)    │
│    └─ Reward Verifier (QA / Citation / Browse)  │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. Trajectory Buffer                            │
│    {(s_t, a_t, r_t)}_{t=1..T} per trajectory    │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. GRPO Update                                  │
│    ├─ Group normalization (G=8 per prompt)      │
│    ├─ Advantage estimation                      │
│    └─ PPO-Clip policy update                    │
└─────────────────────────────────────────────────┘
```

实测在 8×H100 GPU + 64-core CPU server 上，单次 GRPO step 处理 1024 prompts 约需 8-12 分钟。训练一个 7B Deep Research 模型到收敛通常需要 5000-10000 step，即 4-7 天。

::: tip 与 [第 18 章 GRPO](../chapter18_grpo/grpo-practice-and-mechanism) 的衔接
Deep Research 的 RL 训练流水线和 [第 18 章](../chapter18_grpo/grpo-practice-and-mechanism) 讲的 GRPO 没有本质区别——都是 group-normalized advantage + PPO-Clip。差异只在环境（浏览器 vs 文本 sandbox）和奖励（任务完成 vs 答案正确）。如果你已经跑通过 [18.8 金融 API 工具调用 GRPO](../chapter18_grpo/financial-tool-calling-grpo)，迁移到 Deep Research 只需要换 `Environment Wrapper` 和 `Reward Verifier` 两个模块。
:::

## 本节总结

Deep Research 的 harness 工程核心是**五个模块**：环境封装、动作解析、奖励计算、进度跟踪、并行 rollout。其中**环境封装**和**奖励计算**是最难复现的——前者需要真实浏览器工程经验，后者需要任务特定的 verifier 设计。

下一节 [24.3 评测基准与开源项目](./deep-research-eval) 介绍如何衡量 Deep Research Agent 的好坏——你会发现，评测本身比训练更难。
