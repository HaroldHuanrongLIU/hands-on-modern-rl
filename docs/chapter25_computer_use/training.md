# 25.1 GUI Agent 训练实践

> [25.1](./intro) 讲清楚了 Computer Use 的 MDP 建模和 GUI Grounding 的视觉对齐。本节回答下一个工程问题：**怎么把一个 VLM 真正训成 GUI Agent**？这涉及数据合成、课程设计、奖励工程、虚拟环境等一整套工业 pipeline。我们以 2025-2026 年中国实验室的代表性工作为线索——UI-TARS-2、AutoGLM、MobileRL、ComputerRL、CogAgent——对比各家技术路线的优劣。

## 中国实验室的 GUI Agent 集中爆发

2025 年下半年，中国实验室在 GUI Agent RL 训练上集中爆发。这不是偶然——三个条件同时成熟：

1. **VLM 基座成熟**：Qwen2.5-VL、InternVL3、GLM-4.5V 等开源 VLM 提供了高质量起点
2. **虚拟环境工具链**：Android Worldwide、AndroidWorld、OSWorld、WebArena 等benchmark 提供可复现的训练/评测环境
3. **算力成本下降**：4090 / H100 价格稳定，RL 训练 7B 模型成本可承受

代表性工作对比：

| 模型             | 机构        | arXiv      | 参数规模 | 核心创新                            |
| ---------------- | ----------- | ---------- | -------- | ----------------------------------- |
| **UI-TARS-2**    | 字节 Seed   | 2509.02544 | 7B / 72B | 端到端 VLM + 长程任务 RL + 反思增强 |
| **Open-AutoGLM** | 智谱        | 2411.00820 | 9B       | 中英混合 GUI + 移动端 + 完整开源    |
| **MobileRL**     | 腾讯        | 2509.18119 | 7B       | 移动 App 难度课程学习               |
| **ComputerRL**   | 上海 AI Lab | 2508.14040 | 7B       | 反向课程 + 中间探索奖励             |
| **CogAgent-9B**  | 智谱        | 2408.16500 | 9B       | 高分辨率视觉编码 + 双分支融合       |

下文逐一拆解。

## UI-TARS-2 与 端到端 RL 的代表

UI-TARS-2 把 Computer Use 当作**纯 LLM RL 问题**——单一 VLM 同时承担感知、推理、动作输出。模型架构上没有显式的 planner / actor 分工，所有逻辑都在一个 transformer 里。

### 四阶段训练流程

```
Stage 1: 视觉-语言预训练
  └─ GUI 截图 + 文本对 → 基础视觉能力

Stage 2: 监督微调（SFT）
  └─ 人类演示 + 模型自生成轨迹 → 基础动作能力

Stage 3: Reflective RL（反思增强）
  └─ 多候选轨迹 + verifier 选优 → rejection sampling + SFT

Stage 4: Online RL（在线强化学习）
  └─ 真实 GUI 环境 rollout → PPO 优化任务完成率
```

阶段 3 的 rejection sampling 是关键过渡：让模型对同一任务生成 $K=8$ 条轨迹，用程序化 verifier 判定哪些成功，把成功轨迹作为高质量 SFT 数据回灌。这比直接做 online RL 更稳定——online RL 在低成功率时（<10%）几乎学不到信号。

### 反思增强（Reflection Augmentation）

阶段 4 的核心创新是反思机制。让 agent 在失败时显式输出 `<reflection>` 标签：

```
<thought>我需要点击"提交"按钮</thought>
<action>click(450, 320)</action>
<observation>按钮变灰，但没有跳转</observation>
<reflection>可能点击位置偏了。"提交"按钮的可点击区域是 (440-470, 310-330)，我点到了边界外。重试时往中心移动。</reflection>
<action>click(455, 320)</action>
<observation>页面跳转到成功页</observation>
<action>done</action>
```

这种自我纠错能力单靠 SFT 学不到——必须靠 RL 的试错信号。RL 训练时给"成功反思后纠错"的轨迹额外 +0.3 reward，鼓励模型学会反思。

### 多任务 RL 奖励

UI-TARS-2 的总奖励函数：

$$r = r_{\text{task}} + \alpha \cdot r_{\text{format}} + \beta \cdot r_{\text{reflection}} - \gamma \cdot r_{\text{invalid}}$$

- $r_{\text{task}} \in \{0, 1\}$：任务是否完成
- $r_{\text{format}} \in \{0, 1\}$：输出格式是否合法（XML 标签闭合、坐标在范围内）
- $r_{\text{reflection}} \in [0, 0.3]$：成功纠错的反思质量
- $r_{\text{invalid}}$：执行越权动作（如尝试关闭浏览器）

实测权重 $\alpha = 0.1, \beta = 0.3, \gamma = 2.0$。$\gamma$ 故意取大——一次越权动作的代价远高于一次任务完成。

## Open-AutoGLM 与 开源完整 pipeline

智谱的 AutoGLM 系列（Open-AutoGLM 于 2025.12 开源）针对**中文互联网环境**优化——微博、淘宝、微信小程序等中文 App 在英文模型（Operator、Mariner）上效果差。其训练创新包括：

### 中文 GUI 数据合成

英文模型的数据来源是 Common Crawl + RPA 录制，中文 GUI 数据稀缺。Open-AutoGLM 的方案：

1. **微信小程序爬取**：用 Android 自动化框架 Appium 控制 100+ 真机，自动探索小程序，录制每步截图 + 动作
2. **中文电商任务合成**：在淘宝/京东/拼多多上自动生成"搜索商品 → 比价 → 加购 → 下单（不下单）"任务模板
3. **中文社交任务**：微博发帖、抖音评论、小红书收藏等

最终收集了 **2.3M 条中文 GUI 轨迹**，是英文轨迹（800K）的 2.9 倍。

### 多平台统一动作空间

Open-AutoGLM 的关键设计是**跨平台统一**——同一模型可在桌面浏览器、Android App、iOS App（通过 WebDriverAgent）上工作。统一动作空间：

```python
UNIFIED_ACTIONS = {
    "tap":       {"x": float, "y": float},           # 单击/触摸
    "long_press":{"x": float, "y": float, "ms": int},
    "swipe":     {"start": [x,y], "end": [x,y]},     # 滑动/拖拽
    "type":      {"text": str},
    "key":       {"name": str},                       # back, home, enter
    "scroll":    {"dy": int},
    "wait":      {"ms": int},
    "done":      {"summary": str},
}
```

桌面端"click" 和移动端"tap" 统一为同一 action `tap`——不同平台的语义差异由环境适配器处理。

### 完整开源

Open-AutoGLM 把**模型权重、训练数据、环境模拟器、训练脚本**全部开源，是目前最完整的开源 GUI Agent 训练框架：

```bash
git clone https://github.com/THUDM/Open-AutoGLM
cd Open-AutoGLM

# 1. 下载预训练权重
huggingface-cli download zhipuai/Open-AutoGLM-9B

# 2. 启动 Android 模拟器
bash scripts/start_emulator.sh

# 3. RL 训练（单机 8×H100）
bash train.sh \
    --model Open-AutoGLM-9B \
    --algo grpo \
    --platform android \
    --tasks curated-1k.jsonl
```

实测在 8×H100 上，单次 GRPO step 处理 256 prompts 约需 4 分钟。训练 5000 step 到收敛约 14 天。

## MobileRL 与 移动端 RL

腾讯 MobileRL（arXiv:2509.18119）专门解决移动 App 自动化。移动端比桌面端更难，原因有三：

- **屏幕小、元素密集**：一个 App 首页可能有 30 个可点击元素，密集排布
- **手势复杂**：长按、滑动、双指捏合、3D Touch，远比鼠标点击丰富
- **应用切换频繁**：推送、来电、低电量弹窗随时打断任务

### 渐进难度课程

MobileRL 的核心创新是**渐进难度课程**（Curriculum Learning）：

$$\text{Curriculum}(\pi_\theta) = \arg\max_{\text{task } \tau} \; \text{Difficulty}(\tau) \quad \text{s.t.} \quad 0.3 \leq P_\theta(\text{success} \mid \tau) \leq 0.7$$

只在模型当前成功率 30%–70% 的"最近发展区"（Zone of Proximal Development）内采样任务，避免过难任务（信号太稀疏）和过易任务（无学习信号）。

### 任务难度量化

MobileRL 把任务难度定义为四个维度的加权和：

$$\text{Difficulty}(\tau) = w_1 \cdot \text{Steps}(\tau) + w_2 \cdot \text{Apps}(\tau) + w_3 \cdot \text{GestureComplexity}(\tau) + w_4 \cdot \text{Distraction}(\tau)$$

- $\text{Steps}$：完成任务的最少步数（5-50）
- $\text{Apps}$：需要切换的 App 数量（1-4）
- $\text{GestureComplexity}$：所需手势种类数（tap=1, swipe=2, long_press=3, multi-touch=5）
- $\text{Distraction}$：模拟干扰事件数（推送、来电）

实测权重 $w_1=0.4, w_2=0.2, w_3=0.2, w_4=0.2$。

### 课程调度器

```python
class CurriculumSampler:
    def __init__(self, tasks, model):
        self.tasks = tasks
        self.model = model
        self.success_rate = {}  # task_id -> moving average success rate

    def sample(self, batch_size):
        # 1. 评估每个任务在当前模型下的成功率
        for tau in self.tasks:
            if tau.id not in self.success_rate:
                self.success_rate[tau.id] = self._estimate(tau)

        # 2. 过滤出 30%-70% 成功率的任务
        candidates = [t for t in self.tasks
                      if 0.3 <= self.success_rate[t.id] <= 0.7]

        # 3. 按 difficulty 加权采样
        weights = [t.difficulty for t in candidates]
        return weighted_sample(candidates, weights, batch_size)

    def _estimate(self, task):
        # 跑 10 次 rollout 估算成功率
        successes = sum(self._rollout(task) for _ in range(10))
        return successes / 10
```

每个 epoch 重新评估一次任务成功率，让课程跟着模型能力动态调整。

## ComputerRL 与 反向课程 + 探索奖励

上海 AI Lab 的 ComputerRL（arXiv:2508.14040）发现纯任务完成奖励在长程任务（50+ 步）上信号过稀疏。其方案是**反向课程 + 中间探索奖励**。

### 反向课程（Backward Curriculum）

传统课程从易到难——先学 5 步任务，再学 10 步、20 步。反向课程反过来：**从任务终点开始**。

考虑一个 50 步任务 $T = (s_0, a_1, s_1, \ldots, a_{50}, s_{50})$。反向课程的训练顺序：

```
Round 1: 从 s_49 开始，只需执行 a_50 → done（1 步任务）
Round 2: 从 s_48 开始，执行 a_49, a_50 → done（2 步任务）
Round 3: 从 s_47 开始，执行 a_48, a_49, a_50 → done（3 步任务）
...
Round 50: 从 s_0 开始，完整任务（50 步）
```

**为什么有效**？反向课程保证了 RL 永远在"接近奖励"的状态上训练。正向训练时，agent 在 $s_0$ 看不到任何 reward 信号；反向训练时，agent 在 $s_{49}$ 上一步就能拿到 reward。这让 credit assignment 变得简单——刚执行的动作立刻有反馈。

### 中间探索奖励

反向课程解决"终态奖励太远"，但中间步仍然无信号。ComputerRL 加入**中间状态奖励**：

$$r_t = \underbrace{r_{\text{task}}(t=T)}_{\text{稀疏终态奖励}} + \lambda \cdot \underbrace{r_{\text{progress}}(s_t, s_{t+1})}_{\text{密集进度奖励}}$$

其中 $r_{\text{progress}}$ 由一个独立的"进度评估器" LLM 输出：

```python
def compute_progress_reward(s_t, s_{t+1}, task):
    prompt = f"""
    Task: {task}
    State before: {describe(s_t)}
    State after: {describe(s_{t+1})}
    Question: did the agent make progress toward the task?
    Answer with a score in [0, 1]:
    - 1.0: significant progress (e.g., filled a required field)
    - 0.5: minor progress (e.g., navigated closer)
    - 0.0: no progress (e.g., clicked irrelevant element)
    - -0.5: regression (e.g., closed important dialog)
    """
    return float(llm_judge(prompt))
```

这种 LLM-as-judge 的中间奖励类似 [第 20 章 Process Reward Model](../chapter20_prm_search/inference-time-search) 的思想——用 LLM 评估中间步质量。

### 与正向课程的对比

ComputerRL 论文报告了对比实验：

| 方法                    | OSLevel-3 成功率 | 平均步数 | 训练成本 |
| ----------------------- | ---------------- | -------- | -------- |
| 正向课程 + 终态奖励     | 12.3%            | 47       | 1×       |
| 正向课程 + 进度奖励     | 28.7%            | 35       | 2.3×     |
| **反向课程 + 进度奖励** | **51.2%**        | **28**   | 2.8×     |

反向课程把成功率从 12% 拉到 51%，但训练成本也增加 2.8 倍——主要是进度评估器 LLM 的调用开销。

## CogAgent 与 高分辨率视觉的代价

智谱 CogAgent-9B（arXiv:2408.16500）走另一条路：**用更高分辨率视觉编码换准确度**。

### 高分辨率视觉分支

标准 VLM 输入图像分辨率 448×448，CogAgent 用 1120×1120——4 倍像素意味着视觉 token 数翻 4 倍，但能看清 UI 上的小字（如表格内的 9 号字、PowerPoint 工具栏图标）。

CogAgent 的架构巧思是**双分支融合**：

```
┌──────────────────────────────────────────┐
│ 输入截图（1120×1120）                    │
└────────────┬─────────────────────────────┘
             ↓
   ┌─────────┴─────────┐
   │                   │
   ↓                   ↓
高分辨率分支         低分辨率分支
(EVA-CLIP)          (SigLIP)
1120×1120            448×448
→ 3136 tokens        → 256 tokens
   │                   │
   └─────────┬─────────┘
             ↓
        Cross-Attention
             ↓
         LLM Decoder
```

低分辨率分支提供全局上下文（"这是一个购物页面"），高分辨率分支提供细节（"购物车按钮在右上角"）。两者通过 cross-attention 融合，避免 LLM 处理全部 3136 token 的计算开销。

### 准确度 vs 延迟的权衡

代价是计算成本：高分辨率视觉 token 让推理慢 3–5 倍。

| 配置             | 视觉 token | 推理延迟 | OSWorld 准确率 |
| ---------------- | ---------- | -------- | -------------- |
| 448×448 单分支   | 256        | 0.8s     | 38.2%          |
| 1120×1120 单分支 | 3136       | 4.2s     | 47.5%          |
| **双分支融合**   | 3392       | 1.6s     | **46.8%**      |

双分支在保持接近高分辨率准确率的同时，延迟只增加 1 倍。这种 trade-off 是 GUI Agent 的核心工程决策——准确度 vs 延迟。

## 工业落地的三大挑战

把上述系统从论文搬进生产环境，会遇到三个论文里没充分讨论的挑战。

### 环境分布偏移

论文里的训练环境是 OSWorld、AndroidWorld 等可控 benchmark。生产环境是真实用户的电脑——每个人的系统版本、浏览器插件、字体大小都不一样。

**对策**：

- **数据多样化**：UI-TARS-2 收集 50+ 不同 Windows/macOS/Linux 配置的训练环境
- **域随机化**（domain randomization）：训练时随机改 UI 主题、字体、分辨率
- **持续学习**：部署后收集失败案例，周期性 retrain

### 长尾任务

论文 benchmark 都是"主流任务"（订机票、查日历、写邮件）。生产环境中用户会问"帮我把这台电脑的 BIOS 改成 UEFI 模式"——这种任务训练数据极少。

**对策**：

- **任务分层**：常见任务用训练好的策略；罕见任务退化为"tree search + LLM 规划"
- **人在回路**（human-in-the-loop）：低置信度时主动询问用户

### 安全边界

GUI Agent 能执行破坏性操作——删文件、转账、发邮件。生产环境必须有明确的安全边界。

**对策**：

- **白名单动作**：默认禁止 `rm -rf`、转账超过 $100、群发邮件
- **二次确认**：高风险操作前弹窗让用户确认
- **审计日志**：所有操作记录可回溯

详见 [25.3 指令层级与 Prompt Injection 防御](./safety-swarm)。

## 本节总结

中国实验室在 GUI Agent RL 训练上形成了清晰的四条路线：

- **UI-TARS-2**：端到端 VLM + 反思增强，把 Computer Use 当纯 LLM RL 问题
- **Open-AutoGLM**：中文 GUI 数据合成 + 跨平台统一，工程完整度最高
- **MobileRL**：渐进难度课程，专攻移动端 App
- **ComputerRL**：反向课程 + 中间探索奖励，攻长程任务
- **CogAgent**：高分辨率视觉编码，攻小字识别

四条路线不是互斥的——UI-TARS-2 后期也加入了反思课程（类 MobileRL 思想），Open-AutoGLM 也用了反向课程（类 ComputerRL 思想）。**工业级系统往往是多种思想的组合**。

下一节 [25.3 指令层级与 Prompt Injection 防御](./safety-swarm) 转向安全——agent 真正部署到用户电脑后，怎么防止恶意网页、伪造 UI、跨应用攻击劫持。
