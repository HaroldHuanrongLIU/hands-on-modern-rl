# 第 25 章 · Computer Use 与 GUI Agent

> [第 22 章 Agentic RL](../chapter22_agentic/intro) 让 LLM 学会调用工具、阅读工具返回、在多轮交互中纠错——这是单 agent 的形态。但当任务从"写一段函数"升级到"在我电脑里订一张下周三去上海的机票"，agent 必须跨过的鸿沟是：**像人一样看屏幕、点鼠标、敲键盘**。本章解决两件事：(1) Computer Use 范式下，agent 如何把 GUI 像素流映射为原子动作并用 RL 优化（25.1–25.2）；(2) GUI Agent 的训练实践（[25.2](./training)）与安全防御（[25.3](./safety-swarm)）。

## 25.1 Computer Use 范式

[第 22 章工具使用](../chapter22_agentic/tool-use-and-trajectory)中的工具是**结构化 API**——`def search(query): return results`，输入输出都是字符串。但真实世界里大量软件只有一种接口：**GUI**。浏览器、Excel、企业内部 OA、Photoshop、游戏——它们没有公开 API，只有屏幕和鼠标键盘事件。

**Computer Use** 范式把整个操作系统当作 agent 的环境：

- **观察**：屏幕截图 $o_t \in \mathbb{R}^{H \times W \times 3}$（每秒 1–4 帧）
- **动作**：原子 GUI 事件（鼠标移动、点击、滚动、键盘按键、等待）
- **奖励**：任务完成的二值信号（"是否成功订到机票"）

这种 MDP 与传统 RL benchmark 截然不同。CartPole 状态 4 维、动作 2 维、单步奖励稠密；Computer Use 状态上百万维像素、动作空间是混合类型、奖励稀疏到只在最后一步给出。

### 主流产品

| 产品                | 机构           | 发布    | 特征                                  |
| ------------------- | -------------- | ------- | ------------------------------------- |
| **Computer Use**    | Anthropic      | 2024.10 | Claude 3.5 Sonnet 原生支持截图-动作对 |
| **Operator**        | OpenAI         | 2025.01 | CU Agent + GPT-4o 视觉，浏览器专用    |
| **Project Mariner** | Google         | 2024.12 | Gemini 驱动，深度集成 Chrome          |
| **UI-TARS-2**       | ByteDance Seed | 2025.09 | 端到端 VLM + RL 训练                  |
| **Open-AutoGLM**    | 智谱           | 2025.12 | 开源 AutoGLM 升级版                   |

### 核心动作空间

Anthropic Computer Use 的动作原语定义如下（OpenAI Operator、Google Mariner 大同小异）：

```python
ACTIONS = {
    "click":      {"x": int, "y": int, "button": "left|right|middle"},
    "double":     {"x": int, "y": int},
    "drag":       {"start": [x,y], "end": [x,y]},
    "type":       {"text": str},
    "key":        {"keys": "ctrl+c|enter|tab"},   # 组合键
    "scroll":     {"x": int, "y": int, "dy": int},
    "wait":       {"ms": int},
    "screenshot": {},
    "done":       {"summary": str},
}
```

注意三个关键设计：

1. **动作是离散 token + 连续坐标的混合**——`click` 既要选 token 又要预测 $(x, y)$。这是 LLM 难以天然处理的：标准 transformer 输出离散 token，而 $(x, y) \in [0, W] \times [0, H]$ 是连续值
2. **截图频率远低于人眼**——人每秒看到 30–60 帧，Computer Use 每秒 1–4 帧。这意味着状态转移 $P(s_{t+1} \mid s_t, a_t)$ 在两个观察之间有大量隐藏状态变化
3. **等待动作 (`wait`)**——GUI 动画、网络加载、弹窗过渡都需要等待。这是传统 RL 里没有的"主动消耗时间步"动作

### MDP 形式化

定义 Computer Use MDP 为 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, T)$：

$$\mathcal{S} = \{\text{screenshots}\}, \quad \mathcal{A} = \{\text{click, type, scroll, key, wait, done}\}$$

任务描述（如"帮我把这份 PDF 转成 Markdown"）作为初始 prompt $q$ 拼接到每步观察前。策略为条件分布：

$$\pi_\theta(a_t \mid q, o_{1:t}, a_{1:t-1})$$

奖励 $R$ 通常是稀疏二值：$r_T = \mathbb{1}[\text{task completed}]$，中间步 $r_{t<T} = 0$。这让信用分配（credit assignment）极难——一次浏览器自动化任务可能 50 步动作，只有最后一步拿到 reward，前面哪步对、哪步错无从分辨。

::: warning RL 的真正难点
稀疏奖励 + 长时序（50–500 步）+ 高维观察（截图 1344×756 像素）+ 混合动作空间——Computer Use 同时踩中 RL 的所有痛点。这也是为什么 2024 年前几乎所有 Computer Use 系统都是**纯提示工程**（prompt engineering），直到 2025 年 RL 训练才真正进入工业落地。
:::

## GUI Grounding RL

Computer Use 的第一步难题不是决策，而是**定位**：模型怎么知道"提交"按钮在屏幕的哪个 $(x, y)$？

### Set-of-Mark 提示

Yang et al. 2023 提出 **Set-of-Mark (SoM)** 提示：先用 OCR / 目标检测把屏幕上所有可交互元素框出来，编号 $1, 2, \ldots, K$，agent 输出动作时只需引用编号：

```
[屏幕截图 + 框 1: 输入框 "用户名", 框 2: 输入框 "密码", 框 3: 按钮 "登录"]

Agent: type("alice") → click(框 1) → type("***") → click(框 2) → click(框 3)
```

这把连续坐标预测简化为**离散选择**——但代价是依赖外部检测器，且检测器遗漏元素时 agent 无能为力。

### 视觉 Grounding

UI-TARS、CogAgent 等端到端模型走另一条路：**让 VLM 直接输出坐标**。模型架构分两个 head：

$$\text{VLM}(o_t, q) \to \underbrace{(\text{thought}, \text{action token})}_{\text{language head}} + \underbrace{(x, y) \in [0,1]^2}_{\text{grounding head}}$$

grounding head 通常是一个 MLP，输出归一化坐标 $(x, y) \in [0, 1]^2$，再乘以屏幕尺寸映射到像素。

训练 grounding 用**监督模仿**：人工标注"按钮中心点 $(x_i, y_i)$"，loss 为：

$$\mathcal{L}_{\text{ground}} = \frac{1}{N}\sum_i \|\hat{p}_\theta(o_i) - p_i\|_2^2$$

但纯监督有个问题：**模型可能输出空地**。监督只学了"按钮在哪"，没学"按钮要按下去"。RL 在此发挥作用。

### Grounding + 决策的联合 RL

把 grounding 和动作选择放在同一个 PPO 目标里：

$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \gamma^t r_t\right] - \beta \cdot \mathcal{L}_{\text{ground}}(\theta)$$

第二项是 grounding 的监督 loss，作为正则项保留。这种**SFT + RL 联合训练**是 GUI Agent 的标准配方——先模仿学会基础操作，再用 RL 优化任务成功率。

UI-TARS-2 把这个思想推到极致：把思维链（thought）、动作（action）、坐标（coordinate）三部分作为**单一序列**输出，用 RL 同时优化：

```python
def ui_tars_forward(self, screenshot, task):
    # 编码图像
    visual_tokens = self.vision_encoder(screenshot)  # [B, N_vis, d]

    # 拼接 prompt
    prompt = f"<task>{task}</task>\n<image>{visual_tokens}</image>\n"

    # 自回归生成 thought + action + coord
    # 关键：coord 用特殊 token <coord_x> <coord_y> 包裹
    output = self.llm.generate(prompt, max_new_tokens=256)

    # 解析输出："<thought>...</thought>\n<action>click</action>\n<coord>(0.45, 0.62)</coord>"
    thought, action, coord = parse_action(output)
    return thought, action, coord
```

### RL 训练数据生成

真实 GUI 任务没法大规模人工标注——一个 50 步的浏览器任务人工演示成本约 30 分钟。解决方案是**程序化任务生成**：

1. **真实网站爬取**：UI-TARS 收集 200+ 真实 App，每个 App 自动生成 1000+ 任务模板
2. **环境快照**：录制人类操作过程，保存每步截图 + 动作，作为 SFT 数据
3. **任务验证器**：用程序化规则检查任务是否完成（"页面是否出现了成功提示"）
4. **RL rollout**：agent 在虚拟机中执行任务，验证器给出最终 reward

```python
class GUIEnv:
    def reset(self, task_id):
        self.vm.restore_snapshot(task_id)  # 恢复虚拟机到任务初始状态
        self.task = self.tasks[task_id]
        return self.screenshot()

    def step(self, action):
        self.vm.execute(action)            # 鼠标键盘事件注入
        obs = self.screenshot()
        done = self.task.verifier(obs, self.vm.state)
        reward = 1.0 if done else 0.0
        return obs, reward, done, {}
```

::: details 为什么不用真实鼠标
直接控制操作系统的鼠标会让 agent 与人类用户的输入冲突。工业实践是在**虚拟机 + VNC 远程桌面**里跑 agent，鼠标键盘事件通过 RDP/VNC 协议注入，agent 和人类用户隔离。这也是为什么 Computer Use 系统通常 1 秒只能执行 1–2 个动作——截图 + VNC 注入的延迟。
:::

## 本节总结

Computer Use 把 GUI 像素流当作 RL 状态空间，把鼠标键盘事件当作动作空间，这让传统 RL 的所有难题（稀疏奖励、长时序、高维观察）同时放大。**Set-of-Mark** 与**视觉 Grounding** 是解决"定位"问题的两条主流路线：前者依赖外部检测器简化动作空间，后者用 VLM 端到端输出坐标。

下一节 [25.2 GUI Agent 训练实践](./training) 走进工业实战——你会看到 UI-TARS-2、AutoGLM、MobileRL、ComputerRL 等系统如何把这套理论变成可复现的训练 pipeline。
