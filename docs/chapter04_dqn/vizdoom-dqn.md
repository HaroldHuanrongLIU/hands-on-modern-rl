# 4.7 动手：3D 第一人称 ViZDoom

Atari 游戏是平面的——你从上帝视角俯瞰整个场景，球拍和球的位置一览无余。但真实世界不是这样的。你是一个第一人称射击游戏里的士兵，眼前只有屏幕上的一小片视野：走廊转角可能藏着敌人，身后可能传来脚步声，而你只有零点几秒的时间决定是开枪还是躲避。

这就是 ViZDoom 的世界。

回顾一下我们走过的路：CartPole 输入 4 个数字，Atari 输入 2D 像素帧，ViZDoom 则是 3D 第一人称视角——同样的 DQN 算法，但环境复杂度再次跃升。

## 从 Atari 到 ViZDoom：什么变了？

| | Atari | ViZDoom |
|--|--|--|
| 视角 | 上帝视角（俯瞰全场景） | 第一人称（只有眼前的画面） |
| 输入 | 84×84×4 像素帧 | 240×320 RGB（需预处理） |
| 动作空间 | 4-18 个离散动作 | 7 个（前进/后退/左转/右转/射击等） |
| 场景复杂度 | 2D 精灵图 | 3D 渲染（走廊、房间、障碍物） |
| 额外挑战 | 基本无 | 部分可观测性（看不到身后）、导航 |

最大的挑战是**部分可观测性**。在 Pong 里，你永远能看到球和两个球拍。但在 ViZDoom 的走廊里，敌人可能就在你转身之后——你根本看不到。这意味着单帧信息远远不够，帧堆叠在这里不再是锦上添花，而是生存必需。

## ViZDoom 环境简介

ViZDoom 于 2016 年由波兰雅盖隆大学发布，基于经典第一人称射击游戏 Doom。它轻量快速（基于 Doom 引擎，即使没有 GPU 也能快速渲染），高度可定制（可以用 WAD 文件自定义地图），2022 年加入了 Farama 基金会，与 Atari、MuJoCo 等环境并列成为 RL 社区的核心基础设施。

安装：

```bash
pip install vizdoom
```

ViZDoom 内置了多个场景，从简单到复杂：

| 场景 | 描述 | 难度 |
|------|------|------|
| `DEADLY_CORRIDOR` | 在走廊中前进，击杀敌人，躲避攻击 | 低 |
| `DEFEND_THE_CENTER` | 站在圆形竞技场中央，抵御四面八方的敌人 | 中 |
| `HEALTH_GATHERING` | 在有毒的地面上收集医疗包维持生命 | 中 |
| `DEATHMATCH` | 自由地图上的多智能体死斗 | 高 |

我们选 `DEADLY_CORRIDOR` 作为起点——它是最简单的入门场景，但已经比 Atari 的任何游戏都更具挑战性。

```
Deadly Corridor 场景示意（第一人称视角）

┌────────────────────────────────────────┐
│  ┌──┐                                  │
│  │敌│     ← 走廊尽头站着一个敌人        │
│  │人│                                  │
│  └──┘                                  │
│          ● ← 子弹                       │
│                                        │
│                                        │
│           ╱  ← 自己手持的武器           │
│    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━        │
│                                        │
│  HP: 100   AMMO: 50   KILLS: 0         │
└────────────────────────────────────────┘
 你只能看到眼前的画面，身后和转角完全未知
```

## 环境搭建

```python
import gymnasium as gym
import vizdoom.gym  # 注册 ViZDoom 环境

env = gym.make("VizdoomDeadlyCorridor-v0")
print(f"观测空间: {env.observation_space}")  # Box(0, 255, (240, 320, 3), uint8)
print(f"动作空间: {env.action_space}")       # Discrete(7)
```

7 个离散动作：前进、后退、左转、右转、射击、开火、不动。观测空间是一帧 240×320 的 RGB 图像——和人类玩家看到的画面一模一样。

## 预处理：从 240×320 到 84×84×4

和 Atari 一样的思路：灰度化、裁剪、缩放、帧堆叠。

```python
import numpy as np
from collections import deque

def preprocess_frame(frame):
    """灰度化 + 裁剪 HUD + 缩放到 84×84"""
    frame = frame.mean(axis=2)           # 灰度化
    frame = frame[30:-10, 30:-30]        # 裁掉底部 HUD 和两侧黑边
    frame = np.resize(frame, (84, 84))   # 缩放
    frame = frame / 255.0                # 归一化
    return frame

class FrameStack:
    """堆叠最近 4 帧，提供运动信息"""
    def __init__(self, num_stack=4):
        self.frames = deque(maxlen=num_stack)

    def reset(self, frame):
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return self._get_state()

    def step(self, frame):
        self.frames.append(frame)
        return self._get_state()

    def _get_state(self):
        return np.stack(self.frames, axis=0)  # (4, 84, 84)
```

注意 `frame[30:-10, 30:-30]` 裁掉了 ViZDoom 画面底部的 HUD（血量、弹药显示）和两侧的黑边。这些 UI 元素对决策没有帮助，留着只会浪费网络的表示能力。

## CNN Q-Network

网络和 Atari 的 CNN-DQN 完全一样——毕竟任务本质相同：从像素中提取特征，输出每个动作的 Q 值。

```python
import torch
import torch.nn as nn

class CNN_QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

和 Atari 唯一的区别是输入不需要 `/255.0`——因为我们的 `preprocess_frame` 已经做了归一化。

## 完整训练代码

```python
import random

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

def train(env, num_episodes=2000, batch_size=32, gamma=0.99,
          lr=1e-4, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    num_actions = env.action_space.n
    policy_net = CNN_QNetwork(num_actions)
    target_net = CNN_QNetwork(num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    frame_stack = FrameStack()

    epsilon = epsilon_start
    rewards_history = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(preprocess_frame(obs))
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = frame_stack.step(preprocess_frame(obs))
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    target_q = rewards + gamma * target_net(next_states).max(1)[0] * (1 - dones)
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
                optimizer.step()

            if done:
                break

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_history.append(total_reward)
        if episode % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode} | Avg Reward: {avg:.1f} | ε: {epsilon:.3f}")

    return policy_net, rewards_history
```

启动训练：

```python
env = gym.make("VizdoomDeadlyCorridor-v0")
policy_net, rewards = train(env, num_episodes=2000)
```

ViZDoom 的 Deadly Corridor 比 Pong 难得多——首次训练可能需要 1000-2000 个 episode 才能看到明显的学习效果。如果 reward 一直不涨，可以尝试降低学习率（如 5e-5）、增大回放池（100,000）、或延迟训练（先收集 10,000 步经验再开始更新）。

## 测试训练好的智能体

```python
def evaluate(env, policy_net, num_episodes=5):
    frame_stack = FrameStack()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(preprocess_frame(obs))
        total_reward = 0

        while True:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            state = frame_stack.step(preprocess_frame(obs))
            total_reward += reward
            if terminated or truncated:
                print(f"Episode {ep}: Reward = {total_reward:.1f}")
                break

evaluate(env, policy_net)
```

## 三个实战的对比

走完 CartPole、Atari、ViZDoom 三个实战，让我们对比一下 DQN 在不同复杂度环境中的表现：

| | CartPole | Atari Pong | ViZDoom |
|--|--|--|--|
| 状态维度 | 4 | 28,224 | 28,224（但信息密度更低） |
| 网络 | MLP（2 层 128） | CNN（3 层卷积 + 2 层全连接） | 同 Atari |
| 训练时间 | 几分钟（CPU） | 数小时（GPU） | 数小时（GPU） |
| 可观测性 | 完全可观测 | 完全可观测 | 部分可观测 |
| DQN 的瓶颈 | 无（太简单） | CNN 特征提取 | 部分可观测性 + 导航 |
| 核心收获 | 理解 DQN 组件 | CNN 从像素学习 | 真实世界的复杂性 |

算法始终是 DQN，改变的是环境——从简单到复杂，从完全可观测到部分可观测。这揭示了一个重要的洞见：深度强化学习的核心瓶颈往往不在算法本身，而在如何让算法与复杂环境有效交互。ViZDoom 中的部分可观测性、导航需求和多目标平衡，都是真实世界 RL 应用中常见的挑战。

<details>
<summary>思考题：ViZDoom 中只用 4 帧堆叠够吗？有没有更好的方式处理部分可观测性？</summary>

4 帧在简单场景（如 Deadly Corridor）中勉强够用，但在更复杂的地图中可能不足。几种改进方案：

1. **增加帧堆叠数量**：从 4 帧增加到 8 或 16 帧，覆盖更长的时间窗口。但这也增加了输入维度和计算量。
2. **循环神经网络（RNN）**：用 LSTM 或 GRU 替代帧堆叠，让网络维护一个隐藏状态来"记住"历史信息。这在理论上更优雅，但训练更复杂。
3. **记忆增强网络**：如 Memory Networks 或 Transformer 注意力机制，让网络显式地读写外部记忆。

在实际研究中，RNN + 帧堆叠的组合是最常见的方案——帧堆叠捕捉短期运动，RNN 捕捉长期依赖。

</details>
