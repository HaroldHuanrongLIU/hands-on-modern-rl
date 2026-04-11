# 动手：PyBullet 机器人仿真与连续控制

前面的章节里，我们的智能体要么在玩游戏（CartPole、Atari、ViZDoom），要么在抽象的物理世界里奔跑（MuJoCo 的 HalfCheetah）。现在，我们要进入一个更贴近真实应用的世界——**机器人控制**。

PyBullet 是 Bullet3 物理引擎的 Python 接口。Bullet3 最初由游戏行业开发，用于实时碰撞检测和刚体模拟，后来被广泛采用为机器人强化学习的仿真平台。和 MuJoCo 相比，PyBullet 完全免费开源，内置了丰富的机器人模型，是工业界和学术界做连续控制 RL 的常用选择。

## PyBullet vs MuJoCo

| | MuJoCo | PyBullet |
|--|--|--|
| 许可证 | 需要许可证（DeepMind 免费版有限制） | 完全免费开源 |
| 物理精度 | 更高的接触力模拟精度 | 足够用于大部分 RL 任务 |
| 渲染速度 | 快 | 快（基于 OpenGL） |
| 机器人模型 | 少量内置，需要 XML 定义 | 丰富的内置模型（URDF 格式） |
| 社区生态 | RL 社区标准基准 | 机器人社区广泛使用 |
| Gymnasium 集成 | Gymnasium-MuJoCo | PyBullet Gymnasium |

## 环境搭建

```bash
pip install pybullet gymnasium
```

启动一个简单的场景看看效果：

```python
import pybullet as p
import pybullet_data

# 连接物理引擎（GUI 模式，会弹出一个 3D 窗口）
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载地面和机器人
p.loadURDF("plane.urdf")
robot = p.loadURDF("kuka_iiwa/kuka.urdf", [0, 0, 0])

# 查看机器人的关节数量
num_joints = p.getNumJoints(robot)
print(f"Kuka IIWA 机械臂有 {num_joints} 个关节")

# 每个关节的信息
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    print(f"  关节 {i}: {info[1].decode()} | 类型: {info[2]} | 下限: {info[8]:.2f} | 上限: {info[9]:.2f}")

p.disconnect()
```

PyBullet 内置了多种经典机器人模型：

| 模型 | 类型 | 关节数 | 典型任务 |
|------|------|--------|----------|
| Kuka IIWA | 机械臂 | 7 | 抓取、放置 |
| UR5 | 机械臂 | 6 | 灵巧操作 |
| Ant | 四足机器人 | 8 | 行走、奔跑 |
| Humanoid | 人形机器人 | 17 | 步行、平衡 |
| Racecar | 移动机器人 | 2 | 导航 |

## PyBullet Gymnasium：开箱即用的 RL 环境

社区维护了 `pybullet-gymnasium`，提供与 Gymnasium 兼容的标准化接口：

```bash
pip install pybullet-gymnasium
```

```python
import gymnasium as gym
import pybulletgym  # 注册 PyBullet 环境

# 创建 Ant 环境
env = gym.make("AntPyBulletEnv-v0")

print(f"观测空间: {env.observation_space.shape}")  # (28,)
print(f"动作空间: {env.action_space.shape}")        # (8,)  连续值 ∈ [-1, 1]
print(f"动作范围: {env.action_space.low} ~ {env.action_space.high}")

obs, info = env.reset()
for step in range(200):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

注意动作空间是 `Box(8,)`——一个 8 维连续向量，每个分量代表一个关节的目标力矩或角度。这和之前 CartPole 的 `Discrete(2)`（左/右）有本质区别。

## 用 SAC 训练 Ant 行走

以下是用 SAC 算法训练四足机器人行走的完整流程（使用 Stable-Baselines3）：

```python
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import pybulletgym

# 创建训练和评估环境
train_env = gym.make("AntPyBulletEnv-v0")
eval_env = gym.make("AntPyBulletEnv-v0")

# 创建 SAC 模型
model = SAC(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,           # 目标网络软更新系数
    gamma=0.99,
    verbose=1,
)

# 评估回调：每 5000 步评估一次，保存最优模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/sac_ant",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
)

# 开始训练
model.learn(total_timesteps=500_000, callback=eval_callback)

# 保存最终模型
model.save("./models/sac_ant_final")
```

## 观察训练过程

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载最优模型并评估
model = SAC.load("./models/sac_ant_best")

rewards = []
for ep in range(20):
    obs, _ = eval_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    rewards.append(total_reward)

print(f"平均奖励: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
print(f"最高奖励: {np.max(rewards):.1f}")

eval_env.close()
```

## 从仿真到现实：Sim-to-Real 的挑战

在 PyBullet 中训练出能跑的 Ant，并不意味着你造一个真实的四足机器人它就能跑。仿真和现实之间存在**仿真鸿沟（Reality Gap）**：

- **物理参数不精确**：摩擦系数、质量分布、关节阻尼——仿真里的值和真实机器人总有偏差
- **传感器噪声**：真实的 IMU、编码器有噪声和延迟，仿真里通常忽略
- **延迟与通信**：真实系统存在控制延迟，仿真里是即时响应

常见的应对策略：

| 策略 | 思路 |
|------|------|
| 域随机化（Domain Randomization） | 训练时随机化仿真参数，让策略适应各种条件 |
| 噪声注入 | 给观测和动作加噪声，模拟传感器误差和执行器误差 |
| 系统辨识 | 先用真实机器人的数据校准仿真参数 |
| 渐进迁移 | 先在仿真里训练，再在真机上微调 |

## 小结

PyBullet 为连续控制 RL 提供了一个免费、灵活的仿真平台。在这一节中，我们：

- 了解了 PyBullet 作为物理仿真引擎在 RL 中的角色
- 用 PyBullet Gymnasium 创建了连续动作空间的环境
- 用 SAC 训练了四足机器人行走
- 讨论了从仿真到现实的 Sim-to-Real 挑战

核心要点：**连续控制的关键不在于离散还是连续本身，而在于策略如何参数化和探索**——高斯策略（SAC）还是确定性策略（TD3），取决于你的任务特点。
