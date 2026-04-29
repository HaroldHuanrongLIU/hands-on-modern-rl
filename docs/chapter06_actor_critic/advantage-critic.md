# 优势函数与 Critic 训练

第 5 章末尾我们发现：减掉基线 $V(s)$ 可以降低策略梯度的方差，而不改变梯度的方向。本节将深入这个关键洞察，引出优势函数——它是连接 Actor 和 Critic 的桥梁。

## 从基线到优势函数

回忆 REINFORCE 的策略梯度：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$$

$G_t$ 是从当前步到 episode 结束的总回报。问题在于 $G_t$ 波动巨大——同一个策略、同一个状态，跑两次可能拿到完全不同的 $G_t$。

减掉基线 $V(s)$ 后：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot (G_t - V(s))$$

括号里的 $G_t - V(s)$ 就是**优势函数（Advantage Function）** 的一种估计：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

它的含义是：**做了这个动作，比"平均能拿多少分"好了多少。**

- $A > 0$：这个动作比预期好，应该多选
- $A < 0$：这个动作比预期差，应该少选
- $A \approx 0$：这个动作和预期差不多

用下棋类比：$V(s)$ 是"这个棋局整体胜率 60%"，$Q(s, \text{出车})$ 是"走车之后胜率 75%"。优势 $A = 75\% - 60\% = 15\%$，说明走车比平均水平好了 15%，是个好选择。

## 用 TD Error 估计优势

优势函数的理论定义是 $A = Q - V$，但实际中我们通常不直接计算 $Q$。利用 TD Error，可以用一种更高效的方式来估计 $A$：

$$A(s,a) \approx r + \gamma V(s') - V(s) = \delta$$

这就是第 3 章介绍的 TD Error！它衡量的是"走了一步之后，实际结果比预测好了多少"。用 TD Error 替代 $G_t$ 作为策略梯度的信号，有两个好处：

1. **不需要等 episode 结束**——每走一步就能更新（$G_t$ 需要跑完一整局）
2. **方差更低**——$\delta$ 只涉及一步的随机性（$G_t$ 涉及整条轨迹的随机性）

这是 MC → TD 的演进在策略空间的再现：REINFORCE 用 $G_t$（MC），Actor-Critic 用 $\delta$（TD）。

## Critic：V(s) 的神经网络实现

Critic 就是一个估计 $V(s)$ 的神经网络。它的输入是状态 $s$，输出是一个标量——对 $V(s)$ 的估计。

```
Actor（策略网络）           Critic（价值网络）
  输入：状态 s                 输入：状态 s
  输出：π(a|s) 概率分布       输出：V(s) 标量
  作用：选动作                作用：评估状态价值
```

但 Critic 怎么训练？它怎么学会准确估计 $V(s)$？这就要用到第 3 章速览过的 DP、MC、TD 三种方法。

## 训练 Critic 的三种方法

第 3 章介绍了三种估计 $V(s)$ 的方法。现在 Critic 需要学习 $V(s)$，这三种方法就是 Critic 的训练策略。

### DP：知道模型时的最优方案

如果完全知道环境的 $P$ 和 $R$，可以直接用贝尔曼方程迭代：

$$V(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right]$$

精确收敛，但现实中几乎不可能知道完整的 $P$ 和 $R$。DP 更多是理论基准——它告诉你"知道一切时的最优答案"。

### MC：用完整轨迹估计 V

跑完一个完整的 episode，用实际回报 $G_t$ 来更新 Critic：

$$V_\phi(s) \leftarrow V_\phi(s) + \alpha [G_t - V_\phi(s)]$$

$G_t - V_\phi(s)$ 就是 Critic 的预测误差——实际拿了 $G_t$ 分，但之前预测是 $V_\phi(s)$ 分，差多少就补多少。在神经网络实现中，这等价于最小化 Critic 的均方误差损失：

$$L_{\text{Critic}} = (G_t - V_\phi(s))^2$$

MC 给出无偏估计（用的是真实回报），但方差大，且必须等到 episode 结束。

### TD：走一步就更新（实际首选）

用 TD Error 来更新 Critic：

$$V_\phi(s) \leftarrow V_\phi(s) + \alpha [r + \gamma V_\phi(s') - V_\phi(s)]$$

在神经网络实现中，Critic 的训练目标是让 $V_\phi(s)$ 接近 TD Target $r + \gamma V_\phi(s')$：

$$L_{\text{Critic}} = (r + \gamma V_\phi(s') - V_\phi(s))^2 = \delta^2$$

最小化 $\delta^2$ 就是让 Critic 的预测越来越准确。TD 方法的优势是：不需要等 episode 结束，每走一步就能更新；方差比 MC 低得多，因为有 $V_\phi(s')$ 作为稳定的"锚点"。

### 三种方法的对比

|                     | DP       | MC     | TD                 |
| ------------------- | -------- | ------ | ------------------ |
| 用于 Critic 训练？  | 理论基准 | 可以用 | **实际首选**       |
| 需要 episode 结束？ | 不需要   | 需要   | 不需要             |
| 无偏？              | 是       | 是     | 否（有偏但方差低） |

实际中，Actor-Critic 几乎都用 TD 方法来训练 Critic——它允许在线学习，方差低，且与 Actor 的更新节奏一致（都是走一步更新一次）。

## Critic 训练的完整流程

将以上内容整合，Critic 的训练流程如下：

1. 在状态 $s$ 下，Actor 选择动作 $a$，环境返回 $r$ 和 $s'$
2. Critic 计算当前预测 $V_\phi(s)$ 和下一步预测 $V_\phi(s')$
3. 计算 TD Error：$\delta = r + \gamma V_\phi(s') - V_\phi(s)$
4. 用 $\delta^2$ 作为损失更新 Critic 的参数 $\phi$
5. 用 $\delta$ 作为优势估计更新 Actor 的参数 $\theta$

Critic 的参数 $\phi$ 沿着"让 $\delta^2$ 更小"的方向更新——预测越来越准。Actor 的参数 $\theta$ 沿着"让正 $\delta$ 的动作概率更高"的方向更新——选越来越好。两者形成良性循环。

## 下一步

有了优势函数和 Critic 的训练方法，我们就可以搭建完整的 Actor-Critic 架构了。下一节将展示 Actor 和 Critic 如何协作，以及 TD Error 如何替代 $G_t$ 实现逐步更新。[Actor-Critic 架构](./actor-critic)
