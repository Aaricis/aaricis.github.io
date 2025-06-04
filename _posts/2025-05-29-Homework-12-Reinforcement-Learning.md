---
title: 李宏毅-ML2022-HW12-Reinforcement Learning
date: 2025-05-29 15:07:00 +/-8
categories: [Machine Learning 2022]
tags: [Hung-yi Lee]     # TAG names should always be lowercase
math: true
---

## Task Description

使用深度强化学习算法执行OpenAI Gym的Lunar Lander（月球着陆器）任务。LunarLander是一个经典的强化学习环境，模拟航天器在月球表面着陆的任务。通过合理设计策略和调参，智能体可以学会精准着陆，该环境是验证强化学习算法的理想测试平台。

## Environment

Lunar Lander的目标是控制航天器降落在月球表面的两个黄色旗帜之间，同时减少燃料消耗和冲击力。

### 奖励机制（Reward）

- 成功着陆：+100~+140 分
- 坠毁：-100 分
- 每使用一次主引擎：-0.3 分
- 靠近目标点：正奖励；远离：负奖励
- 腿接触地面：+10 分/腿

### 状态空间（State Space）

状态是一个**8维向量**，包括：

| 索引 | 描述                       | 范围       |
| :--- | :------------------------- | :--------- |
| 0    | 航天器X坐标                | [ -∞, +∞ ] |
| 1    | 航天器Y坐标                | [ -∞, +∞ ] |
| 2    | 水平速度                   | [ -∞, +∞ ] |
| 3    | 垂直速度                   | [ -∞, +∞ ] |
| 4    | 角度（弧度，0=竖直）       | [ -π, +π ] |
| 5    | 角速度                     | [ -∞, +∞ ] |
| 6    | 左腿是否触地（1=是，0=否） | {0, 1}     |
| 7    | 右腿是否触地（1=是，0=否） | {0, 1}     |

### 动作空间（Action Space）

| 动作编号 | 描述               |
| :------- | :----------------- |
| 0        | 不点火             |
| 1        | 点燃左方向引擎     |
| 2        | 点燃主引擎（向下） |
| 3        | 点燃右方向引擎     |

### 回合终止条件

- **成功**：航天器平稳着陆（速度与角度在阈值内）；
- **失败**：
  - 坠毁（速度过快或角度过大）；
  - 飞出画面边界；
  - 超过最大步数（默认1000步）。

> 更多详情，参见[官方文档](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)

## 思路

助教发布作业时，Colab的Python版本为3.9，笔者做作业的当下Colab的Python版本已升级为3.11。原先的依赖版本已经不能在Python 3.11环境使用，笔者升级依赖版本，重新配置了运行环境。因此，环境实际Baseline已与助教当时提供的不同，助教Baseline不具有参考意义。

出于学习的目的，下面仅根据助教的提示，实现各个Baseline对应的trick，重点关注score的相对提升，不关注具体的值。

### Simple Baseline (Score: [0, 110])

Your final reward is : 75.46

跑通Sample Code

```python
NUM_BATCH = 1000
```

### Medium Baseline (Score: [110, 180])

Your final reward is : 156.89


- 设置NUM_BATCH;

```python
NUM_BATCH = 500
```

- 将reward计算方式改为discounted reward;

删除原本使用的Immediate Reward:


```python
# rewards.append(reward)
```

改成Discounted Reward:


```python
if done:
    final_rewards.append(reward)
    total_rewards.append(total_reward)

    T = len(seq_rewards)  # total steps
    gamma = 0.99
    discounted_rewards = [0] * T  # initialize the rewards

    # calculated backwards
    cumulative = 0
    for t in reversed(range(T)):
        cumulative = seq_rewards[t] + gamma * cumulative
        discounted_rewards[t] = cumulative

    rewards += discounted_rewards
    break
```

### Strong Baseline (Score: [180, 275])

Your final reward is : 232.88

#### Actor-Critic

Actor-Critic算法是强化学习中的一种混合框架，结合了策略梯度（Actor）和值函数估计（Critic）的优势，既能直接优化策略，又能通过值函数减少训练方差。

- Actor（策略网络）：负责生成动作的策略函数$\pi(a|s;\theta)$，直接控制智能体行为，告诉智能体当前状态下应做什么动作。
  - 输入：状态$s$；
  - 输出：动作概率分布（离散）或动作均值/方差（连续）。
- Critic（值函数网络）：估计当前策略的“未来回报”，指导 Actor 改善。
  - 输入：状态$s$；
  - 输出：标量价值估计。
- 协同机制：Actor根据Critic的评价调整策略，Crtic通过TD误差优化价值估计。

#### 训练流程
1. 采样交互：

  	与环境交互，得到$(s_t, a_t, r_t, s_{t+1})$;
2. 计算TD误差或Advantage:

  	使用Critic来评估当前策略的表现，两种方式：
  - TD-error（Temporal Difference）：

    
    $$
    \delta = r_t + \gamma V(s_{t+1}) - V(s_t)
    $$

  - Advantage（优势函数）：
    
    
    $$
    A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx \delta
    $$
3. 更新Critic（值函数）：

   

   使用TD误差训练Critic的参数，最小化：

   


$$
  L_{critic} = (r_t + \gamma V(s_{t+1}) - V(s_t))^2
$$


4. 更新Actor（策略网络）：

  	使用策略梯度，最大化优势：


$$
  L_{actor} = -log\pi(a_t|s_t;\theta) \cdot A(s_t, a_t)
$$

#### 代码实现
##### Actor

```python
class Actor(nn.Module):
  def __init__(self, state_size=8, action_size=4, hidden_size=64):
    super().__init__()
    self.fc = nn.Sequential(
        nn.Linear(state_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size),
        nn.Softmax(dim=-1)
    )

  def forward(self, state):
    # Returing probability of each action
    return self.fc(state)
```

##### Critic
```python
class Critic(nn.Module):
  def __init__(self, state_size=8, hidden_size=64, drop_prob=0.3):
    super().__init__()
    self.fc = nn.Sequential(
        nn.Linear(state_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_size // 2, 1)
    )

  def forward(self, state):
    # critic: evaluates being in the state s_t
    return self.fc(state)
```
##### Actor Critic Agent
```python
from torch.optim.lr_scheduler import StepLR
from argparse import Namespace

class ActorCriticAgent():
  def __init__(self, actor, critic, **kwargs):
    # Configuration parameters
    self.config = Namespace(**kwargs)

    # Actor-Critic Network
    self.actor = actor
    self.critic = critic
    self.optimizer_actor = getattr(optim, self.config.optimizer)(self.actor.parameters(), lr=self.config.learning_rate)
    self.optimizer_critic = getattr(optim, self.config.optimizer)(self.critic.parameters(), lr=self.config.learning_rate)
    self.loss_fn = nn.SmoothL1Loss()

    # Step and update frequency
    self.step_t = 0
    self.update_freq = self.config.update_freq

    # Records
    self.loss_values = []

    self.empty()

  def step(self, log_probs, rewards, state_values, next_state_values, dones):
    self.step_t = (self.step_t + 1) % self.update_freq

    # Append the experiences
    self.rewards += rewards
    self.log_probs += log_probs
    self.state_values += state_values
    self.next_state_values += next_state_values
    self.dones += dones

    # Update Network
    if self.step_t == 0:
      self.learn(
          torch.stack(self.log_probs),  # log probabilities
          torch.tensor(self.rewards, dtype=torch.float32),  # discounted cumulative rewards
          torch.tensor(self.state_values, requires_grad=True),  # state_values
          torch.tensor(self.next_state_values, requires_grad=True), # next_state_values
          torch.tensor(self.dones, dtype=torch.float32) # dones
      )

      # Empty the experiences
      self.empty()

  def empty(self):
      """
      Empty the experience list
      """
      self.rewards = []
      self.log_probs = []
      self.state_values = []
      self.next_state_values = []
      self.dones = []

  def learn(self, log_probs, rewards, state_values, next_state_values, dones):
    """
    Update value parameters using given experience list.

    Arguments:
      log_probs (torch.Tensor): log probabilities
      rewards (torch.Tensor): discounted cumulative rewards
      state_values (torch.Tensor): predicted current state_values
      next_state_values (torch.Tensor): predicted next state_values
      dones (torch.Tensor): dones

    """
    state_values = state_values.squeeze()
    next_state_values = next_state_values.squeeze()

    gamma = 0.99
    advantages = rewards + gamma * next_state_values * (1 - dones) - state_values

    # Normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

    # Calculate loss
    loss_actor = (-log_probs * advantages).sum()
    loss_critic = self.loss_fn(state_values, rewards)
    self.loss_values.append(loss_actor.detach().item() + loss_critic.detach().item())

    # Backpropagation
    self.optimizer_actor.zero_grad()
    self.optimizer_critic.zero_grad()
    loss_actor.backward()
    loss_critic.backward()
    self.optimizer_actor.step()
    self.optimizer_critic.step()

  def sample(self, state):
    """
    Return action, log_prob, state_value for given state.

    Arguments:
      state(array_like): current state
    """
    action_prob = self.actor(torch.FloatTensor(state))
    state_value = self.critic(torch.FloatTensor(state))

    action_dist = Categorical(action_prob)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action.item(), log_prob, state_value
```

### Boss Baseline (Score: [275, inf))
Your final reward is : 291.98

#### Deep Q-Network (DQN)
LunarLander任务的动作空间是离散的，并且DQN非常适合离散动作的环境，因此实作中选择DQN算法训练LunarLander任务。

Deep Q-Network(DQN)是深度强化学习(DRL)中的一种经典算法，由DeepMind在2013年提出。核心思想是用神经网络近似Q值函数，解决了传统Q-Learning在高维状态空间下的局限性。

##### Q-Learning简述

Q-Learning的目标是学习状态-动作值函数$Q(s, a)$，表示在状态$s$采取动作$a$后的预期回报。更新公式如下：


$$
Q(s_t, a_t) ← Q(s_t, a_t) + \alpha (r_t + \gamma \mathop{\max}\limits_{a^{\prime}} Q(s_{t+1}, a^{\prime}) - Q(s_t, a_t))
$$

##### Deep Q-Network（DQN）核心思想

DQN使用一个神经网络来逼近Q函数：

$$
Q(s, a; \theta) \approx Q^{*}(s, a)
$$
- 输入：当前状态$s$；

- 输出：每个可能动作的Q值$Q(s, a)$；

- 目标：最小化Bellman残差：

  
$$
L(\theta) = (r + \gamma \mathop{\max}\limits_{a^{\prime}} Q(s^{\prime}, a^{\prime}; \theta^{-}) - Q(s, a; \theta))^2
$$

其中$\theta^{-}$是目标网络的参数，是$\theta$的一个延迟副本。

##### 核心机制

DQN的核心机制有：经验回放（Experience Replay）、目标网络（Target Network）、 $\epsilon - Greedy$策略

详情参考原始论文[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

#### 代码实现
DQN是经典的深度强化学习算法，有标准的库可供调用，不必重复造轮子。实作调用Stable-Baselines3库的DQN实现。
##### 导入库

```python
! pip install stable-baselines3[extra]
```
##### 训练
```python
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn

# 定义学习率调度（从 1e-3 线性衰减到 1e-5）
lr_schedule = get_schedule_fn(
    lambda progress: 1e-3 * (1 - progress) + 1e-5 * progress
)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=lr_schedule(0.0),
    buffer_size=500_000,
    learning_starts=10_000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=5_000,
    exploration_fraction=0.2,        # 20% 步骤用于探索
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)

# 训练前评估初始随机策略
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"初始平均奖励: {mean_reward:.2f}")

# 训练
model.learn(total_timesteps=2_000_000)

# 评估
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"✅ Evaluation over 20 episodes: mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

# 保存模型
model.save("dqn_lunarlander_best")

```
##### 测试

```python
# For DQN
fix(env, seed)

del model # remove to demonstrate saving and loading
# 加载模型
model = DQN.load("dqn_lunarlander_best")

NUM_OF_TEST = 5 # Do not revise this !!!
test_total_reward = []
action_list = []


for i in range(NUM_OF_TEST):
  actions = []
  state, _ = env.reset()

  img = plt.imshow(env.render())

  total_reward = 0

  done = False
  while not done:
      action, _states = model.predict(state, deterministic=True)
      actions.append(action.item())
      # state, reward, done, _ = env.step(action)
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      total_reward += reward

      img.set_data(env.render())
      display.display(plt.gcf())
      display.clear_output(wait=True)

  print(total_reward)
  test_total_reward.append(total_reward)

  action_list.append(actions) # save the result of testing
```

## Code

[Boss Baseline](https://github.com/Aaricis/Hung-yi-Lee-ML2022/tree/main/HW12)

## Report

1. **(2分) Implement Advanced RL algorithm**

   **a. Choose one algorithm from Actor-Critic、REINFORCE with baseline、Q Actor-Critic、A2C, A3C or other advance RL algorithms and implement it.**

   在Strong Baseline中实现了Actor-Critic算法，代码详见Strong Baseline部分。

   **b. Please explain the difference between your implementation and Policy Gradient.**

   Actor-Critic算法是强化学习中的一种混合框架，结合了策略梯度（Actor）和值函数估计（Critic）的优势，既能直接优化策略，又能通过值函数减少训练方差。Actor告诉智能体当前状态下应做什么动作，Critic估计当前策略的“未来回报”，指导 Actor 改善。Actor根据Critic的评价调整策略，Crtic通过TD误差优化价值估计。

   Policy Gradient直接优化策略，最大化累计折扣奖励来学习策略。

   **c. Please describe your implementation explicitly (If TAs can’t understand your description, we will check your code directly.** 

   详见Strong Baseline部分。

2. **(2分) How does the objective function of "PPO-ptx" differ from the “PPO” during RL training as used in the [InstructGPT paper](https://arxiv.org/pdf/2203.02155.pdf)? (1 point) Also, what is the potential advantage of using "PPO-ptx" over “PPO” in the [InstructGPT paper](https://arxiv.org/pdf/2203.02155.pdf)? Please provide a detailed analysis from their respective objective functions. (1 point)** 

   PPO-ptx的目标函数额外引入了一个预训练损失$\gamma E_{x\sim D_{pretrain}}[log(\pi^{RL}_{\phi}(x))]$，目的是防止模型在RL优化过程中偏离原始语言模型的分布，避免生成不连贯的文本。

## Reference

[Reinforcement Learning（李宏毅）机器学习 2023 Spring HW12 - 知乎](https://zhuanlan.zhihu.com/p/16270745272)

[李宏毅2022机器学习HW12解析 - 知乎](https://zhuanlan.zhihu.com/p/548920696)

[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/v2.6.0/)

