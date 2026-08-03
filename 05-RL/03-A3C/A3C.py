import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym  # 使用 Gymnasium 替代 Gym
import numpy as np
from collections import deque
import time

# --- 1. Actor-Critic 神经网络 ---
class A3CNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# --- 2. 工作进程 (Worker) ---
class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward,
                 res_queue, env_name, worker_id, gamma=0.99, t_max=5):
        super(Worker, self).__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.res_queue = res_queue
        self.env_name = env_name
        self.worker_id = worker_id
        self.gamma = gamma
        self.t_max = t_max

    def run(self):
        # 每个 Worker 拥有独立的环境
        env = gym.make(self.env_name)
        local_net = A3CNet(env.observation_space.shape[0], env.action_space.n)
        local_net.load_state_dict(self.global_net.state_dict())

        while self.global_episode.value < 1000:  # 训练总回合数
            # ---------- 新版 Gymnasium API ----------
            state, _ = env.reset()          # reset 返回 (obs, info)
            done = False
            episode_reward = 0
            # 存储轨迹数据
            states, actions, rewards, next_states, dones = [], [], [], [], []

            while not done:
                # 同步全局参数到本地网络
                local_net.load_state_dict(self.global_net.state_dict())

                # 执行最多 t_max 步
                for _ in range(self.t_max):
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    action_probs, _ = local_net(state_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()

                    # step 返回 (obs, reward, terminated, truncated, info)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated   # 合并终止信号

                    episode_reward += reward

                    # 存储 transition
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)                # 存储 terminated 标志

                    if done:
                        break
                    state = next_state

                # 计算梯度并更新全局网络
                self._update_global_network(local_net, states, actions, rewards, next_states, dones)

                # 清空轨迹缓存
                states, actions, rewards, next_states, dones = [], [], [], [], []

            # 记录回合结果
            with self.global_episode.get_lock():
                self.global_episode.value += 1
            with self.global_episode_reward.get_lock():
                self.global_episode_reward.value = episode_reward
            self.res_queue.put(episode_reward)

        env.close()

    def _update_global_network(self, local_net, states, actions, rewards, next_states, dones):
        # 将列表转换为张量
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)   # 0/1 张量

        # 计算当前状态价值和下一状态价值
        _, state_values = local_net(states)
        _, next_state_values = local_net(next_states)

        # TD Target 和 Advantage（使用 dones 屏蔽终止状态）
        td_targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = td_targets - state_values.detach()

        # Actor 损失（策略梯度）
        action_probs, _ = local_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions.squeeze())
        actor_loss = -(log_probs.unsqueeze(1) * advantages).mean()

        # Critic 损失（均方误差）
        critic_loss = F.mse_loss(state_values, td_targets.detach())

        # 总损失（可添加熵奖励鼓励探索，此处省略）
        loss = actor_loss + 0.5 * critic_loss

        # 梯度计算与更新
        self.optimizer.zero_grad()
        loss.backward()   # 在本地网络计算梯度

        # 将本地梯度复制到全局网络，然后执行优化器步进
        for local_param, global_param in zip(local_net.parameters(), self.global_net.parameters()):
            if global_param.grad is not None:
                global_param.grad += local_param.grad   # 累加（可选）
            else:
                global_param.grad = local_param.grad
        self.optimizer.step()

# --- 3. 主训练函数 ---
if __name__ == "__main__":
    # 设置多进程启动方式（对 Windows/macOS 兼容）
    mp.set_start_method('spawn', force=True)

    # 超参数
    ENV_NAME = "CartPole-v1"
    NUM_WORKERS = 4          # 并行进程数
    GAMMA = 0.99
    T_MAX = 5
    MAX_EPISODES = 1000

    # 创建环境和全局网络
    env = gym.make(ENV_NAME)
    global_net = A3CNet(env.observation_space.shape[0], env.action_space.n)
    global_net.share_memory()   # 共享内存，使各进程可访问

    # 优化器（RMSprop 是 A3C 论文推荐）
    optimizer = torch.optim.RMSprop(global_net.parameters(), lr=0.001)

    # 进程间共享变量
    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.0)
    res_queue = mp.Queue()

    # 启动所有 Worker
    workers = []
    for worker_id in range(NUM_WORKERS):
        worker = Worker(global_net, optimizer, global_episode, global_episode_reward,
                        res_queue, ENV_NAME, worker_id, GAMMA, T_MAX)
        worker.start()
        workers.append(worker)

    # 主进程监控训练进度
    episode_rewards = deque(maxlen=100)
    start_time = time.time()
    while global_episode.value < MAX_EPISODES:
        if not res_queue.empty():
            reward = res_queue.get()
            episode_rewards.append(reward)
            avg_reward = np.mean(episode_rewards)

            if global_episode.value % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode: {global_episode.value:4d}, "
                      f"Avg Reward (100 eps): {avg_reward:.2f}, "
                      f"Time: {elapsed:.1f}s")

            # CartPole 求解标准：最近100回合平均奖励 ≥ 195
            if avg_reward >= 195.0 and len(episode_rewards) >= 100:
                print(f"Solved in {global_episode.value} episodes!")
                break

    # 等待所有 Worker 结束
    for worker in workers:
        worker.join()

    print("Training finished!")
    env.close()