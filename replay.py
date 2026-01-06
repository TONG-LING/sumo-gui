import random
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self._capacity = capacity
        self._storage = []
        self._num_added = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # 用于控制优先级的平滑程度，alpha=0时退化为均匀采样

    def add(self, state, next_state, action, reward, td_error: float):
        """添加样本，并根据TD-error计算优先级"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(0).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(device)

        transition = Transition(state, action, next_state, reward)

        if len(self._storage) < self._capacity:
            self._storage.append(transition)
        else:
            self._storage[self._num_added % self._capacity] = transition

        # 优先级设置为TD-error的幂次，确保为正
        self._priorities[self._num_added % self._capacity] = (abs(td_error) + 1e-5) ** self.alpha
        self._num_added += 1

    def sample(self, batch_size: int, beta: float = 0.4):
        """根据优先级进行样本采样"""
        if len(self._storage) == 0:
            return None

        # 计算采样的概率分布
        priorities = self._priorities[:len(self._storage)]
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self._storage), batch_size, p=probabilities)
        batch = [self._storage[i] for i in indices]

        # 计算权重 (importance sampling weight)
        total = len(self._storage)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 归一化

        batch = Transition(*zip(*batch))  # 解包为Transition命名元组
        return batch, torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device), indices

    def update_priorities(self, indices, td_errors):
        """更新样本的优先级"""
        for i, td_error in zip(indices, td_errors):
            self._priorities[i] = (abs(td_error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self._storage)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return min(self._num_added, self._capacity)

    @property
    def steps_done(self) -> int:
        return self._num_added