# -*- codeing = utf-8 -*-
"""
@Time : 2024/9/23
@Author : AC
@File : replay.py
@Software : PyCharm
"""
from collections import deque
import random
import numpy as np


# 定义经验仓库类
class replay:
    def __init__(self, maxSize=3000):
        self.buffer = deque(maxlen=maxSize)

    def append(self, exp):
        # obs, action, reward, next_obs, done
        self.buffer.append(exp)

    def sample(self, batch_size):
        # 从buffer中随机取batch_size个样本
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        for exp in mini_batch:
            s, a, r, s_p, done = exp
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        obs_batch = np.array(obs_batch, dtype="float32")
        action_batch = np.array(action_batch, dtype="float32")
        reward_batch = np.array(reward_batch, dtype="float32")
        next_obs_batch = np.array(next_obs_batch, dtype="float32")
        done_batch = np.array(done_batch, dtype="float32")
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)

    def popleft(self):
        self.buffer.popleft()


def main():
    print("hello world")


if __name__ == "__main__":
    main()
