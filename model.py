# -*- codeing = utf-8 -*-
"""
@Time : 2024/9/23
@Author : AC
@File : model.py
@Software : PyCharm
"""
import torch
import torchvision.models as models
import torch.nn as nn
import random
from replay import replay
import numpy as np

INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 3000
# 脑子容量
BATCH_SIZE = 16
GAMMA = 0.9


# 旧的Q值权重

class featureNet(nn.Module):
    def __init__(self, observation_height, observation_width, action_space) -> None:
        super(featureNet, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        # self.net2 = models.mobilenet_v3_small(pretrained=True)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5), padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=(5, 5), padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(int((self.state_w / 4) * (self.state_h / 4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(-1, int((self.state_w / 4) * (self.state_h / 4) * 64))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x

class featureNet2(nn.Module):
    def __init__(self, observation_height, observation_width, action_space) -> None:
        super(featureNet2, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        # self.net2 = models.mobilenet_v3_small(pretrained=True)
        self.net = models.resnet18(pretrained=True)
        # 修改全连接层，将输出类别数改为5
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, action_space)  # 替换为新的全连接层

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, 3, -1, x.shape[-1])
        x = self.net(x)
        return x


class DQN(object):
    def __init__(self, observation_height, observation_width, action_space, model_file, epoch):
        self.model_file = model_file
        # 模型存放地址
        self.target_net = featureNet2(observation_height, observation_width, action_space)
        # 固定Q表的网络
        self.target_net.to("cuda")
        self.eval_net = featureNet2(observation_height, observation_width, action_space)
        # 训练的网络
        self.eval_net.to("cuda")
        self.replay_buffer = replay(REPLAY_SIZE)
        # 经验仓库
        self.epsilon = INITIAL_EPSILON

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        # 优化器
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch // 2)
        # 优化策略
        self.loss = nn.MSELoss()
        # 损失函数
        self.action_dim = action_space
        # 动作空间

    def choose_action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:

            Q_value = self.eval_net(state)
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return torch.argmax(Q_value)

    def store_data(self, state, action, reward, next_state, done):
        # 意思应该是 由state到next_state模型采取了action得到了reward
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # replay_buffer相当于这个模型的脑容量 记住了多少种情况 该做什么决策
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # 如果记住的东西 超出了它的脑容量 就要pop出去 但这里是遗忘最早的state

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(BATCH_SIZE)
        # 从脑子里面 随机取BATCH个情况
        y_batch = []
        Q_value_batch = self.target_net(torch.tensor(next_state_batch).to(dtype=torch.float).to("cuda"))

        for i in range(0, BATCH_SIZE):
            done = done_batch[i]
            # see if the station is the final station
            # 就是不是死了嘛 如果是死了 那么就直接append reward
            # 相当于是最后一步了就只用单步的reward
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                # 如果不是最后就要加上之前的reward
                y_batch.append(reward_batch[i] + GAMMA * torch.max(Q_value_batch[i]))
        # 假设 self.Q_value 是模型输出的Q值，self.action_input 是动作的one-hot编码表示

        action_batch = torch.tensor(action_batch).to(dtype=torch.float).to("cuda")

        Q_eval = self.eval_net(torch.tensor(state_batch).to(dtype=torch.float).to("cuda"))

        Q_action = torch.sum(Q_eval * action_batch, dim=1)

        y_batch = torch.tensor(y_batch).to(dtype=torch.float).to("cuda")

        loss = self.loss(Q_action, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.schedule.step()

    # 保存模型
    def save_model(self):
        torch.save(self.target_net.state_dict(), self.model_file)

    # 更新Q表模型
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


def main():
    print("hello world")


if __name__ == "__main__":
    main()
