# 基于Pytorch的强化学习DQN只狼

> 声明：该代码实现思路是根据B站蓝魔digital的tensorFlow版DQN只狼改进而来。

## 一、运行环境

CPU：12th Gen Intel(R) Core(TM) i5-12450H

GPU:

```
NVIDIA GeForce RTX 3060 Laptop GPU
专用 GPU 内存	6.0 GB
共享 GPU 内存	7.9 GB
GPU 内存	13.9 GB
```
Pytorch版本为2.1.0

## 二、文件说明

replay.py：用于实现DQN中知识储备类。

model.py：用于构建DQN模型。

env.py：用于充当强化学习中环境角色，可获取角色血量与架势条。

DQN.py：用于模型训练。

Tools:

- img.py：用于获取游戏截图。
- keys.py：用于实现键盘输入。
- log.py：用于训练日志的记录与查看。

logs：用于存放训练日志。

model：用于存放训练好的模型。

