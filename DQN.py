# -*- codeing = utf-8 -*-
"""
@Time : 2024/9/23
@Author : AC
@File : DQN.py
@Software : PyCharm
"""
from Tools.keys import *
from Tools.log import save_log
from model import *
from env import *
import os
from datetime import datetime
from math import exp

logDir = r"E:\LZK\codeWork\python\items\DQN_play_blood-main\MyDQN\logs"
modelPath = r"E:\LZK\codeWork\python\items\DQN_play_blood-main\MyDQN\model\model.pth"
actionMap = {0: "飞渡浮舟", 1: "攻击", 2: "跳跃", 3: "防御", 4: "踱步", 5: "右走一步", 6: "左走一步", 7: "前走一步", 8: "后走一步"}
WIDTH = 256
HEIGHT = 256
# 输入图像的宽高
EPOCH = 50
BIG_BATCH_SIZE = 16
UPDATE_STEP = BIG_BATCH_SIZE * 2


# 多少步更新一次旧的Q值网络

def take_action(action):
    # print(action)
    if isinstance(action, torch.Tensor):
        action = action.cpu().item()
    print("action: ", actionMap[action])
    if action == 0:  # 战机
        attack2()
    elif action == 1:  # J
        attack()
    elif action == 2:  # K
        jump()
    elif action == 3:  # Q
        defense()
    elif action == 4:  # O
        dodge()
    elif action == 5:  # D
        go_right()
    elif action == 6:  # A
        go_left()
    elif action == 7:  # W
        go_forward()
    elif action == 8:  # S
        go_back()


# action:做了什么动作

def jude(boss_blood, next_boss_blood, self_blood, next_self_blood, boosDef, next_bossDef, emergence_break, action):
    # boss只有两条命打完就没了
    if emergence_break >= 2:
        return 0, 1, 100

    if next_self_blood < 2:
        # 说明角色死亡
        done = 1
        reward = -10
        return reward, done, emergence_break

    if next_boss_blood < 2:
        # 如果boss死亡
        done = 1
        reward = 20
        emergence_break += 1
        return reward, done, emergence_break

    self_reward = 0
    boss_reward = 0
    done = 0
    if next_boss_blood - boss_blood < -5:
        # 如果给boss造成了一定阈值的伤害
        boss_reward += 15 * exp(1 + abs(next_boss_blood - boss_blood) / boss_blood)
    if next_self_blood - self_blood < -20:
        # 如果给自己造成了一定阈值的伤害
        self_reward -= 5 * exp(1 + abs(next_self_blood - self_blood) / self_blood)
    if next_bossDef - boosDef > 0 and next_bossDef != 0:
        # 如果让boss的驾驶条上去了
        boss_reward += 15 * (1 + (next_bossDef - boosDef)/next_bossDef)
        if action == 3:
            # 如果是因为防御上去的
            boss_reward += 15 * (1 + (next_bossDef - boosDef)/next_bossDef)
            print("---有效防御---")
        if action == 1 or action == 0:
            boss_reward += 15 * (1 + (next_bossDef - boosDef)/next_bossDef) * 0.5
            print("---有效攻击（格挡）---")

    # 如果防御了但是boss条的驾驶条不变 那么为无效防御要减分
    # 应该看自己的驾驶条 而不是
    # if action == 3 and next_bossDef - boosDef <= 0:
    #     self_reward -= 5
    #     print("---无效防御---")


    reward = self_reward + boss_reward
    print("---status---")
    print("next-def:", next_bossDef)
    print("def:", boosDef)
    print("next-self:", next_self_blood)
    print("self:", self_blood)
    print("next-boss:", next_boss_blood)
    print("boss:", boss_blood)
    print("---reward---")
    print("self_reward:", self_reward)
    print("boss_reward:", boss_reward)
    print("------------")
    return reward, done, emergence_break


def main():
    agent = DQN(HEIGHT, WIDTH, len(actionMap), modelPath, EPOCH)

    try:
        param = torch.load(r"E:\LZK\codeWork\python\items\DQN_play_blood-main\MyDQN\model\model.pth")
        agent.eval_net.load_state_dict(param)
        print("--模型参数预加载成功--")
    except Exception as e:
        print("--模型参数预加载失败--")
    finally:
        for i in range(2):
            print(i)
            time.sleep(1)
        print("--start--")
        avg_rewards = []
        # 获取当前时间
        now = datetime.now()
        # 格式化当前时间为字符串
        # 格式说明：%Y-%m-%d %H:%M:%S 分别代表 年-月-日 时:分:秒
        current_time_str = now.strftime("%Y-%m-%d&%H-%M-%S")
        # 用于存储avg_reward值
        logPath = os.path.join(logDir, current_time_str + ".pkl")
        for episode in range(EPOCH):
            emergence_break = 0
            # 用于防止连续帧重复计算reward
            total_reward = 0
            # 总激励
            target_step = 0
            # 记录走了多少步
            last_time = time.time()
            # 记录时间
            while True:
                # 不用截图那么快
                time.sleep(0.05)
                # 获取战斗场景
                state = getFight()
                state = cv2.resize(state, (HEIGHT, WIDTH)).reshape((3, HEIGHT, WIDTH))
                # 获取双方血量
                self, boss = getHP()
                # 获取boss驾驶条
                bossDef = getDef()
                # 目标步加1 表示走了1步
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(dtype=torch.float).to("cuda")
                action = agent.choose_action(state)
                # 模型给出行动
                take_action(action)
                target_step += 1
                print('loop took {} seconds'.format(time.time() - last_time))
                last_time = time.time()

                # 再次获取战斗场景
                next_state = getFight()
                next_state = cv2.resize(next_state, (HEIGHT, WIDTH)).reshape((3, HEIGHT, WIDTH))
                # 获取双方血量
                next_self, next_boss = getHP()
                # 获取boss驾驶条
                next_bossDef = getDef()
                reward, done, emergence_break = jude(boss, next_boss, self, next_self, bossDef, next_bossDef,
                                                     emergence_break, action)

                total_reward += reward

                if emergence_break == 100:
                    # emergence break , save model and paused
                    # 遇到紧急情况，保存数据，并且暂停
                    print("emergence_break")
                    agent.save_model()
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                agent.store_data(state, action, reward, next_state, done)
                # 模型每记住big_BATCH_SIZE个情况就学习一次
                if len(agent.replay_buffer) > BIG_BATCH_SIZE:
                    print("--eval网络训练--")
                    agent.train()

                if target_step % UPDATE_STEP == 0:
                    print("--更新target网络--")
                    agent.update_target()

                if done:
                    print("--done--")
                    break
            if episode % 5 == 0:
                agent.save_model()
            avg_reward = total_reward / target_step
            avg_rewards.append(avg_reward)
            print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)
            restart()
        save_log(avg_rewards, logPath)


if __name__ == "__main__":
    main()
