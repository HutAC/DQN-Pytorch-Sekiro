# -*- codeing = utf-8 -*-
"""
@Time : 2024/9/23
@Author : AC
@File : log.py
@Software : PyCharm
"""
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def save_log(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def show_log(path):
    data = read_pkl(path)
    print(data)
    x = list(range(len(data)))

    # 绘制曲线图
    plt.plot(x, data)
    plt.title("Sine Wave")  # 设置图表标题
    plt.xlabel("x")  # x轴标签
    plt.ylabel("y")  # y轴标签
    plt.show()  # 显示图像

def main():
    show_log(r"E:\LZK\codeWork\python\items\DQN_play_blood-main\MyDQN\logs\2024-09-23&21-54-46.pkl")


if __name__ == "__main__":
    main()
