# -*- codeing = utf-8 -*-
"""
@Time : 2024/9/22
@Author : AC
@File : env.py
@Software : PyCharm
"""
import time

from Tools.img import grab_screen
import cv2
import matplotlib.pyplot as plt

# 自己血量窗口位置
self_window = (57, 550, 55 + 338, 550 + 15)
# 敌人血量窗口位置
boss_window = (57, 77, 222 + 55, 15 + 75)
# 敌人架势条位置
boss_def_window = (330, 63, 700, 70)

# 战斗场景窗口位置
fight_window = (180, 175, 700 + 180, 450 + 125)


# 计算自己的血量
def count_self(img, index=4, channel=3):
    # index = 5  # 用哪一行的像素值
    count = 0
    # BGRA
    if channel == 3:
        red = img[:, :, -2][index]
        # 跳过前面的一些些像素
        red = red[3:]
        for i in red:
            # 110 到 128是分析出来的 血条范围
            if 110 <= i <= 128:
                count += 2
        return count


# 计算Boss的血量
def count_boss(img, index=6, channel=3):
    count = 0
    if channel == 3:
        red = img[:, :, -2][index]
        # 跳过前面的一些些像素
        red = red[3:]
        for i in red:
            if 80 <= i <= 110:
                count += 2
        return count


def count_boss_def(img, index=4, channel=3, b=5):
    count = 0
    if channel == 3:
        red = img[:, :, -2][index]
        mid = len(red) // 2
        # 首先要中心附近的均值达到阈值
        redMid = red[mid-b:mid+b]
        # 如果是有架势条的
        if redMid.mean() > 160:
            for i in red[mid:]:
                if i > 140:
                    count += 1
                else:
                    break
        else:
            count = 0
        return count


# 获取当前自己和Boss的HP
def getHP():
    # 截取血量图片
    selfImg = grab_screen(self_window)
    bossImg = grab_screen(boss_window)
    return count_self(selfImg), count_boss(bossImg)


def getFight():
    return grab_screen(fight_window)[:, :, 0:3]

def getDef():
    bossDef = grab_screen(boss_def_window)
    return count_boss_def(bossDef)

def find_location():
    while True:
        selfImg = grab_screen(boss_window)
        # 要只获取BRG三个通道
        imgNumpy = selfImg[:, :, 0:3]
        # OpenCV使用BGR格式，所以我们需要将RGB转换为BGR
        img_bgr = cv2.cvtColor(imgNumpy, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img_bgr)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()


def main():
    index = 4
    selfImg = grab_screen(boss_def_window)
    imgNumpy = selfImg[:, :, 0:3]
    red = imgNumpy[:, :, 2][index]
    # plt.hist(red, bins=len(set(red)), alpha=0.5, color='blue', edgecolor='black')
    # plt.title('img')
    # plt.xlabel('value')
    # plt.ylabel('count')
    # plt.show()
    print(count_self(selfImg))
    print(red.reshape(-1))

    imgNumpy[:, :, 2][index] = 0
    cv2.imshow('img', imgNumpy)
    cv2.waitKey(0)  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()


def defLoaction():
    boss_index = 4
    boss = grab_screen(boss_def_window)
    boss = boss[:, :, 0:3]
    # 拿到红色和黄色通道
    boss_red = boss[:, :, -1][boss_index]
    boss_yellow = boss[:, :, 1][boss_index]
    plt.hist(boss_yellow, bins=len(set(boss_yellow)), alpha=0.5, color='blue', edgecolor='black')
    plt.title('img')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.show()
    print(boss_red.reshape(-1))
    print(boss_yellow.reshape(-1))
    boss[:, :, -1][boss_index] = 0
    # print(count_boss_def(boss))
    cv2.imshow('img', boss)
    cv2.waitKey(0)  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()


def test():
    import numpy as np
    index = 4
    b = 5
    redMidMean = []
    while True:
        time.sleep(0.5)
        bosse = grab_screen(boss_def_window)
        boss = bosse[:, :, 0:3]
        red = boss[:, :, -1][index]
        green = boss[:, :, 0][index]
        mid = len(red)//2
        redMid = red[mid-b: mid+b]
        greenMid = green[mid-b: mid+b]

        redMidMean.append(redMid.mean())
        # print("red-mean:", red.mean())
        # print(redMid)
        # print("redMid-mead:", redMid.mean())
        # print(greenMid)
        # print("greenMid-mead:", greenMid.mean())
        print(count_boss_def(bosse))
        boss[:, :, -1][index] = 0
        cv2.imshow('img', boss)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    print(np.mean(redMidMean))
cv2.waitKey(0)  # 视频结束后，按任意键退出
cv2.destroyAllWindows()



if __name__ == "__main__":
    # main()
    # defLoaction()
    test()
