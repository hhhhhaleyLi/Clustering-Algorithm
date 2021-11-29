# 开发人员: LiHaoPu
# 开发时间: 2021/11/12 19:23
import numpy as np
import sys
import time

# b = 1 - a
def shadowedSet(u, k):
    min = sys.maxsize   # 获取int类型的最大值  等价于Java中的Integer.MAX_VALUE
    a_optional = []
    resulta = 0
    for i in range(0, k):
        for j in range(0, len(u[i])):
            if u[i][j] <= 0.5:   # 阈值α小于1/2,α∈[0, 1/2]  阈值β大于1/2,β∈[1/2, 1]
                a_optional.append(u[i][j])
    # 时间复杂度太高
    for i in range(0, k):
        temp = 0
        u_max = np.argmax(u[i])   # 获取样本对第i个类簇的隶属度中值最大的那个隶属度值
        for a in a_optional:   # 迭代选取出使目标函数值最小的阈值α
            b = 1 - a
            for ua in u[i] >= a:
                temp += u_max - ua
                # temp += 1 - ua
            for ub in u[i] <= b:
                temp += ub
            for uc in u[i] < a:
                if uc > b:
                    temp -= 1
            if temp < 0:
                temp = -temp
            if temp < min:
                min = temp
                resulta = a
    return resulta, 1-resulta   # 阈值α β
