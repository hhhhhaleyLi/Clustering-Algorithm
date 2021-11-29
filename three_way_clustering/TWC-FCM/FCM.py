# 开发人员: LiHaoPu
# 开发时间: 2021/11/18 15:31
import numpy as np
import time
from skfuzzy.cluster import cmeans


# data: 数据集   K: 数据集中数据个数
def GetMembershipByFCM(data, K):
    maxFPC = 0
    count = 0   # 最终选取的类簇数
    center = []   # 类簇中心

    for k in range(2, K+1):
        # 调用skfuzzy.cluster中的FCM算法
        cnt, u, u0, d, jm, p, fpc = cmeans(data.T, c=k, m=2, error=0.2, maxiter=1000, )
        # input
        # m: 隶属度的指数，是一个加权指数   error: 当隶属度的变化小于error时提前结束迭代   maxiter: 最大迭代次数
        # output
        # cnt: 聚类中心(二维矩阵 行: 类簇数k  列: 属性数)  u: 最终的隶属度矩阵(二维矩阵  行: 类簇数k  列: 样本数)
        # u0: 初始化隶属度矩阵  d: 最终每个数据点到各个类簇中心的欧式距离  jm: 目标函数优化的历史
        # p: 迭代次数   fpc: fuzzy partition coefficient，一个评价分类好坏的指标，范围[0,1]，1最好。用来选取类簇个数k
        if maxFPC < fpc:
            maxFPC = fpc
            count = k
            u_optimal = u   # 隶属度矩阵
            center = cnt
    return count, u_optimal, center   # 类簇数目  隶属度矩阵  类簇中心
