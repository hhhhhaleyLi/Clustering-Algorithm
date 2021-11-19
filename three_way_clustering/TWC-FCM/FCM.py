# 开发人员: LiHaoPu
# 开发时间: 2021/11/18 15:31
import numpy as np
from skfuzzy.cluster import cmeans


def GetMembershipByFCM(data):
    K = 8  # 聚类数-1
    maxFPC = 0
    count = 0
    center = []

    for k in range(2, K):
        cnt, u, u0, d, jm, p, fpc = cmeans(data.T, c=k, m=2, error=0.2, maxiter=1000, )
        if maxFPC < fpc:
            maxFPC = fpc
            count = k
            u_optimal = u
            center = cnt
    return count, u_optimal, center   # 类簇数目  隶属度矩阵  类簇中心