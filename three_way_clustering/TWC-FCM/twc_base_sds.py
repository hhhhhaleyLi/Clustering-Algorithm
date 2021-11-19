# 开发人员: LiHaoPu
# 开发时间: 2021/11/18 15:44


# u: 隶属度矩阵  a、b: 阈值(b=1-a)  k: 聚类之后产生的类簇个数   data: 待测样本
def GetClusteringResult(u, a, b, k, data):
    result = []
    for i in range(0, k):
        resulti = []
        Co = []
        Fr = []
        for j in range(0, len(data)):
            temp = u[i][j]
            if temp > b:
                Co.append(data[j])
            elif a < temp < b:
                Fr.append(data[j])
        resulti.append(Co)
        resulti.append(Fr)
        result.append(resulti)
    return result   # 返回基于阴影集的三支聚类结果  result = {(Co(C1), Fr(C1)), (Co(C2), Fr(C2)), ... , (C0(Ck), Fr(Ck))}