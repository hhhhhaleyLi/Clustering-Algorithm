# 开发人员: LiHaoPu
# 开发时间: 2021/11/18 15:44
import time


# u: 隶属度矩阵  a、b: 阈值(b=1-a)  k: 聚类之后产生的类簇个数   length: 数据集个数
def GetClusteringResult(u, a, b, k, data):
    result = []   # 最终的聚类结果
    for i in range(0, k):   # 对类簇进行迭代
        resulti = []
        Co = []   # 核心域
        Fr = []   # 边界域
        for j in range(0, len(data)):   # 对数据集进行迭代
            temp = u[i][j]   # 第j个数据对第i个类簇的隶属度
            if temp > b:
                Co.append(data[j])
                # Co.append(j)
            elif a < temp < b:
                Fr.append(data[j])
                # Fr.append(j)
        resulti.append(Co)
        resulti.append(Fr)
        result.append(resulti)
    return result   # 返回基于阴影集的三支聚类结果  result = {(Co(C1), Fr(C1)), (Co(C2), Fr(C2)), ... , (C0(Ck), Fr(Ck))}


def GetClusteringResult(u, a, b, k, data, label):
    result = []  # 最终的聚类结果
    for i in range(0, k):  # 对类簇进行迭代
        resulti = []
        Co = []  # 核心域
        Fr = []  # 边界域
        Co_data = []
        Co_label = []
        Fr_data = []
        Fr_label = []
        for j in range(0, len(data)):  # 对数据集进行迭代
            temp = u[i][j]  # 第j个数据对第i个类簇的隶属度
            if temp > b:
                Co_data.append(data[j])
                Co_label.append(label[j])
                # Co.append(data[j])
                # Co.append(j)
            elif a < temp < b:
                Fr_data.append(data[j])
                Fr_label.append(label[j])
                # Fr.append(data[j])
                # Fr.append(j)
        Co.append(Co_data)
        Co.append(Co_label)
        Fr.append(Fr_data)
        Fr.append(Fr_label)
        resulti.append(Co)
        resulti.append(Fr)
        result.append(resulti)
    return result  # 返回基于阴影集的三支聚类结果  result = {(Co(C1), Fr(C1)), (Co(C2), Fr(C2)), ... , (C0(Ck), Fr(Ck))}
    # result = {(Co(data, label), Fr(data, label)), (Co(data, label), Fr(data, label)), ..., (Co(data, label), Fr(data, label)))}
    # Co(data, label)中，data是指属于该类簇核心域的数据对象的集合  label是属于该类簇核心域的数据对象的标签的集合
