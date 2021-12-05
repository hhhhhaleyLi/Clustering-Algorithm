import pandas as pd
import numpy as np
import random
import math
import Parameter.CalcParameter
from sklearn.datasets import load_iris


# data = pd.read_csv('../haberman.csv', names=['a', 'b', 'c', 'd'], usecols=lambda x: x != 'd')
# label = pd.read_csv('../haberman.csv', names=['a', 'b', 'c', 'd'], usecols=['d'])
# temp = data
# data = np.array(data)
# label = np.array(label)
# indices = np.random.permutation(len(data))
# count = math.floor(len(data) * 0.3)  # 样本训练集和测试集分割量
# data_train = data[indices[:-count]]
# label_train = label[indices[:-count]]
# data_test = data[indices[-count:]]
# label_test = label[indices[-count:]]

iris = load_iris()
X = iris.data[:, :2]  # 表示只取特征空间中的后两个维度
y = iris.target
indices = np.random.permutation(len(X))
# permutation接收一个数作为参数(150),产生一个0-149一维数组，只不过是随机打乱的，当然她也可以接收一个一维数组作为参数，结果是直接对这个数组打乱
iris_x_train = X[indices[:-10]]   # 随机选取140个样本作为训练数据集
iris_y_train = y[indices[:-10]]   # 并且选取这140个样本的标签作为训练数据集的标签
iris_x_test = X[indices[-10:]]   # 剩下的10个样本作为测试数据集
iris_y_test = y[indices[-10:]]   # 并且把剩下10个样本对应标签作为测试数据及的标签

k = 3
# region 类簇集合
C = []
for i in range(0, k):
    Ci = []
    Ci_Co = []
    Ci_Fr = []
    Ci.append(Ci_Co)
    Ci.append(Ci_Fr)
    C.append(Ci)
# endregion
# region 将数据集随机分配到k个类簇的正域中
# for i in range(0, len(data_train)):
#     index = random.randint(0, k-1)
#     data_train_i = data_train[i]
#     C[index][0].append(data_train_i)

for i in range(0, len(iris_x_train)):
    index = random.randint(0, k-1)
    data_train_i = iris_x_train[i]
    C[index][0].append(data_train_i)
# endregion
# region 迭代确定类簇正域和边界域
times = 10   # 最多迭代10次
C_rkm = Parameter.CalcParameter.calc_parameter(C, iris_x_train, k, times)
for i in range(0, len(C_rkm)):
    print(C_rkm[i][0])
    print(C_rkm[i][1])
predicts = 0
for c_index in C_rkm[0][0]:
    if iris_y_train[c_index] == 0:
        predicts += 1
for c_index in C_rkm[1][0]:
    if iris_y_train[c_index] == 1:
        predicts += 1
for c_index in C_rkm[2][0]:
    if iris_y_train[c_index] == 2:
        predicts += 1
score = predicts / len(iris_y_train)
print(score)
# todo 聚类准确率0.3 代码实现有问题
# endregion
