# 开发人员: LiHaoPu
# 开发时间: 2021/11/10 10:57
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# KNN算法
# 对未知类别属性的数据集中的每个点(待测样本)依次执行一下操作:
# · 计算已知类别数据集中的点与当前点之间的距离
# · 按照距离递增次序排序
# · 选取与当前点距离最小的k个点
# · 确定前k个点所在类别的出现频率
# · 返回前k个点出现频率最高的类别作为当前点的预测分类
def KNN_Classify(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - dataset
    # 关于tile函数的用法
    # >>> b = [1,3,5]
    # >>> tile(b, (2, 3)]
    # 输出: array([[1,3,5,1,3,5,1,3,5],
    #             [1,3,5,1,3,5,1,3,5]])
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sum(sqDiffMat, axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = argsort(distances)   # argsort()是Numpy模块中的函数，返回的是数组值从小到大的索引值
    # example
    # import numpy as np
    # >>> x = np.array([3, 1, 2])
    # >>> np.argsort(x)
    # 输出: array([1, 2, 0])
    classCount = {}  # 定义一个字典
    # 选择k个最近邻
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        # 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel,
                                               0) + 1  # dict.get(key, default=None) 返回指定键的值，如果值不在字典中返回default值

    # 返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex
