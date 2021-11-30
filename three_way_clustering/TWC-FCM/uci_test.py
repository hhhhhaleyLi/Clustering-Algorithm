import math
import FCMAlgorithm.FCM as fcm
import ShadowSet.shadowedSetTest as sds
import TWCAlgorithm.twc_base_sds as twc
import KNNAlgorithm.KNN_uci_test as knn_test
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # data = pd.read_csv('../spambase_dataset.csv', usecols=lambda x: x != 'class')  # 属性
    # label = pd.read_csv('../spambase_dataset.csv', usecols=["class"])  # 标签
    data = pd.read_csv('../waveform.csv',
                       names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'
                           , 'p', 'q', 'r', 's', 't', 'u', 'v'], usecols=lambda x: x != 'v')
    label = pd.read_csv('../waveform.csv',
                        names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'
                            , 'p', 'q', 'r', 's', 't', 'u', 'v'], usecols=['v'])
    data = np.array(data)
    label = np.array(label)
    label = label.reshape(len(label), )
    # data = data[:, 0:20]   # spambase
    data = data[:, :14]   # waveform
    # region 重复实验（每次实验重新对样本进行随机分割，即每次实验的训练样本和测试样本都不同）
    """
    # 实验重复times次，取结果的平均值作为最后结果
    accuracy = 0.0
    times = 50
    for i in range(1, times+1):
        indices = np.random.permutation(len(data))
        count = math.floor(len(data) * 0.3)  # 样本训练集和测试集分割量
        data_train = data[indices[:-count]]
        label_train = label[indices[:-count]]
        data_test = data[indices[-count:]]
        label_test = label[indices[-count:]]

        # 调用FCM算法获得类簇数目、隶属度矩阵、类簇中心
        k, u, C = fcm.GetMembershipByFCM(data_train, 3)
        # 通过阴影集获取最优阈值α、β
        a, b = sds.shadowedSet(u, k)
        # 获取三支聚类结果
        result = twc.GetClusteringResult(u, a, b, k, data_train, label_train)
        # 进行KNN分类
        accuracy += knn_test.KNNAlgorithm(result, data_train, data_test, label_train, label_test, C, a, b, k, i)

    print("最终结果为: ", accuracy / times)
    """
    # endregion
    # region 重复实验（每次实验的训练样本和测试样本相同）
    indices = np.random.permutation(len(data))
    count = math.floor(len(data) * 0.3)  # 样本训练集和测试集分割量
    data_train = data[indices[:-count]]
    label_train = label[indices[:-count]]
    data_test = data[indices[-count:]]
    label_test = label[indices[-count:]]

    accuracy = 0.0
    times = 10
    for i in range(1, times + 1):
        # 调用FCM算法获得类簇数目、隶属度矩阵、类簇中心
        k, u, C = fcm.GetMembershipByFCM(data_train, 3)
        # 通过阴影集获取最优阈值α、β
        a, b = sds.shadowedSet(u, k)
        # 获取三支聚类结果
        result = twc.GetClusteringResult(u, a, b, k, data_train, label_train)
        # 进行KNN分类
        accuracy += knn_test.KNNAlgorithm(result, data_train, data_test, label_train, label_test, C, a, b, k, i)
    print("最终结果为: ", accuracy / times)
    # endregion
