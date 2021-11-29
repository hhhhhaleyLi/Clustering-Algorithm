import math

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def KNNAlgorithm(result, data_train, data_test, label_train, label_test, center, a, b, k, test_index, m=2):   # test_index是实验次数，仅在函数最后输出准确率时使用
    uci_result = []
    for index in range(len(data_test)):   # 迭代待测样本
        y = data_test[index]
        for i in range(k):   # 迭代类簇中心
            temp = 0   # 计算隶属度使用的临时变量
            fr_data = []   # 使用Fr进行分类时的边界域对象
            fr_label = []   # 使用Fr进行分类时的边界域标签
            is_fr = False   # 是否使用Fr进行分类的标识符
            # 计算待测样本对象y对类簇中心center[i]的隶属度
            for j in range(k):
                temp += ((y - center[i]) / (y - center[j]))
            temp = temp ** (2 / (m-1))
            u_temp = 1 / temp
            # todo KNN分类仍存在问题 偶尔会报错: ValueError: Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 5
            u = 0
            for item in u_temp:
                u += item
            u = u / len(u_temp)
            if u > b:   # 使用Co进行KNN
                neighbor_count = 3
                if neighbor_count > len(result[i][0][0]):
                    neighbor_count = len(result[i][0][0])
                if neighbor_count == 0:
                    neighbor_count = 1
                knn = KNeighborsClassifier(n_neighbors=neighbor_count)  # 定义一个knn分类器对象
                # result[i][0][0] 第i个类簇的核心域中的数据点集合   result[i][0][1] 第i个类簇的核心域中的数据点的标签集合
                knn.fit(result[i][0][0], result[i][0][1])  # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
                uci_y_predict = knn.predict(data_test)
                uci_result.append(uci_y_predict[index])
                break
            elif u > a:   # 使用Fr进行KNN
                is_fr = True
                fr_data.extend(result[i][1][0])
                fr_label.extend(result[i][1][1])

            if i == k-1:   # 待测样本对全部类簇的隶属度都已迭代完 (进行到这一步说明该待测对象不属于任何一个类簇的核心域)
                knn = KNeighborsClassifier(n_neighbors=3)  # 定义一个knn分类器对象
                if is_fr:   # 使用Fr对待测样本进行分类
                    knn.fit(fr_data, fr_label)  # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
                    uci_y_predict = knn.predict(data_test)
                    uci_result.append(uci_y_predict[index])
                else:   # 待测样本不在任何一个类簇的边界域，使用全部训练样本对待测样本进行分类
                    knn.fit(data_train, label_train)  # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
                    uci_y_predict = knn.predict(data_test)
                    uci_result.append(uci_y_predict[index])

    uci_result = np.array(uci_result)
    zeroNum = 0
    for item in (uci_result - label_test):
        if item == 0:
            zeroNum += 1
    # print('uci_y_predict = ')
    # print(uci_result)
    # 输出测试的结果
    # print('uci_y_test = ')
    # print(label_test)
    # 输出原始测试数据集的正确标签，以方便对比
    print('第' + str(test_index) + '次实验 Accuracy:', zeroNum/len(uci_result))   # 计算预测正确的个数占测试集总数的比例 即准确率
    return zeroNum/len(uci_result)