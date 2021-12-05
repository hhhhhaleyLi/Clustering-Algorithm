import math
import operator

import numpy as np


def calc_parameter(C, data, k, times, m=2):
    temp_result = C  # 迭代过程中保存类簇结果
    for i in range(0, times):   # 算法迭代次数最多为times次
        temp_result_index = []  # 迭代过程中保存类簇索引  测试准确率使用 实际聚类中使用temp_result
        for ki in range(0, k):
            temp_result_index.append([[], []])   # 正域 边界域

        C_temp = temp_result  # 计算前一次迭代的聚类结果

        # region 计算类簇中心
        center = []  # 类簇C[index]的类中心
        for index in range(0, k):  # 需要计算出数据点xj对每个类簇的欧式距离
            center_temp = np.tile([0.0], temp_result[index][0][0].shape)  # 类簇C[index]的类中心
            for item in temp_result[index][0]:
                item_shape = item.shape
                arr = np.tile([len(temp_result[index][0])], item_shape[0])   # 构造一个和item形状一样的数组，使用len(temp_result[index][0])作为数组元素填充
                center_temp += (item / arr)
            center.append(center_temp)  # 第index个类簇的类中心
        # endregion

        B = []   # 存储每个数据点xj所属类簇的索引的集合
        G = []   # 存储每个数据点xj对类簇Ci的相对等级gij
        Gpq = []   # 存储每个数据点对于每个类簇Ci的gpj、gqj
        # region 对数据集进行迭代
        for j in range(0, len(data)):   # 对数据集进行迭代，确定每个数据点所属类簇
            xj = data[j]   # 数据点xj
            # region 计算数据点xj所属类簇的集合Bxj
            Bxj = []   # 记录数据点xj所属类簇的index
            for l in range(0, k):
                for n in range(0, len(temp_result[l][0])):
                    if (xj == temp_result[l][0][n]).all():
                        Bxj.append(l)
                        break
                for m in range(0, len(temp_result[l][1])):
                    if (xj == temp_result[l][1][m]).all():
                        Bxj.append(l)
                        break
            B.append(Bxj)
            # endregion
            # region 计算数据点xj对于所属类簇的隶属度的集合Mxj
            Mxj = []   # 数据点xj对于所属类簇的隶属度的集合
            dljs = []   # 数据点xj对所有类中心的欧式距离的集合
            for c_item in center:
                dljs.append(math.sqrt(sum([(x - y) ** 2 for (x, y) in zip(xj, c_item)])))   # 数据点xj与类簇C[index]的欧式距离

            d_temp = 0.0
            for index in Bxj:
                # center_xj = []  # 数据点xj所属类簇C[index]的类中心
                # for item in C[index][0]:
                #     center_xj += item / len(C[index][0])
                dij = math.sqrt(sum([(x - y) ** 2 for (x, y) in zip(xj, center[index])]))   # 数据点xj与类簇C[index]的欧式距离
                for dlj in dljs:
                    d_temp += dij / dlj
                uij = 1 / (d_temp ** (2 / (m-1)))   # 数据集xj对所属类簇Ci的隶属度
                Mxj.append(uij)
            # endregion
            max_index, max_number = max(enumerate(Mxj), key=operator.itemgetter(1))
            fj = max_number   # 数据点xj的特征值（使用数据点xj对各个类簇的隶属度中的最大值作为数据点xj的特征值）
            Gxj = []   # 数据点对所属类簇的相对等级的集合
            for uij in Mxj:
                Gxj.append(uij / fj)
            max_index, gpj = max(enumerate(Gxj), key=operator.itemgetter(1))
            gqj = 0.0
            for gj in Gxj:
                if gqj < gj < gpj:
                    gqj = gj   # 获取G中第二大的值gqj
            G.append(Gxj)
            Gpq_temp = [gpj, gqj]
            Gpq.append(Gpq_temp)
        # endregion
        cutoff_temp = 0.0
        for item in Gpq:
            cutoff_temp += (item[0] - item[1])
        cutoff = cutoff_temp / len(Gpq)   # 截至阈值
        for j in range(0, len(data)):   # 迭代数据集，计算每个数据点的评估值v(mi, xj)
            sumL = 0.0   # sum(L(:, j))
            for gij in G[j]:
                if gij > cutoff:
                    sumL += gij
            vxj = []   # 数据点xj对所属类簇的评价值
            for gij in G[j]:
               vxj.append(gij / sumL)   # 数据点xj对满足G > cutoff的类簇的评价值
            a = 1
            b = cutoff / sumL
            for v_index in (0, len(vxj)):
                c_index = B[j][v_index]  # 数据点xj所属类簇在C中的index
                if vxj[v_index] >= a:
                    temp_result[c_index][0].append(data[j])   # 数据点xj属于类簇正域
                    temp_result_index[c_index][0].append(j)
                    break
                if vxj[v_index] > b:
                    temp_result[c_index][1].append(data[j])   # 数据点xj属于类簇边界域
                    temp_result_index[c_index][1].append(j)
        if temp_result == C_temp:
            break
    # result = temp_result   # 最终处理后的类簇集合
    result = temp_result_index
    return result
