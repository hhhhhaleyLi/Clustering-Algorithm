# 开发人员: LiHaoPu
# 开发时间: 2021/11/18 14:15
import numpy as np
import FCM.FCM as fcm
import shadowedSet.shadowedSetTest as sds
import ThreeWayClustering.twc_base_sds as twc


if __name__ == '__main__':
    # 暂时使用自定义的数据集 .data文件暂时不会操作
    data = 10 * np.random.rand(4, 2)   # 100行2列的随机数组 表示100个拥有2个属性的数据
    # 调用FCM算法获得类簇数目、隶属度矩阵、类簇中心
    k, u, C = fcm.GetMembershipByFCM(data)
    # 通过阴影集获取最优阈值α、β
    a, b = sds.shadowedSet(u, k)
    # 获取三支聚类结果
    result = twc.GetClusteringResult(u, a, b, k, data)
    for i in range(0, len(result)):
        print(result[i])

