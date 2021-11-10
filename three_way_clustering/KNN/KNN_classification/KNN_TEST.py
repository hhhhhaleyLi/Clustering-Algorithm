# 开发人员: LiHaoPu
# 开发时间: 2021/11/10 14:40
import KNN
from numpy import *

# 生成数量集和类别标签
dataSet, labels = KNN.createDataSet()
# 定义一个未知类别的数据
testX = array([1.2, 1.0])
k = 3
# 调用分类函数对未知数据分类
outputLabel = KNN.KNN_Classify(testX, dataSet, labels, k)
print("Your input is: ", testX, " and classified to class: ", outputLabel)

testX = array([0.1, 0.3])
outputLabel = KNN.KNN_Classify(testX, dataSet, labels, k)
print("Your input is: ", testX, " and classified to class: ", outputLabel)