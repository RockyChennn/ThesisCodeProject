import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.cluster import KMeans
from imputation import ZI, MI, kNNI
from DMI import DMI

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_set = [
    "data/Iris=3.txt", "data/Seeds=3.txt", "data/Wine=3.txt",
    "data/Libras=15.txt"
]
data_path = data_set[0]

data_name = re.compile('\w+').findall(data_path)[1]

k = int(re.compile('\w+').findall(data_path)[2])  # 从 data_path 中读取类别个数
print("数据集名称：", data_name)

# print("数据集类别数：", k)


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=" ")
    # print("实例个数：", len(points))
    columns = np.shape(points)[1] - 1  # 除去分类标签为实际的特征个数
    return points, columns


if __name__ == '__main__':
    data, columns = load_data()
    target, columnss = load_data()
    data = data[:, 0:columns]
    target = target[:, 0:columns]

    # groupID = 1  # ZI/MI/kNNI/AMI
    # groupID = 2
    # groupID = 3
    groupID = 4

    typeID = 1  # MAR/MCAR/MNAR
    typeID = 2
    typeID = 3

    if typeID == 1:
        miss_mask = np.loadtxt("miss_mask/" + data_name +
                               "/MAR/MAR-Iris-20.txt",
                               delimiter=" ")
        print("MAR")
    elif typeID == 2:
        miss_mask = np.loadtxt("miss_mask/" + data_name +
                               "/MCAR/MCAR-Iris-20.txt",
                               delimiter=" ")
        print("MCAR")
    else:
        miss_mask = np.loadtxt("miss_mask/" + data_name +
                               "/MNAR/MNAR-Iris-20.txt",
                               delimiter=" ")
        print("MNAR")

    data[miss_mask == 1] = np.nan

    if groupID == 1:
        print("ZI")
        data = ZI(data[:, 0:columns])
    elif groupID == 2:
        print("MI")
        data = MI(data[:, 0:columns])
    elif groupID == 3:
        print("kNNI")
        data = kNNI(data[:, 0:columns])
    else:
        print("AMI")
        data = DMI(data[:, 0:columns])
    print(math.sqrt(mean_squared_error(target, data)))
