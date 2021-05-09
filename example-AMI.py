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

data_path = "data/example.txt"


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=" ")
    # print("实例个数：", len(points))
    columns = np.shape(points)[1]
    return points, columns


if __name__ == '__main__':
    data, columns = load_data()
    target, columnss = load_data()
    data = data[:, 0:columns]
    target = target[:, 0:columns]

    groupID = 1  # ZI/MI/kNNI/DMI
    groupID = 2
    groupID = 3
    groupID = 4
    miss_mask = np.loadtxt("miss_mask/example.txt", delimiter=" ")
    data[miss_mask == 1] = np.nan

    print(data)
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
        print("DMI")
        data = DMI(data[:, 0:columns])
    print(data)
    print(math.sqrt(mean_squared_error(target, data)))
