import numpy as np
import pandas as pd


def DMI(data):
    '''
        计算出每列的均值和方差
    '''
    rows, columns = np.shape(data)
    flagMatrix = pd.isnull(data)
    # 计算每列的均值以及标准
    mean = []  # [5.0, 3.25, 1.4, 0.2]
    std = []  # [0.099, 0.25, 0.0, 0.0]
    for i in range(columns):
        column = data[:, i][flagMatrix[:, i] == False]
        mean.append(
            np.round(np.sum(column) / np.sum(flagMatrix[:, i] == False), 3))
        std.append(np.round(np.std(column), 3))
    for i in range(rows):
        for j in range(columns):
            if (flagMatrix[i, j] == True):
                # 统计可观测数值与均值的大小情况
                left, right = 0, 0
                for n in range(columns):
                    if flagMatrix[i, n] == False:
                        if data[i, n] > mean[n]:
                            right += 1
                        elif data[i, n] < mean[n]:
                            left += 1
                # print(i, j, left, right)
                if left > right:
                    data[i, j] = mean[j] - std[j]
                elif left < right:
                    data[i, j] = mean[j] + std[j]
                else:
                    data[i, j] = mean[j]
    return data


# 导入数据
def load_data():
    miss_mask = np.loadtxt("miss_mask/MCAR-Iris-20.txt", delimiter=" ")
    points = np.loadtxt("data/Iris=3.txt", delimiter=" ")
    row = np.shape(points)[0]  # 除去分类标签为实际的特征个数
    columns = np.shape(points)[1] - 1  # 除去分类标签为实际的特征个数
    return points, columns, miss_mask


if __name__ == '__main__':
    data, columns, miss_mask = load_data()
    data = data[:, 0:columns]
    data[miss_mask == 1] = np.nan
    # DMI(data)
    print(DMI(data))
