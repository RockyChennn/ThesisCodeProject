import numpy as np
import pandas as pd
import math
import re
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_set = [
    "data/Glass=6.txt", "data/Iris=3.txt", "data/Leaf=36.txt",
    "data/LungCancer=3.txt", "data/Libras=15.txt", "data/Seeds=3.txt",
    "data/UserKnowledgeModeling=4.txt", "data/Wine=3.txt"
]
data_path = data_set[4]

data_name = re.compile('\w+').findall(data_path)[1]
k = int(re.compile('\w+').findall(data_path)[2])  # 从 data_path 中读取类别个数
# a = 0.25  # 惩罚项的加权系数
print("数据集名称：", data_name)
print("数据集类别数：", k)


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=" ")
    print("实例个数：", len(points))
    columns = np.shape(points)[1] - 1  # 除去分类标签为实际的特征个数
    return points, columns

def getDistanceMatrix(data, centerPoints):
    """
    计算实例与各个聚类中心的距离，用于后续为每个实例进行分类
    :param data: 样本点
    :param centerPoints: 质点集合
    :return: 质心与样本点距离矩阵和组内平方和
    """
    distanceMatrix = []
    wcss = 0
    for i in range(len(data)):
        distanceMatrix.append([])
        for j in range(k):
            distance = 0
            now = data[i, :]
            count = 0
            for n in range(columns):
                if pd.isnull(now[n]):
                    count += 1
                else:
                    distance += (np.abs(data[i, n] - centerPoints[j, n]))**2
            distanceMatrix[i].append(math.sqrt(distance * columns / (columns - count)))
    for i in range(len(distanceMatrix)):
        wcss += min(distanceMatrix[i])**2
    data_wcss.append(wcss)
    return np.asarray(distanceMatrix)


def divide(data, distanceMatrix):
    """
    对数据点分组，返回每一个数据的聚类结果
    :param data: 样本集合
    :param distanceMatrix: 质心与所有样本的距离
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(distanceMatrix[i])  # 按升序排列，seq存储索引值，seq[0]表示最近的那个实例
        clusterRes[i] = seq[0]
    return np.asarray(clusterRes)


def center(data, clusterRes):
    """
    计算质心
    :param clusterRes: 每个实例被分配到的中心
    :return: 计算得到的质心
    """
    centerNow = []
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterRes == i)
        matrix, mean, var = getMissInfo(data[idx])
        centerNow.append(mean)
    centerNow = np.asarray(centerNow)
    return centerNow[:, 0:columns]


def classfy(data, centerPoints):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param centerPoints: 每个聚类的中心
    :return: 误差, 新质心
    """
    distanceMatrix = getDistanceMatrix(data, centerPoints)
    clusterRes = divide(data, distanceMatrix)
    centerNow = center(data, clusterRes)
    err = sum(sum(abs(centerNow - centerPoints)))
    return err, centerNow, clusterRes


def plotRes(data, clusterRes):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :return:
    """
    nPoints = len(data)
    scatterColors = [
        'red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange', 'brown'
    ]
    for i in range(k):
        color = scatterColors[i]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='o')
    plt.show()


def plotWCSS(data):
    """
    绘制聚类内平方和（Within-Cluster Sum-of-Squares）变化曲线
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data, c="red", label='平方误差和变化曲线', lw=3)
    plt.legend(loc='best')
    fig.suptitle('平方误差和变化曲线', fontsize=14, fontweight='bold')
    # ax.set_title("axes title")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平方误差和")
    plt.show()


def getMissInfo(data):
    '''
        获取缺失数据集的信息，返回缺失值标记矩阵、均值和方差
        均值用于初始化中心点，随机选取的中心点可能会包含缺失值
        标准差var用于作为惩罚项
    '''
    columns = np.shape(data)[1]
    flagMatrix = pd.isnull(data)
    mean = []
    var = []
    # 计算每列的均值以及标准差
    for i in range(columns):
        column = data[:, i][flagMatrix[:, i] == False]
        if np.sum(flagMatrix[:, i] == False) == 0:
            var.append(0)
            mean.append(0)
        else:
            var.append(np.round(np.var(column), 3))
            mean.append(
                np.round(
                    np.sum(column) / np.sum(flagMatrix[:, i] == False), 3))
    return flagMatrix, mean, var


def getInitPoints():
    '''
        初始化聚类中心，如果中心包含缺失值则用该维特征的均值来填充
        :return:
    '''
    # center = np.asarray(random.sample(data[:, 0:columns].tolist(), k))
    # flagMatrix, mean, var = getMissInfo(center)
    # for i in range(k):
    #     for j in range(columns):
    #         if flagMatrix[i, j]:
    #             center[i, j] = mean[j]
    center = np.asarray(random.sample(data_copy[:, 0:columns].tolist(), k))
    for i in range(k):
        idx = np.where(index == i)
        center[i] = np.asarray(random.sample(data_copy[idx].tolist(), 1))
    return center


def startCluster(data, index):
    # 初始化中心点
    centerPoints = getInitPoints()  # 随机取k个质心
    err, centerNow, clusterRes = classfy(data, centerPoints)
    while np.any(abs(err) > 0.00001):
        err, centerNow, clusterRes = classfy(data, centerNow)  # 未满足收敛条件，继续聚类
    distanceMatrix = getDistanceMatrix(data, centerNow)
    clusterResult = divide(data, distanceMatrix)

    # 调整兰德指数计算得分基本用法
    score = metrics.adjusted_rand_score(index, clusterResult)
    return score


if __name__ == '__main__':
    data_wcss = []  # 组内平方和变化曲线
    data, columns = load_data()
    data_copy, columns = load_data()
    index = np.asarray(list(map(int, np.asarray(data[:, columns]) - 1)))
    data = data[:, 0:columns]  # 去除索引，只留数据
    data_copy = data_copy[:, 0:columns]  # 去除索引，只留数据

    btn = 2
    if btn == 0:
        miss_mask = np.loadtxt("miss_mask/MAR/MAR-" + data_name + "-20.txt",
                               delimiter=" ")
        print("MAR")
        print("PDS-kmeans")
    elif btn == 1:
        miss_mask = np.loadtxt("miss_mask/MCAR/MCAR-" + data_name + "-20.txt",
                               delimiter=" ")
        print("MCAR")
        print("PDS-kmeans")
    else:
        miss_mask = np.loadtxt("miss_mask/MNAR/MNAR2-" + data_name + "-20.txt",
                               delimiter=" ")
        print("MNAR")
        print("PDS-kmeans")
    data[miss_mask == 1] = np.nan

    flagMatrix, mean, var = getMissInfo(data)
    score = startCluster(data, index)

    for i in range(20):
        score = startCluster(data, index)
        print(score)

    # plotWCSS(data_wcss)
    # print(data_wcss)