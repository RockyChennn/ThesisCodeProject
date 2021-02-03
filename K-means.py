import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.cluster import KMeans
from imputation import ZI, MI, kNNI
from DMI import DMI

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_set = [
    "data/Glass=6.txt", "data/Iris=3.txt", "data/Leaf=36.txt",
    "data/LungCancer=3.txt", "data/Libras=15.txt", "data/Seeds=3.txt",
    "data/UserKnowledgeModeling=4.txt", "data/Wine=3.txt"
]
data_path = data_set[7]

data_name = re.compile('\w+').findall(data_path)[1]

k = int(re.compile('\w+').findall(data_path)[2])  # 从 data_path 中读取类别个数
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
    计算质点与各个聚类中心的距离，用于后续为每个实例进行分类
    :param data: 样本点
    :param centerPoints: 质点集合
    :return: 质心与样本点距离矩阵和组内平方和
    """
    distanceMatrix = []
    wcss = 0
    for i in range(len(data)):
        distanceMatrix.append([])
        for j in range(k):
            distance = sum((data[i, :] - centerPoints[j, :])**2)
            distanceMatrix[i].append(np.round(distance, 3))
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
        sum = data[idx].sum(axis=0)
        if len(data[idx]) == 0:
            avg_sum = 0
        else:
            avg_sum = sum / len(data[idx])
        centerNow.append(avg_sum)
    centerNow = np.asarray(centerNow)
    return centerNow


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
    ax.plot(data, c="red", label='变化曲线', lw=3)
    plt.legend(loc='best')
    fig.suptitle('组内平方和变化曲线', fontsize=14, fontweight='bold')
    # ax.set_title("axes title")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("组内平方和")
    plt.show()


def startCluster(data, index):
    centerPoints = np.asarray(random.sample(data.tolist(), k))  # 随机取k个质心
    # centerPoints = np.asarray(
    #     ([6.0, 2.9, 4.5, 1.5], [5.1, 3.8, 1.5, 0.3], [6.3, 3.4, 5.6, 2.4]))
    err, centerNow, clusterRes = classfy(data, centerPoints)
    while np.any(abs(err) > 0.00001):
        err, centerNow, clusterRes = classfy(data, centerNow)  # 未满足收敛条件，继续聚类
    distanceMatrix = getDistanceMatrix(data, centerNow)
    clusterResult = divide(data, distanceMatrix)

    # estimator = KMeans(n_clusters=k)  #构造聚类器
    # estimator.fit(data)  #聚类
    # label_pred = estimator.labels_  #获取聚类标签

    # 调整兰德指数计算得分基本用法
    # score = metrics.adjusted_rand_score(index, label_pred)
    ARI = metrics.adjusted_rand_score(index, clusterResult)
    NMI = metrics.normalized_mutual_info_score(index, clusterResult)
    return ARI, NMI


if __name__ == '__main__':
    data, columns = load_data()
    data_wcss = []
    index = np.asarray(list(map(int, np.asarray(data[:, columns]) - 1)))
    data = data[:, 0:columns]  # 去除索引，只留数据

    groupID = 2  # ZI/MI/kNNI/DMI
    typeID = 1  # MAR/MCAR/MNAR
    mode = 20  # 20

    if typeID == 1:
        miss_mask = np.loadtxt("miss_mask/MAR/MAR-" + data_name + "-20.txt",
                               delimiter=" ")
        print("MAR")
    elif typeID == 2:
        miss_mask = np.loadtxt("miss_mask/MCAR/MCAR-" + data_name + "-20.txt",
                               delimiter=" ")
        print("MCAR")
    else:
        miss_mask = np.loadtxt("miss_mask/MNAR/MNAR2-" + data_name + "-20.txt",
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
        print("DMI")
        data = DMI(data[:, 0:columns])

    for i in range(20):
        ARI, NMI = startCluster(data, index)
        print(ARI)
