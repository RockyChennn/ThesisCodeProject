import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_set = [
    "data/Test=2.txt", "data/Glass=6.txt", "data/Iris=3.txt",
    "data/Landsat=7.txt", "data/Leaf=36.txt", "data/Libras=15.txt",
    "data/LungCancer=3.txt", "data/Seeds=3.txt", "data/Sonar=2.txt",
    "data/UserKnowledgeModeling=4.txt", "data/Wine=3.txt"
]
data_path = data_set[2]

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
            distanceMatrix[i].append(
                math.floor(math.sqrt(distance) * 100) / 100)
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


def DMI(data, miss_mask):
    '''
    :param data: 数据集
    :param miss_mask: 缺失分布
    :return newdata: 缺失处理完成的数据集
    '''
    print("缺失标记", data)
    newdata = data
    return newdata


if __name__ == '__main__':
    data, columns = load_data()
    # 用于计算兰德指数得分
    index = np.asarray(list(map(int, np.asarray(data[:, columns]) - 1)))
    # 由既有的缺失标记得到缺失分布
    miss_mask = np.loadtxt("miss_mask/MCAR-Iris-20.txt", delimiter=" ")
    data = DMI(data[:, 0:columns], miss_mask)  # 去除索引，只留数据
    data_wcss = []  # 组内平方和变化曲线

    centerPoints = np.asarray(random.sample(data.tolist(), k))  # 随机取k个质心
    err, centerNow, clusterRes = classfy(data, centerPoints)
    while np.any(abs(err) > 0.00001):
        err, centerNow, clusterRes = classfy(data, centerNow)  # 未满足收敛条件，继续聚类
    distanceMatrix = getDistanceMatrix(data, centerNow)
    clusterResult = divide(data, distanceMatrix)

    # 调整兰德指数计算得分基本用法
    score = metrics.adjusted_rand_score(index, clusterResult)
    print("兰德指数得分", score)
