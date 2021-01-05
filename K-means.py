import numpy as np
import math
import random
import matplotlib.pyplot as plt

data_path = "data/Iris.txt"
k = 3  # 类别个数


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=' ')
    rows = np.shape(points)[1] - 1
    return points, rows


def cal_dis(data, centerPoints, k):
    """
    计算质点与聚类中心的距离
    :param data: 样本点
    :param centerPoints: 质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            # 扩展性不足，按维度来算
            distance = 0
            for m in range(rows):
                distance += (data[i, m] - centerPoints[j, m]) ** 2
            dis[i].append(math.floor(math.sqrt(distance) * 100) / 100)
    return np.asarray(dis)


def divide(data, dis):
    """
    对数据点分组，返回每一个数据的聚类结果
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])  # 按升序排列，seq存储索引值，seq[0]表示最近的那个实例
        clusterRes[i] = seq[0]
    return np.asarray(clusterRes)


def center(data, clusterRes, k):
    """
    计算质心
    :param group: 分组后样本
    :param k: 类别个数
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
    return centerNow[:, 0: rows]


def classfy(data, centerPoints, k):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param centerPoints: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    clulist = cal_dis(data, centerPoints, k)
    clusterRes = divide(data, clulist)
    centerNow = center(data, clusterRes, k)
    err = centerNow - centerPoints
    return err, centerNow, k, clusterRes


def plotRes(data, clusterRes, clusterNum):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param clusterNum: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='o')
    plt.show()


def evaluate(clusterResult, index):
    total = len(index)
    group = []
    groupNum = []
    groupRecord = []
    for idx in set(index):
        currentRange = clusterResult[np.where(index == idx)]
        numInRange = set(currentRange)
        count = 0
        nowIndex = 0
        for num in numInRange:
            nowCount = len(currentRange[np.where(currentRange == num)])
            groupRecord.append([idx, num, nowCount])
            if nowCount >= count:
                count = nowCount
                nowIndex = num
        group.append(count)
        groupNum.append(nowIndex)
    accuracy = math.floor(np.sum(group) / total * 10000) / 100
    print("聚类结果组别：", groupNum)
    # 类别重复的情况下，提示问题
    if len(set(groupNum)) < k:
        print("数据记录", groupRecord)
    return accuracy


if __name__ == '__main__':
    data, rows = load_data()
    centerPoints = np.asarray(random.sample(data[:, 0: rows].tolist(), k))  # 随机取k个质心
    err, centerNow, k, clusterRes = classfy(data, centerPoints, k)

    while np.any(abs(err) > 0.0005):
        # print(centerNow)
        err, centerNow, k, clusterRes = classfy(data, centerNow, k)  # 未满足收敛条件，继续聚类

    clulist = cal_dis(data, centerNow, k)
    clusterResult = divide(data, clulist)
    index = np.asarray(list(map(int, np.asarray(data[:, rows]) - 1)))
    print("聚类准确率：", evaluate(clusterResult, index), "%")

    # plotRes(data, clusterResult, k)  # 可视化
