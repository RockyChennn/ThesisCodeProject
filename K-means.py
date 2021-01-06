import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt

# 以下包为生成excel所用
import xlwt
import datetime
import os
import shutil

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_set = ["data/Test=2.txt",
            "data/Glass=6.txt",
            "data/Iris=3.txt",
            "data/Landsat=7.txt",
            "data/Leaf=36.txt",
            "data/Libras=15.txt",
            "data/LungCancer=3.txt",
            "data/Seeds=3.txt",
            "data/Sonar=2.txt",
            "data/UserKnowledgeModeling=4.txt",
            "data/Wine=3.txt"]
data_path = data_set[2]

data_name = re.compile('\w+').findall(data_path)[1]
k = int(re.compile('\w+').findall(data_path)[2])  # 从 data_path 中读取类别个数
print("数据集名称：", data_name)
print("数据集类别数：", k)


# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=" ")
    rows = np.shape(points)[1] - 1
    return points, rows


def getDistanceMatrix(data, centerPoints, k):
    """
    计算质点与聚类中心的距离
    :param data: 样本点
    :param centerPoints: 质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵和组内平方和
    """
    distanceMatrix = []
    wcss = 0
    for i in range(len(data)):
        distanceMatrix.append([])
        for j in range(k):
            distance = sum((data[i, :rows] - centerPoints[j, :]) ** 2)
            distanceMatrix[i].append(math.floor(math.sqrt(distance) * 100) / 100)
    for i in range(len(distanceMatrix)):
        wcss += min(distanceMatrix[i]) ** 2
    data_wcss.append(wcss)
    return np.asarray(distanceMatrix)


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
    distanceMatrix = getDistanceMatrix(data, centerPoints, k)
    clusterRes = divide(data, distanceMatrix)
    centerNow = center(data, clusterRes, k)
    err = sum(sum(abs(centerNow - centerPoints)))
    return err, centerNow, k, clusterRes


def plotRes(data, clusterRes):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param k: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange', 'brown']
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


def recordResult():
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(data_name)
    nowday = str(datetime.datetime.now()).replace(":", "-")[5:19]
    file_name = data_name + '(' + nowday + ").xls"
    style = xlwt.XFStyle()
    style.num_format_str = 'M/D/YY'  # Other options: D-MMM-YY, D-MMM, MMM-YY, h:mm, h:mm:ss, h:mm, h:mm:ss, M/D/YY h:mm, mm:ss, [h]:mm:ss, mm:ss.0
    worksheet.write(0, 0, datetime.datetime.now(), style)
    workbook.save(file_name)
    aa = os.getcwd()
    # 获取当前文件路径
    file_path = os.path.join(aa, file_name)
    # 移动文件到E盘地方
    target_path = "./result"
    # 使用shutil包的move方法移动文件
    shutil.move(file_path, target_path)


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
    print("聚类结果：", groupNum)
    # 类别重复的情况下，提示问题
    if len(set(groupNum)) < k:
        print("纠错-聚类过程记录", groupRecord)
    return accuracy


if __name__ == '__main__':
    data, rows = load_data()
    index = np.asarray(list(map(int, np.asarray(data[:, rows]) - 1)))
    data_wcss = []

    centerPoints = np.asarray(random.sample(data[:, 0: rows].tolist(), k))  # 随机取k个质心
    err, centerNow, k, clusterRes = classfy(data, centerPoints, k)
    while np.any(abs(err) > 0.00001):
        err, centerNow, k, clusterRes = classfy(data, centerNow, k)  # 未满足收敛条件，继续聚类
    distanceMatrix = getDistanceMatrix(data, centerNow, k)
    clusterResult = divide(data, distanceMatrix)
    print("聚类准确率：", evaluate(clusterResult, index), "%")
    plotWCSS(data_wcss)
    recordResult()
    # plotRes(data, index)  # 原数据分布
    # plotRes(data, clusterResult)  # 聚类结果可视化
