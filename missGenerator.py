import math
import random
import numpy as np


# 导入数据
def load_data():
    points = np.loadtxt("data/Iris=3.txt", delimiter=" ")
    # print(np.shape(points))
    return points


def missGenerator(dataSet, frac, type):
    """
        dataSet_miss返回缺失数据集、缺失数据标记矩阵以及每个特征的缺失率
        generating the pattern of missing features as per type
        frac -> fraction of missingness (0,1)
        type -> type of missingness:
            1: MCAR
            2: MAR
            3: MNAR-I
            4: MNAR-II
    """
    missCounter =  math.floor(np.shape(dataSet)[0] * np.shape(dataSet)[1] * frac)    # 确定丢失的特征个数
    print(missCounter)

    if type == 1 :
        # generating a random pattern of missing features s.t. features are removed completely at random (MCAR)
        # 数据集大小对应一个缺失标记miss_mask，随机抽取missCounter个位置标记为缺失
        miss_mask = np.zeros(np.shape(dataSet), dtype=int)
        while missCounter > 0:
            # size(x,1)获取x的行大小，randi(3)从1至3中随机取一个数
            i = random.randint(0, np.shape(dataSet)[0] - 1)
            j = random.randint(0, np.shape(dataSet)[1] - 1)
            if miss_mask[i][j] != 1:
                miss_mask[i][j] = 1
                missCounter = missCounter - 1

        print('Finished generating MCAR masks.')

#     if type == 2:
#         # generating a random pattern of missing features s.t. features are
#         # removed at random, depending upon the observed features (MAR)
#         miss_tag = randperm(size(x,2)) # 返回一个包含1至n的向量
#         x = [x(:,miss_tag(1:floor(size(x,2)/2))),x(:,miss_tag((floor(size(x,2)/2)+1):end))] # 按列打乱x
#
#         reln_tag = zeros(ceil(size(x,2)/2),floor(size(x,2)/2))
#         P = randi(floor(size(x,2)/2),1,ceil(size(x,2)/2))
#         for i = 1:size(reln_tag,1)
#             reln_tag(i,P(i)) = randi(3)
#         end
#         reln_tag = [reln_tag, zeros(ceil(size(x,2)/2),ceil(size(x,2)/2))]
#
#         miss_mask = zeros(size(x))
#         while (missCounter>0)
#             i = randi(size(x,1))
#             j = randi(ceil(size(x,2)/2)) + floor(size(x,2)/2)
#             if (miss_mask(i,j)~=1)
#                 tag_locn = find(reln_tag((j - floor(size(x,2)/2)),:)~=0)
#                 miss_type = reln_tag((j - floor(size(x,2)/2)),tag_locn)
#                 if (miss_type==1)
#                     muMf = 0
#                     sigmaMf = 0.35
#                 elseif (miss_type==2)
#                     muMf = 1
#                     sigmaMf = 0.35
#                 else
#                     muMf = 2
#                     sigmaMf = 0.35
#                 end
#                 probMiss = gaussmf(abs(x(i,tag_locn)), [sigmaMf, muMf])
#                 probAct = rand(1)
#                 if (probAct<=probMiss)
#                     miss_mask(i,j) = 1
#                     missCounter = missCounter - 1
#                 end
#             end
#         end
#         print('Finished generating MAR masks.')
#
#     if type == 3:
#         # generating a random pattern of missing features s.t. features are
#         # removed based on their values only (MNAR-I)
#         miss_type = randi(3,1,size(x,2))
#         miss_mask = zeros(size(x))
#         while (missCounter>0)
#             i = randi(size(x,1))
#             j = randi(size(x,2))
#             if (miss_mask(i,j)~=1)
#                 if (miss_type(j)==1)
#                     muMf = 0
#                     sigmaMf = 0.35
#                 elseif (miss_type(j)==2)
#                     muMf = 1
#                     sigmaMf = 0.35
#                 else
#                     muMf = 2
#                     sigmaMf = 0.35
#                 end
#                 # 高斯曲线成员函数
#                 probMiss = gaussmf(abs(x(i,j)), [sigmaMf, muMf])
#                 probAct = rand(1)
#                 if (probAct<=probMiss)
#                     miss_mask(i,j) = 1
#                     missCounter = missCounter - 1
#                 end
#             end
#         end
#         miss_count = sum(miss_mask,2) miss_count = miss_count'
#         clearvars -except x labels K k miss_count miss_mask alpha Loops loop type alpha
#         print('Finished generating MNAR-I masks.')
#
#     if type == 4:
#         # generating a random pattern of missing features s.t. features are removed according to MNAR-II
#         miss_tag = randperm(size(x,2))
#         x = [x(:,miss_tag(1:floor(size(x,2)/2))),x(:,miss_tag((floor(size(x,2)/2)+1):end))]
#
#         reln_tag = zeros(ceil(size(x,2)/2),floor(size(x,2)/2)) %required for MAR missingness
#         P = randi(floor(size(x,2)/2),1,ceil(size(x,2)/2))
#         for i = 1:size(reln_tag,1)
#             reln_tag(i,P(i)) = randi(3)
#         end
#         reln_tag = [reln_tag, zeros(ceil(size(x,2)/2),ceil(size(x,2)/2))] %required for MNAR missingness
#
#         miss_type2 = randi(3,1,floor(size(x,2)/2))
#
#         miss_mask = zeros(size(x))
#         while (missCounter>0)
#             flag = round(rand(1))
#             if (flag)
#                 %dependence on observed features
#                 i = randi(size(x,1))
#                 j1 = randi(ceil(size(x,2)/2))
#                 j = j1 + floor(size(x,2)/2)
#                 if (miss_mask(i,j)~=1)
#                     miss_type1 = reln_tag(j1,P(j1))
#                     if (miss_type1==1)
#                         muMf = 0
#                         sigmaMf = 0.35
#                     elseif (miss_type1==2)
#                         muMf = 1
#                         sigmaMf = 0.35
#                     else
#                         muMf = 2
#                         sigmaMf = 0.35
#                     end
#                     probMiss = gaussmf(abs(x(i,P(j1))), [sigmaMf, muMf])
#                     probAct = rand(1)
#                     if (probAct<=probMiss)
#                         miss_mask(i,j) = 1
#                         missCounter = missCounter - 1
#                     end
#                 end
#             else
#                 %dependence on unobserved features
#                 i = randi(size(x,1))
#                 j = randi(floor(size(x,2)/2))
#                 if (miss_mask(i,j)~=1)
#                     if (miss_type2(j)==1)
#                         muMf = 0
#                         sigmaMf = 0.35
#                     elseif (miss_type2(j)==2)
#                         muMf = 1
#                         sigmaMf = 0.35
#                     else
#                         muMf = 2
#                         sigmaMf = 0.35
#                     end
#                     probMiss = gaussmf(abs(x(i,j)), [sigmaMf, muMf])
#                     probAct = rand(1)
#                     if (probAct<=probMiss)
#                         miss_mask(i,j) = 1
#                         missCounter = missCounter - 1
#                     end
#                 end
#             end
#         end
#         print('Finished generating MNAR-II masks.')




        # 用 maxValue 填充缺失值，miss_mask标记为1的数据是缺失值
        maxValue = max(map(max, dataSet)) + 1000
        dataSet_miss = np.array(dataSet)
        dataSet_miss[miss_mask == 1] = maxValue

        # 计算每个特征的缺失比例，用于特征加权
        prob_miss = np.sum(miss_mask, axis=0) / len(miss_mask)
        print(np.sum(miss_mask, axis=0))
        print(prob_miss)
        return dataSet_miss, miss_mask



if __name__ == '__main__':
    data = load_data()
    missGenerator(data, 0.1, 1)