import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


def ZI(data):
    imp = SimpleImputer(missing_values=np.nan,
                        strategy='constant',
                        fill_value=0)
    imp.fit(data)
    SimpleImputer()
    return imp.transform(data)


def MI(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)
    SimpleImputer()
    return imp.transform(data)


def kNNI(data):
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
    imputer.fit(data)
    return imputer.transform(data)


# 导入数据
def load_data():
    miss_mask = np.loadtxt("miss_mask/MCAR-Iris-20.txt", delimiter=" ")
    points = np.loadtxt("data/Iris=3.txt", delimiter=" ")
    columns = np.shape(points)[1] - 1  # 除去分类标签为实际的特征个数
    return points, columns, miss_mask


if __name__ == '__main__':
    data, columns, miss_mask = load_data()
    data = data[:, 0:columns]
    data[miss_mask == 1] = np.nan
    print(data)
    print(kNNI(data))