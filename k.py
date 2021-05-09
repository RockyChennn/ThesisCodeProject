import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

data = np.loadtxt("data/Iris=3.txt", delimiter=" ", encoding='gbk')
data = data[:, 0:np.shape(data)[1] - 1] 
SSE = []

# 插补处理
miss_mask = np.loadtxt("miss_mask/Iris/MCAR/MCAR-Iris-40.txt",
                               delimiter=" ")
data[miss_mask == 1] = np.nan
imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
imputer.fit(data)
data = imputer.transform(data)

for k in range(1,9):
    km = KMeans(n_clusters=k,random_state=100) 
    km.fit(data)
    SSE.append(km.inertia_) 
print(SSE)

plt.plot(range(1,9),SSE,'o-')
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()