#先随机在数据中扔几个点，作为核心。再计算每一个点距离核心的位置。
#每个核心肯定会有边境点，之后计算每个核心分离的中间位置，把核心偏移过去。重新形成新的边境
#多次循环后核心就会稳定
import pandas as pda
import numpy as np
from sklearn.cluster import KMeans as KM
if __name__=='__main__':
    data=pda.read_csv('luqu.csv')
    x=data.iloc[:,1:4].as_matrix()
    km=KM(n_clusters=2,n_jobs=2)
    print(km.fit_predict(x))

