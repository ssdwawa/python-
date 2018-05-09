# import pandas as pda
# #1.逻辑回顾
# file='luqu.csv'
# data=pda.read_csv(file)
# x=data.iloc[:,1:3].as_matrix()#获取2到3列 同时有标题
# y=data.iloc[:,0:1].as_matrix()#admit:[400列]
#
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.linear_model import RandomizedLogisticRegression as RLR #用来做特征筛选 降纬
#
#
# print(x)
# r2=LR(C=1e5)
# r2.fit(x,y)
# print(r2.predict([[500,2]]))


import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[400,3],[600,3],[550,2],[700,2],[300,3],[750,2]])
y = np.array([0, 1, 0, 1, 0, 1])



print(X)
clf = LinearRegression()
clf.fit(X,y)

print(clf.predict([[600,3]]))
