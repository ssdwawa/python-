
import pandas as pda

file='luqu.csv'
data=pda.read_csv(file)
x=data.iloc[:5,1:4].as_matrix()#获取2到3列 同时有标题
y=data.iloc[:5,0:1].as_matrix()#admit:[400列]



import numpy as np
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
knn.fit(x, y.ravel())  # 导入数据进行训练
acc=knn.score(x,y) #根据给定数据与标签返回正确率的均值
print('模型评价:',acc)
data=knn.predict([[600,3.5,2]])
print(data)