from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier  # K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.model_selection import cross_val_score # K折交叉验证模块
import matplotlib.pyplot as plt

#加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

#分割数据并赋值
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4) #是随机种子数

# #使用K折交叉验证模块
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
#
# #将5次的预测准确率打印出
# print(scores)
# # [ 0.96666667  1.          0.93333333  0.96666667  1.        ]
#
# #将5次的预测准确平均率打印出
# print(scores.mean())
# # 0.973333333333


#建立测试参数集
k_range = range(1, 31)

k_scores = []

#藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') #cv是把数据分为10段，每次取10分之一当测试集
    k_scores.append(scores.mean())

#可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


#一般来说平均方差(Mean squared error)会用于判断回归(Regression)模型的好坏。