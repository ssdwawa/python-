import numpy as np
import pandas as pda
import matplotlib.pylab as pyl
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

train_df=pda.read_csv('./data/train.csv', index_col=0)
test_df=pda.read_csv('./data/test.csv', index_col=0)

# #采用np.log1p的方法把数据标准化，之后通过pandas的hist()方法把直方图存到内存中 通过pyl.show()来展示
# prices = pda.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
# prices.hist()
# pyl.show()

#把SalePrice取出并标准化，定义为y_train,原数组改变
y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pda.concat((train_df, test_df), axis=0)
print(y_train)

#有些时候数据为数字会对我们产生误导因此把它改为字符串离散化
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)


#把全部的数据给他标签化
all_dummy_df = pda.get_dummies(all_df)

#处理数据缺失，本案例采用平均值来填充缺失
##打印缺失值
#print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
mean_cols = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_cols)
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))


#对数字分类再进行标准化，但不需要对0,1类的再进行分类
numeric_cols = all_df.columns[all_df.dtypes != 'object']
all_dummy_df.loc[:,numeric_cols]=np.log1p(all_dummy_df.loc[:,numeric_cols])

# prices = pda.DataFrame({"LotArea":all_df["LotArea"], "LotArea(price + 1)":all_dummy_df["LotArea"]})
# prices.hist()
# pyl.show()



#进行建模分析
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

X_train = dummy_train_df.values
X_test = dummy_test_df.values



# alphas = np.logspace(-3, 2, 50)
# test_scores = []
# for alpha in alphas:
#     clf = Ridge(alpha) #该模型选用了Ridge，而该模型的参数就是alpha的值
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# pyl.plot(alphas, test_scores)
# pyl.title("Alpha vs CV Error")
# pyl.show()
# 大概alpha=15的时候，可以把score达到0.135左右。

# max_features = [.1, .3, .5, .7, .9, .99]
# test_scores = []
# for max_feat in max_features:
#     clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# pyl.plot(max_features, test_scores)
# pyl.title("Alpha vs CV Error")
# pyl.show()
#参数为.3的时候效果最好


#结合俩个模型的最优解，进行拟合
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
#用expm1还原之前优化的函数
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
#一个正经的Ensemble是把这群model的预测结果作为新的input，再做一次预测。这里我们简单的方法，就是直接『平均化』。
y_final = (y_ridge + y_rf) / 2

#提交数据
submission_df = pda.DataFrame(data={'Id':test_df.index,'price':y_final})



#用xgboost进行
from xgboost import XGBRegressor

# params = [1,2,3,4,5,6]
# test_scores = []
# for param in params:
#     clf = XGBRegressor(max_depth=param)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# pyl.plot(params, test_scores)
# pyl.title("max_depth vs CV Error")
# pyl.show()

xgb = XGBRegressor(max_depth=5)
xgb.fit(X_train, y_train)
y_final = np.expm1(xgb.predict(X_test))
print(y_final)
