import numpy as np
import pandas as pd
import matplotlib.pyplot as pyl
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


df=pd.read_csv('./data/train_small.csv', index_col=0)


test1 = df.groupby(['banner_pos','click'])['click'].count().astype(float)
print(test1)
#
# df=df.drop(['hour','site_id', 'app_id', 'device_id', 'device_ip', 'site_domain',
#                   'site_category', 'app_domain', 'app_category', 'device_model'],axis=1)
#
#
#
# all_dummy_df = pd.get_dummies(df)
# all_dummy_df['C20'][(all_dummy_df['C20']==-1)]=0
# # all_dummy_df.loc[:,1:-1]=np.log1p(all_dummy_df.loc[:,1:-1])
#
# # img=pd.DataFrame({"C18-g":all_dummy_df["C20"],"C18":df["C20"]})
# # img.hist()
# # plt.show()
#
#
# train_data=all_dummy_df.values
# X_train, X_test, y_train, y_test = train_test_split(train_data[0::, 1::], train_data[0::, 0],
#                                                     test_size=0.3, random_state=0)
# # LR = LogisticRegression()
# # LR.fit(X_train,y_train)
# # LR.predict(X_test)
# # print(LR.score(X_test,y_test))
#
# max_features = [1, 10, 100, 1000]
# test_scores = []
# for max_feat in max_features:
#     clf = LogisticRegression(C=max_feat,penalty='l1')
#     test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# pyl.plot(max_features, test_scores)
# pyl.title("Alpha vs CV Error")
# pyl.show()
