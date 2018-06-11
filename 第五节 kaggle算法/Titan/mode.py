# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # K折交叉验证模块

# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC

from  sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



initle_df = pd.read_csv('./input/train.csv', index_col=0)
test_df = pd.read_csv('./input/test.csv', index_col=0)
y_train = initle_df['Survived']
train_df=initle_df

# train_df = train_df.drop(['Ticket'], axis=1)
# test_df = test_df.drop(['Ticket'], axis=1)
all_df=pd.concat((train_df,test_df),axis=0)



all_df['Cabin'][(all_df['Cabin'].isnull())]='U20'

all_df['Fare'][(all_df['Fare'].isnull())]=all_df['Fare'].mean()


age_df = all_df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
# #采用随机森林来补全年龄
# age_df_notnull = age_df.loc[(all_df['Age'].notnull())] #选择age不为空的行，并且把它们保存起来
# age_df_isnull = age_df.loc[(all_df['Age'].isnull())]
#
# X = age_df_notnull.values[:,1:]
# Y = age_df_notnull.values[:,0]
#
# RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
# RFR.fit(X,Y)
# PRE = RFR.predict(age_df_isnull.values[:,1:])
#
# all_df['Age'][(all_df['Age'].isnull())]=PRE.astype(int)
# all_df['Title']= all_df['Name'].str.extract(', (.*?)\.', expand=False)

#根据姓名来取年龄
all_df['Title']= all_df['Name'].str.extract(', (.*?)\.', expand=False)
# print(all_df.groupby('Title')['Age'].size())
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'the Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
all_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = all_df.groupby('Title')['Age'].median()[titles.index(title)]
    all_df.loc[(all_df['Age'].isnull()) & (all_df['Title'] == title), 'Age'] = age_to_impute


# #分析姓名的影响
# initle_df['Title']= initle_df['Name'].str.extract(', (.*?)\.', expand=False)
# initle_df[['Title','Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()


#分析有无亲人影响，并且改变数据
# all_df['SibSp'][(all_df['SibSp']!=0)]=1
# all_df['Parch'][(all_df['Parch']!=0)]=1
#创建家族幸存的影响
all_df['Family_Size']=all_df['SibSp']+ all_df['Parch']
all_df['Last_Name'] = all_df['Name'].apply(lambda x: str.split(x, ",")[0])
DEFAULT_SURVIVAL_VALUE = 0.5
all_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
#看同姓人群中，家族是否有人幸存
for grp, grp_df in all_df.groupby(['Last_Name', 'Fare']):
    if(len(grp_df)!=1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            if (smax == 1.0):
                all_df.loc[ind,'Family_Survival'] = 1
            elif (smin == 0.0):
                all_df.loc[ind,'Family_Survival'] = 0
            # print(all_df[ind - 1:ind]['Family_Survival'])

#看买同一张票人群中，家族是否有人幸存
for _, grp_df in all_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                if (smax == 1.0):
                    all_df.loc[ind, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    all_df.loc[ind, 'Family_Survival'] = 0

#特征工程，改变数据，使之离散化
all_df=all_df.drop('Name',axis=1)
# 改变港口的数据
all_df['Embarked']=pd.factorize(all_df['Embarked'])[0]
# #新增母亲字段
def add_mother(x,y):
    if x == 'Mrs' and  y>1:
        return 1
    else:
        return 0
all_df['mother']=list(map(lambda x,y:add_mother(x,y),all_df['Title'],all_df['Parch']))
print(all_df['mother'])
#新增儿童字段
all_df['Child']=all_df['Age']
all_df['Child'][(all_df['Child']<=12)]=1
all_df['Child'][(all_df['Child']>12)]=0
# 改变性别
all_df['Sex']=pd.factorize(all_df['Sex'])[0]
# 改变甲板
all_df['Cabin']=pd.factorize(all_df['Cabin'])[0]
# 改变Title
all_df['Title']=pd.factorize(all_df['Title'])[0]
# 改变票价
all_df['FareBin'] = pd.qcut(all_df['Fare'], 5)

label = LabelEncoder()
all_df['FareBin_Code'] = label.fit_transform(all_df['FareBin'])
# all_df['Fare'][(all_df['Fare']<=20)]=0
# all_df['Fare'][(all_df['Fare']>20) & (all_df['Fare']<=120 )]=1
# all_df['Fare'][(all_df['Fare']>120)]=2
# all_df['Fare'] = pd.factorize(fare_change)[0]

# # 改变年龄
all_df['AgeBin'] = pd.qcut(all_df['Age'], 4)
label = LabelEncoder()
all_df['AgeBin_Code'] = label.fit_transform(all_df['AgeBin'])

all_df=all_df.drop(['Fare','Age','AgeBin','FareBin','Embarked','Cabin','Title','SibSp','Parch','Ticket','Last_Name','Survived','mother','Child'], axis=1)


print(all_df.head())

# 准备训练样本
from sklearn.model_selection import KFold

titanic_train_data_X = all_df.loc[train_df.index]
titanic_test_data_X = all_df.loc[test_df.index]
titanic_train_data_Y = y_train

ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)
x_train = titanic_train_data_X.values # Creates an array of the train data
x_test = titanic_test_data_X.values # Creats an array of the test data
y_train = titanic_train_data_Y.values

std_scaler = StandardScaler()
x_train = std_scaler.fit_transform(x_train)
x_test = std_scaler.transform(x_test)

# def get_out_fold(clf, x_train, y_train, x_test):#本函数输入了4个参数，第一个是你选的模型，然后是你的训练和label，最后是你的测试数据
#     oof_train = np.zeros((ntrain,))
#     oof_test = np.zeros((ntest,))
#     oof_test_skf = np.empty((NFOLDS, ntest))
#
#     #在循环中对数据进行切分，会把整个数据切成很多份，每一份再分成训练和标签,同时也会再分离出一个测试样本
#     #每次循环都会训练一下，训练分为两部分，一部分是对K分割的数据进行训练，一部分是对测试集训练和预测
#     for i, (train_index, test_index) in enumerate(kf.split(x_train)):
#         x_tr = x_train[train_index]
#         y_tr = y_train[train_index]
#         x_te = x_train[test_index]
#
#         clf.fit(x_tr, y_tr)
#
#         oof_train[test_index] = clf.predict(x_te)
#         oof_test_skf[i, :] = clf.predict(x_test)
#     oof_test[:] = oof_test_skf.mean(axis=0)
#     #生成了经过处理的891份训练集预测结果和418份测试集预测结果，输出之后进行整合。
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#
#
# # dt_est = RandomForestClassifier(random_state=0)
# # dt_param_grid = {'n_estimators':[x for x in range(1,1000,100)]}
# # dt_grid = GridSearchCV(dt_est, dt_param_grid, cv=10, verbose=1)
# # dt_grid.fit(x_train, y_train)
# # print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
# # print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
# # print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
#
# rf = RandomForestClassifier(n_estimators=600, warm_start=True, max_features='sqrt',max_depth=6,
#                             min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
#
# ada = AdaBoostClassifier(n_estimators=650, learning_rate=0.1)
#
# et = ExtraTreesClassifier(n_estimators=650, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
#
# gb = GradientBoostingClassifier(n_estimators=600, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
#
# dt = DecisionTreeClassifier(max_depth=4,min_samples_split=2)
#
# knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
#                            metric_params=None, n_jobs=1, n_neighbors=6, p=2,
#                            weights='uniform')
#
# svm = SVC(kernel='linear', C=0.025)
#
# lr = LogisticRegression(C=5.0, penalty='l1', tol=1e-6)
# #
#
# # 采用STACKING方法
#
# rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
# ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost
# et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
# gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
# dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
# knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
# svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector
# lr_oof_train, lr_oof_test = get_out_fold(lr, x_train, y_train, x_test) # LR Vector
#
# x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train,svm_oof_train,lr_oof_train,knn_oof_train), axis=1)
# x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test,svm_oof_test,lr_oof_test,knn_oof_test), axis=1)
#
#
# from xgboost import XGBClassifier
#
# gbm = XGBClassifier(n_estimators= 500, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8,
#                         colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1)
# gbm.fit(x_train, y_train)
# predictions = gbm.predict(x_test)
#
# scores = cross_val_score(gbm, x_train, y_train, cv=10, scoring='accuracy')
# print(scores.mean())
#
# StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': predictions})
# StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')
#
# # dt_est = XGBClassifier(random_state=0)
# # dt_param_grid = {'n_estimators':[x for x in range(1,5000,500)],'max_depth': [4], 'min_child_weight':[2], 'gamma':[0.9], 'subsample':[0.8],
# #                         'colsample_bytree':[0.8], 'nthread': [-1], 'scale_pos_weight':[1]}
# # dt_grid = GridSearchCV(dt_est, dt_param_grid, cv=5, verbose=1)
# # dt_grid.fit(x_train, y_train)
# # print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
# # print('Top N Features Best DT Score:' + str(dt_grid.best_score_))








# #采用VOTE方法
# voting_est =VotingClassifier(estimators=[('ada', ada), ('dt', dt), ('knn', knn), ('rf', rf), ('gbm', gb), ('et', et)],
#                                        voting='hard')
# print('----------------------1')
# voting_est.fit(x_train, y_train)
# print('----------------------2')
# predictions = voting_est.predict(x_test)
#
# scores = cross_val_score(voting_est, x_train, y_train, cv=10, scoring='accuracy')
# print(scores.mean())
# StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': predictions})
# StackingSubmission.to_csv('StackingSubmission-3.csv',index=False,sep=',')

#
# # #采用Bagging方法
# std_scaler = StandardScaler()
# x_train = std_scaler.fit_transform(x_train)
# x_test = std_scaler.transform(x_test)
#
# clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf = BaggingRegressor(base_estimator=clf, n_estimators=20 ,max_samples=0.8 , max_features=1.0 , bootstrap=True ,bootstrap_features=False,)
# # print('1')
# # scores = cross_val_score(bagging_clf, x_train, y_train, cv=10, scoring='accuracy')
# # print('2')
# # print(scores.mean())
# bagging_clf.fit( x_train, y_train)
# predictions=bagging_clf.predict(x_test)
# #
# #
# #
# #
# #
# StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': predictions.astype(np.int32)})
# StackingSubmission.to_csv('StackingSubmission-3.csv',index=False,sep=',')




# all_df['Age'].hist(bins=70)
# plt.show()

# https://blog.csdn.net/Koala_Tree/article/details/78725881

# knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
#                            metric_params=None, n_jobs=1, n_neighbors=6, p=2,
#                            weights='uniform')
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
# print(scores.mean())
# StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': y_pred.astype(np.int32)})
# StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')
#
# from xgboost import XGBClassifier
#
# gbm = XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)
# grid ={'max_delta_step': 0, 'max_depth': 10, 'min_child_weight': 2, 'n_estimators': 280, 'colsample_bytree': 0.65, 'gamma': 2}
# gbm.set_params(**grid)
# gbm.fit(x_train, y_train)
# y_pred = gbm.predict(x_test)
#
# # dt_est = XGBClassifier(random_state=0)
# # dt_param_grid = {'n_estimators':[240,280,320],
# #             'max_depth':[10,11,12],
# #             'gamma':[0,1,2,3],
# #             'max_delta_step':[0,1,2],
# #             'min_child_weight':[1,2,3],
# #             'colsample_bytree':[0.55,0.6,0.65],
# #             'learning_rate':[0.1,0.2,0.3]
# #             }
# #
# # dt_grid = GridSearchCV(dt_est, dt_param_grid, verbose=1)
# # dt_grid.fit(x_train, y_train)
# # print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
# # print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
#
#
# scores = cross_val_score(gbm, x_train, y_train, cv=10, scoring='accuracy')
# print(scores.mean())
# StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': y_pred.astype(np.int32)})
# StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')


#EXT 试试
ext = ExtraTreesClassifier()
grid ={'class_weight': 'balanced', 'n_jobs': 2, 'random_state': None, 'warm_start': True}
ext.set_params(**grid)
ext.fit(x_train, y_train)
y_pred = ext.predict(x_test)
scores = cross_val_score(ext, x_train, y_train, cv=10, scoring='accuracy')
print(scores.mean())
StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': y_pred.astype(np.int32)})
StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')




#https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/notebook