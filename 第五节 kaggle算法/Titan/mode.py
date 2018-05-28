# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
initle_df = pd.read_csv('./input/train.csv', index_col=0)
test_df = pd.read_csv('./input/test.csv', index_col=0)
y_train = initle_df['Survived']
train_df=initle_df.drop(['Survived'],axis=1)

train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
all_df=pd.concat((train_df,test_df),axis=0)


all_df['Cabin'][(all_df['Cabin'].notnull())]='Yes'
all_df['Cabin'][(all_df['Cabin'].isnull())]='No'

all_df['Fare'][(all_df['Fare'].isnull())]=all_df['Fare'].mean()


age_df = all_df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(all_df['Age'].notnull())] #选择age不为空的行，并且把它们保存起来
age_df_isnull = age_df.loc[(all_df['Age'].isnull())]

X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]

RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X,Y)
PRE = RFR.predict(age_df_isnull.values[:,1:])

all_df['Age'][(all_df['Age'].isnull())]=PRE.astype(int)
all_df['Title']= all_df['Name'].str.extract(', (.*?)\.', expand=False)



# #分析姓名的影响
# initle_df['Title']= initle_df['Name'].str.extract(', (.*?)\.', expand=False)
# initle_df[['Title','Survived']].groupby(['Title']).mean().plot.bar()
# plt.show()


#分析有无亲人影响，并且改变数据
all_df['SibSp'][(all_df['SibSp']!=0)]=1
all_df['Parch'][(all_df['Parch']!=0)]=1


#特征工程，改变数据，使之离散化
all_df=all_df.drop('Name',axis=1)
# 改变港口的数据
all_df['Embarked']=pd.factorize(all_df['Embarked'])[0]
# 改变性别
all_df['Sex']=pd.factorize(all_df['Sex'])[0]
# 改变甲板
all_df['Cabin']=pd.factorize(all_df['Cabin'])[0]
# 改变Title
all_df['Title']=pd.factorize(all_df['Title'])[0]
# 改变票价
all_df['Fare'].astype(int)
all_df['Fare'][(all_df['Fare']<=20)]=0
all_df['Fare'][(all_df['Fare']>20) & (all_df['Fare']<=120 )]=1
all_df['Fare'][(all_df['Fare']>120)]=2
# # 改变年龄
change_age=pd.qcut(all_df['Age'], 5)
all_df['Age'] = pd.factorize(change_age)[0]


#准备训练样本
from sklearn.model_selection import KFold

titanic_train_data_X = all_df.loc[train_df.index]
titanic_test_data_X = all_df.loc[test_df.index]
titanic_train_data_Y = y_train
print(titanic_test_data_X.index)

ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors = 2)

svm = SVC(kernel='linear', C=0.025)

x_train = titanic_train_data_X.values # Creates an array of the train data
x_test = titanic_test_data_X.values # Creats an array of the test data
y_train = titanic_train_data_Y.values


rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector

x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId':titanic_test_data_X.index, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')



# all_df['Age'].hist(bins=70)
# plt.show()

# https://blog.csdn.net/Koala_Tree/article/details/78725881


