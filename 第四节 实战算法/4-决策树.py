#决策树：先算出每条树枝的信息熵再计算每一个条件的信息增益
#嫁的个数为6个，占1/2，那么信息熵为-1/2log1/2-1/2log1/2 = -log1/2=0.301
# H(Y|X = 矮) = -1/7log1/7-6/7log6/7=0.178
# H(Y|X=中) = -1log1-0 = 0
# H(Y|X=高） = -1log1-0=0
# p(X = 矮) = 7/12,p(X =中) = 2/12,p(X=高) = 3/12
# 则可以得出条件熵为：7/12*0.178+2/12*0+3/12*0 = 0.103
# 那么我们知道信息熵与条件熵相减就是我们的信息增益，为0.301-0.103=0.198
#我们可以知道，本来如果我对一个男生什么都不知道的话，作为他的女朋友决定是否嫁给他的不确定性有0.301这么大。
#当我们知道男朋友的身高信息后，不确定度减少了0.198.也就是说，身高这个特征对于我们广大女生同学来说，决定嫁不嫁给自己的男朋友是很重要的。
import pandas as pda
data=pda.read_csv('lesson.csv',encoding='gbk')
data=data.stack()
data[data=='高']=1
data[data=='是']=1
data[data=='多']=1
data[data=='低']=0
data[data=='否']=0
data[data=='少']=0
data=data.unstack()


from sklearn.tree import DecisionTreeClassifier as DTC
dtc=DTC(criterion='entropy')
x=data.iloc[:,1:5].as_matrix()
z=data.iloc[:,5].as_matrix()
y=data.iloc[:,5].as_matrix().astype(float)
# print(z)
# print(y)
dtc.fit(x,y)
print(dtc.predict([[0,0,1,1]]))
