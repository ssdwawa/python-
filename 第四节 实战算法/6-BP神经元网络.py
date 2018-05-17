#具体原理见https://www.zhihu.com/question/24827633
#分为三层神经元，层层之间有权重,中间层和输出曾可以再区分为进和出
import pandas as pda
from sklearn.neural_network import MLPClassifier
data=pda.read_csv('lesson.csv',encoding='gbk')
data=data.stack()
data[data=='高']=1
data[data=='是']=1
data[data=='多']=1
data[data=='低']=0
data[data=='否']=0
data[data=='少']=0
data=data.unstack()

x=data.iloc[:,1:5].as_matrix()
y=data.iloc[:,5].as_matrix().astype(float)
mlp =MLPClassifier(hidden_layer_sizes=(30,),activation='relu',max_iter=100)
mlp.fit(x,y)
print(mlp.predict([[1,0,0,1]]))

