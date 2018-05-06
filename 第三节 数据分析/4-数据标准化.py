import pymysql
import pandas as pda
import numpy as np
conn=pymysql.connect(host='127.0.0.1',user='root',passwd='dfb10e69a2',db='ssddatalearn')
sql="SELECT price,comment FROM taob"
data=pda.read_sql(sql,conn)

#1.数据的离差标准化 x1=(x-min)/(max-min)
data2=(data-data.min())/(data.max()-data.min())
#print(data2)

#2.标准差标准化 x1=(x-平均数)/标准差
data3=(data-data.mean())/data.std()
#print(data3)


#3.小数定标规范化 以数据中最大的数为基本参照，看看整体数据除以几个0
k=np.ceil(np.log10(data.abs().max()))
data4=data/10**k
#print(data4)


#离散化 用数字代表数量的大小
#1.等宽离散化
data5=data[u'price'].copy()
data6=data5.T
data7=data6.values
c1=pda.cut(data7,3,labels=['便宜','适中','贵'])
#2.等频离散化
c2=pda.cut(data7,[10,50,100,500],labels=['便宜','适中','贵'])
#print(c1)


#创造新数据
# ch=data['comment']/data['price']
# print(ch)
# data[u'评点比']=ch
# data.to_excel('taob.xls',index=False)

datax=pda.read_excel('taob.xls')
print(datax['price'])

#主成分分析
from sklearn.decomposition import PCA
pca=PCA()#写2就是降到二维
pca.fit(datax)
#降纬
reduceDirection=pca.transform(datax)
print(reduceDirection)
#特征值
C=pca.components_
#贡献率
rate=pca.explained_variance_ratio_
#还原三围
pca.inverse_transform(reduceDirection)

#https://blog.csdn.net/guyuealian/article/details/68483213


