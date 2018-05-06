import pymysql
import pandas as pda
import numpy
import matplotlib.pylab as pl
conn=pymysql.connect(host='127.0.0.1',user='root',passwd='dfb10e69a2',db='ssddatalearn')
sql="SELECT * FROM taob"
data=pda.read_sql(sql,conn)
#find empty value
data['price'][(data['price']==0)]=36
#print(data.describe())
##find wrong data by sandiantu
# dataT=data.T
# priceX=dataT.values[2]
# commentY=dataT.values[3]
# pl.plot(priceX,commentY,'o')
# pl.savefig('four.png')

#筛选数据
# data2=data[(data["price"]<2000) & (data["comment"]<200000) ]
# dataT=data2.T
# priceX=dataT.values[2]
# commentY=dataT.values[3]
# pl.plot(priceX,commentY,'o')
# pl.savefig('four.png')

data['price'][(data['price']>2000)]=36
data3=data[(data["price"]<2000) | (data["comment"]<200000) ]
# print(data3)



#分布分析
#1.求极值
max=data['price'].max()
min=data['price'].min()
step=(max-min)/12
print(step)
sty=pl.arange(min,max,step)
pl.hist(data3['price'],sty)
pl.savefig('five.png')


#数据集合
#data['price']=numpy.concatenate(a,b)
# #更多https://blog.csdn.net/jt1123/article/details/50086595
