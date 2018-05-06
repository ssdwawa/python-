import numpy
x=numpy.array(["a","2","4","9"])
y=numpy.array([[2,3,3,5],[7,8,9,11],[111,4,2,1]])
print(x)
#排序
y.sort()
print(y)
#取最大值
y1=y.max()
print(y1)
#切片
x1=x[0:3]#[:]不写就是从头到尾
print(x1)

#slice 2darry
y1=y[0:2]#[:]不写就是从头到尾
print(y1)
y2=y[2]
print(y2)

import pandas as pa
#列表式
a=pa.Series([4,2,1,3,1],index=['a','b','c','d','e'])
print(a)
#数据框
b=pa.DataFrame([[2,3,3,5],[7,8,9,11],[111,4,2,1]]) #columns= 参数指定x轴
print(b)
#按列分布
c=pa.DataFrame({
    'a':4,
    'b':[3,2,1],
})
print(c)
#取前几行
print(b.head(2))#b.tail后几行
#统计 按列统计
print(b.describe())
#转制 b.T


#random
data=numpy.random.random_integers(1,20,3)
print(data)

#normal
data2=numpy.random.normal(5,1.0,10)
print(data2)

