import pandas as pda
data1=pda.read_csv("hexun.csv")#read_excel() #read_html()
#print(data1.describe())
#以某一列排序 不改变原来的数据
data2=data1.sort_values(by='21')
print(data2)

