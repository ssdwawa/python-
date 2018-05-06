from numpy import *
import os
import operator

def loadData(fileName):
    arr=[]
    fh=open('traindata/'+fileName)
    for i in range(0,32):
        line=fh.readline()
        for j in range(0,32):
            arr.append(int(line[j]))
    return  arr

def readFileName(fileName):
    num=fileName.split('_')
    return num[0]

def loadTestData():
    arr=[]
    labels=[]
    fileList=os.listdir('traindata')#拿到全部的文件
    num=len(fileList)
    for i in range(0,num):
        arrItem=loadData(fileList[i])
        name=readFileName(fileList[i])
        arr.append(arrItem)
        labels.append(name)
    return labels,arr




# from sklearn import neighbors
#
#
# labels,arr=loadTestData()
# knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
# knn.fit(arr, labels)  # 导入数据进行训练
# print(knn.predict([loadData('6_6.txt')]))


