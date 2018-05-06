import numpy as np
import operator
import os

class Bayes:
    def __init__(self):
        self.length=-1
        self.labelCount=dict()
        self.vectorGroup=dict()
    def fit(self,testData,testLabels):
        self.length=len(testData[0])
        sumLabesls=len(testLabels)
        noRepLabesls=set(testLabels)
        for item in noRepLabesls: #计算出各个标签占总标签的比例
            self.labelCount[item]=testLabels.count(item)/sumLabesls
        for vector,label in zip(testData,testLabels):#将测试的数据和标签对应到字典中
            if (label not in self.vectorGroup):
                self.vectorGroup[label]=[] #{'3': 0.10289555325749741, '7': 0.10392967942088935, '0': 0.09772492244053775, '2': 0.10082730093071354, '5': 0.09669079627714582, '1': 0.10237849017580145, '9': 0.10548086866597725, '4': 0.09617373319544985, '8': 0.09307135470527404, '6': 0.10082730093071354}
            self.vectorGroup[label].append(vector) #{0:[[.][.][.][.]100多个]}
        return  self
    def test(self,inputData,testLabels):
        #计算inputData为各个labels的概率
        pdDict=dict()
        for label in testLabels:
            p=1
            thisLabel = self.labelCount[label]#获取当前类别占总类别的比率
            thisVector = self.vectorGroup[label]#获取当前的全部向量
            num=len(thisVector)
            thisVector=np.array(thisVector).T
            for index in range(0,len(inputData)):#thisVector=[[],[][][],[],[],[]]
                vertor=list(thisVector[index])#vertor=[0,0,0,0,0,0,0]
                p*=vertor.count(inputData[index])/num #统计一下在比对向量中出现的概率 #inputData=[0,0,0,0,0,0,0]
            pdDict[label]=p*thisLabel
        result = sorted(pdDict.items(), key=operator.itemgetter(1), reverse=True)
        return result[0]



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


labels,arr=loadTestData()
bys=Bayes()
bys.fit(arr,labels)
labelsall=['0','1','2','3','4','5','6','7','8','9']
thisdata=loadData('6_6.txt')
rst=bys.test(thisdata,labelsall)
print(rst)