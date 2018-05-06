from numpy import *
import operator
#求距离，根据距离接近的k数求预期结果
def knn(k,testData,trainData,labels):
    #取得行数
    dataSize=trainData.shape[0]
    #把测试数据扩展到训练数据的行数
    dif=tile(testData,(dataSize,1))-trainData
    sqdif=dif**2
    sumsqdif=sqdif.sum(axis=1)#对每一行的的数据求和
    distance=sumsqdif**0.5
    sortDistance=distance.argsort()
    count={}
    for i in range(0,k):                #labels[a,b,c]
        #使用argsort方法将A数组不改变数组位置的排序 [1,3,2]=>[0,2,1] 然后用新的数组去对应label,得到的就是排序后的label
        vote=labels[sortDistance[i]]
        #用get方法进行统计
        count[vote]=count.get(vote,0)+1
    print(count)
    #对字典进行排序
    result=sorted(count.items(), key=operator.itemgetter(1),reverse=True)
    return  result[0]
data=array([[164,55],[180,80],[166,81]])
de=knn(3,[176,75],data,['女','男','女'])
print(de)



# import numpy as np
# from sklearn import neighbors
#
# knn = neighbors.KNeighborsClassifier()  # 取得knn分类器
# data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])  # data对应着打斗次数和接吻次数
# labels = np.array([1, 1, 1, 2, 2, 2])  # labels则是对应Romance和Action
# knn.fit(data, labels)  # 导入数据进行训练
# print(knn.predict([18, 90]))