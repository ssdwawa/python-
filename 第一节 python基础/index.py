# # 写一个乘法口诀表
# for i in range(1,10):
#     for j in range(1,i+1):
#         print('{}x{}={}\t'.format(i,j,i*j),end='')
#     print()
#
# #写一个冒泡排序
# arr=[1,5,7,11,2,3]
# for i in range(len(arr)-1):
#     for j in range(0,len(arr)-1-i):
#         if arr[j+1]>arr[j]:
#             arr[j + 1],arr[j]=arr[j],arr[j + 1]
# print(arr)

# #写一个简单的爬虫
# import  urllib as url
# data = url.request.urlopen("https://www.baidu.com/").read()
# print(data)

#操作文件
## open打开或者创建文件
# fh=open('D:/py work/py learn/python-/第一节 python基础/file.txt','w')
# fh.writelines('hello world')
# fh.close()

# # 文件的读取 一行一行的读
# fh = open('D:/py work/py learn/python-/第一节 python基础/file.txt','r')
# while True:
#     line = fh.readline()
#     if(len(line)==0):
#         break
#     print(line)
# fh.close()

# #异常处理
# try:
#     print('haha')
#     sad()
# except Exception as err:
#     print(err)
