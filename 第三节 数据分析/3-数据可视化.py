import matplotlib.pylab as pyl
import numpy as np

# x=[1,4,5,6,7]
# y=[5,7,2,1,5]
# # show img-line2D and you get know that the line will merge
# pyl.plot(x,y)#pyl.plot(x,y,'o')mean non-line #pyl.plot(x,y,'--c') change color and dansh line and -. and .
# pyl.title('xxx')
# pyl.xlabel('aaa')
# pyl.ylabel('ccc')
# pyl.xlim(0,8)
# pyl.xlim(2,8)
# pyl.savefig("one.png")
# # if you want to draw couple image you need pyl.plot(x1,y2) again


# #hist count data
# data=np.random.normal(5,1.0,1000)
# pyl.hist(data)
# #sty=pyl.arange(start,end,step)
# #pyl.hist(data,sty) #to limt data range
# pyl.savefig("two.png")


#mul-image
pyl.subplot(2,2,2)#2 line 2 line cloum and choice second image

#2line 1cloum
pyl.subplot(2,2,1)
#print in 221
x1=[1,4,5,6,7]
y2=[5,7,2,1,5]
pyl.plot(x1,y2)
pyl.subplot(2,2,2)
#print in 222
x1=[1,4,5,6,7]
y2=[5,7,2,1,5]
pyl.plot(x1,y2)
pyl.subplot(2,2,2)
pyl.subplot(2,1,2)
#print in 212
x1=[1,4,5,6,7]
y2=[5,7,2,1,5]
pyl.plot(x1,y2)
pyl.subplot(2,2,2)
pyl.savefig("three.png")