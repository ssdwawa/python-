import pandas as pad
import numpy as np
import matplotlib.pylab as pl


data=pad.read_csv('hexun.csv')
print(data.shape)
# print(data.values[2][2])
data2=data.T
book_id=data2.values[0]
read_count=data2.values[3]
talk_count=data2.values[4]
pl.plot(book_id,read_count)
pl.savefig('four.png')
#print(read_count[0])
