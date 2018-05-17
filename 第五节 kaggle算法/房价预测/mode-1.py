import numpy as np
import pandas as pda
import matplotlib.pylab as pyl


train_df=pda.read_csv('./data/train.csv')
test_df=pda.read_csv('./data/test.csv')

#采用np.log1p的方法把数据标准化，之后通过pandas的hist()方法把直方图存到内存中 通过pyl.show()来展示
# prices = pda.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
# prices.hist()
# pyl.show()

#把SalePrice取出并标准化，定义为y_train,原数组改变
y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pda.concat((train_df, test_df), axis=0)
