import  urllib.request as re
keyWord='宋世达'
keyWord=re.quote(keyWord)
url='http://www.baidu.com/s?wd='+keyWord
re.urlretrieve(url,'1.html')


# 发起请求 get
req=re.Request(url)
#读取请求
data=re.urlopen(req).read().decode()

#post请求
import urllib.parse as pa
madata=pa.urlencode({
    'name':'ssd',
    'pass':'1234'
}).encode('utf-8')
...
