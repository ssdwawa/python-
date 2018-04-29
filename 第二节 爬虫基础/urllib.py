import  urllib.request as re
# keyWord='宋世达'
# keyWord=re.quote(keyWord)
# url='http://www.baidu.com/s?wd='+keyWord
# re.urlretrieve(url,'1.html')


# # 发起请求 get
# req=re.Request(url)
# #读取请求
# data=re.urlopen(req).read().decode()
#
# #post请求
# import urllib.parse as pa
# mydata=pa.urlencode({
#     'name':'ssd',
#     'pass':'1234'
# }).encode('utf-8')

#异常处理
# import urllib.error
# try:
#     re.urlopen('...')
# except urllib.error.URLError as e:
#     if hasattr(e,'code'):
#         print(e.code)
#     if hasattr(e,'reason'):
#         print(e.reason)

#带请求头的爬
url="https://blog.csdn.net/ssdwawa/article/details/79269062"
header={
'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"
}
req=re.Request(url,headers=header)
data=re.urlopen(req).read().decode()
print(data)