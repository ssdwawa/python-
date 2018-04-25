import  urllib.request as re
import requests
url="http://www.printfriendly.com/print/?source=homepage&url=http://news.163.com/18/0425/19/DG8UJRE40001875O.html"
headers = requests.head(url).headers
work=headers['Location']
re.urlretrieve(work,'test.html')
print('done!')

#http://docs.python-requests.org/zh_CN/latest/user/quickstart.html#url