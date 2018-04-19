import urllib.request as req
import  re
pat ="<span>http://.*</span>"
data=req.urlopen('https://edu.csdn.net/huiyiCourse/detail/450').read()
arr=re.compile(pat).findall(str(data))
print(arr)