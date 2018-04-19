import re
#看字符串中有没有想要的字符串
test1 = 'hahahadahahaha'
re.search('da',test1)

#\w 正常字符 \d 数字  \s 空白  大写就相反
test2 = "dasdasdhahahadaddad"
print(re.search('\whahaha\w',test2))
#[abc]该字符位可以是abc
#.是除了换行符外的任何字符 |是可有可无 贪婪是搜到底 懒惰是只找一次
test3 = "ggggg123ahchadaddabcgggg"
print(re.search('123(.*)abc',test3))
#全局搜索 re.compile(条件).findall(字符串)