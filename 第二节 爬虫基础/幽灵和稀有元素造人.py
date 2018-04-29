from selenium import webdriver
import os
import re
# driver = webdriver.PhantomJS(executable_path="C:/Users/Administrator/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe")
# driver.set_page_load_timeout(5)
# driver.get('http://zj.news.163.com/')
#
#
#
# try:
#     driver.get('http://zj.news.163.com/')
#     data = driver.page_source
#     pat1='<div class="news_title">.*?href="(http://zj.news.163.com/.*?\.html)'
#     aList=re.compile(pat1).findall(data)
#     print(aList)
# except Exception as e:
#     print(e)
# driver.quit()

#driver = webdriver.PhantomJS(executable_path="C:/Users/Administrator/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe")







# cookie={
# 'domain':'.weibo.com'
# }
# rush="SINAGLOBAL=6001712172292.173.1500302710458; UOR=,,www.baidu.com; login_sid_t=f72b9d66eaf71a55a2eb81c984eec583; cross_origin_proto=SSL; YF-Ugrow-G0=b02489d329584fca03ad6347fc915997; YF-V5-G0=46bd339a785d24c3e8d7cfb275d14258; _s_tentry=www.baidu.com; Apache=9662239190958.26.1524885790952; ULV=1524885790958:2:1:1:9662239190958.26.1524885790952:1500302710465; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5yAxxrNCOE-dLQor8_ZhxW5JpX5K2hUgL.FozfeoB0ShqEShB2dJLoIX2LxKBLBonL1h5LxK.LBKzL1KMLxK-LB-zL1h2LxKBLB.2LB.2LxKML1-2L1hBLxKML1h-L12-LxKnL1-zL12ikP7tt; ALF=1556421819; SSOLoginState=1524885819; SUHB=03oEYBdt6vfyFP; un=13488476320; wvr=6; YF-Page-G0=46f5b98560a83dd9bfdd28c040a3673e"
# rushList=rush.split(';')
# for i in rushList:
#     data=i.split('=')
#     cookie[data[0]]=data[-1]
# driver = webdriver.PhantomJS(executable_path="C:/Users/Administrator/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe")
# driver.set_page_load_timeout(5)
# driver.get('https://weibo.com')
# driver.delete_all_cookies()
# print(cookie)
# driver.add_cookie(cookie)
# driver.get('https://weibo.com')
# print(driver.page_source)

# driverOptions = webdriver.ChromeOptions()
# driverOptions.add_argument(r"user-data-dir=C:\Users\Administrator\AppData\Local\Google\Chrome\User Data")
# driver = webdriver.Chrome("chromedriver",0,driverOptions)
# driver.set_page_load_timeout(5)
# driver.get('https://weibo.com')



profile_ff="C:/Users/Administrator/AppData/Roaming/Mozilla/Firefox/Profiles/mw04aw5i.default"
fp = webdriver.FirefoxProfile(profile_ff)
driver = webdriver.Firefox(fp)
driver.set_page_load_timeout(5)
driver.get('https://weibo.com')

