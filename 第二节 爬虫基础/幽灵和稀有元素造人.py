from selenium import webdriver
import re
driver = webdriver.PhantomJS(executable_path="C:/Users/Administrator/Desktop/phantomjs-2.1.1-windows/bin/phantomjs.exe")
driver.set_page_load_timeout(5)
driver.get('http://zj.news.163.com/')



try:
    driver.get('http://zj.news.163.com/')
    data = driver.page_source
    pat1='<div class="news_title">.*?href="(http://zj.news.163.com/.*?\.html)'
    aList=re.compile(pat1).findall(data)
    print(aList)
except Exception as e:
    print(e)
driver.quit()