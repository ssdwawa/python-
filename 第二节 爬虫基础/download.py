import  requests

url='https://www.printfriendly.com/pdfs/1524753938_812b10/download'
r = requests.get(url)
with open("demo3.pdf", "wb") as code:
     code.write(r.content)