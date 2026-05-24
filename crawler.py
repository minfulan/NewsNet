import urllib.request
from bs4 import BeautifulSoup
import useMod

news_url = [
    "https://www.sina.com.cn/",
    "https://news.163.com/",
    "https://www.sohu.com/",
    "https://www.qq.com/",
]

tlist = []
hlist = []

# 设置请求头，模拟浏览器访问，避免被部分网站拒绝
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for url in news_url:
    try:
        # 创建请求对象
        req = urllib.request.Request(url, headers=headers)
        # 发送请求并获取响应
        response = urllib.request.urlopen(req)
        # 读取内容并解码为字符串
        html_content = response.read().decode('utf-8', errors='ignore')
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for tag in soup.find_all('a'):
            href = tag.get('href')
            text = tag.text
            text = text.strip()
            if len(text) > 10:
                tlist.append(text)
                hlist.append(href)
                
    except Exception as e:
        print(f"Error fetching {url}: {e}")

preBit=useMod.predict(tlist)
for i in range(len(preBit)):
    if preBit[i]:
        print(tlist[i],hlist[i])