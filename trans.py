import pandas as pd

def j2c():
    # 读取 JSON 文件
    df = pd.read_json('trainedNews.json')
    # 保存为 CSV
    df.to_csv('新闻数据.csv', index=False, encoding='utf-8-sig')

def c2j():
    # 读取 CSV 文件
    df = pd.read_csv('新闻数据.csv')
    # 保存为 JSON
    df.to_json('trainedNews.json', orient='records', force_ascii=False, indent=2)

if __name__ == '__main__':
    x=input("请选择转换方向：1.JSON to CSV 2.CSV to JSON")
    if x=='1':
        j2c()
    elif x=='2':
        c2j()
    else:
        print("输入错误")