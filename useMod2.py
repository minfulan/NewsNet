import pandas as pd
import joblib
from tokenizer_utils import chinese_tokenizer
import re

# 1. 加载数据
df = pd.read_json('trainedNews.json')

loaded_vectorizer = joblib.load('my_news_vectorizer.joblib')
loaded_model = joblib.load('my_news_model.joblib')

# 2. 用模型为每条新闻预测“有价值”的概率
X = loaded_vectorizer.transform(df['text'])
# predict_proba返回的概率中，第1列（索引[0]）通常是“无价值”，第2列（索引[1]）是“有价值”
# 具体顺序取决于你训练时标签的顺序，这里假设1.0是“有价值”
probability_of_valuable = loaded_model.predict_proba(X)[:, 1]

# 3. 将概率值作为新分数，并添加到数据中
df['model_score'] = probability_of_valuable

# 2. 定义你的评分规则函数
def calculate_manual_score(text):
    score = 0
    text_lower = text.lower()

    # 规则示例：出现以下关键词加分
    valuable_keywords = ['突破', '成功', '发现', '创新', '发射', '成就', '进展', '首飞', '下线', 'AI', '芯片', '航天']
    for word in valuable_keywords:
        if word in text:
            score += 2  # 每个关键词加2分

    # 规则示例：出现以下关键词减分
    non_valuable_keywords = ['娱乐', '八卦', '明星', '恋情', '穿搭', '房价', '星座', '运势', '广告', '促销']
    for word in non_valuable_keywords:
        if word in text:
            score -= 2  # 每个关键词减2分

    # 规则示例：根据文本长度微调（假设长新闻更严肃）
    if len(text) > 20:
        score += 1
    elif len(text) < 10:
        score -= 1

    # 确保分数不为负
    return max(0, score)

# 3. 为所有新闻计算手动评分
df['model_score'] += df['text'].apply(calculate_manual_score)

# 4. 按手动评分从高到低排序，选取前20条
top_20_by_rule = df.sort_values(by='model_score', ascending=False).head(20)

print("=== 基于手动规则评分的前20条新闻 ===")
print(top_20_by_rule[['text', 'model_score', 'label']].to_string(index=False))

# 5. （可选）保存带评分的新数据
df.to_json('trainedNews_with_scores.json', orient='records', force_ascii=False, indent=2)
print(f"\n已保存带评分的文件: trainedNews_with_scores.json")