import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import jieba
import joblib
import json
from tokenizer_utils import chinese_tokenizer

#训练数据读取与统计
filePath= r"C:\Users\minfulan\Desktop\software\自制\NewsNet\trainedNews.json"
with open(filePath, 'r', encoding='utf-8') as f:
    data = json.load(f)


df = pd.DataFrame(data)

# 【关键修改】1. 删除 label 为空的行
initial_count = len(df)
df = df.dropna(subset=['label'])
dropped_count = initial_count - len(df)
if dropped_count > 0:
    print(f"已删除 {dropped_count} 条标签为空的数据")

# 【关键修改】2. 确保 label 是整数类型 (0 或 1)，防止浮点数问题
df['label'] = df['label'].astype(int)

print(f"总样本数: {len(df)}")
print(f"有价值(1)数量: {sum(df['label']==1)}")
print(f"无价值(0)数量: {sum(df['label']==0)}")
if len(df) > 0:
    print(f"正负样本比例: {sum(df['label']==1)/len(df):.2%}")
else:
    print("没有有效数据可训练")
    exit()


def clean_text(text):
    import re
    # 保留：中文汉字(\u4e00-\u9fa5)、英文字母(a-zA-Z)、数字(0-9)、空格(\s)
    # [^\u4e00-\u9fa5a-zA-Z0-9\s] 表示匹配所有“非”上述字符的内容（即标点符号等）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text.strip()

df['text_clean'] = df['text'].apply(clean_text)

# 向量化器
word_vectorizer = TfidfVectorizer(
    tokenizer=chinese_tokenizer,  # 使用中文分词
    max_features=1000,             # 取最重要的1000个词
    stop_words=['的', '了', '是', '在', '和', '与', '及','吧']  # 中文停用词
)

# 2. 提取特征
X = word_vectorizer.fit_transform(df['text_clean'])
y = df['label']

print(f"特征维度: {X.shape[1]} 个字符特征")                        

# 标签

# 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,stratify=y
)

# 4. 尝试朴素贝叶斯（适合小样本）
model = MultinomialNB(alpha=0.5)  # alpha越小，对特征越敏感
model.fit(X_train, y_train)

if __name__ == '__main__':
    # 5. 评估
    y_pred = model.predict(X_test)
    print("=== 朴素贝叶斯（字符级）===")
    print(classification_report(y_test, y_pred, 
                            target_names=['无价值', '有价值']))
    print(f"准确率: {model.score(X_test, y_test):.2%}")
    
    # 1. 保存TF-IDF向量化器
    joblib.dump(word_vectorizer, 'news_vectorizer.joblib')

    # 2. 保存分类模型
    joblib.dump(model, 'news_model.joblib')

    print("模型保存成功！生成两个文件：")
    print("1. my_news_vectorizer.joblib")
    print("2. my_news_model.joblib")