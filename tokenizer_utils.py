import jieba
def chinese_tokenizer(text):
    return [w for w in jieba.cut(text) if w.strip()]