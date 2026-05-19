import joblib
from tokenizer_utils import chinese_tokenizer
import jieba
import sys

# 【 hack 】如果模型是在 __main__ 中训练的，pickle 会在 __main__ 中查找该函数。
# 当 crawler.py 运行时，__main__ 就是 crawler.py 模块。
# 我们将当前模块 (useMod) 中的 chinese_tokenizer 复制到 __main__ 模块中，以防万一。
if '__main__' in sys.modules:
    main_module = sys.modules['__main__']
    # 只有当 __main__ 中没有这个函数时才添加，避免覆盖
    if not hasattr(main_module, 'chinese_tokenizer'):
        main_module.chinese_tokenizer = chinese_tokenizer

# 1. 加载保存的组件
try:
    loaded_vectorizer = joblib.load('my_news_vectorizer.joblib')
    loaded_model = joblib.load('my_news_model.joblib')
except AttributeError as e:
    print(f"加载失败: {e}")
    print("提示：请确保 chinese_tokenizer 的定义与训练时完全一致。")
    raise

if __name__ == '__main__':
    # 测试代码...
    new_news = ['测试新闻']
    X_new = loaded_vectorizer.transform(new_news)
    prediction = loaded_model.predict(X_new)
    print(prediction)

def predict(new_news:list):
    """
    对新新闻进行预测
    :param new_news: 新闻列表
    :return: 预测结果列表(1/0)
    """
    X_new = loaded_vectorizer.transform(new_news)
    prediction = loaded_model.predict(X_new)
    return prediction