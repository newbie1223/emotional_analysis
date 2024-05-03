from transformers import pipeline

text_classification_pipeline = pipeline(
    model = "llm-book/bert-base-japanese-v3-marc_ja"
)

positive_text = "私は今日も元気です。"
print(text_classification_pipeline(positive_text)[0])

negative_text = "私は今日も悲しいです。"
print(text_classification_pipeline(negative_text)[0])