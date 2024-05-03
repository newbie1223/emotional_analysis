from transformers import AutoTokenizer

model_name = "cl-tohoku/bert-base-japanese-v3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(type(tokenizer).__name__)
tokenizer.tokenize("私は今日も元気です。")