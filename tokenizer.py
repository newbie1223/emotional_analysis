from transformers import AutoTokenizer

model_name = "cl-tohoku/bert-base-japanese-v3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# print(type(tokenizer).__name__)
tokenizer.tokenize("私は今日も元気です。")

encoded_input = tokenizer("私は今日も元気です。")
print(type(encoded_input).__name__)
pprint(encoded_input)

tokenizer.convert_ids_to_tokens(encoded_input["input_ids"])