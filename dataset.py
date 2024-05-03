from pprint import pprint
from datasets import load_dataset

train_dataset = load_dataset(
    "llm-book/wrime-sentiment", split="train",
    trust_remote_code=True
)

valid_dataset = load_dataset(
    "llm-book/wrime-sentiment", split="validation",
    trust_remote_code=True
)

pprint(train_dataset[0])
pprint(train_dataset.features)