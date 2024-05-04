from transformers import BatchEncoding
from pprint import pprint

def preprocess_text_classification(
        examples: dict[str, str | int]
) -> BatchEncoding:
    
    encoded_example = tokenizer(examples["sentence"], max_length=512)
    encoded_example["labels"] = examples["label"]
    return encoded_example

encoded_train_dataset = train_dataset.map(
    preprocess_text_classification,
    remove_columns=train_dataset.column_names,
)

encoded_valid_dataset = valid_dataset.map(
    preprocess_text_classification,
    remove_columns=valid_dataset.column_names,
)

pprint(encoded_train_dataset[0])