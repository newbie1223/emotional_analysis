from transformers import DataCollatorWithPadding
from pprint import pprint

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch_inputs = data_collator(encoded_train_dataset[0:4])
pprint({name: tensor.size() for name, tensor in batch_inputs.items()})