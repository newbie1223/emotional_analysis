from collections import Counter
import japanize_matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm
from pprint import pprint
from emotional_analysis.tokenizer import tokenizer
from emotional_analysis.data import train_dataset, valid_dataset

plt.rcParams["font.size"] = 18

def visualize_text_length(dataset: Dataset):
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["sentence"]))
        length_counter[length] += 1
    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()

# visualize_text_length(train_dataset)
# visualize_text_length(valid_dataset)

# for data in valid_dataset:
#   if len(tokenizer.tokenize(data["sentence"])) < 10:
#     pprint(data)

def visualize_labels(dataset: Dataset):
    label_counter = Counter()
    for data in dataset:
        label_id = data["label"]
        label_name = dataset.features["label"].names[label_id]
        label_counter[label_name] += 1
    plt.bar(label_counter.keys(), label_counter.values(), width=1.0)
    plt.xlabel("ラベル")
    plt.ylabel("事例数")
    plt.show()

visualize_labels(train_dataset)
visualize_labels(valid_dataset)