from transformers import AutoModelForSequenceClassification
from emotional_analysis.data import endcoded_train_dataset, data_collator, train_dataset
from emotional_analysis import model_name

class_label = train_dataset.features['label']
label2id = {label: id for id, label in enumerate(class_label.names)}
id2label = {id: label for id, label in enumerate(class_label.names)}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=class_label.num_classes,
    label2id=label2id,
    id2label=id2label
)

print(type(model).__name__)
print(model.forward(**data_collator(endcoded_train_dataset[0:4])))