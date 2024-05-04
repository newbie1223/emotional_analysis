from transformers import Trainer
from pprint import pprint
from emotional_analysis.training_arguments import training_args
from emotional_analysis.model import model
from emotional_analysis.data import encoded_train_dataset, encoded_valid_dataset, data_collator, compute_accuracy

trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_accuracy,
)
trainer.train()

eval_metrics = trainer.evaluate(encoded_valid_dataset)
pprint(eval_metrics)