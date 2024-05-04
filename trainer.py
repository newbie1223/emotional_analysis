from transformers import Trainer
from pprint import pprint

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