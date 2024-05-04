import numpy as np

def compute_accuracy(
        eval_pred: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}