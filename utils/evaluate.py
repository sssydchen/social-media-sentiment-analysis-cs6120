from sklearn.metrics import classification_report, accuracy_score

results = {}


def compute_accuracy(name: str, y_true, y_pred) -> float:
    """
    Compute accuracy and store it in the global `results` dict.

    Parameters
    ----------
    name : str
        The name under which to store this model’s accuracy.
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        The computed accuracy (between 0 and 1).
    """
    acc = accuracy_score(y_true, y_pred)
    results[name] = acc
    return acc


def print_report(name: str, y_true, y_pred):
    """
    Print the classification report and accuracy for a model.

    This calls `compute_accuracy` internally to both print the
    final accuracy and record it in `results`.

    Parameters
    ----------
    name : str
        A label for the model (e.g. "Bi-LSTM").
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    """
    print(f"=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = compute_accuracy(name, y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
