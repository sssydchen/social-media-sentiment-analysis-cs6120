from sklearn.metrics import classification_report, accuracy_score

results = {}

def evaluate(name: str, y_true, y_pred):
    """
    Print classification report and store accuracy in results.
    """
    print(f"=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    results[name] = acc