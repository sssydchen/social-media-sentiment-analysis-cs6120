import pandas as pd

def save_predictions(model_name, X, y_true, y_pred):
    """
    Save a CSV with columns: text, true_label, predicted_label
    so that downstream scripts (e.g. confusion matrix) can consume it.
    """
    df = pd.DataFrame({
        'text': X,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")
