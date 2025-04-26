import os
import pandas as pd

def save_predictions(model_name, X, y_true, y_pred):
    """
    Save a CSV with columns: text, true_label, predicted_label
    into the sibling 'predictions/' folder so downstream scripts can consume it.
    """
    # build the output folder path
    here = os.path.dirname(__file__) 
    out_dir = os.path.join(here, os.pardir, 'predictions')
    out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    filename = f"{model_name.lower().replace(' ', '_')}_predictions.csv"
    fullpath = os.path.join(out_dir, filename)

    # save the DataFrame
    df = pd.DataFrame({
        'text': X,
        'true_label': y_true,
        'predicted_label': y_pred
    })
    df.to_csv(fullpath, index=False)
    print(f"Saved predictions to {fullpath}")
