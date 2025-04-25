import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
    csv_path: str,
    text_col: str = 'text',
    label_col: str = 'airline_sentiment',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load CSV, extract text and labels, and split into train/test.
    Returns X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(csv_path)
    X = df[text_col].astype(str).values
    y = df[label_col].values
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )