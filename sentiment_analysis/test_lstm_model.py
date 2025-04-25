import pytest
from models.lstm_model import LSTMModel

@pytest.fixture
def toy_data():
    X = ["one two three", "four five six", "seven eight"]
    y = ["A", "B", "A"]
    return X, y

def test_fit_and_predict(toy_data):
    X, y = toy_data
    lstm = LSTMModel(max_vocab=20, max_len=5, embed_dim=8)
    lstm.fit(X, y)
    preds = lstm.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({"A", "B"})

