"""
test_bert_model.py

Unit tests for the BERTModel class in bert_model.py.

Tests:
    - test_init: verifies correct attribute initialization.
    - test_fit_and_predict: verifies fit() and predict() behaviors with dummy data.
"""

import sys
import numpy as np
from unittest.mock import patch, MagicMock

# 1) Mock the BertTokenizer.from_pretrained method to return a callable tokenizer mock
tokenizer_mock = MagicMock()
# Configure the tokenizer mock to return dummy input_ids and attention_mask arrays when called
tokenizer_mock.return_value = {
    'input_ids': np.array([[1], [2]]),        # batch_size=2, seq_len=1
    'attention_mask': np.array([[1], [1]])    # attention masks corresponding to the inputs
}
patch('bert_model.BertTokenizer.from_pretrained',
      return_value=tokenizer_mock).start()

# 2) Mock the TFBertForSequenceClassification.from_pretrained method to return a dummy model
dummy_model = MagicMock()
# Provide MagicMock implementations for compile, fit, and predict methods
dummy_model.compile = MagicMock()
dummy_model.fit     = MagicMock()
# Create a dummy output object with a logits attribute for model.predict
dummy_output = MagicMock(logits=np.array([[0.2, 0.8], [0.7, 0.3]]))
dummy_model.predict = MagicMock(return_value=dummy_output)

patch('bert_model.TFBertForSequenceClassification.from_pretrained',
      return_value=dummy_model).start()

# Now safely import the BERTModel class without triggering actual downloads or TensorFlow code
from bert_model import BERTModel  

def test_init():
    """
    Test the BERTModel constructor.

    Verifies:
        - model attribute is None before training.
        - model_name, max_len, and learning_rate match provided values.
        - tokenizer attribute is present.

    Input: None
    Output: AssertionError if any check fails.
    """
    model = BERTModel(model_name='test-model', max_len=64, learning_rate=1e-3)
    # Model should not have been loaded yet
    assert model.model is None
    # Check that the provided parameters are stored properly
    assert model.model_name == 'test-model'
    assert model.max_len == 64
    assert model.learning_rate == 1e-3
    # Ensure a tokenizer attribute was created
    assert hasattr(model, 'tokenizer')

def test_fit_and_predict():
    """
    Test the fit and predict methods of BERTModel using dummy data.

    - fit(): should call compile() and fit() on the dummy model.
    - predict(): should return a sequence of predicted labels matching the input length.

    Input:
        X (list[str]): sample texts ['hello', 'world']
        y (list[str]): sample labels ['pos', 'neg']
    Output:
        preds (list or np.ndarray): predicted labels of length 2
        AssertionError if behavior deviates.
    """
    bmodel = BERTModel()
    X = ['hello', 'world']
    y = ['pos', 'neg']
    # Call fit, which should invoke compile() and fit() on the dummy model
    bmodel.fit(X, y, epochs=1, batch_size=1)
    assert dummy_model.compile.called
    assert dummy_model.fit.called

    # Call predict, which should tokenize inputs and call the dummy model's predict
    preds = bmodel.predict(X, batch_size=1)
    # The return value should be a list or numpy array of length 2
    assert isinstance(preds, (list, np.ndarray))
    assert len(preds) == 2

if __name__ == '__main__':
    test_init()
    test_fit_and_predict()
    print("All tests passed.")
