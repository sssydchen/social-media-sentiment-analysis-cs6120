# test_bert_model.py

import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# 1) Mock the BertTokenizer.from_pretrained method
tokenizer_mock = MagicMock()
tokenizer_mock.return_value = {
    'input_ids': np.array([[1], [2]]),        # batch_size=2, seq_len=1
    'attention_mask': np.array([[1], [1]])    # attention masks corresponding to the inputs
}
patch('models.bert_model.BertTokenizer.from_pretrained',
      return_value=tokenizer_mock).start()

# 2) Mock the TFBertForSequenceClassification.from_pretrained method
dummy_model = MagicMock()
dummy_model.compile = MagicMock()
dummy_model.fit     = MagicMock()
dummy_output = MagicMock(logits=np.array([[0.2, 0.8], [0.7, 0.3]]))
dummy_model.predict = MagicMock(return_value=dummy_output)
patch('models.bert_model.TFBertForSequenceClassification.from_pretrained',
      return_value=dummy_model).start()

# Now import under test
from models.bert_model import BERTModel

class TestBERTModel(unittest.TestCase):
    def test_init(self):
        """
        Test the BERTModel constructor:
          - model is None before training
          - model_name, max_len, learning_rate are set
          - tokenizer attribute exists
        """
        model = BERTModel(model_name='test-model', max_len=64, learning_rate=1e-3)
        self.assertIsNone(model.model)
        self.assertEqual(model.model_name, 'test-model')
        self.assertEqual(model.max_len, 64)
        self.assertEqual(model.learning_rate, 1e-3)
        self.assertTrue(hasattr(model, 'tokenizer'))

    def test_fit_and_predict(self):
        """
        Test fit() and predict() methods using dummy data:
          - fit() calls compile() and fit() on the underlying model
          - predict() returns the correct number of labels
        """
        bmodel = BERTModel()
        X = ['hello', 'world']
        y = ['pos', 'neg']

        # run fit
        bmodel.fit(X, y, epochs=1, batch_size=1)
        self.assertTrue(dummy_model.compile.called, "compile() was not called")
        self.assertTrue(dummy_model.fit.called, "fit() was not called")

        # run predict
        preds = bmodel.predict(X, batch_size=1)
        self.assertIsInstance(preds, (list, np.ndarray))
        self.assertEqual(len(preds), 2, "predict() returned wrong number of labels")

if __name__ == '__main__':
    unittest.main()
