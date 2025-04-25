import unittest
from models.lstm_model import LSTMModel

class TestLSTMModel(unittest.TestCase):
    """
    Unit tests for the LSTMModel class to ensure correct training and prediction.
    """
    def setUp(self):
        """
        Prepare a small LSTMModel and toy dataset for testing.
        """
        # Use small hyperparameters to speed up tests
        self.model = LSTMModel(max_vocab=50, max_len=10, embed_dim=8)
        # Toy training data: simple two-class problem
        self.X_train = [
            "good movie",
            "bad movie",
            "i love it",
            "i hate it"
        ]
        self.y_train = ["pos", "neg", "pos", "neg"]
        # Test data
        self.X_test = ["good film", "hate this"]
        self.y_true = ["pos", "neg"]

    def test_fit_changes_state(self):
        """
        After fitting, internal components should be populated:
          - model.model is a trained Keras model
          - tokenizer.word_index is populated
          - label_encoder.classes_ matches training labels
        """
        # Before fit, model.model should be None
        self.assertIsNone(self.model.model)
        self.assertEqual(self.model.tokenizer.word_index, {})

        # Fit the model
        self.model.fit(self.X_train, self.y_train)

        # After fit, model.model should be set
        self.assertIsNotNone(self.model.model)
        # Tokenizer vocabulary must be non-empty
        self.assertTrue(len(self.model.tokenizer.word_index) > 0)
        # Label encoder should know the two classes
        classes = set(self.model.label_encoder.classes_)
        self.assertEqual(classes, {"pos", "neg"})

    def test_predict_returns_valid_labels(self):
        """
        Predictions should have same length as X_test and only valid class labels.
        """
        self.model.fit(self.X_train, self.y_train)
        preds = self.model.predict(self.X_test)
        # Length matches test inputs
        self.assertEqual(len(preds), len(self.X_test))
        # Predicted labels subset of trained classes
        self.assertTrue(set(preds).issubset({"pos", "neg"}))

    def test_predict_consistency(self):
        """
        On training data, model should achieve at least chance-level accuracy.
        """
        self.model.fit(self.X_train, self.y_train)
        train_preds = self.model.predict(self.X_train)
        # At least one correct prediction
        correct = sum(p == t for p, t in zip(train_preds, self.y_train))
        self.assertGreaterEqual(correct, 1)


if __name__ == "__main__":
    unittest.main()