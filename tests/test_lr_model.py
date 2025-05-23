# This file contains unit tests for the LRModel class in the sentiment analysis module.

import unittest
from models.lr_model import LRModel


# ===============================
# Class: TestLRModel
# What it does:
#   - Unit tests for the LRModel class.
#   - Verifies correct behavior for fit and predict methods.
# ===============================
class TestLRModel(unittest.TestCase):
    # ===============================
    # Function: setUp
    # What it does:
    #   - Prepare a fresh LRModel and sample datasets for testing.
    #
    # Inputs:
    #   - None
    #
    # Output:
    #   - Initializes model and sample data for tests
    # ===============================
    def setUp(self):
        self.model = LRModel()
        self.X_train = ["I love dogs", "I hate cats", "Dogs are the best"]
        self.y_train = [1, 0, 1]
        self.X_test = ["I love animals", "Cats are evil"]

    # ===============================
    # Function: test_fit_predict
    # What it does:
    #   - Test model training and prediction functionality.
    #
    # Inputs:
    #   - X_train, y_train: training data
    #   - X_test: test data for prediction
    #
    # Output:
    #   - Asserts predictions length and label validity
    # ===============================
    def test_fit_predict(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))

    # ===============================
    # Function: test_fit_changes_state
    # What it does:
    #   - Verify that fitting updates vectorizer state.
    #
    # Inputs:
    #   - X_train, y_train: training data
    #
    # Output:
    #   - Asserts vocabulary_ exists after fitting
    # ===============================
    def test_fit_changes_state(self):
        # Before fitting, vocabulary_ attribute should not exist
        self.assertFalse(hasattr(self.model.vectorizer, "vocabulary_"))

        self.model.fit(self.X_train, self.y_train)

        # After fitting, vocabulary_ attribute should exist and be populated
        self.assertTrue(hasattr(self.model.vectorizer, "vocabulary_"))
        self.assertIsInstance(self.model.vectorizer.vocabulary_, dict)
        self.assertGreater(len(self.model.vectorizer.vocabulary_), 0)


if __name__ == "__main__":
    unittest.main()
