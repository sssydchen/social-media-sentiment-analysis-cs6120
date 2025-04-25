# This file contains unit tests for the NBModel class in the sentiment analysis module.

import unittest
from nb_model import NBModel

"""
    Unit tests for the NBModel class.

    This class tests the fitting and prediction behavior of the NBModel,
    ensuring the internal state changes correctly and that predictions are returned properly.
"""


class TestNBModel(unittest.TestCase):
    def setUp(self):
        """
        Setup a fresh NBModel instance and some sample data before each test case.

        Inputs:
            None
        Outputs:
            self.model (NBModel): A new NBModel instance ready for testing.
            self.X_train (list of str): Sample training text data.
            self.y_train (list of int): Sample training labels.
            self.X_test (list of str): Sample test text data.
        """
        self.model = NBModel()
        self.X_train = ["I love dogs", "I hate cats", "Dogs are the best"]
        self.y_train = [1, 0, 1]
        self.X_test = ["I love animals", "Cats are evil"]

    def test_fit_predict(self):
        """
        Test that the model can fit training data and correctly predict labels for new data.

        Inputs:
            self.X_train (list of str): Training text samples.
            self.y_train (list of int): Training labels.
            self.X_test (list of str): New text samples for prediction.

        Outputs:
            Asserts that:
                - The number of predictions matches the number of test samples.
                - The predicted labels are within the expected label set {0, 1}.
        """
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_fit_changes_state(self):
        """
        Test that fitting the model properly updates the internal state of the vectorizer.

        Inputs:
            self.X_train (list of str): Training text samples.
            self.y_train (list of int): Training labels.

        Outputs:
            Asserts that:
                - Before fitting, the vectorizer has no vocabulary.
                - After fitting, the vectorizer has a populated vocabulary dictionary.
        """
        self.assertFalse(hasattr(self.model.vectorizer, "vocabulary_"))

        self.model.fit(self.X_train, self.y_train)

        self.assertTrue(hasattr(self.model.vectorizer, "vocabulary_"))
        self.assertIsInstance(self.model.vectorizer.vocabulary_, dict)
        self.assertGreater(len(self.model.vectorizer.vocabulary_), 0)


if __name__ == "__main__":
    unittest.main()
