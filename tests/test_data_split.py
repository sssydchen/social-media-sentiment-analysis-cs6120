import unittest
import tempfile
import os
import pandas as pd
from utils.dataset import load_and_split

class TestDataset(unittest.TestCase):
    """
    Unit tests for the load_and_split function in dataset.py
    """

    def setUp(self):
        """
        Create a temporary CSV file with balanced labels for testing.
        """
        # Sample data: 8 rows, 4 of class 0 and 4 of class 1
        self.df = pd.DataFrame({
            'text': [f'text{i}' for i in range(8)],
            'label': [0]*4 + [1]*4
        })
        # Create temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w')
        self.csv_path = tmp.name
        self.df.to_csv(self.csv_path, index=False)
        tmp.close()

    def tearDown(self):
        """
        Remove temporary file after tests.
        """
        os.unlink(self.csv_path)

    def test_split_size_and_stratify(self):
        """
        Verify that train/test sizes and label proportions are correct.
        """
        X_train, X_test, y_train, y_test = load_and_split(
            self.csv_path,
            text_col='text',
            label_col='label',
            test_size=0.25,
            random_state=0
        )
        # Check sizes: 75% of 8 = 6 train, 2 test
        self.assertEqual(len(X_train), 6)
        self.assertEqual(len(X_test), 2)
        # Check stratification: test set should have 1 of each label
        counts = pd.Series(y_test).value_counts().to_dict()
        self.assertEqual(counts.get(0, 0), 1)
        self.assertEqual(counts.get(1, 0), 1)

    def test_invalid_columns(self):
        """
        Passing wrong column names should raise KeyError.
        """
        with self.assertRaises(KeyError):
            load_and_split(self.csv_path, text_col='foo', label_col='label')
        with self.assertRaises(KeyError):
            load_and_split(self.csv_path, text_col='text', label_col='bar')

if __name__ == '__main__':
    unittest.main()
