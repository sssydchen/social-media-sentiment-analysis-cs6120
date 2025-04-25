# File: sentiment_analysis/test_save_predictions.py

"""
Unit tests for save_predictions(model_name, X, y_true, y_pred).

These tests verify:
  1. The CSV file is created with the correct filename.
  2. The CSV contains exactly the columns 'text', 'true_label', 'predicted_label'
     and the values match the inputs.
  3. A confirmation message is printed to stdout.
"""

import unittest
import os
import tempfile
import pandas as pd

# Import the function under test. Adjust module path if necessary.
from save_predictions import save_predictions

class TestSavePredictions(unittest.TestCase):
    """
    Test suite for save_predictions function.
    """

    def setUp(self):
        """
        Create a temporary directory and switch into it, so that
        any files created by save_predictions are isolated.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        """
        Return to the original working directory and clean up the temp folder.
        """
        os.chdir(self.orig_cwd)
        self.temp_dir.cleanup()

    def test_file_creation_and_content(self):
        """
        What it tests:
            - save_predictions creates a CSV named "{model_name}_predictions.csv"
            - The file has exactly the columns 'text', 'true_label', 'predicted_label'
            - The rows match the supplied lists X, y_true, y_pred

        Input:
            model_name = "Test Model"
            X = ["tweet1", "tweet2"]
            y_true = [0, 1]
            y_pred = [1, 0]

        Expected output:
            - A file named "test_model_predictions.csv"
            - CSV columns and values match the inputs.
        """
        model_name = "Test Model"
        X = ["tweet1", "tweet2"]
        y_true = [0, 1]
        y_pred = [1, 0]

        save_predictions(model_name, X, y_true, y_pred)

        expected_filename = "test_model_predictions.csv"
        # Check file existence
        self.assertTrue(os.path.isfile(expected_filename),
                        f"Expected {expected_filename} to be created")

        # Read and verify content
        df = pd.read_csv(expected_filename)
        self.assertListEqual(df["text"].tolist(), X)
        self.assertListEqual(df["true_label"].tolist(), y_true)
        self.assertListEqual(df["predicted_label"].tolist(), y_pred)

    def test_prints_confirmation_message(self):
        """
        What it tests:
            save_predictions prints "Saved predictions to <filename>"

        Input:
            model_name = "Another Model"
            X, y_true, y_pred = empty lists

        Expected output:
            stdout contains "Saved predictions to another_model_predictions.csv"
        """
        model_name = "Another Model"
        X, y_true, y_pred = [], [], []

        # Capture stdout output
        from io import StringIO
        import sys
        buf = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buf
        try:
            save_predictions(model_name, X, y_true, y_pred)
        finally:
            sys.stdout = sys_stdout

        output = buf.getvalue().strip()
        expected_text = "Saved predictions to another_model_predictions.csv"
        self.assertIn(expected_text, output,
                      f"Expected printout to contain '{expected_text}'")


if __name__ == "__main__":
    unittest.main()
