import unittest
import os
import tempfile
from utils.confusion_matrix import plot_confusion_matrix


# ===============================
# Class: TestPlotConfusionMatrix
# What it does:
#   - Unit tests for the plot_confusion_matrix function.
#   - Verifies that the function runs without error and optionally saves the file correctly.
# ===============================


class TestPlotConfusionMatrix(unittest.TestCase):
    # ===============================
    # Function: setUp
    # What it does:
    #   - Prepare sample true and predicted labels for testing.
    #
    # Inputs:
    #   - None
    #
    # Output:
    #   - Initializes sample label data
    # ===============================
    def setUp(self):
        self.true_labels = [
            "positive",
            "neutral",
            "negative",
            "positive",
            "neutral",
        ]
        self.pred_labels = [
            "positive",
            "negative",
            "negative",
            "positive",
            "neutral",
        ]
        self.model_name = "TestModel"
        self.accuracy = 0.8

    # ===============================
    # Function: test_plot_without_saving
    # What it does:
    #   - Test that plot_confusion_matrix runs without saving a file.
    #
    # Inputs:
    #   - true_labels, pred_labels, model_name, accuracy
    #
    # Output:
    #   - Asserts that no exceptions are raised during plotting
    # ===============================
    def test_plot_without_saving(self):
        try:
            plot_confusion_matrix(
                self.true_labels,
                self.pred_labels,
                self.model_name,
                self.accuracy,
            )
        except Exception as e:
            self.fail(f"plot_confusion_matrix raised an exception: {e}")

    # ===============================
    # Function: test_plot_with_saving
    # What it does:
    #   - Test that plot_confusion_matrix saves a file when save_path is provided.
    #
    # Inputs:
    #   - true_labels, pred_labels, model_name, accuracy, temporary save path
    #
    # Output:
    #   - Asserts that the output file exists after saving
    # ===============================
    def test_plot_with_saving(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                self.true_labels,
                self.pred_labels,
                self.model_name,
                self.accuracy,
                save_path=save_path,
            )
            self.assertTrue(
                os.path.isfile(save_path),
                "Confusion matrix image was not saved.",
            )


if __name__ == "__main__":
    unittest.main()
