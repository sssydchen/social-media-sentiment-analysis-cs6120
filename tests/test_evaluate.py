import unittest
import io
import sys
from utils.evaluate import compute_accuracy, print_report, results


class TestEvaluate(unittest.TestCase):
    """
    Unit tests for compute_accuracy and print_report functions in evaluate.py
    """
    def setUp(self):
        # Clear results dict before each test
        results.clear()

    def test_compute_accuracy_stores_and_returns(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 0, 0, 1]
        acc = compute_accuracy("test_model", y_true, y_pred)
        # 3 correct out of 4 = 0.75 accuracy
        self.assertAlmostEqual(acc, 0.75)
        # Should store in results
        self.assertIn("test_model", results)
        self.assertAlmostEqual(results["test_model"], 0.75)

    def test_print_report_output_and_results(self):
        y_true = ["a", "b", "a"]
        y_pred = ["a", "a", "b"]
        # Capture stdout
        buffer = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer

        print_report("dummy", y_true, y_pred)

        # Restore stdout
        sys.stdout = sys_stdout
        output = buffer.getvalue()

        # Check that header and metrics appear
        self.assertIn("=== dummy ===", output)
        self.assertIn("precision", output.lower())
        self.assertIn("accuracy", output.lower())
        # Ensure results updated
        self.assertIn("dummy", results)
        self.assertTrue(0 <= results["dummy"] <= 1)

    def test_zero_division_handling(self):
        # True labels all one class, predictions all another class
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        # compute_accuracy should return 0.0
        acc = compute_accuracy("zerotest", y_true, y_pred)
        self.assertAlmostEqual(acc, 0.0)

        # print_report should not raise and should mention 0.00 accuracy
        buffer = io.StringIO()
        sys_stdout = sys.stdout
        sys.stdout = buffer

        print_report("zerotest", y_true, y_pred)

        sys.stdout = sys_stdout
        output = buffer.getvalue()
        self.assertIn("zerotest", output)
        self.assertIn("0.00", output)


if __name__ == '__main__':
    unittest.main()
