# File: sentiment_analysis/test_preprocessing.py

"""
Unit tests for the clean_text function defined in preprocessing.py.

Each test verifies one aspect of the text cleaning pipeline:
- Lowercasing
- URL, mention, hashtag removal
- Non-alphabetic character filtering
- Stopword removal with exceptions
- Negation retention
- Lemmatization behavior
"""

import unittest
from preprocessing import clean_text


class TestCleanText(unittest.TestCase):
    """
    Test suite for the clean_text function.

    Methods:
        test_empty_string        -- clean_text("") returns ""
        test_lowercase_conversion-- upper-case input is lowercased
        test_remove_url          -- URLs are removed from text
        test_remove_mentions     -- @mentions are stripped
        test_remove_hashtag_symbol-- hashtag symbol removed but text preserved
        test_remove_non_alpha    -- non-alphabetic chars removed
        test_stopword_removal    -- common stopwords are filtered out
        test_negation_retention  -- negation words are retained
        test_lemmatization       -- verbs are correctly lemmatized
        test_multiple_negations  -- multiple negations handled correctly
    """

    def test_empty_string(self):
        """
        What it tests:
            clean_text on an empty string.
        Input:
            ""
        Expected output:
            ""
        """
        self.assertEqual(clean_text(""), "")

    def test_lowercase_conversion(self):
        """
        What it tests:
            Conversion of uppercase to lowercase.
        Input:
            "HELLO WORLD"
        Expected output:
            "hello world"
        """
        self.assertEqual(clean_text("HELLO WORLD"), "hello world")

    def test_remove_url(self):
        """
        What it tests:
            Removal of URLs from text.
        Input:
            "Check this out: https://example.com now"
        Expected output:
            "check"
        """
        inp = "Check this out: https://example.com now"
        self.assertEqual(clean_text(inp), "check")

    def test_remove_mentions(self):
        """
        What it tests:
            Stripping of @mentions.
        Input:
            "@user Hello there"
        Expected output:
            "hello"
        """
        self.assertEqual(clean_text("@user Hello there"), "hello")

    def test_remove_hashtag_symbol(self):
        """
        What it tests:
            Removal of '#' symbol but retention of hashtag text.
        Input:
            "#hashtag fun"
        Expected output:
            "hashtag fun"
        """
        self.assertEqual(clean_text("#hashtag fun"), "hashtag fun")

    def test_remove_non_alpha(self):
        """
        What it tests:
            Filtering out non-alphabetic characters.
        Input:
            "Numbers 1234 and punctuation!!!"
        Expected output:
            "number punctuation"
        """
        self.assertEqual(clean_text("Numbers 1234 and punctuation!!!"), "number punctuation")

    def test_stopword_removal(self):
        """
        What it tests:
            Removal of common stopwords.
        Input:
            "This is a simple test"
        Expected output:
            "simple test"
        """
        self.assertEqual(clean_text("This is a simple test"), "simple test")

    def test_negation_retention(self):
        """
        What it tests:
            Retention of negation words (e.g., 'not').
        Input:
            "Do not remove negations"
        Expected output:
            "not remove negation"
        """
        self.assertEqual(clean_text("Do not remove negations"), "not remove negation")

    def test_lemmatization(self):
        """
        What it tests:
            Proper lemmatization of verbs (e.g., running -> running or run).
        Input:
            "Running cars"
        Expected output:
            "running car"
        """
        self.assertEqual(clean_text("Running cars"), "running car")

    def test_multiple_negations(self):
        """
        What it tests:
            Handling of multiple negations and contractions.
        Input:
            "No no never won't"
        Expected output:
            "no no never"
        """
        self.assertEqual(clean_text("No no never won't"), "no no never")


if __name__ == "__main__":
    unittest.main()

