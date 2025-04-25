# 文件名: test_text_cleaning.py

import unittest
from preprocessing import clean_text

class TestCleanText(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")

    def test_lowercase_conversion(self):
        self.assertEqual(clean_text("HELLO WORLD"), "hello world")

    def test_remove_url(self):
        inp = "Check this out: https://example.com now!"
        # "check this out" 中的 "this"、"out" 为停用词，会被移除
        self.assertEqual(clean_text(inp), "check now")

    def test_remove_mentions(self):
        self.assertEqual(clean_text("@user Hello there"), "hello there")

    def test_remove_hashtag_symbol(self):
        # 只会移除 "#"，保留词本身
        self.assertEqual(clean_text("#hashtag fun"), "hashtag fun")

    def test_remove_non_alpha(self):
        self.assertEqual(clean_text("Numbers 1234 and punctuation!!!"), "number punctuation")

    def test_stopword_removal(self):
        # "this"、"is"、"a" 为停用词，应被移除
        self.assertEqual(clean_text("This is a simple test"), "simple test")

    def test_negation_retention(self):
        # "not" 在 NEGATIONS 中，应被保留
        self.assertEqual(clean_text("Do not remove negations"), "not remove negation")

    def test_lemmatization(self):
        # "running" → "run", "cars" → "car"
        self.assertEqual(clean_text("Running cars"), "run car")

    def test_only_negations(self):
        # 连续多个否定词均应保留
        self.assertEqual(clean_text("No no never won't"), "no no never won't")

if __name__ == "__main__":
    unittest.main()
