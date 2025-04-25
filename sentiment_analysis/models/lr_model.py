from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

"""
    A simple Logistic Regression text classification model using TfidfVectorizer.

    This class provides methods to train (fit) a Logistic Regression classifier
    on text data and make predictions on new text inputs.
"""


class LRModel:
    def __init__(self):
        """
        Initializes the LRModel with a TfidfVectorizer and a LogisticRegression classifier.

        Inputs:
            None
        Outputs:
            An initialized LRModel instance with a fresh vectorizer and classifier.
        """
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(max_iter=200)

    def fit(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        Inputs:
            X_train (list of str): List of text samples to train on.
            y_train (list or array-like): Corresponding labels for the training samples.

        Outputs:
            None. The model is fitted internally (modifies the internal vectorizer and classifier).
        """
        Xv = self.vectorizer.fit_transform(X_train)
        self.clf.fit(Xv, y_train)

    def predict(self, X):
        """
        Predicts labels for the given text inputs.

        Inputs:
            X (list of str): List of text samples to classify.

        Outputs:
            preds (array): Array of predicted labels corresponding to each input sample.
        """
        Xv = self.vectorizer.transform(X)
        return self.clf.predict(Xv)
