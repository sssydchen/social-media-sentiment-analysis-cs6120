from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

"""
    A simple Naive Bayes text classification model using CountVectorizer.

    This class provides methods to train (fit) a Multinomial Naive Bayes classifier
    on text data and make predictions on new text inputs.
"""


class NBModel:
    def __init__(self):
        """
        Initializes the NBModel with a CountVectorizer and a MultinomialNB classifier.

        Inputs:
            None
        Outputs:
            An initialized NBModel instance with a fresh vectorizer and classifier.
        """
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()

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
