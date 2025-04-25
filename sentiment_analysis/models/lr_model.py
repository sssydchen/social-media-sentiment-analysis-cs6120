from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LRModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = LogisticRegression(max_iter=200)

    def fit(self, X_train, y_train):
        Xv = self.vectorizer.fit_transform(X_train)
        self.clf.fit(Xv, y_train)

    def predict(self, X):
        Xv = self.vectorizer.transform(X)
        return self.clf.predict(Xv)