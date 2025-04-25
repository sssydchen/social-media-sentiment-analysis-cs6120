from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class NBModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.clf = MultinomialNB()

    def fit(self, X_train, y_train):
        Xv = self.vectorizer.fit_transform(X_train)
        self.clf.fit(Xv, y_train)

    def predict(self, X):
        Xv = self.vectorizer.transform(X)
        return self.clf.predict(Xv)