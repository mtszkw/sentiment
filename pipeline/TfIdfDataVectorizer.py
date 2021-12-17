import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfDataVectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)

    def train_and_transform(self, X_train, X_test):
        self.vectorizer.fit(X_train)
        train_transformed = self.vectorizer.transform(X_train)
        test_transformed = self.vectorizer.transform(X_test)
        return train_transformed, test_transformed

    def serialize(self):
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.joblib')
