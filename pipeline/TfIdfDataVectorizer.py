import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class TfIdfDataVectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)

    def train_and_transform(self, X_train: List[str], X_test: List[int]):
        self.vectorizer.fit(X_train)
        train_transformed = self.vectorizer.transform(X_train)
        test_transformed = self.vectorizer.transform(X_test)
        return train_transformed, test_transformed

    def serialize(self, filename: str):
        joblib.dump(self.vectorizer, filename)
