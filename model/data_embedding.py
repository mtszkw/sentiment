import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdfDataVectorizer():
    def __init__(self):
        pass

    def __call__(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
        tfidf.fit(X_train)
        train_transformed = tfidf.transform(X_train)
        test_transformed = tfidf.transform(X_test)
        return train_transformed, test_transformed
