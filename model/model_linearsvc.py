from joblib import dump, load
import pandas as pd
from sklearn.svm import LinearSVC


class LinearSVCModel():
    def __init__(self):
        self.svc_model = LinearSVC()

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.svc_model.fit(X_train, y_train)
        
    def predict(self, X: pd.DataFrame):
        self.y_pred = self.svc_model.predict(X)
        return self.y_pred

    def save_model(self, path_with_extension: str):
        dump(self.svc_model, path_with_extension)
        print(f"Saving LinearSVC model to {path_with_extension}")

    def load_model(self, path_with_extension: str):
        self.svc_model = load(path_with_extension)
