import tempfile

import joblib
from metaflow import S3
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel():
    def __init__(self, seed: int):
        self.model = GradientBoostingClassifier(random_state=seed, n_estimators=500, learning_rate=0.1)

    @staticmethod
    def from_local(full_local_path: str):
        gbc = GradientBoostingClassifier()
        gbc.model = joblib.load(full_local_path)
        return gbc

    @staticmethod
    def from_s3(full_s3_path: str):
        model = GradientBoostingClassifier()
        with S3() as s3:
            model_local_path = s3.get(full_s3_path)
            model.svc_model = joblib.load(model_local_path)
        return model

    def preprocess(self):
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(X_train, y_train)
        
    def predict(self, X: pd.DataFrame):
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def save_model_local(self, path_with_extension: str):
        joblib.dump(self.model, path_with_extension)
        print(f"Saving Gradient Boosting model to {path_with_extension}")

    def save_model_s3(self, full_s3_path: str):
        with S3() as s3:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(self.model, fp)
                fp.seek(0)
                s3.put(key=full_s3_path, obj=fp.read(), overwrite=True)