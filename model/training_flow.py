from metaflow import FlowSpec, Parameter, step
import pandas as pd
import scipy
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch

from data_embedding import TfIdfDataVectorizer
from data_processing import Sentiment140Preprocessing
from model_linearsvc import LinearSVCModel
from model_mlp import MLPModel

class TrainingFlow(FlowSpec):
    data_path = Parameter('data_path', help='Path to CSV file with input data', required=True)
    rnd_seed = Parameter('rnd_seed', help='Seed for Random Number Generator', default=42)
    test_size = Parameter('test_size', help='Size of test set for train-test split (in %)', default=0.1)

    @step
    def start(self):
        self.next(self.get_data)

    @step
    def get_data(self):
        print(f'Downloading data from {self.data_path}')
        try:
            data_columns  = ["sentiment", "ids", "date", "flag", "user", "text"]
            self.df_data_raw = pd.read_csv(self.data_path, names=data_columns)
            print(self.df_data_raw.head())
        except Exception as exc:
            print(f"Exception: {exc}")
        self.next(self.process_data)

    @step
    def process_data(self):
        data_proc = Sentiment140Preprocessing()
        self.df_text, self.df_sentiment = data_proc(
            self.df_data_raw,
            target='sentiment')
        self.next(self.train_test_split)
    
    @step
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df_text,
            self.df_sentiment,
            test_size=self.test_size,
            random_state=self.rnd_seed)
        print(f"Train set type: f{type(self.X_train)}")
        print(f"Train set size: {self.X_train.shape}, test set: {self.X_test.shape}")
        self.next(self.tfidf_vectorization)

    @step
    def tfidf_vectorization(self):
        tfidf = TfIdfDataVectorizer()
        self.X_train, self.X_test = tfidf(self.X_train, self.X_test)
        print(f"Train set type: f{type(self.X_train)}")
        print(f"Train set size: {self.X_train.shape}, test set: {self.X_test.shape}")
        self.next(self.train_linearsvc, self.train_mlp)

    @step
    def train_linearsvc(self):
        svc_model = LinearSVCModel()
        svc_model.train(self.X_train, self.y_train)
        self.y_pred = svc_model.predict(self.X_test)
        self.next(self.visualize_results)

    @step
    def train_mlp(self):
        mlp_model = MLPModel(
            input_size=self.X_train.shape[1],
            hidden_sizes=[1024, 512])
        X_train_tensor = torch.Tensor(scipy.sparse.csr_matrix.todense(self.X_train)).float()
        y_train_tensor = torch.Tensor(self.y_train.values)
        X_test_tensor = torch.Tensor(scipy.sparse.csr_matrix.todense(self.X_test)).float()
        mlp_model.train(X_train_tensor, y_train_tensor)
        self.y_pred = mlp_model.predict(X_test_tensor)
        self.next(self.visualize_results)

    @step
    def visualize_results(self, inputs):
        print(f"LinearSVC:\n{classification_report(self.y_test, inputs.train_linearsvc.y_pred)}")
        print(f"Torch MLP:\n{classification_report(self.y_test, inputs.train_mlp.y_pred)}")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TrainingFlow()