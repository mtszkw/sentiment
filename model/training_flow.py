from metaflow import FlowSpec, Parameter, step
from sklearn.model_selection import train_test_split
import pandas as pd

from data_processing import Sentiment140Preprocessing


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
        print(f"Train set size: {self.X_train.shape}, test set: {self.X_test.shape}")
        self.next(self.train_convnet)

    @step
    def train_convnet(self):
        print('Training Convolutional Neural Network...')
        self.next(self.visualize_results)

    @step
    def visualize_results(self):
        print('Visualizing results...')
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TrainingFlow()