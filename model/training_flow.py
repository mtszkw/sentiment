from metaflow import FlowSpec, Parameter, step, S3
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from data_embedding import TfIdfDataVectorizer
from data_processing import Sentiment140Preprocessing
from model_linearsvc import LinearSVCModel

class TrainingFlow(FlowSpec):
    # data_path = Parameter('data_path', help='Path to CSV file with input data', required=True)
    s3_input_csv_path = Parameter('s3_input_csv_path', help='Path to input CSV file stored in S3 bucket, starts with s3://', required=True)
    rnd_seed = Parameter('rnd_seed', help='Seed for Random Number Generator', default=42)
    test_size = Parameter('test_size', help='Size of test set for train-test split (in %)', default=0.1)
    quickrun_pct = Parameter('quickrun_pct', help='% of data to use for quick run (e.g. 0.1, 1 to use all)', default=1.0)

    @step
    def start(self):
        self.next(self.get_data)

    @step
    def get_data(self):
        print(f'Downloading data from {self.s3_input_csv_path}')
        try:
            data_columns  = ["sentiment", "ids", "date", "flag", "user", "text"]
            skip_every_nthline = int(1/self.quickrun_pct) if self.quickrun_pct < 1.0 else 1
            print(f"Skipping every {skip_every_nthline}th line thus reading {self.quickrun_pct*100}% of data")

            with S3() as s3:
                print(f"Downloading input file from {self.s3_input_csv_path}")
                self.s3_csv_file = s3.get(self.s3_input_csv_path)
                self.df_raw = pd.read_csv(
                    self.s3_csv_file.path,
                    names=data_columns,
                    skiprows=lambda i: i % skip_every_nthline != 0)
                print(f"Found {self.df_raw.shape} data frame in {self.s3_csv_file.path}")
        except Exception as exc:
            raise(f"Failure when reading data: {exc}")
        self.next(self.process_data)

    @step
    def process_data(self):
        data_proc = Sentiment140Preprocessing()
        self.df_text, self.df_sentiment = data_proc(
            self.df_raw,
            target='sentiment')
        self.next(self.train_test_split)
    
    @step
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df_text,
            self.df_sentiment,
            test_size=self.test_size,
            random_state=self.rnd_seed)
        print(f"Train set size: [{self.X_train.shape}, {self.y_train.shape}], test set: [{self.X_test.shape}, {self.y_test.shape}]")
        self.next(self.tfidf_vectorization)

    @step
    def tfidf_vectorization(self):
        tfidf = TfIdfDataVectorizer()
        self.X_train, self.X_test = tfidf(self.X_train, self.X_test)
        print(f"Train set size: {self.X_train.shape}, test set: {self.X_test.shape}")
        self.next(self.train_linearsvc)

    @step
    def train_linearsvc(self):
        svc_model = LinearSVCModel()
        svc_model.train(self.X_train, self.y_train)
        self.linearsvc_y_pred = svc_model.predict(self.X_test)
        print(f"Linear SVC accuracy on test set = {accuracy_score(self.y_test, self.linearsvc_y_pred)}")
        self.next(self.visualize_results)

    @step
    def visualize_results(self):
        print(f"LinearSVC:\n{classification_report(self.y_test, self.linearsvc_y_pred)}")
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    TrainingFlow()