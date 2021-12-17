import time
from typing import Text

# External deps
from metaflow import FlowSpec, Parameter, step
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Internal deps
from TextPreprocessing import *
from TfIdfDataVectorizer import *
from LinearSVCModel import *
from s3_handler import S3Handler


class TrainingFlow(FlowSpec):
    s3_input_csv_path = Parameter(
        's3_input_csv_path',
        help='Path to input CSV file stored in S3 bucket, starts with s3://',
        required=True)

    rnd_seed = Parameter(
        'rnd_seed',
        help='Seed for Random Number Generator',
        default=42)

    test_size = Parameter(
        'test_size',
        help='Size of test set for train-test split (in %)',
        default=0.1)

    quickrun_pct = Parameter(
        'quickrun_pct',
        help='% of data to use for quick run (e.g. 0.1, 1 to use all)',
        default=1.0)


    @step
    def start(self):
        self.next(self.get_data)


    @step
    def get_data(self):
        data_columns  = ["sentiment", "ids", "date", "flag", "user", "text"]
        skip_every_nthline = int(1/self.quickrun_pct) if self.quickrun_pct < 1.0 else 1

        self.df_raw = S3Handler.download_s3_and_read(self.s3_input_csv_path, data_columns, skip_every_nthline)
        self.next(self.dataset_cleanup)


    @step
    def dataset_cleanup(self):
        self.texts_list = self.df_raw['text'].tolist()
        self.sentiments_list = self.df_raw['sentiment'].tolist()
        self.next(self.train_test_split)

    
    @step
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.texts_list,
            self.sentiments_list,
            test_size=self.test_size,
            random_state=self.rnd_seed)

        print(f"Train set size: {len(self.X_train)}, test: {len(self.X_test)}")
        print("Train features sample:\n", self.X_train[:5])
        print("Train labels sample:\n", self.y_train[:5])
        self.next(self.text_preprocessing)


    @step
    def text_preprocessing(self):
        preprocessor = TextPreprocessing()
        map(preprocessor.convert_to_lowercase, self.X_train)
        map(preprocessor.remove_stopwords, self.X_train)
        map(preprocessor.replace_regex_patterns, self.X_train)

        map(preprocessor.convert_to_lowercase, self.y_train)
        map(preprocessor.remove_stopwords, self.y_train)
        map(preprocessor.replace_regex_patterns, self.y_train)

        tfidf = TfIdfDataVectorizer()
        self.X_train, self.X_test = tfidf.train_and_transform(self.X_train, self.X_test)
        print(f"Train set size: {self.X_train.shape}, test set: {self.X_test.shape}")

        timestr = time.strftime("%Y%m%d-%H%M%S")
        tfidf.serialize(f'tfidf_vectorizer_{timestr}.joblib')
        self.next(self.train_linearsvc)


    @step
    def train_linearsvc(self):
        self.svc_model = LinearSVCModel()
        self.svc_model.train(self.X_train, self.y_train)
        self.linearsvc_y_pred = self.svc_model.predict(self.X_test)

        self.test_acc = accuracy_score(self.y_test, self.linearsvc_y_pred)
        print(f"Linear SVC accuracy on test set = {self.test_acc}")
        self.next(self.save_model_to_s3)


    @step
    def save_model_to_s3(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.svc_model.save_model_local(f'linearsvc_{timestr}_acc{self.test_acc}.joblib')
        # self.svc_model.save_model_s3(full_s3_path=f's3://mtszkw-github/sentiment/linearsvc_{timestr}.joblib')
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