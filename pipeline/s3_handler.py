from metaflow import S3
import pandas as pd

from typing import List


class S3Handler():
    def __init__(self):
        pass

    @staticmethod
    def download_s3_and_read(s3_input_csv_path: str, data_columns: List[str], skip_every_nthline=None):
        print(f'Downloading data from {s3_input_csv_path}')
        if skip_every_nthline:
            print(f"Skipping every {skip_every_nthline}th line...")

        with S3() as s3:
            print(f"Downloading input file from {s3_input_csv_path}")
            s3_csv_file = s3.get(s3_input_csv_path)
            df_raw = pd.read_csv(
                s3_csv_file.path,
                names=data_columns,
                skiprows=lambda i: i % skip_every_nthline != 0)
            print(f"Found {df_raw.shape} data frame in {s3_csv_file.path}")
            return df_raw
