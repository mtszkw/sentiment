import re

import nltk
from nltk import text
from nltk.corpus import stopwords
import pandas as pd


class Sentiment140Preprocessing():
    def __init__(self):
        nltk.download('stopwords')

    def __call__(self, df_raw: pd.DataFrame, target: str):
        df_raw = self._remove_unnecessary_columns(df_raw)
        df_raw['text'] = df_raw['text'].apply(self._convert_to_lowercase)
        df_raw['text'] = df_raw['text'].apply(self._replace_regex_patterns)
        df_raw['text'] = df_raw['text'].apply(self._remove_stopwords)
        print(f"Processed data: {df_raw.head()}")
        return df_raw['text'], df_raw['sentiment']

    def _remove_unnecessary_columns(self, df_reviews: pd.DataFrame):
        # Only sentiment (target) and text columns are important here.
        return df_reviews[['sentiment', 'text']]

    def _convert_to_lowercase(self, text_entry: str):
        return text_entry.lower()

    def _replace_regex_patterns(self, text_entry: str):
        urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern       = '@[^\s]+'
        alphaPattern      = "[^a-zA-Z0-9]"
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        text_entry = re.sub(urlPattern, ' URL', text_entry)
        text_entry = re.sub(userPattern, ' USER', text_entry)
        text_entry = re.sub(alphaPattern, " ", text_entry)
        text_entry = re.sub(sequencePattern, seqReplacePattern, text_entry)
        return text_entry

    def _remove_stopwords(self, text_entry: str):
        # Remove stop words
        text_entry = text_entry.split()
        stop_words = stopwords.words('english')
        text_entry = " ".join([word for word in text_entry if not word in stop_words])
        return text_entry
