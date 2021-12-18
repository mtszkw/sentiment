import argparse
import joblib


import sys
sys.path.append("./pipeline")
sys.path.append("../../pipeline")

# from sklearn.svm import LinearSVC
from TextPreprocessing import TextPreprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_joblib_path", help="Path to the serialized model")
    parser.add_argument("--vectorizer_joblib_path", help="Path to the serialized vectorizer")
    parser.add_argument("--input_text", help="Input text to classify")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    clf = joblib.load(args.model_joblib_path)
    vectorizer = joblib.load(args.vectorizer_joblib_path)

    # Preprocessing
    input_text = args.input_text
    preprocessor = TextPreprocessing()
    map(preprocessor.convert_to_lowercase, input_text)
    map(preprocessor.remove_stopwords, input_text)
    map(preprocessor.replace_regex_patterns, input_text)
    print(f"Input text after preprocessing: {input_text}")

    text_embeddings = vectorizer.transform([input_text])
    print(f"Text embeddings: {text_embeddings}")

    output = clf.predict(text_embeddings)[0]

    label_map = {0: 'Negative', 2: 'Neutral', 4: 'Positive'}
    print(f"Model output: {label_map[output]}")

    assert label_map[output] == 'Positive'
