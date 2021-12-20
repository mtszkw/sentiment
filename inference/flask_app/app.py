import joblib
from flask import Flask, request, json

from TextPreprocessing import TextPreprocessing


MODEL_FILE_NAME = '/usr/model.joblib'
VECTORIZER_FILE_NAME = '/usr/vectorizer.joblib'

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    payload = json.loads(request.get_data().decode('utf-8'))
    prediction = predict(payload['payload'])
    data = {}
    data['data'] = prediction[-1]
    return json.dumps(data)

def load_model():
    return joblib.load(MODEL_FILE_NAME)

def load_vectorizer():
    return joblib.load(VECTORIZER_FILE_NAME)

def predict(data):
    # Preprocessing
    preprocessor = TextPreprocessing()
    map(preprocessor.convert_to_lowercase, data)
    map(preprocessor.remove_stopwords, data)
    map(preprocessor.replace_regex_patterns, data)
    # print(f"Input text after preprocessing: {data}")

    text_embeddings = load_vectorizer().transform([data])
    # print(f"Text embeddings: {text_embeddings}")

    output = load_model().predict(text_embeddings)
    return output

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')