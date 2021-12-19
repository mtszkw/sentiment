FROM python:3.8-alpine

WORKDIR /

COPY model_*.joblib model.joblib
COPY vectorizer_*.joblib vectorizer.joblib
COPY TextPreprocessing.py TextPreprocessing.py 

RUN pip install scikit-learn==1.0.1 flask

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]