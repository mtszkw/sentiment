# https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_learn_bring_your_own_model/code

import os
import joblib

def predict_fn(input_object, model):
    print("calling model")
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model