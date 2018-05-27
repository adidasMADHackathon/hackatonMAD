from flask import Flask
from flask import jsonify
from flask import request
import json

import iris_data
import premade_estimator
app = Flask(__name__)

BATCH_SIZE = 100
TRAIN_STEPS = 1000

@app.route("/forecast", methods=['POST'])
def forecast():
    payload = request.get_json()
    # data = json.loads(payload)[0]
    # predict_x = {
        # "anger": [0.5360],
        # "contempt": [0.1651],
        # "disgust": [0.1309],
        # "fear": [0.0271],
        # "happiness": [0.0286],
        # "neutral": [0.0804],
        # "sadness": [0.0183],
        # "surprise": [0.0136]
    # }

    classifier = premade_estimator.loadEstimator(BATCH_SIZE, TRAIN_STEPS)
    score, probability = premade_estimator.predict(classifier, BATCH_SIZE, payload)

    return json.dumps([score, 100 * probability])

@app.route("/dataset")
def dataset():
    dataset, _ = iris_data.load_data()
    return dataset.to_json(orient = "records")