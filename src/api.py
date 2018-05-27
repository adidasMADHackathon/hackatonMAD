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

    data = {
        'anger': [payload['anger']],
        'contempt': [payload['contempt']],
        'disgust': [payload['disgust']],
        'fear': [payload['fear']],
        'happiness': [payload['happiness']],
        'neutral': [payload['neutral']],
        'sadness': [payload['sadness']],
        'surprise': [payload['surprise']]
    }

    classifier = premade_estimator.loadEstimator(BATCH_SIZE, TRAIN_STEPS)
    score, probability = premade_estimator.predict(classifier, BATCH_SIZE, data)

    return json.dumps([score, 100 * probability])

@app.route("/dataset")
def dataset():
    dataset, _ = iris_data.load_data()
    return dataset.to_json(orient = "records")