#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=100, type=int,
                    help='number of training steps')

def loadEstimator(batch_size, train_steps):
# Fetch the data
    (train_x, train_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda:iris_data.train_input_fn(train_x, train_y,
                                                 batch_size),
        steps=train_steps)

    return classifier

def predict(classifier, batch_size, emotions):

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(emotions,
                                                labels=None,
                                                batch_size=batch_size))


    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        
    return iris_data.SCORES[class_id], probability

def main(argv):
    args = parser.parse_args(argv[1:])

    # Generate predictions from the model
    predict_x = {
        'anger': [0.5360],
        'contempt': [0.1651],
        'disgust': [0.1309],
        'fear': [0.0271],
        'happiness': [0.0286],
        'neutral': [0.0804],
        'sadness': [0.0183],
        'surprise': [0.0136]
    }

    classifier = loadEstimator(args.batch_size, args.train_steps)
    score, probability = predict(classifier, args.batch_size, predict_x)

    template = ('\nPrediction is "{}" ({:.1f}%)')
    print(template.format(score, 100 * probability))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
