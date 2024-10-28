import os

from flask import Flask, request

from sms_predictor_API import SMSPredictor
from flask import jsonify

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/predict_classification/', methods=['POST']) # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_input = request.get_json()
    return dp.predict_classification(prediction_input)


dp = SMSPredictor()
# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port = 5000)
