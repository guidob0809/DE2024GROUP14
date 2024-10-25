# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/predict_sms', methods=["GET", "POST"])
def predict_sms():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        # Get the SMS message from the form input
        prediction_input = [
            {
                "message": request.form.get("message")  # getting input with name = message in HTML form
            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        # Use an environment variable to find the value of the SMS prediction API
        predictor_api_url = os.environ['PREDICTOR_API']

        # Use requests library to execute the prediction service API by sending an HTTP POST request
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        # Extract prediction result from the response
        prediction_value = res.json()['predictions'][0]
        logging.info("Prediction Output : %s", prediction_value)

        # Return the prediction result to the response page
        return render_template("response_page.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # If HTTP method other than GET or POST is used

# The code within this conditional block will only run if the python file is executed as a script
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
