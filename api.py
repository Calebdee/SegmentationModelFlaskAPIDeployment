from flask import Flask, request
from flasgger import Swagger
import logging

import pandas as pd
import pickle
import json

with open("model_utils.txt", "r") as file:
    codecontent = file.read()
exec(codecontent)

app = Flask(__name__)
# Set up swagger object for API documentation at /apidocs
swagger = Swagger(app)
# Set up logger that writes logs to local "record.log"
logging.basicConfig(filename="record.log", level=logging.DEBUG)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict outcome of purchase from input data
    ---
    parameters:
        - name: x5
          required: true
          type: string
          description: Day of the week in lowercase format (e.g. monday)
        - name: x12
          required: true
          type: string
          description: Currency amount, with or without $ (e.g. $1200.00, 12000.00)
        - name: x31
          required: true
          type: string
          description: Location of event (valid fields are "asia", "germany", "japan", "america")
        - name: x63
          required: true
          type: string
          description: Percentage, with or without % (e.g. 16.1% or 16.1)
        - name: x81
          required: true
          type: string
          description: Month of the year capitalized (e.g. January)
        - name: x82
          required: true
          type: string
          description: Gender capitalized (e.g. Male)
        - name: x0-x99 not explained above
          required: true
          type: string
          description: Numerical float fields
    responses:
        200:
            description: Prediction successfully returned
        410:
            description: Invalid input data size in input data
        420:
            description: Invalid day of week value in input data
        421:
            description: Invalid location of event value in input data
        422:
            description: Invalid month of the year value in input data
        423:
            description: Invalid gender value in input data
    """
    # Gets request body from a POST operation
    data = request.json

    # Batch operation
    if type(data) == list:
        data = pd.DataFrame(data, columns=data[0].keys())
    # Single row
    else:
        data = pd.DataFrame([data], columns=data.keys())

    # Validate user input
    results, code = validate_input(data)
    if results != "Valid":
        return results, code
    
    # Prepare input for our model, converting certain fields to floats, imputing missing fields, scaling, and subsetting to selected features
    data_model = prepare_model_input(data)
    pred = result.predict(data_model)
    
    # Prepare output, returing all model variables, probability, and outcome class at the 75th percentile for each prediction
    response = data_model.to_dict(orient="records")
    for i in range(len(pred)):
        response[i]["business_outcome"] = int(pred[i] >= 0.75)
        response[i]["phat"] = pred[i]
    
    return json.dumps(response, sort_keys=True), 201

# Function to return the app for unit testing in the tests directory
def test_model(): 
    return app

# Runs the Flask application at port 1313
# The following two functions are not considered for unit testing coverage (pragma: no cover)
def run_model(): # pragma: no cover
    app.logger.info("Setting up API on port 1313")
    app.run(port=1313, host='0.0.0.0')

if __name__ == '__main__': # pragma: no cover
    run_model()
    
