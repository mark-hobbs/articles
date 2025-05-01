from flask import Flask, request

from . import services

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests via POST and return the result as JSON
    """
    return services.predict(request.json)