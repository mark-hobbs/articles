import os

from flask import jsonify

from .model import GPR
from .utils import json_to_ndarray

model = GPR()
model.load(os.path.join("service", "pretrained-model.pkl"))


def predict(input):
    """
    Predict... using a pre-trained model

    Args:
        - input

    Returns:
        - JSON response
    """
    try:
        mean, var = model.predict(json_to_ndarray(input))
        return jsonify({"mean": float(mean[0, 0]), "variance": float(var[0, 0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
