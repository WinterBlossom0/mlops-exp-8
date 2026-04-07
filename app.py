from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model1.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "model1.pkl was not found. Run `python model1.py` before starting the Flask app."
    )

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


@app.get("/")
def home() -> str:
    return "ML Model API is running"


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    values = payload.get("input")

    if not isinstance(values, list) or not values:
        return jsonify({"error": "Input must be a non-empty list, for example: [5]"}), 400

    try:
        features = np.array(values, dtype=float).reshape(1, -1)
    except ValueError:
        return jsonify({"error": "Input must contain only numeric values."}), 400

    prediction = model.predict(features).ravel().tolist()
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
