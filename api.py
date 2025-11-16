import os
import pandas as pd
import mlflow
from flask import Flask, request, jsonify

# --- CONFIGURATION
# Load MLFLOW_TRACKING_URI from environment variable
# This will be set in your GKE deployment
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI environment variable not set")

mlflow.set_tracking_uri(tracking_uri)

MODEL_NAME = "iris-classifier"
MODEL_STAGE = "production"
model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"

# --- LOAD MODEL ---
print(f"Loading model '{MODEL_NAME}' (Stage: {MODEL_STAGE}) from {tracking_uri}...")
try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load model. Error: {e}")
    model = None # Handle error gracefully

# --- CREATE FLASK APP ---
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify service is running."""
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    if not model:
        return jsonify({"error": "Model is not loaded"}), 503

    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Convert to DataFrame (model expects this format)
        # Assumes input like: {"columns": [...], "data": [[...]]}
        # Or simplified: {"data": [5.1, 3.5, 1.4, 0.2]}

        if "data" not in data:
             return jsonify({"error": "JSON data must contain a 'data' key"}), 400

        if isinstance(data["data"][0], list):
            # Multiple predictions
            input_data = pd.DataFrame(
                data["data"], 
                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
        else:
            # Single prediction
            input_data = pd.DataFrame(
                [data["data"]], 
                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )

        # Make prediction
        prediction = model.predict(input_data)

        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Gunicorn will run the app
    # This block is for local testing (e.g., `python api.py`)
    app.run(host="0.0.0.0", port=8080)