import pytest
import pandas as pd
import mlflow
import dvc.api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import subprocess
import shutil

# --- 1. CONFIGURATION ---
# (REQUIRED) SET THIS TO YOUR 3002 PROXY URL
MLFLOW_TRACKING_URI = "http://34.170.7.37:5000/"

# Model name in the MLflow Model Registry
MODEL_NAME = "iris-classifier"
MODEL_STAGE = "production"

# DVC path for the evaluation data
DATA_PATH = 'data/iris.csv'
DVC_REPO_URL = 'https://github.com/kun101/mlops-w3'
DVC_TAG = 'v1.0' # Use the same data version you trained on

if os.path.exists("mlops-w3"):
    shutil.rmtree("mlops-w3")
    
if not os.path.exists("mlops-w3"):
    subprocess.run(["git", "clone", DVC_REPO_URL], check=True)
    
repo_path = os.path.abspath("mlops-w3")

# --- 2. FIXTURES ---
@pytest.fixture(scope="module")
def model():
    """Fixture to load the Production model from MLflow Registry."""
    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        pytest.fail(f"Failed to load model from MLflow: {e}")

@pytest.fixture(scope="module")
def data():
    """Fixture to load evaluation data from DVC."""
    print("Loading data from DVC...")
    try:
        data_url = dvc.api.get_url(
            path="data/iris.csv",
            repo=repo_path,
            remote="gcs-remote",
            rev="v1.0"
        )
        data = pd.read_csv(data_url)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        pytest.fail(f"Failed to load data from DVC: {e}")

# --- 3. TESTS ---
def test_model_predict(model, data):
    """Test if the model can make predictions without error."""
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    try:
        predictions = model.predict(X.head())
        assert predictions is not None
        print("\nPrediction test passed.")
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")

def test_model_accuracy(model, data):
    """Test if the model accuracy is above the 0.90 threshold."""
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    # Use the same split as in training for a consistent eval set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.3f}")

    assert accuracy >= 0.90, f"Accuracy {accuracy:.3f} is below threshold (0.90)"
    print("Accuracy test passed.")