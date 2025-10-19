import pytest
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dvc.api import DVCFileSystem # Import DVCFileSystem

# --- Configuration (UPDATE THESE) ---
DVC_REPO_URL = "https://github.com/kun101/mlops-w3.git" # Your GitHub repo URL
DATA_PATH_IN_DVC = "data/iris.csv" # Path to data folder in your DVC config
MODEL_PATH_IN_DVC = "artifacts/model.joblib" # Path to model in your DVC config
DVC_TAG_TO_TEST = os.getenv("DVC_REV", "v1.0")

# --- Data Loading Function (using DVCFileSystem) ---
def load_dvc_data(path_in_dvc, tag):
    fs = DVCFileSystem(DVC_REPO_URL, rev=tag)
    with fs.open(path_in_dvc, 'rb') as f:
        if path_in_dvc.endswith('.csv'):
            return pd.read_csv(f)
        elif path_in_dvc.endswith('.joblib'):
            return joblib.load(f)
        else:
            raise ValueError("Unsupported file type for DVC loading.")

# --- Tests ---
@pytest.fixture(scope="module")
def setup_data_and_model():
    """Fixture to load data and model for tests."""
    try:
        data = load_dvc_data(DATA_PATH_IN_DVC, DVC_TAG_TO_TEST)
        model = load_dvc_data(MODEL_PATH_IN_DVC, DVC_TAG_TO_TEST)
        return data, model
    except Exception as e:
        pytest.fail(f"Failed to load data or model from DVC: {e}")

def test_data_columns(setup_data_and_model):
    """Test if the dataset has expected columns."""
    data, _ = setup_data_and_model
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in data.columns for col in expected_columns), "Data is missing expected columns"
    print(f"\nData columns: {data.columns.tolist()} (OK)")

def test_data_shape(setup_data_and_model):
    """Test if the dataset has the expected number of rows (e.g., at least 100 for Iris)."""
    data, _ = setup_data_and_model
    assert len(data) >= 100, "Dataset has fewer than 100 rows"
    print(f"Data shape: {data.shape} (OK)")

def test_model_predicts(setup_data_and_model):
    """Test if the model can make predictions without error."""
    data, model = setup_data_and_model
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    try:
        predictions = model.predict(X.head())
        assert predictions is not None, "Model failed to make predictions"
        print(f"Sample predictions made successfully: {predictions.tolist()} (OK)")
    except Exception as e:
        pytest.fail(f"Model prediction failed: {e}")

def test_model_accuracy_threshold(setup_data_and_model):
    """Test if the model's accuracy meets a minimum threshold."""
    data, model = setup_data_and_model
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    # Split data for evaluation (mimic training split)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    min_accuracy_threshold = 0.90 # Set your desired minimum accuracy
    assert accuracy >= min_accuracy_threshold, f"Model accuracy {accuracy:.3f} is below {min_accuracy_threshold}"
    print(f"Model accuracy: {accuracy:.3f} (Above {min_accuracy_threshold} threshold - OK)")