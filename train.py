import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dvc.api
import os
import subprocess
import shutil

# --- 1. SET UP ---

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
# (REQUIRED) SET THIS TO YOUR 3002 PROXY URL
MLFLOW_TRACKING_URI = "http://127.0.0.1:3002"

# Path to the data in your DVC remote (GCS)
# This assumes you are in the 'week_1' directory
path = 'data/iris.csv'
repo = 'https://github.com/kun101/mlops-w3' # Your GitHub repo
version = 'v1.0' # The DVC tag for your data

if os.path.exists("mlops-w3"):
    shutil.rmtree("mlops-w3")
    
if not os.path.exists("mlops-w3"):
    subprocess.run(["git", "clone", repo], check=True)

repo_path = os.path.abspath("mlops-w3")

# MLflow experiment name
EXPERIMENT_NAME = "iris-classifier-hyper-tuning"

def main(max_depth, min_samples_leaf):
    """
    Main training loop.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        print(f"Starting run: {run.info.run_id}")

        # --- 2. LOAD DATA (from DVC) ---
        print("Loading data from DVC...")
        data_url = dvc.api.get_url(
            path="data/iris.csv",
            repo=repo_path,
            remote="gcs-remote",
            rev="v1.0"
        )
        data = pd.read_csv(data_url)

        # --- 3. TRAIN MODEL (with Hyperparameters) ---
        print(f"Training model with max_depth={max_depth}, min_samples_leaf={min_samples_leaf}...")
        train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
        X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_train = train.species
        X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y_test = test.species

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=1)
        model.fit(X_train, y_train)

        # --- 4. LOG TO MLFLOW ---
        print("Logging to MLflow...")

        # Log hyperparameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        # Calculate and log metrics
        prediction = model.predict(X_test)
        accuracy = accuracy_score(prediction, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        # This is the line that replaces `dvc add`
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # This creates a 'model' folder inside the run's artifacts
            registered_model_name="iris-classifier" # This registers the model
        )

        print(f"Run complete. Accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=3, help="Max depth of the decision tree")
    parser.add_argument("--min_samples_leaf", type=int, default=1, help="Min samples per leaf")
    args = parser.parse_args()

    main(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)