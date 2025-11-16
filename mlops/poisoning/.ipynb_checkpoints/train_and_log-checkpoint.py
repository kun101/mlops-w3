#!/usr/bin/env python3
"""
train_and_log.py
Train a simple classifier on provided CSV and log to MLflow.

Usage:
python train_and_log.py --data data/poisoned/iris_feature_noise_05.csv --exp poison_experiment --run-name fn-05
"""

import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['target']
    return X, y

def train_and_log(csv_path, experiment_name, run_name, seed=42):
    mlflow.set_experiment(experiment_name)
    X, y = load_data(csv_path)
    # split into train/test deterministic
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    with mlflow.start_run(run_name=run_name):
        params = {
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": seed,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "data_file": os.path.basename(csv_path)
        }
        mlflow.log_params(params)
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, preds)

        mlflow.log_metrics({"accuracy": acc, "precision_macro": prec, "recall_macro": rec})
        # save confusion matrix as artifact
        cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
        cm_path = "confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True)
        mlflow.log_artifact(cm_path)

        # log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged run: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp", default="iris_poisoning")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_log(args.data, args.exp, args.run_name or os.path.basename(args.data), seed=args.seed)
