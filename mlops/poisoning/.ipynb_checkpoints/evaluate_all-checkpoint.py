#!/usr/bin/env python3
"""
evaluate_all.py
Run training across datasets and summarize metrics from MLflow, produce plots.
"""

import os
import subprocess
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import glob

DATA_DIR = "data/poisoned"
EXPERIMENT = "iris_poisoning"
OUT_DIR = "results/poisoning"
os.makedirs(OUT_DIR, exist_ok=True)

def run_all_training():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    for f in files:
        name = os.path.basename(f).replace(".csv", "")
        print("Training", name)
        subprocess.run(["python", "mlops/poisoning/train_and_log.py", "--data", f, "--exp", EXPERIMENT, "--run-name", name], check=True)

def collect_mlflow_metrics():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit("Experiment not found, run training first")
    runs = client.search_runs(exp.experiment_id)
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName"),
            "accuracy": r.data.metrics.get("accuracy"),
            "precision_macro": r.data.metrics.get("precision_macro"),
            "recall_macro": r.data.metrics.get("recall_macro"),
            "data_file": r.data.params.get("data_file"),
            "training_rows": r.data.params.get("training_rows")
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
    return df

def plot_metrics(df):
    df['poison_type'] = df['data_file'].str.extract(r'iris_([^_]+)_')[0]
    df['poison_level'] = df['data_file'].str.extract(r'_(\d{2})\.csv$')[0].astype(int)
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    for i, metric in enumerate(['accuracy','precision_macro','recall_macro']):
        for ptype in df['poison_type'].unique():
            sub = df[df['poison_type']==ptype].sort_values('poison_level')
            ax[i].plot(sub['poison_level'], sub[metric], marker='o', label=ptype)
        ax[i].set_title(metric)
        ax[i].set_xlabel('poison level (%)')
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metrics_vs_poison.png"))
    print("Saved plot to results/poisoning/metrics_vs_poison.png")

if __name__ == "__main__":
    # run_all_training()  # uncomment to run training automatically
    df = collect_mlflow_metrics()
    print(df)
    plot_metrics(df)
