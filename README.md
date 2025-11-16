# Poisoning Module – File Descriptions

This folder contains the code used to generate poisoned versions of the Iris dataset, train models on those datasets, and evaluate the effect of poisoning through MLflow-tracked experiments.

## 1. `poison_data.py`
Generates clean and poisoned variations of the Iris dataset.

### Purpose
To produce multiple dataset versions with controlled levels of corruption so we can study how poisoning affects model performance.

### What It Does
- Loads the original Iris dataset into a DataFrame.  
- Applies multiple poisoning strategies:
  - **Feature noise**: replaces selected rows’ numerical features with random values within column ranges.  
  - **Label flipping**: randomly reassigns class labels for a percentage of rows.  
  - **Clean data**: generates an unmodified baseline dataset.
- Creates datasets with poisoning levels: **0%, 5%, 10%, 50%**.
- Saves the resulting datasets as CSV files in a specified output directory.

### Outputs
CSV files such as:
iris_clean_00.csv
iris_feature_noise_05.csv
iris_label_flip_10.csv
iris_feature_noise_50.csv

---

## 2. `train_and_log.py`
Trains a model on a chosen poisoned dataset and logs the full experiment to MLflow.

### Purpose
To run a reproducible training job on any dataset variant and capture all experiment metadata for comparison.

### What It Does
- Loads a CSV produced by `poison_data.py`.  
- Performs a stratified train/test split.  
- Trains a **RandomForestClassifier**.  
- Evaluates on a clean test subset, computing:
  - Accuracy  
  - Macro Precision  
  - Macro Recall  
  - Confusion matrix  
- Logs the following to MLflow:
  - Model parameters  
  - Metrics  
  - Confusion matrix artifact  
  - Serialized model using `mlflow.sklearn.log_model`  

### Inputs
One poisoned dataset CSV.

### Outputs
A fully logged MLflow run stored under the selected experiment name.

---

## 3. `evaluate_all.py`
Collects results across all training runs and generates summary plots.

### Purpose
To compare how poisoning affects model performance across different methods and severity levels.

### What It Does
- Optionally runs training for all poisoned datasets.  
- Queries MLflow for all runs in the experiment.  
- Extracts metrics (accuracy, precision, recall) and associated parameters.  
- Saves a summary table of all results.  
- Produces plots comparing metrics versus poisoning levels for each poisoning method.

### Outputs
Located under `results/poisoning/`:

summary_metrics.csv
metrics_vs_poison.png


These outputs allow visual and quantitative analysis of how data poisoning impacts model robustness.