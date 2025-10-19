# Iris Classification Pipeline with DVC and CI

This repository demonstrates a complete MLOps pipeline for the **Iris classification task**, integrating **DVC for data & model versioning** and **GitHub Actions for continuous integration**.  

## Repository Structure

- **`artifacts/`**  
  Stores the trained model files (e.g., `model.joblib`) tracked by DVC.

- **`data.dvc`**  
  DVC-tracked pointer to the updated Iris dataset (`v2`).

- **`data/`**  
  Local copy of the dataset pulled via DVC.

- **`test_pipeline.py`**  
  Unit tests for:
  - Data validation (columns, row count)
  - Model evaluation (predictions and accuracy threshold)

- **`.dvc/`**  
  DVC metadata and configuration directory, tracking data versions and pipeline stages.

- **`.dvcignore`**  
  Files/folders ignored by DVC.

- **`.gitignore`**  
  Files/folders ignored by Git (e.g., virtual environments, temporary files).

- **`.github/workflows/ci.yaml`**  
  GitHub Actions workflow for CI:
  - Pulls data and models from DVC
  - Runs unit tests
  - Posts results using CML
