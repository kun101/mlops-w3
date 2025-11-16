#!/usr/bin/env python3
"""
poison_data.py
Generate poisoned variations of the Iris dataset.

Usage:
  python poison_data.py --outdir data/poisoned --seed 42
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle

def load_iris_df():
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.copy()
    # Ensure consistent column names
    df = df.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
        'target': 'target'
    })
    # If target is numeric, keep numeric
    return df

def poison_features(df, fraction, seed=None):
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(seed)
    k = int(np.round(fraction * n))
    if k == 0:
        return df
    idx = rng.choice(n, size=k, replace=False)
    # For each feature, sample from uniform range of that column
    for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        lo = df[col].min()
        hi = df[col].max()
        df.loc[df.index[idx], col] = rng.uniform(lo, hi, size=k)
    return df

def flip_labels(df, fraction, seed=None):
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(seed)
    k = int(np.round(fraction * n))
    if k == 0:
        return df
    idx = rng.choice(n, size=k, replace=False)
    unique_labels = sorted(df['target'].unique())
    for i in idx:
        current = df.at[i, 'target']
        choices = [l for l in unique_labels if l != current]
        df.at[i, 'target'] = rng.choice(choices)
    return df

def no_change(df, fraction, seed=None):
    return df

methods = {
    'clean': no_change,
    'feature_noise': poison_features,
    'label_flip': flip_labels
}

def main(outdir, seed=42):
    os.makedirs(outdir, exist_ok=True)
    df = load_iris_df()
    levels = [0.0, 0.05, 0.10, 0.50]

    for meth_name, func in methods.items():
        for lvl in levels:
            out = func(df, lvl, seed=seed)   # seed now works for all
            frac_str = str(int(lvl*100)).zfill(2)
            fname = f"iris_{meth_name}_{frac_str}.csv"
            path = os.path.join(outdir, fname)
            out.to_csv(path, index=False)
            print(f"Wrote {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/poisoned", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.outdir, args.seed)
