"""
Utility helper functions for the Bank Marketing Decision Tree project.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime


def load_pickle(path: str):
    """Load any pickled object."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    """Save any object as pickle."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_column_stats(df: pd.DataFrame) -> dict:
    """Return basic stats for each column."""
    stats = {}
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique()),
        }
        if df[col].dtype in [np.float64, np.int64]:
            col_stats.update({
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            })
        stats[col] = col_stats
    return stats


def predict_single(clf, feature_values: dict, feature_order: list) -> dict:
    """
    Predict for a single record.
    feature_values: dict mapping feature name -> encoded value
    feature_order: list of feature names in training order
    """
    row = np.array([[feature_values.get(f, 0) for f in feature_order]])
    pred = clf.predict(row)[0]
    proba = clf.predict_proba(row)[0]
    return {
        "prediction": int(pred),
        "label": "Yes (Will Subscribe)" if pred == 1 else "No (Won't Subscribe)",
        "probability_no": round(float(proba[0]), 4),
        "probability_yes": round(float(proba[1]), 4),
    }


def encode_input(raw_input: dict, encoders: dict) -> dict:
    """Encode raw string inputs using stored label encoders."""
    encoded = {}
    for k, v in raw_input.items():
        if k in encoders:
            try:
                encoded[k] = int(encoders[k].transform([str(v)])[0])
            except Exception:
                encoded[k] = 0
        else:
            try:
                encoded[k] = float(v)
            except Exception:
                encoded[k] = 0
    return encoded


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
