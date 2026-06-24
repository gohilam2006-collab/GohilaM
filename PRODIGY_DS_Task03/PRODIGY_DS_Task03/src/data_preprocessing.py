"""
Data Preprocessing for Bank Marketing Dataset
Handles cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath, sep=',')
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and handle missing/unknown values."""
    df = df.drop_duplicates()
    # Replace 'unknown' strings with NaN for categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].replace('unknown', np.nan)
    # Fill NaN categoricals with mode
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
    print(f"[INFO] After cleaning: {len(df)} rows.")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[INFO] Encoded columns: {list(cat_cols)}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features."""
    df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
    df['high_balance'] = (df['balance'] > df['balance'].median()).astype(int)
    df['long_duration'] = (df['duration'] > 180).astype(int)
    return df


def preprocess_pipeline(raw_path: str, save_path: str) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved cleaned data to {save_path}")
    return df


if __name__ == "__main__":
    preprocess_pipeline(
        raw_path="data/raw/bank.csv",
        save_path="data/processed/cleaned_data.csv"
    )
