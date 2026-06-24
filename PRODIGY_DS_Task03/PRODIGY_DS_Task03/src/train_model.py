"""
Train Decision Tree Classifier on Bank Marketing Dataset.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder


def prepare_features(df: pd.DataFrame):
    """Split into features X and target y."""
    target = 'y'
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train(X_train, y_train, max_depth=5, random_state=42) -> DecisionTreeClassifier:
    """Train and return a DecisionTreeClassifier."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=3,
        criterion='gini',
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    print(f"[INFO] Model trained. Tree depth: {clf.get_depth()}")
    return clf


def save_model(clf, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str) -> DecisionTreeClassifier:
    with open(path, 'rb') as f:
        clf = pickle.load(f)
    return clf


def train_pipeline(cleaned_path: str, model_path: str):
    """Full training pipeline."""
    df = pd.read_csv(cleaned_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = train(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=min(5, len(df) // 5), scoring='accuracy')
    print(f"[INFO] CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    save_model(clf, model_path)
    return clf, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_pipeline(
        cleaned_path="data/processed/cleaned_data.csv",
        model_path="models/decision_tree_model.pkl"
    )
