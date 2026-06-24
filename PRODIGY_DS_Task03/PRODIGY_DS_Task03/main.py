"""
main.py — Run the full Bank Marketing Decision Tree pipeline.
Usage: python main.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import preprocess_pipeline
from src.train_model import train_pipeline, load_model
from src.evaluate_model import (
    evaluate,
    plot_confusion_matrix,
    plot_decision_tree,
    plot_correlation_heatmap,
    plot_feature_importance,
)

RAW_DATA    = "data/raw/bank.csv"
CLEAN_DATA  = "data/processed/cleaned_data.csv"
MODEL_PATH  = "models/decision_tree_model.pkl"
VISUALS_DIR = "visuals"


def run():
    print("\n" + "="*60)
    print("  Bank Marketing — Decision Tree Classifier Pipeline")
    print("="*60 + "\n")

    # 1. Preprocess
    print("[STEP 1] Preprocessing data...")
    df = preprocess_pipeline(RAW_DATA, CLEAN_DATA)

    # 2. Train
    print("\n[STEP 2] Training Decision Tree...")
    clf, X_train, X_test, y_train, y_test = train_pipeline(CLEAN_DATA, MODEL_PATH)

    # 3. Evaluate
    print("\n[STEP 3] Evaluating model...")
    results = evaluate(clf, X_test, y_test)

    # 4. Generate visuals
    print("\n[STEP 4] Generating visuals...")
    os.makedirs(VISUALS_DIR, exist_ok=True)
    plot_confusion_matrix(
        results["cm"],
        os.path.join(VISUALS_DIR, "confusion_matrix.png")
    )
    plot_decision_tree(
        clf, list(X_test.columns),
        os.path.join(VISUALS_DIR, "decision_tree_plot.png")
    )
    plot_correlation_heatmap(
        df,
        os.path.join(VISUALS_DIR, "correlation_heatmap.png")
    )
    plot_feature_importance(
        clf, list(X_test.columns),
        os.path.join(VISUALS_DIR, "feature_importance.png")
    )

    print("\n" + "="*60)
    print(f"  ✅ Pipeline complete! Accuracy: {results['accuracy']:.4f}")
    print("  Visuals saved in /visuals/")
    print("  Model saved in /models/")
    print("="*60 + "\n")
    print("  Run the web UI with:  python app.py")
    print("  Then open:            http://localhost:5000\n")


if __name__ == "__main__":
    run()
