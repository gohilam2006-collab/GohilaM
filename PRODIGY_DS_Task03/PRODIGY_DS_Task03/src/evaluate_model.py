"""
Evaluate Decision Tree model and generate visuals.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import plot_tree
import os


def evaluate(clf, X_test, y_test) -> dict:
    """Return evaluation metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[INFO] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return {"accuracy": acc, "report": report, "cm": cm, "y_pred": y_pred}


def plot_confusion_matrix(cm, save_path: str):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


def plot_decision_tree(clf, feature_names, save_path: str):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf, feature_names=feature_names,
        class_names=['No', 'Yes'], filled=True,
        rounded=True, fontsize=9, ax=ax
    )
    ax.set_title('Decision Tree Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Decision tree plot saved to {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap='coolwarm', ax=ax, linewidths=0.5,
        annot_kws={"size": 8}
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Heatmap saved to {save_path}")


def plot_feature_importance(clf, feature_names, save_path: str):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    ax.bar(range(len(feature_names)), importances[indices], color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances', fontsize=14, fontweight='bold')
    ax.set_ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance plot saved to {save_path}")
