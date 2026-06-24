"""
app.py — Flask Web UI for Bank Marketing Decision Tree Classifier
Run: python app.py  →  http://localhost:5000
"""

import os
import sys
import json
import pickle
import base64
import numpy as np
import pandas as pd
from io import BytesIO

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

MODEL_PATH = "models/decision_tree_model.pkl"
CLEAN_DATA = "data/processed/cleaned_data.csv"
VISUALS_DIR = "visuals"

# ─── Lazy model / data loading ───────────────────────────────────────────────

_clf = None
_feature_names = None
_df = None
_trained = False


def ensure_trained():
    """Auto-train if model doesn't exist yet."""
    global _clf, _feature_names, _df, _trained
    if _trained:
        return

    if not os.path.exists(MODEL_PATH):
        print("[APP] Model not found. Running pipeline first...")
        from src.data_preprocessing import preprocess_pipeline
        from src.train_model import train_pipeline
        from src.evaluate_model import (
            plot_confusion_matrix, plot_decision_tree,
            plot_correlation_heatmap, plot_feature_importance, evaluate
        )
        df = preprocess_pipeline("data/raw/bank.csv", CLEAN_DATA)
        clf, X_train, X_test, y_train, y_test = train_pipeline(CLEAN_DATA, MODEL_PATH)
        results = evaluate(clf, X_test, y_test)
        os.makedirs(VISUALS_DIR, exist_ok=True)
        plot_confusion_matrix(results["cm"], f"{VISUALS_DIR}/confusion_matrix.png")
        plot_decision_tree(clf, list(X_test.columns), f"{VISUALS_DIR}/decision_tree_plot.png")
        plot_correlation_heatmap(df, f"{VISUALS_DIR}/correlation_heatmap.png")
        plot_feature_importance(clf, list(X_test.columns), f"{VISUALS_DIR}/feature_importance.png")
        print("[APP] Pipeline complete.")

    with open(MODEL_PATH, 'rb') as f:
        _clf = pickle.load(f)

    if os.path.exists(CLEAN_DATA):
        _df = pd.read_csv(CLEAN_DATA)
        _feature_names = [c for c in _df.columns if c != 'y']
    else:
        _feature_names = []

    _trained = True


def get_model_stats():
    ensure_trained()
    if _df is None or _clf is None:
        return {}
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    X = _df.drop(columns=['y'])
    y = _df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = _clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(report.get('weighted avg', {}).get('precision', 0) * 100, 2),
        "recall": round(report.get('weighted avg', {}).get('recall', 0) * 100, 2),
        "f1": round(report.get('weighted avg', {}).get('f1-score', 0) * 100, 2),
        "tree_depth": _clf.get_depth(),
        "n_leaves": _clf.get_n_leaves(),
        "n_features": len(_feature_names),
        "n_samples": len(_df),
    }


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    ensure_trained()
    stats = get_model_stats()
    visuals = []
    if os.path.exists(VISUALS_DIR):
        visuals = [f for f in os.listdir(VISUALS_DIR) if f.endswith('.png')]
    return render_template('index.html', stats=stats, visuals=visuals)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ensure_trained()
    result = None
    if request.method == 'POST':
        try:
            # Categorical mappings (must match LabelEncoder order from preprocessing)
            job_map = {
                'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
                'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
                'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11
            }
            marital_map = {'divorced': 0, 'married': 1, 'single': 2}
            education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
            binary_map = {'no': 0, 'yes': 1}
            contact_map = {'cellular': 0, 'telephone': 1, 'unknown': 2}
            month_map = {
                'apr': 0, 'aug': 1, 'dec': 2, 'feb': 3, 'jan': 4,
                'jul': 5, 'jun': 6, 'mar': 7, 'may': 8, 'nov': 9, 'oct': 10, 'sep': 11
            }
            poutcome_map = {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3}

            age = int(request.form.get('age', 35))
            job = request.form.get('job', 'management')
            marital = request.form.get('marital', 'married')
            education = request.form.get('education', 'secondary')
            default = request.form.get('default', 'no')
            balance = float(request.form.get('balance', 0))
            housing = request.form.get('housing', 'yes')
            loan = request.form.get('loan', 'no')
            contact = request.form.get('contact', 'unknown')
            day = int(request.form.get('day', 15))
            month = request.form.get('month', 'may')
            duration = int(request.form.get('duration', 200))
            campaign = int(request.form.get('campaign', 1))
            pdays = int(request.form.get('pdays', -1))
            previous = int(request.form.get('previous', 0))
            poutcome = request.form.get('poutcome', 'unknown')

            # Derived features
            was_contacted_before = 1 if pdays != -1 else 0
            high_balance = 1 if balance > 500 else 0
            long_duration = 1 if duration > 180 else 0

            features = np.array([[
                age,
                job_map.get(job, 4),
                marital_map.get(marital, 1),
                education_map.get(education, 1),
                binary_map.get(default, 0),
                balance,
                binary_map.get(housing, 1),
                binary_map.get(loan, 0),
                contact_map.get(contact, 2),
                day,
                month_map.get(month, 8),
                duration,
                campaign,
                pdays,
                previous,
                poutcome_map.get(poutcome, 3),
                was_contacted_before,
                high_balance,
                long_duration,
            ]])

            pred = _clf.predict(features)[0]
            proba = _clf.predict_proba(features)[0]

            result = {
                "prediction": int(pred),
                "label": "✅ Yes — Customer will subscribe!" if pred == 1 else "❌ No — Customer won't subscribe.",
                "prob_yes": round(float(proba[1]) * 100, 1),
                "prob_no": round(float(proba[0]) * 100, 1),
                "color": "#22c55e" if pred == 1 else "#ef4444",
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template('predict.html', result=result)


@app.route('/visuals')
def visuals_page():
    ensure_trained()
    imgs = []
    if os.path.exists(VISUALS_DIR):
        for fname in sorted(os.listdir(VISUALS_DIR)):
            if fname.endswith('.png'):
                label = fname.replace('.png', '').replace('_', ' ').title()
                imgs.append({"file": fname, "label": label})
    return render_template('visuals.html', imgs=imgs)


@app.route('/visuals/<filename>')
def visual_file(filename):
    return send_from_directory(VISUALS_DIR, filename)


@app.route('/data')
def data_page():
    ensure_trained()
    raw_df = pd.read_csv("data/raw/bank.csv")
    cols = list(raw_df.columns)
    rows = raw_df.head(20).values.tolist()
    shape = raw_df.shape
    return render_template('data.html', cols=cols, rows=rows, shape=shape)


@app.route('/api/stats')
def api_stats():
    return jsonify(get_model_stats())


@app.route('/api/retrain', methods=['POST'])
def retrain():
    global _clf, _feature_names, _df, _trained
    _trained = False
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    ensure_trained()
    return jsonify({"status": "ok", "stats": get_model_stats()})


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🌳 Bank Marketing Decision Tree — Web UI")
    print("="*55)
    print("  Starting server at http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
