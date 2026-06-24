"""
PRODIGY_DS_Task02 – Titanic EDA Web App
Run: python app.py  →  open http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source_code'))

from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

# Load data once at startup
from eda import load_and_clean, get_summary_stats
from eda import (chart_survival_by_gender, chart_age_distribution,
                 chart_pclass_analysis, chart_correlation_heatmap,
                 chart_family_embark)

CSV_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'titanic.csv')
df, cleaning_log = load_and_clean(CSV_PATH)
stats = get_summary_stats(df)


@app.route('/')
def index():
    return render_template('index.html', stats=stats)


@app.route('/api/cleaning-log')
def api_cleaning_log():
    return jsonify(cleaning_log)


@app.route('/api/stats')
def api_stats():
    return jsonify(stats)


@app.route('/api/chart/gender')
def api_gender():
    return jsonify({'image': chart_survival_by_gender(df)})


@app.route('/api/chart/age')
def api_age():
    return jsonify({'image': chart_age_distribution(df)})


@app.route('/api/chart/pclass')
def api_pclass():
    return jsonify({'image': chart_pclass_analysis(df)})


@app.route('/api/chart/heatmap')
def api_heatmap():
    return jsonify({'image': chart_correlation_heatmap(df)})


@app.route('/api/chart/family')
def api_family():
    return jsonify({'image': chart_family_embark(df)})


@app.route('/api/data-preview')
def api_preview():
    preview = df.head(20).to_dict(orient='records')
    columns = list(df.columns)
    return jsonify({'columns': columns, 'rows': preview})


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🚢 PRODIGY DS Task-02 – Titanic EDA Dashboard")
    print("  ➜  Open  http://localhost:5000  in your browser")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
