"""
app.py - Flask web application for Social Media Sentiment Analysis Dashboard
"""

import os
import sys
import json
import base64
import traceback
import pandas as pd
from datetime import datetime
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, redirect, url_for, flash)
from werkzeug.utils import secure_filename

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from data_cleaning import load_and_clean
from sentiment_model import analyze_dataframe, evaluate_model
from visualization import generate_all_visuals
from utils import get_summary_stats, df_to_records, file_hash, timestamp_now

app = Flask(__name__)
app.secret_key = 'sentiment_analysis_secret_2024'

# Config
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'data', 'processed')
VISUALS_FOLDER = os.path.join(BASE_DIR, 'visuals')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

for d in [UPLOAD_FOLDER, PROCESSED_FOLDER, VISUALS_FOLDER,
          os.path.join(BASE_DIR, 'models')]:
    os.makedirs(d, exist_ok=True)

# Cache
_cache = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_cleaned_df():
    """Load or build the cleaned+analyzed dataframe."""
    cleaned_path = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
    raw_path = os.path.join(UPLOAD_FOLDER, 'social_media_data.csv')
    
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        if 'predicted_sentiment' not in df.columns:
            df = analyze_dataframe(df)
            df.to_csv(cleaned_path, index=False)
        return df
    
    if os.path.exists(raw_path):
        df = load_and_clean(raw_path)
        df = analyze_dataframe(df)
        df.to_csv(cleaned_path, index=False)
        return df
    
    return pd.DataFrame()


def run_full_pipeline(raw_path):
    """Run the complete analysis pipeline."""
    cleaned_path = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
    
    df = load_and_clean(raw_path)
    df = analyze_dataframe(df)
    df.to_csv(cleaned_path, index=False)
    
    visuals = generate_all_visuals(df, VISUALS_FOLDER)
    
    metrics = evaluate_model(df)
    model_info_path = os.path.join(BASE_DIR, 'models', 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Clear cache
    _cache.clear()
    
    return df, visuals, metrics


def get_visuals_with_data():
    """Get visual images as base64 encoded data."""
    visual_files = {
        'sentiment_distribution': 'sentiment_distribution.png',
        'platform_sentiment': 'platform_sentiment.png',
        'brand_sentiment': 'brand_sentiment.png',
        'wordcloud_positive': 'wordcloud_positive.png',
        'wordcloud_negative': 'wordcloud_negative.png',
        'hashtag_analysis': 'hashtag_analysis.png',
        'engagement': 'engagement_sentiment.png',
        'trends': 'topic_trends.png',
    }
    
    result = {}
    for key, fname in visual_files.items():
        path = os.path.join(VISUALS_FOLDER, fname)
        if os.path.exists(path):
            result[key] = f'/visuals/{fname}?v={file_hash(path)}'
    
    return result


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    df = get_cleaned_df()
    stats = {}
    visuals = {}
    records = []
    metrics = {}
    
    if len(df) > 0:
        stats = get_summary_stats(df)
        visuals = get_visuals_with_data()
        records = df_to_records(df, max_rows=100)
        
        model_info_path = os.path.join(BASE_DIR, 'models', 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path) as f:
                metrics = json.load(f)
    
    # Check if visuals exist, generate if not
    if len(df) > 0 and not visuals:
        try:
            generate_all_visuals(df, VISUALS_FOLDER)
            visuals = get_visuals_with_data()
        except Exception as e:
            print(f"Could not generate visuals: {e}")
    
    return render_template('index.html',
                           stats=stats,
                           visuals=visuals,
                           records=records,
                           metrics=metrics,
                           has_data=len(df) > 0,
                           last_updated=timestamp_now())


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        try:
            df, visuals, metrics = run_full_pipeline(save_path)
            flash(f'✅ Analysis complete! Processed {len(df)} posts.', 'success')
        except Exception as e:
            flash(f'❌ Error during analysis: {str(e)}', 'error')
            traceback.print_exc()
    else:
        flash('Please upload a CSV file', 'error')
    
    return redirect(url_for('index'))


@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze a single text input."""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    from data_cleaning import clean_text
    from sentiment_model import (get_vader_analyzer, analyze_vader, vader_label,
                                  analyze_textblob)
    
    cleaned = clean_text(text)
    
    # VADER
    analyzer = get_vader_analyzer()
    vader = analyze_vader(cleaned, analyzer)
    vader_sent = vader_label(vader['compound'])
    
    # TextBlob
    tb = analyze_textblob(cleaned)
    
    # Final
    final = vader_sent if analyzer else tb['label']
    
    return jsonify({
        'original_text': text,
        'cleaned_text': cleaned,
        'sentiment': final,
        'vader_compound': round(vader['compound'], 4),
        'vader_scores': {k: round(v, 4) for k, v in vader.items()},
        'textblob_polarity': round(tb['polarity'], 4),
        'textblob_subjectivity': round(tb['subjectivity'], 4),
        'confidence': round(abs(vader['compound']) if analyzer else abs(tb['polarity']), 4)
    })


@app.route('/api/data')
def api_data():
    """Return analyzed data as JSON."""
    df = get_cleaned_df()
    if len(df) == 0:
        return jsonify({'error': 'No data available'}), 404
    
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    
    # Filters
    sentiment_filter = request.args.get('sentiment')
    platform_filter = request.args.get('platform')
    entity_filter = request.args.get('entity')
    
    if sentiment_filter:
        df = df[df[sentiment_col] == sentiment_filter]
    if platform_filter and 'platform' in df.columns:
        df = df[df['platform'] == platform_filter]
    if entity_filter and 'entity' in df.columns:
        df = df[df['entity'] == entity_filter]
    
    return jsonify({
        'total': len(df),
        'records': df_to_records(df, max_rows=500)
    })


@app.route('/api/stats')
def api_stats():
    df = get_cleaned_df()
    if len(df) == 0:
        return jsonify({'error': 'No data'}), 404
    return jsonify(get_summary_stats(df))


@app.route('/api/regenerate', methods=['POST'])
def regenerate_visuals():
    """Regenerate all visualizations."""
    df = get_cleaned_df()
    if len(df) == 0:
        return jsonify({'error': 'No data available'}), 404
    try:
        visuals = generate_all_visuals(df, VISUALS_FOLDER)
        return jsonify({'success': True, 'generated': list(visuals.keys())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/visuals/<filename>')
def serve_visual(filename):
    return send_from_directory(VISUALS_FOLDER, filename)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': timestamp_now()})


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Social Media Sentiment Analysis Dashboard")
    print("="*55)
    print(f"  → Open your browser: http://localhost:5000")
    print("="*55 + "\n")
    
    # Run initial pipeline if data exists but no processed output
    raw_path = os.path.join(UPLOAD_FOLDER, 'social_media_data.csv')
    cleaned_path = os.path.join(PROCESSED_FOLDER, 'cleaned_data.csv')
    
    if os.path.exists(raw_path) and not os.path.exists(cleaned_path):
        print("Running initial pipeline...")
        try:
            run_full_pipeline(raw_path)
            print("Initial pipeline complete.")
        except Exception as e:
            print(f"Pipeline warning: {e}")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
