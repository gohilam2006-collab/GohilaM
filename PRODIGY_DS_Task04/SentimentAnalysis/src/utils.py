"""
utils.py - Helper utility functions
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(*dirs):
    """Ensure directories exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_df_safe(path, fallback_path=None):
    """Load CSV safely with fallback."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if fallback_path and os.path.exists(fallback_path):
        try:
            return pd.read_csv(fallback_path)
        except Exception as e:
            print(f"Error loading fallback {fallback_path}: {e}")
    return pd.DataFrame()


def get_summary_stats(df):
    """Get summary statistics from dataframe."""
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    
    stats = {
        'total_posts': len(df),
        'platforms': df['platform'].nunique() if 'platform' in df.columns else 0,
        'brands': df['entity'].nunique() if 'entity' in df.columns else 0,
        'sentiment_dist': {},
        'avg_likes': 0,
        'avg_retweets': 0,
        'date_range': '',
        'top_platform': '',
        'top_brand': '',
    }
    
    if sentiment_col in df.columns:
        counts = df[sentiment_col].value_counts()
        total = len(df)
        stats['sentiment_dist'] = {
            k: {'count': int(v), 'pct': round(v/total*100, 1)}
            for k, v in counts.items()
        }
    
    if 'likes' in df.columns:
        stats['avg_likes'] = round(df['likes'].mean(), 1)
    if 'retweets' in df.columns:
        stats['avg_retweets'] = round(df['retweets'].mean(), 1)
    
    if 'timestamp' in df.columns or 'date' in df.columns:
        col = 'date' if 'date' in df.columns else 'timestamp'
        try:
            dates = pd.to_datetime(df[col])
            stats['date_range'] = f"{dates.min().strftime('%b %d')} – {dates.max().strftime('%b %d, %Y')}"
        except Exception:
            pass
    
    if 'platform' in df.columns:
        stats['top_platform'] = df['platform'].value_counts().index[0]
    if 'entity' in df.columns:
        stats['top_brand'] = df['entity'].value_counts().index[0]
    
    return stats


def df_to_records(df, max_rows=200):
    """Convert dataframe to JSON-serializable records."""
    cols = [c for c in [
        'textID', 'text', 'sentiment', 'platform', 'entity',
        'timestamp', 'user', 'likes', 'retweets', 'country',
        'hashtags', 'predicted_sentiment', 'vader_compound',
        'textblob_polarity', 'textblob_subjectivity', 'confidence'
    ] if c in df.columns]
    
    subset = df[cols].head(max_rows).fillna('')
    records = subset.to_dict(orient='records')
    # Convert non-serializable types
    for r in records:
        for k, v in r.items():
            if hasattr(v, 'item'):
                r[k] = v.item()
    return records


def file_hash(path):
    """Get MD5 hash of file for cache busting."""
    if not os.path.exists(path):
        return '0'
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
