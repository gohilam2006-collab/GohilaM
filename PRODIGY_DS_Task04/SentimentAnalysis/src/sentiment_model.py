"""
sentiment_model.py - Sentiment analysis using VADER and TextBlob
"""

import pandas as pd
import os
import pickle
import json


def get_vader_analyzer():
    """Get VADER sentiment analyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except Exception:
            return None


def analyze_vader(text, analyzer):
    """Analyze sentiment with VADER."""
    if analyzer is None:
        return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
    scores = analyzer.polarity_scores(str(text))
    return scores


def vader_label(compound_score):
    """Convert compound score to label."""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_textblob(text):
    """Analyze sentiment with TextBlob."""
    try:
        from textblob import TextBlob
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        return {'polarity': polarity, 'subjectivity': subjectivity, 'label': label}
    except ImportError:
        return {'polarity': 0, 'subjectivity': 0.5, 'label': 'Neutral'}


def simple_keyword_sentiment(text):
    """Fallback keyword-based sentiment analysis."""
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'best', 'perfect', 'awesome', 'brilliant', 'outstanding',
        'superb', 'incredible', 'beautiful', 'happy', 'joy', 'pleased',
        'satisfied', 'delighted', 'impressive', 'exceptional', 'fabulous'
    ])
    negative_words = set([
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing',
        'poor', 'disappoints', 'fail', 'failure', 'broken', 'useless', 'waste',
        'problem', 'issue', 'angry', 'upset', 'frustrated', 'disgusting',
        'unacceptable', 'ridiculous', 'outrageous', 'scam', 'fraud', 'crashed'
    ])
    
    words = str(text).lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    score = (pos_count - neg_count) / max(len(words), 1)
    
    if score > 0.02:
        return {'label': 'Positive', 'score': score}
    elif score < -0.02:
        return {'label': 'Negative', 'score': score}
    else:
        return {'label': 'Neutral', 'score': score}


def analyze_dataframe(df, text_column='cleaned_text'):
    """Run sentiment analysis on full dataframe."""
    analyzer = get_vader_analyzer()
    use_vader = analyzer is not None
    
    # VADER analysis
    if use_vader:
        vader_results = df[text_column].apply(lambda x: analyze_vader(x, analyzer))
        df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
        df['vader_pos'] = vader_results.apply(lambda x: x['pos'])
        df['vader_neu'] = vader_results.apply(lambda x: x['neu'])
        df['vader_neg'] = vader_results.apply(lambda x: x['neg'])
        df['vader_sentiment'] = df['vader_compound'].apply(vader_label)
    
    # TextBlob analysis
    tb_results = df[text_column].apply(analyze_textblob)
    df['textblob_polarity'] = tb_results.apply(lambda x: x['polarity'])
    df['textblob_subjectivity'] = tb_results.apply(lambda x: x['subjectivity'])
    df['textblob_sentiment'] = tb_results.apply(lambda x: x['label'])
    
    # Final predicted sentiment (VADER preferred, else TextBlob, else keyword)
    if use_vader:
        df['predicted_sentiment'] = df['vader_sentiment']
    else:
        df['predicted_sentiment'] = df['textblob_sentiment']
    
    # Confidence score (0-1)
    if use_vader:
        df['confidence'] = df['vader_compound'].abs()
    else:
        df['confidence'] = df['textblob_polarity'].abs()
    
    return df


def evaluate_model(df, true_col='sentiment', pred_col='predicted_sentiment'):
    """Compute accuracy metrics if ground truth is available."""
    if true_col not in df.columns:
        return {}
    
    valid = df[[true_col, pred_col]].dropna()
    valid = valid[valid[true_col].isin(['Positive', 'Negative', 'Neutral'])]
    
    if len(valid) == 0:
        return {}
    
    accuracy = (valid[true_col] == valid[pred_col]).mean()
    
    metrics = {'accuracy': round(accuracy, 4), 'total_samples': len(valid)}
    
    # Per-class metrics
    for label in ['Positive', 'Negative', 'Neutral']:
        subset = valid[valid[true_col] == label]
        if len(subset) > 0:
            class_acc = (subset[true_col] == subset[pred_col]).mean()
            metrics[f'{label.lower()}_accuracy'] = round(class_acc, 4)
            metrics[f'{label.lower()}_count'] = len(subset)
    
    return metrics


def save_model_info(metrics, output_dir):
    """Save model evaluation info."""
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return info_path


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base, "data", "processed", "cleaned_data.csv")
    
    if not os.path.exists(cleaned_path):
        from data_cleaning import load_and_clean
        raw_path = os.path.join(base, "data", "raw", "social_media_data.csv")
        df = load_and_clean(raw_path)
        df.to_csv(cleaned_path, index=False)
    else:
        df = pd.read_csv(cleaned_path)
    
    df = analyze_dataframe(df)
    
    metrics = evaluate_model(df)
    print("Model Metrics:", metrics)
    
    save_model_info(metrics, os.path.join(base, "models"))
    df.to_csv(cleaned_path, index=False)
    print("Analysis complete. Results saved.")
