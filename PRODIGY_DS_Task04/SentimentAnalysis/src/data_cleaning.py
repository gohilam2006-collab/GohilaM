"""
data_cleaning.py - Text cleaning and preprocessing for social media data
"""

import re
import pandas as pd
import string
import os

# Optional NLTK import
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    for resource in ['stopwords', 'punkt', 'wordnet', 'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)


def remove_emojis(text):
    """Remove emojis and special unicode characters."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', str(text))


def remove_mentions_hashtag_symbols(text):
    """Remove @ mentions and # symbols (keep hashtag text)."""
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'#', '', str(text))
    return text


def remove_punctuation(text):
    """Remove punctuation."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    """Remove standalone numbers."""
    return re.sub(r'\b\d+\b', '', str(text))


def remove_extra_spaces(text):
    """Remove extra whitespace."""
    return re.sub(r'\s+', ' ', str(text)).strip()


def remove_stopwords_simple(text):
    """Remove common stopwords using built-in list."""
    words = text.split()
    return ' '.join([w for w in words if w.lower() not in STOP_WORDS])


def clean_text(text):
    """Full text cleaning pipeline."""
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_mentions_hashtag_symbols(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords_simple(text)
    text = remove_extra_spaces(text)
    return text


def extract_hashtags(text):
    """Extract hashtags from original text."""
    hashtags = re.findall(r'#(\w+)', str(text))
    return hashtags


def load_and_clean(filepath):
    """Load CSV and apply full cleaning pipeline."""
    df = pd.read_csv(filepath)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Extract hashtags if not already present
    if 'hashtags' not in df.columns or df['hashtags'].isna().all():
        df['hashtags'] = df['text'].apply(lambda x: ' '.join(extract_hashtags(x)))
    
    # Standardize sentiment labels
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
        df['sentiment'] = df['sentiment'].replace({
            'Pos': 'Positive', 'Neg': 'Negative', 'Neu': 'Neutral',
            '4': 'Positive', '0': 'Negative', '2': 'Neutral'
        })
    
    # Parse timestamps if present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()
        except Exception:
            pass
    
    # Drop rows with empty cleaned text
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base, "data", "raw", "social_media_data.csv")
    out_path = os.path.join(base, "data", "processed", "cleaned_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = load_and_clean(raw_path)
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to {out_path}")
    print(df[['text', 'cleaned_text', 'sentiment']].head())
