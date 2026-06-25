"""
visualization.py - Generate charts and visualizations for sentiment analysis
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter


COLORS = {
    'Positive': '#22c55e',
    'Negative': '#ef4444',
    'Neutral':  '#f59e0b',
}
BG_COLOR = '#0f172a'
CARD_COLOR = '#1e293b'
TEXT_COLOR = '#f1f5f9'
GRID_COLOR = '#334155'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': CARD_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'grid.color': GRID_COLOR,
    'font.family': 'DejaVu Sans',
})


def _sentiment_colors(labels):
    return [COLORS.get(l, '#94a3b8') for l in labels]


def plot_sentiment_distribution(df, output_path):
    """Pie + bar chart of sentiment distribution."""
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    counts = df[sentiment_col].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    colors = _sentiment_colors(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': TEXT_COLOR, 'fontsize': 11},
        wedgeprops={'edgecolor': BG_COLOR, 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color('#ffffff')
    ax1.set_title('Sentiment Distribution', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)

    # Bar chart
    bars = ax2.bar(labels, sizes, color=colors, edgecolor=BG_COLOR, linewidth=2, width=0.5)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax2.set_title('Count by Sentiment', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel('Number of Posts', color=TEXT_COLOR)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_platform_sentiment(df, output_path):
    """Stacked bar chart of sentiment per platform."""
    if 'platform' not in df.columns:
        return None
    
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    pivot = df.groupby(['platform', sentiment_col]).size().unstack(fill_value=0)
    
    for s in ['Positive', 'Negative', 'Neutral']:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[['Positive', 'Neutral', 'Negative']]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    bottom = np.zeros(len(pivot))
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        vals = pivot[sentiment].values
        bars = ax.bar(pivot.index, vals, bottom=bottom,
                      color=COLORS[sentiment], label=sentiment,
                      edgecolor=BG_COLOR, linewidth=1.5)
        for bar, v, b in zip(bars, vals, bottom):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2., b + v/2.,
                        str(v), ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
        bottom += vals

    ax.set_title('Sentiment by Platform', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Platform', color=TEXT_COLOR)
    ax.set_ylabel('Post Count', color=TEXT_COLOR)
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_brand_sentiment(df, output_path):
    """Grouped bar chart by brand/entity."""
    entity_col = None
    for c in ['entity', 'brand', 'topic']:
        if c in df.columns:
            entity_col = c
            break
    if entity_col is None:
        return None

    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    pivot = df.groupby([entity_col, sentiment_col]).size().unstack(fill_value=0)

    for s in ['Positive', 'Negative', 'Neutral']:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[['Positive', 'Neutral', 'Negative']]

    x = np.arange(len(pivot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_COLOR)

    for i, sentiment in enumerate(['Positive', 'Neutral', 'Negative']):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, pivot[sentiment].values, width,
                      label=sentiment, color=COLORS[sentiment],
                      edgecolor=BG_COLOR, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha='right')
    ax.set_title('Sentiment by Brand / Entity', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Post Count', color=TEXT_COLOR)
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_wordcloud(df, sentiment_filter, output_path):
    """Word cloud for a specific sentiment."""
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    
    subset = df[df[sentiment_col] == sentiment_filter]
    if len(subset) == 0:
        return None
    
    all_text = ' '.join(subset[text_col].dropna().tolist())
    words = [w for w in all_text.split() if len(w) > 2]
    word_freq = Counter(words)
    top_words = word_freq.most_common(30)
    
    if not top_words:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Manual word layout
    base_color = COLORS.get(sentiment_filter, '#94a3b8')
    max_freq = top_words[0][1] if top_words else 1
    
    np.random.seed(42)
    x_positions = np.random.uniform(0.05, 0.95, len(top_words))
    y_positions = np.random.uniform(0.1, 0.9, len(top_words))

    for (word, freq), x, y in zip(top_words, x_positions, y_positions):
        size = 8 + int((freq / max_freq) * 28)
        alpha = 0.5 + 0.5 * (freq / max_freq)
        ax.text(x, y, word, fontsize=size, color=base_color, alpha=alpha,
                ha='center', va='center', fontweight='bold',
                transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Top Words — {sentiment_filter} Posts',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_hashtag_analysis(df, output_path):
    """Top hashtags bar chart."""
    hashtag_col = 'hashtags' if 'hashtags' in df.columns else None
    text_col = 'text' if 'text' in df.columns else None
    
    import re
    all_tags = []
    
    if hashtag_col:
        for val in df[hashtag_col].dropna():
            tags = re.findall(r'#?(\w+)', str(val))
            all_tags.extend([t.lower() for t in tags if len(t) > 1])
    elif text_col:
        for val in df[text_col].dropna():
            tags = re.findall(r'#(\w+)', str(val))
            all_tags.extend([t.lower() for t in tags])
    
    if not all_tags:
        return None
    
    top_tags = Counter(all_tags).most_common(15)
    tags, counts = zip(*top_tags)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    
    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(tags)))
    bars = ax.barh(tags, counts, color=colors, edgecolor=BG_COLOR, linewidth=1.5)
    
    for bar, val in zip(bars, counts):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2.,
                str(val), va='center', color=TEXT_COLOR, fontsize=10)
    
    ax.set_title('Top Hashtags', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency', color=TEXT_COLOR)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_engagement_by_sentiment(df, output_path):
    """Box plot of likes/retweets by sentiment."""
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    metric_cols = [c for c in ['likes', 'retweets'] if c in df.columns]
    
    if not metric_cols:
        return None
    
    metric = metric_cols[0]
    sentiments = ['Positive', 'Neutral', 'Negative']
    data = [df[df[sentiment_col] == s][metric].dropna().tolist() for s in sentiments]
    data = [d if d else [0] for d in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))
    
    for patch, sentiment in zip(bp['boxes'], sentiments):
        patch.set_facecolor(COLORS[sentiment])
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(sentiments)
    ax.set_title(f'Engagement ({metric.title()}) by Sentiment',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel(metric.title(), color=TEXT_COLOR)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def plot_sentiment_over_time(df, output_path):
    """Line chart of sentiment trends over time."""
    if 'timestamp' not in df.columns and 'date' not in df.columns:
        return None
    
    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'predicted_sentiment'
    time_col = 'date' if 'date' in df.columns else 'timestamp'
    
    df[time_col] = pd.to_datetime(df[time_col])
    daily = df.groupby([df[time_col].dt.date, sentiment_col]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in daily.columns:
            ax.plot(daily.index, daily[sentiment],
                    label=sentiment, color=COLORS[sentiment],
                    linewidth=2.5, marker='o', markersize=5)
    
    ax.set_title('Sentiment Trends Over Time', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Date', color=TEXT_COLOR)
    ax.set_ylabel('Post Count', color=TEXT_COLOR)
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    return output_path


def generate_all_visuals(df, visuals_dir):
    """Generate all visualizations and return paths."""
    os.makedirs(visuals_dir, exist_ok=True)
    generated = {}

    paths = [
        ('sentiment_distribution', plot_sentiment_distribution,
         os.path.join(visuals_dir, 'sentiment_distribution.png')),
        ('platform_sentiment', plot_platform_sentiment,
         os.path.join(visuals_dir, 'platform_sentiment.png')),
        ('brand_sentiment', plot_brand_sentiment,
         os.path.join(visuals_dir, 'brand_sentiment.png')),
        ('wordcloud_positive', lambda df, p: plot_wordcloud(df, 'Positive', p),
         os.path.join(visuals_dir, 'wordcloud_positive.png')),
        ('wordcloud_negative', lambda df, p: plot_wordcloud(df, 'Negative', p),
         os.path.join(visuals_dir, 'wordcloud_negative.png')),
        ('hashtag_analysis', plot_hashtag_analysis,
         os.path.join(visuals_dir, 'hashtag_analysis.png')),
        ('engagement', plot_engagement_by_sentiment,
         os.path.join(visuals_dir, 'engagement_sentiment.png')),
        ('trends', plot_sentiment_over_time,
         os.path.join(visuals_dir, 'topic_trends.png')),
    ]

    for key, func, path in paths:
        try:
            result = func(df, path)
            if result:
                generated[key] = path
        except Exception as e:
            print(f"Warning: Could not generate {key}: {e}")

    return generated


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_path = os.path.join(base, "data", "processed", "cleaned_data.csv")
    visuals_dir = os.path.join(base, "visuals")
    
    df = pd.read_csv(cleaned_path)
    paths = generate_all_visuals(df, visuals_dir)
    print("Generated visuals:", list(paths.keys()))
