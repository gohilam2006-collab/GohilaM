# 📊 Social Media Sentiment Analysis Dashboard

A complete end-to-end pipeline for analyzing and visualizing public sentiment from social media data, with an interactive web dashboard.

## 🚀 Quick Start (Windows)

**Double-click `run.bat`** — it installs everything and opens the dashboard.

Then open your browser at: **http://localhost:5000**

---

## ✨ Features

- **Automatic Sentiment Analysis** using VADER + TextBlob NLP models
- **8 Interactive Charts**: distribution, trends over time, platform/brand breakdowns, word clouds, hashtag analysis, engagement metrics
- **Live Text Analyzer**: type any text and get instant sentiment predictions
- **CSV Upload**: drag & drop your own dataset
- **Data Explorer**: filterable table with search
- **REST API**: `/api/data`, `/api/stats`

---

## 📁 Project Structure

```
SentimentAnalysis/
├── data/
│   ├── raw/                     # Input CSV files
│   └── processed/               # Cleaned + analyzed data
├── src/
│   ├── data_cleaning.py         # Text preprocessing
│   ├── sentiment_model.py       # VADER + TextBlob analysis
│   ├── visualization.py         # Chart generation (matplotlib)
│   └── utils.py                 # Helper functions
├── templates/
│   └── index.html               # Dashboard UI
├── visuals/                     # Generated chart PNGs
├── models/                      # Model metadata
├── app.py                       # Flask web server
├── main.py                      # CLI pipeline runner
├── requirements.txt
└── run.bat                      # Windows launcher
```

---

## 📋 CSV Format

Your CSV should have these columns (all optional except `text`):

| Column | Description |
|--------|-------------|
| `text` | The post content (required) |
| `sentiment` | Ground truth label: Positive / Negative / Neutral |
| `platform` | Twitter / Facebook / Instagram |
| `entity` | Brand or topic name |
| `timestamp` | Date/time of post |
| `likes` | Engagement count |
| `retweets` | Share count |
| `country` | Post origin |
| `hashtags` | Associated hashtags |

---

## 🛠️ Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline + start server
python main.py --serve

# Or run them separately
python main.py
python app.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data` | GET | All analyzed records (JSON) |
| `/api/stats` | GET | Summary statistics |
| `/api/regenerate` | POST | Regenerate all charts |
| `/analyze` | POST | Analyze a single text |

---

## 📦 Dependencies

- **Flask** — Web framework
- **VADER Sentiment** — Rule-based social media NLP
- **TextBlob** — Lexicon sentiment + subjectivity
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Chart generation

---

## 🎓 Dataset

The sample dataset uses the [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) format from Kaggle. You can drop any compatible CSV into `data/raw/`.
