# 🌳 Bank Marketing Decision Tree Classifier

A full-stack machine learning web application that predicts whether a bank customer will subscribe to a term deposit, using a Decision Tree classifier on the UCI Bank Marketing dataset.

---

## 🚀 Quick Start (Windows)

1. Make sure **Python 3.8+** is installed → [python.org](https://python.org)
2. Double-click **`run.bat`**
3. Browser opens automatically at **http://localhost:5000**

### Manual Start

```bash
pip install -r requirements.txt
python main.py        # train model + generate visuals
python app.py         # start web server
```

---

## 📁 Project Structure

```
BankMarketingDT/
├── data/
│   ├── raw/bank.csv                  # Original dataset
│   └── processed/cleaned_data.csv    # After preprocessing
├── src/
│   ├── data_preprocessing.py         # Cleaning, encoding, feature engineering
│   ├── train_model.py                # Decision Tree training
│   ├── evaluate_model.py             # Metrics + visualizations
│   └── utils.py                      # Helper functions
├── models/
│   └── decision_tree_model.pkl       # Saved trained model
├── visuals/
│   ├── confusion_matrix.png
│   ├── decision_tree_plot.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── templates/                        # Flask HTML templates
├── static/                           # CSS + JS
├── app.py                            # Flask web application
├── main.py                           # Full pipeline runner
├── run.bat                           # Windows one-click launcher
└── requirements.txt
```

---

## 🌐 Web UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Model metrics, accuracy, quick actions |
| Predict | `/predict` | Enter customer details, get prediction |
| Visuals | `/visuals` | All charts and plots |
| Dataset | `/data` | Browse raw data |

---

## 🧠 Model Details

- **Algorithm:** Decision Tree (CART)
- **Criterion:** Gini Impurity
- **Max Depth:** 5
- **Features:** age, job, marital, education, balance, housing, loan, contact, duration, campaign, pdays, previous, poutcome + 3 engineered features

## 📊 Dataset

UCI Bank Marketing Dataset — 45,211 records (sample of 60 included, replace `data/raw/bank.csv` with full dataset from UCI/GitHub).

Full dataset: https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%203

---

## 🔄 Replace with Full Dataset

1. Download `bank.csv` from the link above
2. Replace `data/raw/bank.csv`
3. Re-run: `python main.py` or click **Retrain** in the dashboard

---

## 🛠 Tech Stack

- Python 3.8+
- scikit-learn — Decision Tree classifier
- pandas + numpy — Data processing
- matplotlib + seaborn — Visualizations
- Flask — Web framework
