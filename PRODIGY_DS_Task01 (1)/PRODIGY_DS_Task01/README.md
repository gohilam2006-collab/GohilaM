# PRODIGY_DS_Task01 — Distribution Visualizer

> **Prodigy InfoTech Data Science Internship · Task 1**
> Visualize the distribution of categorical and continuous variables using bar charts, histograms, and more.

---

## 📁 Project Structure

```
PRODIGY_DS_Task01/
├── main.py                  ← Flask web app (run this)
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv          ← Titanic-style population dataset
├── outputs/
│   ├── bar_chart.png        ← Auto-generated on first run
│   └── histogram.png        ← Auto-generated on first run
├── notebooks/
│   └── task01_analysis.ipynb
└── templates/
    └── index.html           ← Dashboard UI
```

---

## 🚀 Quick Start (Windows)

### 1 · Install Python 3.10+
Download from https://www.python.org/downloads/ — tick **"Add to PATH"** during install.

### 2 · Open a terminal in the project folder
Right-click the folder → **Open in Terminal** (or use Command Prompt / PowerShell).

### 3 · Install dependencies
```
pip install -r requirements.txt
```

### 4 · Run the app
```
python main.py
```

### 5 · Open your browser
Navigate to → **http://127.0.0.1:5000**

---

## 🖥️ Features

| Feature | Detail |
|---|---|
| **Chart types** | Histogram, Bar, Pie, Box Plot, Violin Plot |
| **Any column** | Select from categorical or numeric columns via dropdown |
| **Bin control** | Slider for histogram bin count (5 – 50) |
| **Color picker** | 7 accent color swatches |
| **Stats panel** | Mean, median, std, quantiles for numeric; value counts for categorical |
| **Data preview** | First 10 rows displayed as a table |
| **Save to disk** | "Save to outputs/" button exports PNG to `outputs/` folder |

---

## 📊 Dataset

The sample dataset (`data/dataset.csv`) is modelled after the classic Titanic passenger dataset from the Prodigy InfoTech GitHub:
https://github.com/Prodigy-InfoTech/data-science-datasets/tree/main/Task%201

Columns: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

You can replace this file with any CSV — the app will automatically detect columns.

---

## 🔄 Using Your Own Data

Drop any `.csv` file into `data/` and rename it `dataset.csv`. Relaunch `main.py`.

---

## 🛠️ Tech Stack

- **Flask** — lightweight Python web framework
- **Pandas** — data loading and statistics
- **Matplotlib** — chart rendering
- **Seaborn** — statistical plots
- **Vanilla JS + CSS** — no build step, runs in any browser
