# 🚢 PRODIGY DS Task-02 — Titanic EDA Dashboard

**Data cleaning and Exploratory Data Analysis on the Titanic dataset**  
*PRODIGY InfoTech – Data Science Internship*

---

## 📋 Overview

This project performs comprehensive **data cleaning** and **exploratory data analysis (EDA)** on the classic Titanic dataset. It ships as a **local web app** (Flask) with an interactive dashboard showing:

- Survival breakdown by **gender, age, passenger class, family size, and embarkation port**
- A **correlation heatmap** of all numeric features
- A detailed **data cleaning log**
- An interactive **data preview table**
- A **Key Insights** panel summarising findings

---

## 🗂️ Project Structure

```
PRODIGY_DS_Task02/
├── app.py                  ← Flask web application entry point
├── requirements.txt        ← Python dependencies
├── README.md
├── report.md               ← EDA findings report
│
├── dataset/
│   └── titanic.csv         ← Titanic passenger data (891 rows)
│
├── source_code/
│   └── eda.py              ← Data cleaning + chart generation logic
│
├── templates/
│   └── index.html          ← Dashboard UI (dark theme)
│
├── screenshots/
│   ├── survival_by_gender.png
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   └── data_cleaning.png
│
└── notebooks/
    └── eda_analysis.ipynb  ← Jupyter notebook walkthrough
```

---

## 🚀 Quick Start (Windows)

### Option A — Double-click launcher
1. Run `run.bat` (included in the zip)

### Option B — Manual
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the app
python app.py

# 3. Open browser
http://localhost:5000
```

---

## 🔍 EDA Highlights

| Feature | Key Finding |
|---|---|
| **Gender** | Women: ~74% survival vs Men: ~19% |
| **Passenger Class** | 1st class: ~63% vs 3rd class: ~24% |
| **Age** | Children had higher survival rates |
| **Family Size** | Sweet spot: 2–4 members |
| **Fare** | Higher fare → higher survival (class proxy) |
| **Embarkation** | Cherbourg (C) passengers had best rates |

---

## 🧹 Data Cleaning Steps

1. **Age** — Filled missing values with median grouped by `Pclass` & `Sex`
2. **Fare** — Filled 2 missing values with overall median
3. **Embarked** — Filled 2 missing values with mode (Southampton)
4. **Cabin** — Converted to binary `Has_Cabin` flag (687 nulls → feature)
5. **Feature Engineering** — Added `FamilySize`, `IsAlone`, `Title`, `AgeGroup`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Flask** — web server
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — chart generation
- **HTML / CSS / JS** — dashboard UI (no external frameworks)

---

*Submitted by: [Your Name] | PRODIGY InfoTech Data Science Internship*
