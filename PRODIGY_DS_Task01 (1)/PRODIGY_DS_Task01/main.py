"""
PRODIGY_DS_Task01 - Data Science Visualization Dashboard
Author: Prodigy InfoTech Internship Task 1
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import io, base64, os, json
import numpy as np

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv')
OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), 'outputs')

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#7C3AED",
    "accent":    "#F59E0B",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "bars":      ["#2563EB", "#7C3AED", "#F59E0B", "#10B981", "#EF4444",
                  "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#8B5CF6"],
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0F172A', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor('#1E293B')
    ax.set_title(title, color='#F1F5F9', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color='#94A3B8', fontsize=11)
    ax.set_ylabel(ylabel, color='#94A3B8', fontsize=11)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.grid(axis='y', color='#334155', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    df = load_data()
    summary = {
        "total_rows": int(len(df)),
        "total_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "numeric_cols": list(df.select_dtypes(include='number').columns),
        "categorical_cols": list(df.select_dtypes(include='object').columns),
        "missing": int(df.isnull().sum().sum()),
        "survived_pct": round(df['Survived'].mean() * 100, 1) if 'Survived' in df.columns else None,
    }
    return render_template('index.html', summary=summary)


@app.route('/api/columns')
def api_columns():
    df = load_data()
    cols = {
        "all": list(df.columns),
        "numeric": list(df.select_dtypes(include='number').columns),
        "categorical": list(df.select_dtypes(include='object').columns),
    }
    return jsonify(cols)


@app.route('/api/chart', methods=['POST'])
def api_chart():
    body   = request.get_json()
    col    = body.get('column', 'Age')
    ctype  = body.get('chart_type', 'histogram')   # histogram | bar | pie | box | violin
    bins   = int(body.get('bins', 15))
    color  = body.get('color', PALETTE['primary'])
    save   = body.get('save', False)

    df = load_data()

    if col not in df.columns:
        return jsonify({"error": f"Column '{col}' not found"}), 400

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0F172A')

    try:
        # ── HISTOGRAM ──────────────────────────────────────────────────────────
        if ctype == 'histogram':
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            n, bin_edges, patches = ax.hist(data, bins=bins, color=color,
                                            edgecolor='#0F172A', linewidth=0.8, alpha=0.9)
            # gradient fill
            for i, p in enumerate(patches):
                p.set_facecolor(PALETTE['bars'][i % len(PALETTE['bars'])])
            style_axes(ax, f'Distribution of {col}', col, 'Frequency')
            # stats overlay
            ax.axvline(data.mean(), color='#F59E0B', linewidth=1.5,
                       linestyle='--', label=f'Mean: {data.mean():.1f}')
            ax.axvline(data.median(), color='#10B981', linewidth=1.5,
                       linestyle='--', label=f'Median: {data.median():.1f}')
            ax.legend(facecolor='#1E293B', labelcolor='#F1F5F9', fontsize=9)

        # ── BAR CHART ──────────────────────────────────────────────────────────
        elif ctype == 'bar':
            counts = df[col].value_counts().head(15)
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=PALETTE['bars'][:len(counts)],
                          edgecolor='#0F172A', linewidth=0.8)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + counts.max() * 0.01,
                        str(val), ha='center', va='bottom',
                        color='#94A3B8', fontsize=8)
            style_axes(ax, f'Distribution of {col}', col, 'Count')
            plt.xticks(rotation=30, ha='right')

        # ── PIE ────────────────────────────────────────────────────────────────
        elif ctype == 'pie':
            counts = df[col].value_counts().head(8)
            wedge_props = {'linewidth': 2, 'edgecolor': '#0F172A'}
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index.astype(str),
                autopct='%1.1f%%',
                colors=PALETTE['bars'][:len(counts)],
                startangle=140,
                wedgeprops=wedge_props,
                pctdistance=0.82,
            )
            for t in texts:
                t.set_color('#94A3B8')
                t.set_fontsize(9)
            for at in autotexts:
                at.set_color('#F1F5F9')
                at.set_fontsize(8)
            ax.set_facecolor('#0F172A')
            ax.set_title(f'Distribution of {col}', color='#F1F5F9',
                         fontsize=14, fontweight='bold', pad=14)

        # ── BOX ────────────────────────────────────────────────────────────────
        elif ctype == 'box':
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            bp = ax.boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor=PALETTE['primary'], color='#94A3B8'),
                            medianprops=dict(color='#F59E0B', linewidth=2),
                            whiskerprops=dict(color='#94A3B8'),
                            capprops=dict(color='#94A3B8'),
                            flierprops=dict(marker='o', color=PALETTE['danger'],
                                            alpha=0.5, markersize=4))
            style_axes(ax, f'Box Plot of {col}', '', col)

        # ── VIOLIN ─────────────────────────────────────────────────────────────
        elif ctype == 'violin':
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            for pc in parts.get('bodies', []):
                pc.set_facecolor(PALETTE['secondary'])
                pc.set_alpha(0.7)
            parts['cmeans'].set_color('#F59E0B')
            parts['cmedians'].set_color('#10B981')
            style_axes(ax, f'Violin Plot of {col}', '', col)

        else:
            plt.close(fig)
            return jsonify({"error": "Unknown chart_type"}), 400

    except Exception as e:
        plt.close(fig)
        return jsonify({"error": str(e)}), 500

    plt.tight_layout(pad=1.5)

    # optionally save to outputs/
    if save:
        fname = f"{ctype}_{col}.png"
        fpath = os.path.join(OUTPUTS_PATH, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight',
                    facecolor='#0F172A', edgecolor='none')

    img_b64 = fig_to_b64(fig)
    return jsonify({"image": img_b64, "column": col, "chart_type": ctype})


@app.route('/api/stats', methods=['POST'])
def api_stats():
    body = request.get_json()
    col  = body.get('column', 'Age')
    df   = load_data()

    if col not in df.columns:
        return jsonify({"error": f"Column '{col}' not found"}), 400

    series = df[col]
    is_num = pd.api.types.is_numeric_dtype(series)

    if is_num:
        num = pd.to_numeric(series, errors='coerce')
        result = {
            "type": "numeric",
            "count": int(num.count()),
            "missing": int(num.isna().sum()),
            "mean": round(float(num.mean()), 3) if not num.empty else None,
            "median": round(float(num.median()), 3) if not num.empty else None,
            "std": round(float(num.std()), 3) if not num.empty else None,
            "min": round(float(num.min()), 3) if not num.empty else None,
            "max": round(float(num.max()), 3) if not num.empty else None,
            "q25": round(float(num.quantile(0.25)), 3) if not num.empty else None,
            "q75": round(float(num.quantile(0.75)), 3) if not num.empty else None,
        }
    else:
        vc = series.value_counts()
        result = {
            "type": "categorical",
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "top": str(vc.index[0]) if not vc.empty else None,
            "top_freq": int(vc.iloc[0]) if not vc.empty else None,
            "value_counts": {str(k): int(v) for k, v in vc.head(10).items()},
        }
    return jsonify(result)


@app.route('/api/preview')
def api_preview():
    df = load_data()
    return jsonify({
        "columns": list(df.columns),
        "rows": df.head(10).fillna('').to_dict(orient='records'),
    })


# ── Generate default saved charts on first run ──────────────────────────────

def generate_default_charts():
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    df = load_data()

    # Bar chart – Gender
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0F172A')
    counts = df['Sex'].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE['primary'], PALETTE['secondary']],
                  edgecolor='#0F172A', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(val),
                ha='center', color='#94A3B8', fontsize=10)
    style_axes(ax, 'Gender Distribution', 'Gender', 'Count')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_PATH, 'bar_chart.png'),
                dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close(fig)

    # Histogram – Age
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0F172A')
    ages = pd.to_numeric(df['Age'], errors='coerce').dropna()
    n, bins, patches = ax.hist(ages, bins=12, color=PALETTE['primary'],
                                edgecolor='#0F172A', linewidth=0.8)
    for i, p in enumerate(patches):
        p.set_facecolor(PALETTE['bars'][i % len(PALETTE['bars'])])
    ax.axvline(ages.mean(), color='#F59E0B', linewidth=1.5,
               linestyle='--', label=f'Mean: {ages.mean():.1f}')
    ax.axvline(ages.median(), color='#10B981', linewidth=1.5,
               linestyle='--', label=f'Median: {ages.median():.1f}')
    ax.legend(facecolor='#1E293B', labelcolor='#F1F5F9', fontsize=9)
    style_axes(ax, 'Age Distribution', 'Age', 'Frequency')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_PATH, 'histogram.png'),
                dpi=150, bbox_inches='tight', facecolor='#0F172A')
    plt.close(fig)

if __name__ == '__main__':
    generate_default_charts()
    print("\n" + "="*55)
    print("  PRODIGY_DS_Task01  ─  Data Visualization Dashboard")
    print("  Open  ➜  http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
