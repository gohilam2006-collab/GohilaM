"""
PRODIGY_DS_Task02 - Titanic EDA
Data cleaning and exploratory data analysis on the Titanic dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE = {'survived': '#22c55e', 'died': '#ef4444',
           'accent': '#6366f1', 'bg': '#0f172a', 'text': '#f1f5f9'}
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({'figure.facecolor': '#1e293b', 'axes.facecolor': '#1e293b',
                     'axes.labelcolor': '#cbd5e1', 'xtick.color': '#94a3b8',
                     'ytick.color': '#94a3b8', 'text.color': '#f1f5f9',
                     'grid.color': '#334155', 'grid.alpha': 0.5})


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_and_clean(path='dataset/titanic.csv'):
    df = pd.read_csv(path)

    cleaning_log = []

    # Missing value report (before)
    before_nulls = df.isnull().sum().to_dict()
    cleaning_log.append({'step': 'Before Cleaning – Null Counts',
                          'detail': {k: int(v) for k, v in before_nulls.items() if v > 0}})

    # Fill Age with median grouped by Pclass & Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    df['Age'].fillna(df['Age'].median(), inplace=True)
    cleaning_log.append({'step': 'Age', 'detail': 'Filled with median by Pclass & Sex group'})

    # Fill Fare median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    cleaning_log.append({'step': 'Fare', 'detail': 'Filled with overall median'})

    # Fill Embarked mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    cleaning_log.append({'step': 'Embarked', 'detail': 'Filled with mode (S)'})

    # Cabin – flag presence
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    df.drop(columns=['Cabin'], inplace=True)
    cleaning_log.append({'step': 'Cabin', 'detail': 'Converted to binary Has_Cabin flag; original dropped'})

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].str.strip()
    df['AgeGroup'] = pd.cut(df['Age'],
                             bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
    cleaning_log.append({'step': 'Feature Engineering',
                          'detail': 'Added FamilySize, IsAlone, Title, AgeGroup'})

    after_nulls = df.isnull().sum().to_dict()
    cleaning_log.append({'step': 'After Cleaning – Null Counts',
                          'detail': {k: int(v) for k, v in after_nulls.items() if v > 0} or 'No nulls remaining'})

    return df, cleaning_log


# ── Chart Generators ──────────────────────────────────────────────────────────

def chart_survival_by_gender(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('#1e293b')

    counts = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
    counts.columns = ['Died', 'Survived']
    counts.plot(kind='bar', ax=axes[0],
                color=[PALETTE['died'], PALETTE['survived']],
                edgecolor='none', width=0.6)
    axes[0].set_title('Survival Count by Gender', fontsize=13, pad=10)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(facecolor='#334155', edgecolor='none')

    rates = df.groupby('Sex')['Survived'].mean() * 100
    bars = axes[1].bar(rates.index, rates.values,
                        color=[PALETTE['accent'], '#ec4899'], edgecolor='none', width=0.4)
    axes[1].set_title('Survival Rate by Gender (%)', fontsize=13, pad=10)
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, rates.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                      f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    fig.tight_layout(pad=2)
    return fig_to_base64(fig)


def chart_age_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1e293b')

    for val, label, color in [(0, 'Died', PALETTE['died']),
                               (1, 'Survived', PALETTE['survived'])]:
        subset = df[df['Survived'] == val]['Age']
        axes[0].hist(subset, bins=28, alpha=0.65, label=label,
                      color=color, edgecolor='none')
    axes[0].set_title('Age Distribution by Survival', fontsize=13, pad=10)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].legend(facecolor='#334155', edgecolor='none')

    ag = df.groupby('AgeGroup')['Survived'].mean() * 100
    axes[1].bar(ag.index.astype(str), ag.values,
                 color=PALETTE['accent'], edgecolor='none', width=0.55)
    axes[1].set_title('Survival Rate by Age Group (%)', fontsize=13, pad=10)
    axes[1].set_ylabel('Survival Rate (%)')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(ag.values):
        axes[1].text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=10)

    fig.tight_layout(pad=2)
    return fig_to_base64(fig)


def chart_pclass_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1e293b')

    counts = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    counts.columns = ['Died', 'Survived']
    counts.plot(kind='bar', ax=axes[0],
                color=[PALETTE['died'], PALETTE['survived']],
                edgecolor='none', width=0.6)
    axes[0].set_title('Survival by Passenger Class', fontsize=13, pad=10)
    axes[0].set_xlabel('Passenger Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(facecolor='#334155', edgecolor='none')

    fare_pclass = df.groupby('Pclass')['Fare'].median()
    axes[1].bar([f'Class {c}' for c in fare_pclass.index], fare_pclass.values,
                 color=['#f59e0b', '#64748b', '#475569'], edgecolor='none', width=0.5)
    axes[1].set_title('Median Fare by Passenger Class', fontsize=13, pad=10)
    axes[1].set_ylabel('Median Fare (£)')
    for i, v in enumerate(fare_pclass.values):
        axes[1].text(i, v + 0.5, f'£{v:.1f}', ha='center', fontsize=11)

    fig.tight_layout(pad=2)
    return fig_to_base64(fig)


def chart_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')

    num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch',
                 'Fare', 'FamilySize', 'IsAlone', 'Has_Cabin']
    corr = df[num_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, linecolor='#1e293b',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9})
    ax.set_title('Correlation Heatmap', fontsize=14, pad=12)

    fig.tight_layout()
    return fig_to_base64(fig)


def chart_family_embark(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1e293b')

    fam = df.groupby('FamilySize')['Survived'].mean() * 100
    axes[0].plot(fam.index, fam.values, marker='o',
                  color=PALETTE['accent'], linewidth=2.5, markersize=7,
                  markerfacecolor='white')
    axes[0].fill_between(fam.index, fam.values, alpha=0.15, color=PALETTE['accent'])
    axes[0].set_title('Survival Rate by Family Size', fontsize=13, pad=10)
    axes[0].set_xlabel('Family Size (including self)')
    axes[0].set_ylabel('Survival Rate (%)')

    emb = df.groupby('Embarked')['Survived'].mean() * 100
    colors = ['#22c55e', '#6366f1', '#f59e0b']
    axes[1].bar(emb.index, emb.values, color=colors, edgecolor='none', width=0.45)
    axes[1].set_title('Survival Rate by Embarkation Port', fontsize=13, pad=10)
    axes[1].set_ylabel('Survival Rate (%)')
    for i, v in enumerate(emb.values):
        axes[1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=11)

    fig.tight_layout(pad=2)
    return fig_to_base64(fig)


def get_summary_stats(df):
    total = len(df)
    survived = int(df['Survived'].sum())
    return {
        'total_passengers': total,
        'survived': survived,
        'died': total - survived,
        'survival_rate': round(df['Survived'].mean() * 100, 1),
        'avg_age': round(df['Age'].mean(), 1),
        'avg_fare': round(df['Fare'].mean(), 2),
        'missing_before': {'Age': 177, 'Cabin': 687, 'Embarked': 2},
        'female_survival': round(df[df['Sex']=='female']['Survived'].mean()*100, 1),
        'male_survival': round(df[df['Sex']=='male']['Survived'].mean()*100, 1),
        'class1_survival': round(df[df['Pclass']==1]['Survived'].mean()*100, 1),
        'class3_survival': round(df[df['Pclass']==3]['Survived'].mean()*100, 1),
    }
