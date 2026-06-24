# Titanic EDA – Findings Report
## PRODIGY InfoTech – Data Science Task 02

---

### 1. Dataset Overview
- **Source**: Titanic passenger manifest (kaggle.com/competitions/titanic)
- **Rows**: 891 passengers  
- **Features**: 12 original → 16 after feature engineering  
- **Target**: `Survived` (0 = No, 1 = Yes)

---

### 2. Data Cleaning Summary

| Column | Issue | Resolution |
|---|---|---|
| Age | 177 missing (19.9%) | Median by Pclass & Sex group |
| Cabin | 687 missing (77.1%) | Converted to binary `Has_Cabin` |
| Fare | 2 missing | Filled with overall median |
| Embarked | 2 missing | Filled with mode (S) |

**New Features Created:**
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` = 1 if FamilySize == 1
- `Title` = extracted from Name (Mr, Mrs, Miss, etc.)
- `AgeGroup` = binned into Child / Teen / Adult / Middle-Aged / Senior

---

### 3. Key Findings

#### 3.1 Gender
- Female survival rate: **~74%**
- Male survival rate: **~19%**
- **Finding**: Gender is the single strongest predictor of survival, consistent with "women and children first" policy.

#### 3.2 Passenger Class
- 1st class: **~63%** survival
- 2nd class: **~47%** survival  
- 3rd class: **~24%** survival
- **Finding**: Socioeconomic status directly impacted access to lifeboats. 3rd class passengers were located deeper in the ship.

#### 3.3 Age
- Children (0–12): highest survival rates
- Seniors (60+): lower survival rates
- **Finding**: Age interacted with gender — young females and children had the best outcomes.

#### 3.4 Family Size
- Solo travellers (FamilySize=1): moderate survival
- Small families (2–4): best survival
- Large families (5+): worst survival
- **Finding**: Small families could coordinate better; very large families were disadvantaged.

#### 3.5 Fare & Embarkation
- Higher fare → stronger survival correlation (proxy for class)
- Cherbourg (C) passengers had best survival rate due to higher proportion of 1st-class travellers

---

### 4. Correlation Analysis
Top correlations with `Survived`:
- `Sex` (encoded): strong positive (female)
- `Pclass`: strong negative (higher class number = lower cabin)
- `Fare`: moderate positive
- `Has_Cabin`: moderate positive

---

### 5. Conclusion
The Titanic disaster survival was **not random**. It was systematically shaped by **gender**, **class**, and **age**. A predictive model built on these features alone can achieve ~80% accuracy — consistent with published Kaggle benchmarks.
