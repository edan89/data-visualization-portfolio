# 🤖 Week 5 Tutorial: Supervised Machine Learning — Full Build Guide

> **Who is this for?** Complete beginners. We explain every concept, every tool, every piece of math, and every line of code from start to finish.

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Big Picture: What is Machine Learning?](#2-the-big-picture)
3. [Tools & Libraries We Use (and Why)](#3-tools--libraries)
4. [Step 0 — Imports & Page Setup](#step-0)
5. [Step 1 — Loading & Preparing the Data](#step-1)
6. [Step 2 — The Sidebar: Interactive Controls](#step-2)
7. [Step 3 — Data Preparation for ML](#step-3)
8. [Step 4 — Exploratory Data Analysis (EDA)](#step-4)
9. [Step 5 — Regression: Predicting Tip Amount](#step-5)
10. [Step 6 — Classification: Predicting Tip Quality](#step-6)
11. [Step 7 — Interpretation & Discussion](#step-7)
12. [Key Math Concepts Cheat Sheet](#math-cheat-sheet)
13. [Full Glossary](#glossary)

---

## 1. What Are We Building? <a name="1-what-are-we-building"></a>

We're building an **interactive web app** using **Streamlit** that:

- Takes a restaurant tips dataset
- Performs **two types of prediction**:
  - **Regression** — predict the exact tip amount (a number)
  - **Classification** — predict whether the tip will be Low, Medium, or High (a category)
- Trains multiple Machine Learning models
- Compares them against a "dumb" baseline
- Visualizes results with interactive charts
- Lets the user tweak settings via a sidebar

**The final result:** A deployed web page at `http://YOUR_VM_IP/week5` that satisfies all assignment requirements from `In_week5.md`.

---

## 2. The Big Picture: What is Machine Learning? <a name="2-the-big-picture"></a>

### The Simplest Explanation

Imagine you're a waiter. After months of work, you *intuitively* know: "Big table + expensive dinner = big tip." You learned this from **experience** (data).

**Machine Learning does the same thing**, but with math:

1. We give the computer **lots of examples** (past restaurant visits)
2. Each example has **inputs** (bill amount, party size, etc.) and a **correct answer** (the actual tip)
3. The computer finds **mathematical patterns** connecting inputs to answers
4. It uses those patterns to **predict answers for new data** it has never seen

### Two Types of Prediction

| Type | What we predict | Example | Math Output |
|------|----------------|---------|-------------|
| **Regression** | A **number** | "The tip will be $4.50" | Continuous value |
| **Classification** | A **category** | "The tip will be High" | One of several labels |

### Why Do We Need a Baseline?

A **baseline** is the "dumbest possible prediction":
- **Regression baseline:** "I'll just guess the average tip every time" (~$3.00)
- **Classification baseline:** "I'll just guess the most common category every time"

If our fancy model can't beat the baseline, it's useless! The baseline is our "minimum bar."

### What is Overfitting?

Think of a student who **memorizes** the exact answers to practice problems but can't solve new ones. That's overfitting:

- **Training data:** The practice problems the model learns from
- **Test data:** The "exam" — new problems it has never seen
- **Overfitting:** Model performs great on training data but poorly on test data
- **Good model:** Performs similarly on both

---

## 3. Tools & Libraries We Use (and Why) <a name="3-tools--libraries"></a>

| Library | What it does | Why we chose it |
|---------|-------------|-----------------|
| `streamlit` | Creates the web app | Easy to build interactive dashboards with Python — no HTML/CSS needed |
| `pandas` | Data manipulation | The #1 tool for tabular data in Python (think "Excel for programmers") |
| `numpy` | Math operations | Fast numerical computation (arrays, statistics) |
| `seaborn` | Dataset source + plotting | Comes with built-in datasets (like "tips") and pretty statistical plots |
| `plotly` | Interactive charts | Creates beautiful, zoomable, hoverable charts (better than static images) |
| `scikit-learn` | Machine Learning | The most popular Python ML library — has every model and tool we need |

### Why scikit-learn specifically?

scikit-learn gives us a **consistent API** for everything:
```python
# Every model works the same way:
model.fit(X_train, y_train)       # Step 1: Learn from training data
predictions = model.predict(X_test)  # Step 2: Make predictions
```

This consistency means you can swap models with minimal code changes.

---

## Step 0 — Imports & Page Setup <a name="step-0"></a>

### The Code (Lines 1–52)

```python
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
```

### What Each Import Does

**Data & Visualization:**
- `streamlit` → The web framework — all `st.something()` calls create UI elements
- `pandas` → DataFrames (tables). Think of `pd.DataFrame` as a spreadsheet
- `numpy` → `np.sqrt()`, `np.number`, arrays — raw math power
- `seaborn` → We use `sns.load_dataset("tips")` to get our data
- `plotly.express` → Quick charts: `px.bar()`, `px.imshow()`
- `plotly.graph_objects` → Custom charts: `go.Scatter()`, `go.Histogram()`
- `make_subplots` → Put multiple charts side by side

**Machine Learning (from scikit-learn):**
- `train_test_split` → Splits data into training and testing portions
- `cross_val_score` → Tests a model multiple times on different data slices
- `LabelEncoder` → Converts text like "Male"/"Female" to numbers (0/1)
- `LinearRegression` → Draws a "best fit line" through data points
- `LogisticRegression` → Classifies data using an S-shaped curve
- `RandomForestRegressor/Classifier` → Uses many decision trees voting together
- `DummyRegressor/Classifier` → The "dumb" baselines
- All the metrics → Ways to measure how good a model is (explained later)

**Utility:**
- `warnings.filterwarnings('ignore')` → Hides noisy warnings that would clutter the app

### Page Configuration (Line 22)

```python
st.set_page_config(page_title="Week 5: Machine Learning", page_icon="🤖", layout="wide")
```

- `page_title` → What appears in the browser tab
- `page_icon` → The emoji in the tab
- `layout="wide"` → Uses the full browser width (default is a narrow centered column)

### Custom CSS (Lines 25–52)

```python
st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        padding: 18px; border-radius: 10px;
        border-left: 5px solid #43a047; margin: 12px 0;
        color: #1b5e20 !important;
    }
    /* ... more styles ... */
</style>
""", unsafe_allow_html=True)
```

**Why?** Streamlit's default styles are plain. We created three custom boxes:

| Class | Color | Purpose |
|-------|-------|---------|
| `.insight-box` | 🟢 Green | Results and findings |
| `.task-box` | 🔵 Blue | Task descriptions |
| `.warn-box` | 🟠 Orange | Warnings and cautions |

`unsafe_allow_html=True` tells Streamlit to render raw HTML (normally it escapes it for security).

---

## Step 1 — Loading & Preparing the Data <a name="step-1"></a>

### The Code (Lines 72–82)

```python
@st.cache_data
def load_data():
    df = sns.load_dataset("tips")
    # Create classification target: tip quality
    df['tip_quality'] = pd.cut(
        df['tip'], bins=[0, 2, 3.5, df['tip'].max()],
        labels=['Low', 'Medium', 'High'], include_lowest=True
    )
    return df

df = load_data()
```

### Line-by-Line Breakdown

**`@st.cache_data`** — This is a **decorator** (a special wrapper). It tells Streamlit: "Run this function once, save the result, and reuse it instead of re-running every time the page refreshes." Without this, the data would reload on every user interaction, making the app slow.

**`sns.load_dataset("tips")`** — Seaborn includes several built-in datasets. The "tips" dataset has 244 restaurant visits with these columns:

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `total_bill` | Number | 16.99 | Total bill amount ($) |
| `tip` | Number | 1.01 | Tip amount ($) |
| `sex` | Category | Male/Female | Customer's gender |
| `smoker` | Category | Yes/No | Whether they smoke |
| `day` | Category | Sun/Sat/Thur/Fri | Day of the week |
| `time` | Category | Lunch/Dinner | Meal time |
| `size` | Number | 2 | Party size (number of people) |

**`pd.cut()`** — This is how we **create the classification target**. The assignment says we need a categorical variable, but our dataset only has `tip` (a number). So we **bin** (group) the tip values:

```
pd.cut(df['tip'], bins=[0, 2, 3.5, max], labels=['Low', 'Medium', 'High'])
```

This means:
- **Low:** $0 to $2.00
- **Medium:** $2.01 to $3.50
- **High:** Above $3.50

`include_lowest=True` makes the first bin include 0 (otherwise $0 tips would be excluded).

### Why This Dataset?

The assignment requires:
- ✅ At least 200 rows → We have 244
- ✅ Multiple predictor variables → We have 6
- ✅ A numeric target for regression → `tip`
- ✅ A categorical target for classification → `tip_quality` (created by binning)

---

## Step 2 — The Sidebar: Interactive Controls <a name="step-2"></a>

### The Code (Lines 87–110)

```python
st.sidebar.header("⚙️ Model Settings")

available_features = ['total_bill', 'size', 'sex', 'smoker', 'day', 'time']
selected_features = st.sidebar.multiselect(
    "📋 Features to use", available_features, default=available_features,
    help="Pick which information the model can use to make predictions"
)
if not selected_features:
    selected_features = available_features
    st.sidebar.warning("Need at least one feature — using all.")

n_estimators = st.sidebar.slider(
    "🌲 Number of trees (Random Forest)", 10, 200, 100, step=10
)
max_depth = st.sidebar.slider(
    "📏 Max tree depth", 2, 20, 5
)
test_size = st.sidebar.slider(
    "📊 Test set size (%)", 20, 40, 30, step=5
) / 100
```

### What Each Control Does

**Feature Selection (`multiselect`):**
The user picks which columns the model can "see." This lets them experiment:
- What if the model only knows the bill amount? (Just `total_bill`)
- What if we add party size? Does it improve?
- What if we remove all categorical features?

The `if not selected_features` check is a safety net — we need at least one feature.

**Number of Trees (`n_estimators`):**
Random Forest works by creating many decision trees and averaging their votes. More trees = usually better but slower. Range: 10 to 200, default 100.

**Max Tree Depth (`max_depth`):**
How many questions each tree can ask. Deep trees memorize training data (overfitting); shallow trees are too simple (underfitting). Range: 2 to 20, default 5.

**Test Size:**
What percentage of data we save for testing. The `/ 100` at the end converts the percentage (e.g., 30) to a decimal (0.30) because `train_test_split` expects a decimal.

### Why a Sidebar?

The sidebar keeps controls separate from results. Users can tweak settings on the left while seeing results update on the right. It satisfies the assignment's "interactive" requirement.

---

## Step 3 — Data Preparation for ML <a name="step-3"></a>

### The Code (Lines 115–151)

```python
@st.cache_data
def prepare_features(df, selected_features, test_size):
    feature_df = df[selected_features].copy()
    
    # Encode categorical columns
    le_dict = {}
    for col in feature_df.select_dtypes(include=['category', 'object']).columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col])
        le_dict[col] = le
    
    X = feature_df.values
    feature_names = list(feature_df.columns)
    
    # Regression target
    y_reg = df['tip'].values
    
    # Classification target
    le_target = LabelEncoder()
    y_cls = le_target.fit_transform(df['tip_quality'])
    class_names = le_target.classes_
    
    # Split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=test_size, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=test_size, random_state=42
    )
    
    return (X_train_r, X_test_r, y_train_r, y_test_r,
            X_train_c, X_test_c, y_train_c, y_test_c,
            feature_names, class_names)
```

### The Concepts Step by Step

#### 3a. Label Encoding — Making Text into Numbers

ML models only understand numbers. So "Male" and "Female" need to become 0 and 1:

```
Before:  sex = ["Male", "Female", "Male", "Male", "Female"]
After:   sex = [1, 0, 1, 1, 0]
```

**How `LabelEncoder` works:**
```python
le = LabelEncoder()
le.fit_transform(["Male", "Female", "Male"])
# Output: [1, 0, 1]  (alphabetical order: Female=0, Male=1)
```

We store each encoder in `le_dict` so we could decode back if needed.

#### 3b. Features (X) vs Target (y)

This is the fundamental ML setup:

```
X = features = the INPUTS (what the model sees)
    total_bill, size, sex, smoker, day, time

y = target = the ANSWER (what the model predicts)
    For regression:      tip (a dollar amount)
    For classification:  tip_quality (Low/Medium/High)
```

`.values` converts the pandas DataFrame into a NumPy array (a plain grid of numbers), which is what scikit-learn expects.

#### 3c. Train-Test Split — The Most Important Step

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
```

**What this does:**
```
244 restaurant visits
    ├── 70% → Training set (≈171 rows) → Model LEARNS from these
    └── 30% → Test set (≈73 rows) → Model is EVALUATED on these
```

**Why split?** If we test the model on the same data it learned from, that's like giving a student the exam answers beforehand — it proves nothing. The test set is data the model has **never seen**.

**`random_state=42`** → Makes the split reproducible. Every time you run the code, you get the exact same split. The number 42 is arbitrary (it's a pop culture reference to *The Hitchhiker's Guide to the Galaxy*).

**Why do we split twice (lines 138-143)?** Because regression and classification have different target variables (`tip` vs `tip_quality`). Even though X is the same, y is different, so `train_test_split` needs to pair them correctly.

---

## Step 4 — Exploratory Data Analysis (EDA) <a name="step-4"></a>

### The Code Overview (Lines 156–221)

EDA means "look at the data before you model it." It's like reading the ingredients list before cooking.

### 4a. Basic Metrics (Lines 165–169)

```python
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Rows", df.shape[0])
with col2: st.metric("Columns", df.shape[1])
with col3: st.metric("Numeric Vars", len(df.select_dtypes(include=[np.number]).columns))
with col4: st.metric("Missing Values", df.isnull().sum().sum())
```

`st.columns(4)` creates 4 equal-width columns. `st.metric()` displays a number in a large, prominent format.

`df.isnull().sum().sum()` → First `.sum()` counts nulls per column, second `.sum()` adds all columns together. Result: 0 (no missing data!).

### 4b. Distribution Charts (Lines 177–209)

**Histograms (left column):**
```python
fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Total Bill ($)", "Tip Amount ($)"])
fig_dist.add_trace(
    go.Histogram(x=df['total_bill'], marker_color='#3366cc', opacity=0.75, name="Bill"),
    row=1, col=1
)
```

A **histogram** groups values into bins and counts how many fall in each bin. It answers: "What's the typical bill/tip?" Most bills are $10–$30; most tips are $2–$4.

**Tip Quality Bar Chart (right column):**
```python
tip_counts = df['tip_quality'].value_counts().reindex(['Low', 'Medium', 'High'])
fig_tq = px.bar(x=tip_counts.index, y=tip_counts.values, ...)
```

This shows how many tips fall into each category. If one category dominates (e.g., "Low" has 60% of the data), that's **class imbalance** — an important problem we address later.

### 4c. Correlation Heatmap (Lines 212–221)

```python
corr = numeric_df.corr()
fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r')
```

#### The Math: Correlation

**Correlation** measures how two variables move together. It ranges from **-1 to +1**:

| Value | Meaning | Example |
|-------|---------|---------|
| +1.0 | Perfect positive: when A goes up, B always goes up | - |
| +0.68 | Strong positive | Total bill ↔ Tip |
| 0.0 | No relationship | - |
| -0.5 | Moderate negative: when A goes up, B goes down | - |
| -1.0 | Perfect negative | - |

The formula (Pearson correlation):

```
r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / √(Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²)
```

In plain English: "How much do X and Y move together, relative to how much they move individually?"

**Key finding:** `total_bill` and `tip` have correlation ≈0.68. This tells us the bill amount is a strong predictor of the tip — which makes sense, since tips are usually a percentage of the bill.

---

## Step 5 — Regression: Predicting Tip Amount <a name="step-5"></a>

This is the heart of the assignment. We train three models and compare them.

### 5a. Baseline Model — "Just Guess the Average" (Lines 255–267)

```python
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
```

**What this does:** Ignores ALL features and always predicts the mean tip (~$3.00).

**Why?** It sets the "floor." If our real model can't beat this, it hasn't learned anything useful.

**Metrics computed for the baseline:**
```python
'rmse': np.sqrt(mean_squared_error(y_test, y_pred_base)),   # ~$1.38
'mae': mean_absolute_error(y_test, y_pred_base),             # ~$1.06
'r2': r2_score(y_test, y_pred_base),                         # 0.000
```

### 5b. Linear Regression — "Draw the Best Line" (Lines 269–281)

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

#### The Math: Linear Regression

Linear Regression finds the equation:

```
tip = w₁ × total_bill + w₂ × size + w₃ × sex + ... + b
```

Where:
- `w₁, w₂, w₃...` are **weights** (how important each feature is)
- `b` is the **bias** (the starting point)

The algorithm finds the weights that **minimize the total squared error**:

```
Error = Σ(actual_tipᵢ - predicted_tipᵢ)²
```

**Why squared?** Two reasons:
1. It penalizes big mistakes more than small ones ($5 off is punished 25×, not 5×)
2. It makes the math smooth (calculus can find the minimum)

**Visually:** Imagine plotting total_bill (x-axis) vs tip (y-axis). Linear Regression draws the straight line that best fits through all the points.

### 5c. Random Forest Regressor — "Many Trees Vote" (Lines 283–301)

```python
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

#### The Math: Decision Trees & Random Forests

**A single Decision Tree** asks yes/no questions:
```
Is total_bill > $20?
├── YES: Is size > 3?
│   ├── YES: Predict tip = $4.50
│   └── NO:  Predict tip = $3.80
└── NO:  Is smoker == Yes?
    ├── YES: Predict tip = $2.10
    └── NO:  Predict tip = $2.50
```

**A Random Forest** creates many trees (100 by default), each slightly different:
1. Each tree sees a **random subset** of the data
2. At each decision point, each tree considers a **random subset** of features
3. Final prediction = **average of all trees' predictions**

**Why is this better?** Individual trees are unpredictable. But averaging 100 slightly different trees cancels out individual errors — like asking 100 people to estimate something and averaging their answers.

**Hyperparameters:**
- `n_estimators=100` → 100 trees in the forest
- `max_depth=5` → Each tree can ask at most 5 questions deep
- `random_state=42` → Same randomness every run (reproducible)

### 5d. Cross-Validation (Line 290)

```python
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
```

#### The Math: 5-Fold Cross-Validation

Instead of one train/test split, we do 5:

```
Fold 1: [ TEST ] [Train] [Train] [Train] [Train]
Fold 2: [Train] [ TEST ] [Train] [Train] [Train]
Fold 3: [Train] [Train] [ TEST ] [Train] [Train]
Fold 4: [Train] [Train] [Train] [ TEST ] [Train]
Fold 5: [Train] [Train] [Train] [Train] [ TEST ]
```

Each time, 1/5 of the training data is held out as a "mini test set." We train on the other 4/5 and evaluate on the held-out portion.

**Why?** A single train/test split might be "lucky" or "unlucky." Cross-validation gives us the **mean** and **standard deviation** of the score, telling us how **consistent** the model is.

### 5e. Regression Metrics — How We Measure Success

#### RMSE (Root Mean Squared Error)

```
RMSE = √(Σ(actualᵢ - predictedᵢ)² / n)
```

Example with 3 predictions:
```
Actual:    [$3.00, $5.00, $2.00]
Predicted: [$3.50, $4.00, $2.50]
Errors:    [-$0.50, $1.00, -$0.50]
Squared:   [$0.25, $1.00, $0.25]
Mean:      $0.50
RMSE:      √0.50 = $0.707
```

**Interpretation:** "On average, our predictions are off by about $0.71."

#### MAE (Mean Absolute Error)

```
MAE = Σ|actualᵢ - predictedᵢ| / n
```

Same example:
```
|Errors|:  [$0.50, $1.00, $0.50]
MAE:       $0.667
```

**MAE vs RMSE:** RMSE penalizes large errors more (because of squaring). If RMSE >> MAE, you have some very large errors.

#### R² (R-squared / Coefficient of Determination)

```
R² = 1 - (Σ(actualᵢ - predictedᵢ)²) / (Σ(actualᵢ - mean)²)
```

**Interpretation:**
- R² = 0.0 → Model is as good as just guessing the mean (baseline)
- R² = 0.5 → Model explains 50% of the variation in tips
- R² = 1.0 → Perfect predictions

### 5f. Regression Visualizations

**Predicted vs Actual Plot (Lines 386–414):**

```python
fig_pva = go.Figure()
fig_pva.add_trace(go.Scatter(x=y_test_r, y=best_preds, mode='markers'))
# Add 45-degree reference line
fig_pva.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode='lines', line=dict(dash='dash')))
```

The 45-degree line represents **perfect predictions** (predicted = actual). Points near the line = good predictions; scattered points = errors.

**Residual Plot (Lines 416–432):**

```python
residuals = y_test_r - best_preds  # actual - predicted
```

**Residuals** are the errors. We plot them against predicted values. In a good model, residuals should be **random** (no pattern). If you see a funnel shape or a curve, the model is missing something systematic.

**Feature Importance (Lines 434–452):**

```python
importances = rf_model.feature_importances_
```

Random Forest can tell us which features it relied on most. `feature_importances_` gives a percentage for each feature (they sum to 1.0). This is computed based on how much each feature reduces prediction error across all trees.

---

## Step 6 — Classification: Predicting Tip Quality <a name="step-6"></a>

### 6a. Creating the Classification Target (Recap)

We already created `tip_quality` by binning: Low (≤$2), Medium ($2-$3.50), High (>$3.50).

### 6b. Class Distribution Check (Lines 469–491)

```python
class_counts = pd.Series(y_train_c).value_counts().sort_index()
```

**Why check this?** If 80% of tips are "Low," a model could just always guess "Low" and be "80% accurate." That's misleading! We need to check for **class imbalance**.

The code checks if any class has >60% of the data and shows a warning if so.

### 6c. Baseline Classifier — "Always Guess the Most Common" (Lines 498–511)

```python
baseline = DummyClassifier(strategy='most_frequent')
```

Always predicts the most common class. If "Low" has 40% of data, baseline accuracy = 40%.

### 6d. Logistic Regression — "S-Curve Classification" (Lines 513–526)

```python
log_reg = LogisticRegression(max_iter=1000, random_state=42)
```

#### The Math: Logistic Regression

Despite the name, this is a **classification** algorithm. It uses the **sigmoid function** to convert any number into a probability between 0 and 1:

```
P(class) = 1 / (1 + e^(-z))

where z = w₁x₁ + w₂x₂ + ... + b
```

The **sigmoid curve** looks like an S:
```
Probability
1.0  ─────────────────────────╭─────
                             ╱
0.5  ─────────────────────╱─────
                         ╱
0.0  ──────────────────╯─────────
                    z value →
```

For **multi-class** (Low/Medium/High), it uses "one-vs-rest": three separate models, each deciding "is it this class or not?"

`max_iter=1000` → Maximum iterations for the optimizer to converge (find the best weights).

### 6e. Random Forest Classifier (Lines 528–547)

Same concept as the Random Forest Regressor, but instead of averaging predictions, the trees **vote**:

```
Tree 1: "I think it's High"
Tree 2: "I think it's Medium"
Tree 3: "I think it's High"
...
Tree 100: "I think it's High"

Final answer: High (got the most votes)
```

### 6f. Classification Metrics — How We Measure Success

#### Accuracy

```
Accuracy = correct predictions / total predictions
```

**Problem:** With imbalanced data, a model that always predicts "Low" can be 40% accurate without being useful.

#### Precision

```
Precision = true positives / (true positives + false positives)
```

"Of all the times the model said 'High', how many were actually High?"

High precision = few false alarms.

#### Recall (Sensitivity)

```
Recall = true positives / (true positives + false negatives)
```

"Of all the actual 'High' tips, how many did the model catch?"

High recall = few missed cases.

#### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The **harmonic mean** of precision and recall. It balances both:
- If precision is high but recall is low (or vice versa), F1 will be moderate
- F1 is high only when BOTH precision and recall are high

**Why F1 over accuracy?** In our code, we prioritize F1 because our classes are imbalanced. F1 ensures the model is good at finding ALL categories, not just the most common one.

#### Weighted Average

Since we have 3 classes (Low/Medium/High), each class gets its own precision, recall, and F1. The **weighted average** combines them, weighting each class by how many samples it has.

### 6g. Confusion Matrix (Lines 624–636)

```python
fig_cm = px.imshow(best_cm, text_auto=True, x=class_names, y=class_names)
```

A confusion matrix is a grid showing:
```
                 Predicted
                 Low  Med  High
Actual  Low    [[ 25    3    0 ]
        Med     [  5   15    2 ]
        High    [  1    4   18 ]]
```

- **Diagonal** (top-left to bottom-right) = correct predictions
- **Off-diagonal** = mistakes
- Read rows to see: "What did the model predict for actual Low/Medium/High?"

### 6h. ROC Curves (Lines 638–670)

```python
y_prob = model.predict_proba(X_test_c)  # Get probability for each class
fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
roc_auc = auc(fpr, tpr)
```

#### The Math: ROC Curve

ROC = **Receiver Operating Characteristic**. For each class, it plots:
- **X-axis: False Positive Rate** (FPR) = % of non-members incorrectly classified as members
- **Y-axis: True Positive Rate** (TPR) = % of actual members correctly identified

**AUC** (Area Under the Curve):
- AUC = 1.0 → Perfect classifier
- AUC = 0.5 → Random guessing (the diagonal line)
- AUC < 0.5 → Worse than random (something is wrong)

The gray dashed diagonal line represents random guessing. Our model's curve should bow toward the top-left corner.

For multi-class, we use **One-vs-Rest**: separate ROC curves for "Low vs Not-Low," "Medium vs Not-Medium," etc.

`label_binarize()` converts our labels to binary format:
```
[0, 1, 2] → [[1,0,0], [0,1,0], [0,0,1]]  (one-hot encoding)
```

---

## Step 7 — Interpretation & Discussion <a name="step-7"></a>

### The Code (Lines 689–762)

This section ties everything together with:

1. **Summary Table** — All models from both tasks in one place
2. **Key Insights** — Written in plain language:
   - Which model won and by how much
   - Whether the improvement is meaningful
   - What features matter most
3. **Limitations** — Honest assessment:
   - Small dataset (244 rows)
   - Single restaurant
   - Few features
   - Arbitrary category boundaries
   - Historical data

### Dynamic Insights (Lines 716–718)

```python
lr_rmse = reg_results['Linear Regression']['rmse']
rf_rmse = reg_results['Random Forest']['rmse']
rmse_diff_pct = abs(lr_rmse - rf_rmse) / lr_rmse * 100
```

The code calculates how different the models are and **generates different text** based on the result:
- If difference < 15%: "The relationship is mostly linear"
- If difference ≥ 15%: "There are non-linear patterns"

This is smarter than hardcoding a single conclusion — the text adapts to the actual results!

### The Overfitting Checks (Lines 359–379 and 602–617)

```python
gap = r['r2_train'] - r['r2']
status = "✅ Good" if abs(gap) < 0.15 else "⚠️ Slight overfitting" if abs(gap) < 0.3 else "❌ Overfitting"
```

We compare train vs test performance:
- **Gap < 0.15** → ✅ The model generalizes well
- **Gap 0.15 – 0.3** → ⚠️ Slight overfitting — might need less complexity
- **Gap > 0.3** → ❌ The model memorized the training data

---

## Key Math Concepts Cheat Sheet <a name="math-cheat-sheet"></a>

| Concept | Formula | What it means |
|---------|---------|---------------|
| **Mean** | Σxᵢ / n | Average value |
| **RMSE** | √(Σ(actual-predicted)²/n) | Average error magnitude (penalizes big errors) |
| **MAE** | Σ\|actual-predicted\|/n | Average error magnitude (treats all errors equally) |
| **R²** | 1 - SS_res/SS_total | % of variance explained (0=bad, 1=perfect) |
| **Accuracy** | correct / total | % of correct predictions |
| **Precision** | TP / (TP+FP) | "When it says yes, is it right?" |
| **Recall** | TP / (TP+FN) | "Does it find all the yes cases?" |
| **F1** | 2×P×R/(P+R) | Balance of precision and recall |
| **AUC** | Area under ROC curve | Overall classification quality (0.5=random, 1=perfect) |
| **Correlation** | Σ((x-x̄)(y-ȳ))/√(Σ(x-x̄)²Σ(y-ȳ)²) | How strongly two variables move together (-1 to +1) |
| **Sigmoid** | 1/(1+e^(-z)) | Converts any number to probability (0–1) |

---

## Full Glossary <a name="glossary"></a>

| Term | Simple Definition |
|------|------------------|
| **Supervised Learning** | Teaching a computer with labeled examples (input + correct answer) |
| **Feature** | An input variable (column) that the model uses to make predictions |
| **Target** | The variable we want to predict |
| **Training Set** | The data the model learns from |
| **Test Set** | The data we evaluate the model on (never seen during training) |
| **Baseline** | The simplest possible prediction (e.g., always guess the average) |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Underfitting** | Model is too simple to capture the patterns in the data |
| **Label Encoding** | Converting text categories to numbers (e.g., "Male"=1, "Female"=0) |
| **Cross-Validation** | Testing a model multiple times on different data slices |
| **Hyperparameter** | A setting YOU choose (e.g., number of trees) vs a parameter the model learns |
| **Residual** | The error: actual value minus predicted value |
| **Confusion Matrix** | A grid showing correct vs incorrect predictions for each class |
| **ROC Curve** | A plot showing the tradeoff between true positives and false positives |
| **AUC** | Area Under the ROC Curve — overall classification quality |
| **Decision Tree** | A model that makes predictions by asking yes/no questions |
| **Random Forest** | Many decision trees that vote to make a final prediction |
| **Binning** | Grouping a continuous number into categories (e.g., tips → Low/Medium/High) |
| **Decorator** | A `@something` wrapper that adds behavior to a function (like `@st.cache_data`) |
| **Class Imbalance** | When one category has way more samples than others |

---

## How the File Connects to the Assignment Requirements

| Assignment Requirement | Where in week5.py | Lines |
|----------------------|-------------------|-------|
| Dataset ≥200 rows, multiple features | "tips" dataset (244 rows, 7 columns) | 73–74 |
| Define regression & classification tasks | Header text explaining both tasks | 58–67, 228–236, 459–467 |
| Data preparation & train-test split | `prepare_features()` function | 115–151 |
| Baseline models | `DummyRegressor` and `DummyClassifier` | 255–267, 498–511 |
| ≥2 regression models | Linear Regression + Random Forest | 269–301 |
| ≥2 classification models | Logistic Regression + Random Forest | 513–547 |
| Regression metrics (RMSE, MAE, R²) | Computed for all regression models | 262–267, 276–281, 294–300 |
| Classification metrics (Accuracy, F1, etc.) | Computed for all classification models | 505–510, 520–525, 540–546 |
| Cross-validation | 5-fold CV on Random Forest | 290, 535 |
| Predicted vs Actual plot | Scatter plot with 45° reference line | 386–414 |
| Residual plot | Error vs predicted values | 416–432 |
| Feature importance | Horizontal bar charts | 434–452, 672–687 |
| Confusion matrix | Heatmap | 624–636 |
| ROC curves | One-vs-Rest curves with AUC | 638–670 |
| Overfitting check (train vs test) | Tables with gap analysis | 359–379, 602–617 |
| Interpretation in words | Dynamic insight text | 720–744 |
| Limitations & next steps | Table + bullet points | 746–762 |
| Interactive sidebar controls | Feature selection, hyperparameters | 87–110 |
| Class imbalance discussion | Warning box + F1 explanation | 480–491, 596–599 |

---

> **🎓 Tip:** The best way to learn from this code is to **change things and see what happens.** Try:
> - Remove `total_bill` from the features — watch all models get worse
> - Set `max_depth` to 20 — watch Random Forest overfit
> - Set test size to 40% — see how less training data affects results
> - Change the tip quality bins in `pd.cut()` — see how classification changes
