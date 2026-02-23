# 🔬 Week 6 Tutorial: Unsupervised Machine Learning — Full Build Guide

> **Who is this for?** Complete beginners. We explain every concept, every tool, every piece of math, and every line of code from start to finish.

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Big Picture: What is Unsupervised Learning?](#2-the-big-picture)
3. [Tools & Libraries We Use (and Why)](#3-tools--libraries)
4. [Step 0 — Imports & Page Setup](#step-0)
5. [Step 1 — Loading & Preparing the Data](#step-1)
6. [Step 2 — The Sidebar: Interactive Controls](#step-2)
7. [Step 3 — Standardization: Why Scale Matters](#step-3)
8. [Step 4 — Exploratory Data Analysis (EDA)](#step-4)
9. [Step 5 — Factor Analysis: Finding Hidden Variables](#step-5)
10. [Step 6 — Clustering: Grouping Similar Wines](#step-6)
11. [Step 7 — Joint Interpretation: Connecting Factors & Clusters](#step-7)
12. [Step 8 — Discussion & Reflection](#step-8)
13. [Key Math Concepts Cheat Sheet](#math-cheat-sheet)
14. [Full Glossary](#glossary)

---

## 1. What Are We Building? <a name="1-what-are-we-building"></a>

We're building an **interactive web app** using **Streamlit** that:

- Takes a wine quality dataset (1,599 red wines from Portugal)
- Performs **two unsupervised learning techniques**:
  - **Factor Analysis** — discover hidden variables behind 11 chemical measurements
  - **Clustering** — group wines into natural "taste profiles"
- Optionally compares Factor Analysis with PCA
- Evaluates clusters with multiple metrics
- Connects the two analyses together (Joint Interpretation)
- Lets the user tweak settings via a sidebar

**The final result:** A deployed web page at `http://YOUR_VM_IP/week6` that satisfies all assignment requirements from `In_week6.md`.

---

## 2. The Big Picture: What is Unsupervised Learning? <a name="2-the-big-picture"></a>

### The Simplest Explanation

In **Week 5 (Supervised Learning)**, we had a teacher:
- "Here are 244 restaurant visits with tips. **Learn to predict tips**."
- The model had correct answers to learn from.

In **Week 6 (Unsupervised Learning)**, there's **no teacher**:
- "Here are 1,599 wines. **Find interesting patterns on your own.**"
- No correct answers. The algorithm explores freely.

**Real-life analogy:** Imagine you dump 1,000 unsorted photos on a table and ask a friend to organize them. Without any instructions, they'd naturally group photos by theme — vacations, family, food. That's unsupervised learning.

### Two Techniques We Use

| Technique | What it does | Analogy |
|-----------|-------------|---------|
| **Factor Analysis** | Finds **hidden variables** behind the data | "These 11 wine measurements can be summarized by 3 hidden concepts" |
| **Clustering** | Groups **similar items** together | "These 1,599 wines naturally fall into 3 taste groups" |

### Supervised vs Unsupervised — Quick Comparison

| | Supervised (Week 5) | Unsupervised (Week 6) |
|-|--------------------|-----------------------|
| **Has labels?** | ✅ Yes (tip amount, tip quality) | ❌ No labels |
| **Goal** | Predict something specific | Discover hidden structure |
| **Train/Test split?** | ✅ Yes (need test data to evaluate) | ❌ No (use all data) |
| **Success metric** | RMSE, Accuracy, F1 | Silhouette score, Eigenvalues |
| **Examples** | Regression, Classification | Factor Analysis, Clustering |

---

## 3. Tools & Libraries We Use (and Why) <a name="3-tools--libraries"></a>

| Library | What it does | Why we chose it |
|---------|-------------|-----------------|
| `streamlit` | Creates the web app | Easy to build interactive dashboards with Python — no HTML/CSS needed |
| `pandas` | Data manipulation | The #1 tool for tabular data in Python (think "Excel for programmers") |
| `numpy` | Math operations | Fast numerical computation — eigenvalues, arrays, statistics |
| `plotly` | Interactive charts | Creates beautiful, zoomable, hoverable charts |
| `scikit-learn` | Machine Learning | Provides Factor Analysis, PCA, K-Means, and evaluation metrics |
| `scipy` | Scientific computing | We use it for hierarchical clustering linkage and dendrograms |
| `matplotlib` | Static plotting | Required for rendering the dendrogram (scipy's `dendrogram()` uses it) |

### Why Two Plotting Libraries?

We use **Plotly** for most charts because its interactive features (zoom, hover, pan) are superior. But the **dendrogram** from `scipy.cluster.hierarchy` only works with **matplotlib**, so we use that for one specific chart.

---

## Step 0 — Imports & Page Setup <a name="step-0"></a>

### The Code (Lines 1–55)

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
```

### What Each Import Does

**Data & Visualization:**
- `streamlit` → The web framework — all `st.something()` calls create UI elements
- `pandas` → DataFrames (tables). Think of `pd.DataFrame` as a spreadsheet
- `numpy` → `np.linalg.eigh()` for eigenvalues, `np.cov()` for covariance, arrays
- `plotly.express` → Quick charts: `px.imshow()` for heatmaps, `px.scatter()` for scatter plots
- `plotly.graph_objects` → Custom charts: `go.Scatter()`, `go.Histogram()`
- `make_subplots` → Put multiple charts side by side

**Machine Learning (from scikit-learn):**
- `StandardScaler` → Standardizes data to mean=0, variance=1 (critical for unsupervised learning!)
- `FactorAnalysis` → Finds hidden latent variables behind correlated features
- `PCA` → Principal Component Analysis — an alternative dimensionality reduction method
- `KMeans` → Clustering algorithm that partitions data into k spherical groups
- `AgglomerativeClustering` → Hierarchical clustering (builds a tree of merges)
- `silhouette_score` → Measures how well-separated clusters are
- `calinski_harabasz_score` → Between-cluster vs within-cluster variance ratio
- `davies_bouldin_score` → Average similarity between clusters (lower = better)

**Scientific Computing (from scipy):**
- `dendrogram` → Draws the tree diagram showing how clusters merge
- `linkage` → Computes the hierarchical clustering linkage matrix

**Utility:**
- `matplotlib.use('Agg')` → Tells matplotlib to render charts as images (not interactive windows), which is needed for Streamlit
- `warnings.filterwarnings('ignore')` → Hides noisy convergence warnings

### Page Configuration (Line 18)

```python
st.set_page_config(page_title="Week 6: Unsupervised ML", page_icon="🔬", layout="wide")
```

- `page_title` → What appears in the browser tab
- `page_icon` → The emoji in the tab (microscope for exploration/discovery)
- `layout="wide"` → Uses the full browser width

### Custom CSS (Lines 21–55)

```python
st.markdown("""
<style>
    .insight-box { ... }   /* Green — results and findings */
    .task-box { ... }      /* Blue — task descriptions */
    .warn-box { ... }      /* Orange — warnings and cautions */
    .learn-box { ... }     /* Purple — educational explanations */
</style>
""", unsafe_allow_html=True)
```

**Why?** Streamlit's default styles are plain. We created four custom colored boxes:

| Class | Color | Purpose |
|-------|-------|---------|
| `.insight-box` | 🟢 Green | Results and findings |
| `.task-box` | 🔵 Blue | Task descriptions |
| `.warn-box` | 🟠 Orange | Warnings and cautions |
| `.learn-box` | 🟣 Purple | Educational explanations (new in Week 6!) |

`unsafe_allow_html=True` tells Streamlit to render raw HTML (normally it escapes it for security).

---

## Step 1 — Loading & Preparing the Data <a name="step-1"></a>

### The Code (Lines 74–99)

```python
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return df

df = load_data()

feature_descriptions = {
    'fixed acidity': 'Tartaric acid level (affects tartness)',
    'volatile acidity': 'Acetic acid level (too high = vinegar taste)',
    'citric acid': 'Adds freshness & flavor',
    # ... etc for all 11 features
}

feature_cols = [c for c in df.columns if c != 'quality']
X_raw = df[feature_cols]
```

### Line-by-Line Breakdown

**`@st.cache_data`** — This is a **decorator** (a special wrapper). It tells Streamlit: "Run this function once, save the result, and reuse it instead of re-running every time the page refreshes." Without this, the data would re-download from UCI on every user interaction.

**`pd.read_csv(url, sep=";")`** — Downloads and reads the CSV file. The `sep=";"` is important because this dataset uses semicolons (`;`) as separators instead of commas (`,`). If we used the default comma separator, all columns would be smashed into one.

**The Wine Quality Dataset has these columns:**

| Column | Range | Meaning |
|--------|-------|---------|
| `fixed acidity` | 4.6–15.9 | Tartaric acid (g/L) — makes wine tart |
| `volatile acidity` | 0.12–1.58 | Acetic acid (g/L) — too much = vinegar |
| `citric acid` | 0.0–1.0 | Citric acid (g/L) — adds freshness |
| `residual sugar` | 0.9–15.5 | Sugar after fermentation (g/L) |
| `chlorides` | 0.01–0.61 | Salt content (g/L) |
| `free sulfur dioxide` | 1–72 | Free SO₂ (mg/L) — prevents microbes |
| `total sulfur dioxide` | 6–289 | Total SO₂ (mg/L) — free + bound |
| `density` | 0.99–1.004 | Density (g/cm³) |
| `pH` | 2.74–4.01 | Acidity level (lower = more acidic) |
| `sulphates` | 0.33–2.0 | Potassium sulphate (g/L) |
| `alcohol` | 8.4–14.9 | Alcohol (% vol) |
| `quality` | 3–8 | Expert rating (NOT used — it's a label!) |

**`feature_cols = [c for c in df.columns if c != 'quality']`** — We exclude `quality` because it's a human-assigned label (a rating from 3–8). In unsupervised learning, we intentionally don't use labels — the whole point is to let the algorithm discover patterns without being told the answer.

### Why This Dataset?

The assignment requires:
- ✅ At least 200 observations → We have **1,599**
- ✅ 6-8+ numeric variables → We have **11**
- ✅ Reasonable correlations between variables → pH ↔ acidity, density ↔ alcohol
- ✅ Not used in previous weeks → Fresh dataset

---

## Step 2 — The Sidebar: Interactive Controls <a name="step-2"></a>

### The Code (Lines 104–124)

```python
st.sidebar.header("⚙️ Controls")

# Factor Analysis controls
st.sidebar.subheader("Factor Analysis")
n_factors = st.sidebar.slider(
    "🔢 Number of factors", 2, 6, 3,
    help="How many hidden variables to extract"
)
show_pca = st.sidebar.checkbox(
    "📊 Compare with PCA", value=False,
    help="PCA finds directions of maximum spread; Factor Analysis finds hidden causes"
)

# Clustering controls
st.sidebar.subheader("Clustering")
cluster_algo = st.sidebar.selectbox(
    "🧮 Algorithm", ["K-Means", "Hierarchical (Ward)"],
    help="K-Means: fast, spherical clusters. Hierarchical: builds a tree of merges"
)
n_clusters = st.sidebar.slider(
    "🎯 Number of clusters (k)", 2, 8, 3,
    help="How many groups to create"
)
```

### What Each Control Does

**Number of Factors (`slider`, 2–6, default 3):**
How many hidden variables Factor Analysis should find. More factors = more detail but harder to interpret. The Scree plot and Kaiser criterion help you decide.

**Compare with PCA (`checkbox`):**
When checked, the app shows PCA results side-by-side with Factor Analysis. This is an optional part of the assignment — it helps you understand the difference between the two methods.

**Algorithm (`selectbox`):**
Choose between K-Means and Hierarchical clustering. Both are valid; we compare them automatically.

**Number of Clusters (`slider`, 2–8, default 3):**
How many groups to divide the wines into. The Elbow and Silhouette plots help you pick the right number.

### Why These Specific Ranges?

- **Factors 2–6:** With 11 features, more than 6 factors defeats the purpose of reduction. Less than 2 isn't useful for visualization.
- **Clusters 2–8:** Fewer than 2 is meaningless. More than 8 with 11 features creates groups too small to interpret.

---

## Step 3 — Standardization: Why Scale Matters <a name="step-3"></a>

### The Code (Lines 129–135)

```python
@st.cache_data
def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

X_scaled, scaler = standardize(X_raw)
```

### The Math: Standardization (Z-Score Normalization)

For each feature value:

```
z = (x - μ) / σ

where:
  x = original value
  μ = mean of that feature
  σ = standard deviation of that feature
```

**Concrete example with alcohol:**

```
Alcohol values: [9.4, 9.8, 9.8, 11.2, 9.4, ...]
Mean (μ): 10.42
Standard deviation (σ): 1.07

For a wine with alcohol = 12.5:
z = (12.5 - 10.42) / 1.07 = 1.94  ← "1.94 standard deviations above average"

For a wine with alcohol = 9.0:
z = (9.0 - 10.42) / 1.07 = -1.33  ← "1.33 standard deviations below average"
```

After standardization:
- Every feature has mean = 0 and standard deviation = 1
- Values are now in "how many standard deviations from the mean" units

### Why Is This Critical?

Look at the raw data ranges:

| Feature | Range | Scale |
|---------|-------|-------|
| Alcohol | 8.4 – 14.9 | Small numbers |
| Total sulfur dioxide | 6 – 289 | Huge numbers! |
| Density | 0.99 – 1.004 | Tiny numbers! |

Without standardization, the algorithm would think total sulfur dioxide is "more important" just because its numbers are bigger. **Standardization puts all features on the same playing field.**

**`fit_transform()` is a shortcut for two steps:**
```python
scaler.fit(X)          # Step 1: Calculate μ and σ for each feature
X_scaled = scaler.transform(X)  # Step 2: Apply (x - μ) / σ to every value
```

---

## Step 4 — Exploratory Data Analysis (EDA) <a name="step-4"></a>

### The Code Overview (Lines 140–206)

EDA means "look at the data before you model it." It's like reading the ingredient list before cooking.

### 4a. Basic Metrics (Lines 150–154)

```python
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Wines", df.shape[0])
with col2: st.metric("Features", len(feature_cols))
with col3: st.metric("Numeric Vars", len(df.select_dtypes(include=[np.number]).columns))
with col4: st.metric("Missing Values", df.isnull().sum().sum())
```

`st.columns(4)` creates 4 equal-width columns. `st.metric()` displays a number in a large, prominent format.

Results:
- **1,599** wines
- **11** features (excluding quality)
- **12** numeric variables (including quality)
- **0** missing values (no data cleaning needed!)

### 4b. Distribution Charts (Lines 170–185)

```python
fig_dist = make_subplots(rows=2, cols=2,
    subplot_titles=["Alcohol (%)", "Volatile Acidity", "Citric Acid", "Residual Sugar"])
```

**Histograms** group values into bins and count how many fall in each. They answer: "What's the typical value for each feature?"

Key observations:
- **Alcohol** is roughly normally distributed, centered around 10%
- **Volatile acidity** is right-skewed (most wines have low values, a few have very high)
- **Citric acid** has a notable spike at 0 (some wines have none)
- **Residual sugar** is very right-skewed (most wines are dry, a few are sweet)

### 4c. Correlation Heatmap (Lines 188–195)

```python
corr = df[feature_cols].corr()
fig_corr = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r',
                      zmin=-1, zmax=1)
```

#### The Math: Correlation

**Correlation** measures how two variables move together. It ranges from **-1 to +1**:

| Value | Meaning | Example in our data |
|-------|---------|---------------------|
| +1.0 | Perfect positive: when A goes up, B always goes up | — |
| +0.67 | Strong positive | Free SO₂ ↔ Total SO₂ |
| 0.0 | No relationship | — |
| -0.68 | Strong negative | Fixed acidity ↔ pH |
| -1.0 | Perfect negative | — |

**Key correlations that matter for Factor Analysis:**
- **Fixed acidity ↔ pH** (r ≈ -0.68): More acid = lower pH. Chemistry makes sense!
- **Fixed acidity ↔ citric acid** (r ≈ +0.67): Both are types of acid.
- **Free SO₂ ↔ Total SO₂** (r ≈ +0.67): One is a component of the other.
- **Density ↔ alcohol** (r ≈ -0.50): Alcohol is lighter than water, lowering density.

These correlations are **exactly** what Factor Analysis is designed to exploit — correlated features should "load onto" the same factor.

---

## Step 5 — Factor Analysis: Finding Hidden Variables <a name="step-5"></a>

This is the first major unsupervised technique. It's all about **dimensionality reduction** — going from many features to fewer, more meaningful ones.

### 5a. The Concept

**Imagine this scenario:** You measure 11 properties of wine in a lab. But really, those 11 numbers boil down to a few underlying concepts:
- **"Acidity profile"** — fixed acidity, citric acid, and pH all move together because they're all about acid
- **"Body/strength"** — alcohol, density, and sugar are all about how heavy/strong the wine feels
- **"Preservation chemistry"** — sulfur dioxide types and sulphates are about winemaking practices

Factor Analysis finds these hidden groupings **automatically from the data**.

### 5b. Determining the Number of Factors (Lines 234–279)

```python
@st.cache_data
def compute_eigenvalues(X_scaled):
    cov_mat = np.cov(X_scaled, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_mat)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues
```

#### The Math: Covariance Matrix

The **covariance matrix** is an 11×11 table where each cell shows how much two features vary together:

```
                   fixed_acid  vol_acid  citric  sugar  ...
fixed_acidity      [ 1.00     -0.26     0.67   0.11   ...]
volatile_acidity   [-0.26      1.00    -0.55   0.00   ...]
citric_acid        [ 0.67     -0.55     1.00   0.14   ...]
...
```

`np.cov(X_scaled, rowvar=False)` computes this matrix. `rowvar=False` means "each column is a variable" (not each row).

#### The Math: Eigenvalues

**Eigenvalues** tell us how much "information" (variance) each potential factor captures.

Think of it like this:
- You have 11 features = 11 units of information
- The first eigenvalue might be 2.8 → Factor 1 captures the information equivalent of 2.8 original features
- The second eigenvalue might be 1.9 → Factor 2 captures 1.9 features' worth
- By the 5th eigenvalue it might be 0.6 → Factor 5 captures less than one feature's worth

**`np.linalg.eigh(cov_mat)`** — Computes the eigenvalues and eigenvectors of the covariance matrix. `eigh` is for **symmetric** matrices (covariance matrices are always symmetric).

**`np.sort(eigenvalues)[::-1]`** — Sort eigenvalues from largest to smallest (most important first).

#### Two Methods to Choose the Number of Factors

**1. Kaiser Criterion:**
```python
n_kaiser = int(np.sum(eigenvalues > 1))
```
Rule: Keep only factors with eigenvalue > 1. Why? An eigenvalue of 1 means the factor captures exactly as much information as a single original variable. If it captures less, why bother?

**2. Scree Plot:**
A line chart of eigenvalues. You look for the **"elbow"** — the point where the curve stops falling steeply and flattens out. Factors after the elbow add little value.

```
Eigenvalue
3.0  ●
      \
2.0    ●
        \
1.0  ----●-----  ← Kaiser line (eigenvalue = 1)
          ●  ●  ●  ●  ●  ●  ●  ← flat = not worth keeping
     1  2  3  4  5  6  7  8  9  10  11
     Factor Number
```

### 5c. Fitting Factor Analysis (Lines 284–304)

```python
@st.cache_data
def fit_fa(X_scaled, n_factors, feature_names):
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_scores = fa.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=feature_names,
        columns=[f"Factor {i+1}" for i in range(n_factors)]
    )
    var_communalities = pd.Series(
        np.sum(fa.components_.T**2, axis=1),
        index=feature_names
    )
    explained_var = np.sum(fa.components_**2, axis=1) / X_scaled.shape[1]
    return fa, fa_scores, loadings, var_communalities, explained_var
```

#### What Each Part Does

**`FactorAnalysis(n_components=3)`** — We're asking scikit-learn: "Find 3 hidden factors that best explain the correlations in our data."

**`fa.fit_transform(X_scaled)`** — Two things happen:
1. **fit:** The model finds the factor structure (loadings)
2. **transform:** Each wine gets a "score" on each factor

**`fa.components_.T`** → The **loadings matrix**. This is the heart of Factor Analysis. It's a table showing how strongly each original feature connects to each factor:

```
                Factor 1    Factor 2    Factor 3
fixed acidity     +0.85       +0.12       -0.05
pH               -0.78       +0.08       +0.10
citric acid       +0.68       +0.15       -0.12
alcohol           +0.10       +0.82       -0.03
...
```

**Reading the loadings:**
- **+0.85** = Strong positive connection (when the factor goes up, this feature goes up)
- **-0.78** = Strong negative connection (when the factor goes up, this feature goes down)
- **+0.10** = Weak connection (this feature is mostly unrelated to this factor)

The threshold for "strong" is usually |loading| > 0.4 or 0.5.

#### Communalities

```python
var_communalities = np.sum(fa.components_.T**2, axis=1)
```

**Communality** = the sum of squared loadings for a feature across all factors.

For example, if fixed acidity has loadings [0.85, 0.12, -0.05]:
```
Communality = 0.85² + 0.12² + (-0.05)²
            = 0.7225 + 0.0144 + 0.0025
            = 0.7394
```

**Interpretation:** 73.9% of fixed acidity's variation is explained by the 3 factors. The remaining 26.1% is unique noise or information that doesn't share a pattern with other features.

| Communality | Interpretation |
|-------------|----------------|
| > 0.4 | Well explained ✅ |
| 0.2 – 0.4 | Partially explained ⚠️ |
| < 0.2 | Poorly explained — this feature is mostly independent |

#### Variance Explained

```python
explained_var = np.sum(fa.components_**2, axis=1) / X_scaled.shape[1]
```

This tells us what fraction of the total information each factor captures. If Factor 1 explains 20%, Factor 2 explains 15%, and Factor 3 explains 10%, together they explain 45% of all the variation in 11 features.

### 5d. Factor Interpretation (Lines 330–350)

```python
for i in range(n_factors):
    col_name = f"Factor {i+1}"
    top_positive = loadings[col_name].nlargest(3)
    top_negative = loadings[col_name].nsmallest(2)
    ...
```

This code automatically identifies the strongest connections for each factor. We look at which features have the largest positive and negative loadings.

**Example interpretation:**
- **Factor 1** loads on: fixed acidity (+0.85), citric acid (+0.68), pH (-0.78) → **"Acidity Profile"**
- **Factor 2** loads on: alcohol (+0.82), density (-0.65) → **"Body/Strength"**
- **Factor 3** loads on: free SO₂ (+0.75), total SO₂ (+0.70) → **"Preservation Chemistry"**

The naming is subjective — we look at what the top features have in common and assign a meaningful label.

### 5e. Factor Visualizations (Lines 353–384)

**Loadings Heatmap:**
```python
fig_heat = px.imshow(loadings.values.round(2), text_auto=True,
                      color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
```

A visual version of the loadings table. Dark red = strong positive loading, dark blue = strong negative, white = no connection. This makes it easy to see which features "belong" to which factor at a glance.

**Factor Scores Scatter Plot:**
```python
fa_df = pd.DataFrame(fa_scores[:, :2], columns=['Factor 1', 'Factor 2'])
fig_fa_scatter = px.scatter(fa_df, x='Factor 1', y='Factor 2')
```

Each dot is a wine, plotted by its scores on Factor 1 and Factor 2. Wines close together are similar in the hidden factor dimensions. Clusters or patterns here preview what the Clustering section will find.

### 5f. Optional: PCA Comparison (Lines 387–416)

When the user checks "Compare with PCA," we also run:

```python
pca = PCA(n_components=n_factors, random_state=42)
pca_scores = pca.fit_transform(X_scaled)
pca_var = pca.explained_variance_ratio_
```

#### PCA vs Factor Analysis — The Key Difference

| | PCA | Factor Analysis |
|-|-----|----------------|
| **Goal** | Find directions of maximum variance | Find hidden variables that cause correlations |
| **Assumption** | No model — just math | There ARE hidden factors causing the data |
| **Noise** | Mixes signal + noise | Separates signal from noise |
| **Variance explained** | Usually higher | Usually lower (but more interpretable) |
| **Best for** | Data compression, visualization | Understanding latent structure |

**Analogy:**
- **PCA** is like taking a photo of a landscape from the angle that captures the most scenery. It maximizes what you see but doesn't explain WHY the landscape looks that way.
- **Factor Analysis** is like figuring out that the landscape is shaped by erosion, tectonic plates, and weather — the hidden causes behind what you see.

---

## Step 6 — Clustering: Grouping Similar Wines <a name="step-6"></a>

This is the second major technique. Instead of reducing features, we're now **grouping observations**.

### 6a. The Concept

**Clustering** asks: "Are there natural groups in this data?"

Imagine you have 1,599 wines plotted in an 11-dimensional space (one axis for each chemical feature). Even though we can't visualize 11 dimensions, the algorithms can detect "clumps" of similar wines.

### 6b. Determining the Number of Clusters (Lines 444–525)

```python
@st.cache_data
def compute_cluster_metrics(X_scaled, max_k=8):
    inertias = []
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return K_range, inertias, silhouettes, ...
```

This loop tries k=2, 3, 4, 5, 6, 7, 8 clusters and records two metrics for each.

#### Method 1: The Elbow Method

**Inertia** = the total distance from each point to its nearest cluster center. Lower = tighter clusters.

```
Inertia
8000  ●
       \
6000    ●
         \
4000      ●   ← "elbow" — diminishing returns start here
           \___
3200         ● ● ● ● ●  ← flat = adding clusters helps less
       2  3  4  5  6  7  8
       Number of Clusters (k)
```

You look for the "elbow" — the point where the curve bends and flattens. More clusters always reduces inertia, but after the elbow, the improvement isn't worth the added complexity.

#### Method 2: Silhouette Score

```
Silhouette Score = (b - a) / max(a, b)

where:
  a = average distance to other points in SAME cluster
  b = average distance to points in NEAREST OTHER cluster
```

| Silhouette Score | Meaning |
|-----------------|---------|
| 1.0 | Perfect — each item is far from other clusters |
| 0.5 – 0.7 | Good — clear cluster structure |
| 0.25 – 0.5 | Moderate — some overlap between clusters |
| < 0.25 | Weak — clusters are not well separated |

The code picks the k with the highest silhouette score as the "suggested" optimal:
```python
best_k = K_range[np.argmax(silhouettes)]
```

### 6c. K-Means Algorithm (Lines 530–538)

```python
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)
```

#### The Math: How K-Means Works

K-Means is beautifully simple. It repeats these steps:

```
Step 1: Place k "centroids" (center points) randomly

Step 2: ASSIGN — each wine goes to its nearest centroid
        distance = √(Σ(wine_featureᵢ - centroid_featureᵢ)²)

Step 3: UPDATE — move each centroid to the average position of its members
        new_centroid = mean of all wines assigned to it

Step 4: Repeat steps 2-3 until centroids stop moving
```

**Visual (2D example):**
```
Iteration 1:            Iteration 2:            Final:
  ★ . . .               .★. . .                 .★. . .
  . . .                  . . .                   . . .
        . . .                .★.                      .★.
  . . .                  . . .                   . . .
  ★ = centroid, . = data point
```

**Parameters:**
- `n_clusters` → How many groups to create (k)
- `random_state=42` → Reproducible results
- `n_init=10` → Run the algorithm 10 times with different random starting positions and keep the best result (avoids bad starting positions)

### 6d. Hierarchical Clustering (Lines 536–537)

```python
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = model.fit_predict(X_scaled)
```

#### The Math: How Hierarchical Clustering Works

Hierarchical clustering works **bottom-up** (agglomerative):

```
START:  Each wine is its own cluster (1,599 clusters)

Step 1: Find the two most similar clusters → merge them (now 1,598 clusters)
Step 2: Find the two most similar → merge (1,597 clusters)
Step 3: Repeat...
...
Step N: Continue until you have k clusters remaining
```

**Ward's method** decides "similarity" by minimizing the total within-cluster variance. When merging two groups, Ward picks the pair that increases the total variance the **least**.

**vs K-Means:**

| | K-Means | Hierarchical |
|-|---------|-------------|
| **Speed** | Fast (good for large data) | Slow (computes all pairwise distances) |
| **Number of k** | Must specify k in advance | Can explore different k from the same tree |
| **Cluster shape** | Assumes spherical clusters | No shape assumption |
| **Deterministic?** | No (random start) | Yes (always same result) |
| **Extra output** | Just labels | Full dendrogram tree |

### 6e. Evaluation Metrics (Lines 546–574)

We compare both algorithms on three metrics:

```python
silhouette_score(X_scaled, labels)         # Higher = better
calinski_harabasz_score(X_scaled, labels)   # Higher = better
davies_bouldin_score(X_scaled, labels)      # Lower = better
```

#### Silhouette Score (already explained above)
- Ranges from -1 to 1
- Measures how similar items are to their own cluster vs. other clusters

#### Calinski-Harabasz Index (Variance Ratio Criterion)

```
CH = (between-cluster variance / within-cluster variance) × (n - k) / (k - 1)
```

**In plain English:** "How spread apart are the cluster centers compared to how spread out the points are within each cluster?"

High CH = clusters are dense internally but far from each other = good.

#### Davies-Bouldin Index

```
DB = (1/k) × Σ max(similarity between cluster i and cluster j)
```

**In plain English:** "On average, how similar is each cluster to its most similar neighbor?"

Low DB = clusters are distinct and well-separated = good. (This metric is the opposite — lower is better!)

### 6f. Cluster Interpretation (Lines 577–631)

```python
cluster_df = df[feature_cols].copy()
cluster_df['Cluster'] = labels
cluster_means = cluster_df.groupby('Cluster').mean()
```

We compute the **average value** of each feature per cluster. This is how we understand what makes each cluster different.

**Standardized cluster means heatmap:**
```python
cluster_means_std = (cluster_means - cluster_means.mean()) / cluster_means.std()
```

This re-standardizes the means so we can compare across features (some features naturally have higher values). Red = above average for that feature, blue = below average.

**Dynamic cluster descriptions (Lines 620–631):**
```python
for c in range(n_clusters):
    row = cluster_means.loc[c]
    top_high = cluster_means_std.loc[c].nlargest(2)
    top_low = cluster_means_std.loc[c].nsmallest(2)
```

For each cluster, we find the two features where it's highest and lowest, then generate a natural-language description like:
> **Cluster 0** (534 wines): High in **alcohol** (11.5), **sulphates** (0.75). Low in **volatile acidity** (0.35), **chlorides** (0.07).

### 6g. Cluster Visualizations (Lines 633–687)

**Scatter in Factor Space:**
```python
scatter_df = pd.DataFrame({
    'Factor 1': fa_scores[:, 0],
    'Factor 2': fa_scores[:, 1],
    'Cluster': [f"Cluster {l}" for l in labels]
})
fig = px.scatter(scatter_df, x='Factor 1', y='Factor 2', color='Cluster')
```

This connects our two analyses — we project the wines onto the Factor Analysis dimensions and color them by cluster. If Factor Analysis and Clustering agree, you'll see colored groups that are clearly separated.

**Dendrogram:**
```python
Z = linkage(X_scaled[sample], method='ward')
dendrogram(Z, truncate_mode='lastp', p=30, ...)
ax.axhline(y=Z[-n_clusters+1, 2], color='red', linestyle='--')
```

The **dendrogram** is a tree showing the merging history:
```
Distance
    │
 20 │     ┌──────┐
    │  ┌──┤      │
 15 │  │  └──┐   │
    │──┤     │   │
 10 │  │  ┌──┤   │
    │  └──┤  │   │
  5 │     └──┘   │
    │            │
  0 └────────────┘
    wine1 wine2 wine3 wine4 wine5
```

The **red dashed line** shows where we "cut" the tree to get our k clusters. Everything below the cut that's connected forms one cluster.

`truncate_mode='lastp', p=30` — We only show the last 30 merges (otherwise 1,599 wines would make an unreadable chart). We also sample 200 wines for performance.

---

## Step 7 — Joint Interpretation: Connecting Factors & Clusters <a name="step-7"></a>

### The Code (Lines 692–740)

This is where everything comes together. We ask: **"Do the clusters correspond to the factors?"**

```python
fa_cluster_df = pd.DataFrame(fa_scores[:, :3], columns=['Factor 1', 'Factor 2', 'Factor 3'])
fa_cluster_df['Cluster'] = labels
fa_cluster_means = fa_cluster_df.groupby('Cluster').mean()
```

We compute the **average factor score** for each cluster. This tells us where each cluster "lives" in factor space.

```python
factor_spreads = fa_cluster_means.std()
dominant_factor = factor_spreads.idxmax()
```

**`std()`** of the cluster means tells us which factor shows the **most difference** between clusters. The factor with the highest standard deviation across clusters is the "dominant" one — it's the dimension that best separates the groups.

**Example result:**
```
              Factor 1    Factor 2    Factor 3
Cluster 0      -0.85        0.32        0.10
Cluster 1       0.12        0.95       -0.20
Cluster 2       0.65       -0.76        0.15

Spread (std)    0.75        0.86        0.18
                            ↑ highest = Factor 2 is dominant
```

If Factor 2 is dominant, the insight is: "The hidden variable represented by Factor 2 (e.g., body/strength) is the primary characteristic that separates wine groups."

### 3D Scatter (Lines 724–740)

```python
fig_3d = px.scatter_3d(scatter_3d, x='Factor 1', y='Factor 2', z='Factor 3',
                        color='Cluster', opacity=0.5)
```

When you have 3+ factors, we create a **3D rotating scatter plot** showing all three factor dimensions colored by cluster. This is one of the most powerful visualizations — you can rotate it to see the cluster separation from different angles.

---

## Step 8 — Discussion & Reflection <a name="step-8"></a>

### The Code (Lines 745–858)

This section provides the narrative required by the assignment. It dynamically generates insights based on the actual results:

```python
sil_val = silhouette_score(X_scaled, labels)
```

The silhouette score determines whether we call the clustering "moderate" or "good":
```python
{"moderate" if sil_val < 0.4 else "good"}
```

### What the Assignment Requires in This Section

1. **Were clusters meaningful?** → Yes, distinct chemical profiles exist
2. **Did Factor Analysis work?** → Yes, factors correspond to intuitive wine concepts
3. **Stability across methods** → K-Means and Hierarchical give similar results
4. **One expected result** → Alcohol is the dominant differentiator
5. **One surprising result** → Volatile acidity matters more than expected
6. **Limitations & next steps** → Honest assessment in a table format

### The Learning Section (Lines 790–858)

This final section provides **plain-language explanations** of every key concept:

- **Unsupervised Learning** — exploring without labels
- **Factors** — hidden variables causing correlations
- **Clusters** — groups of similar items
- **Standardization** — putting features on the same scale

These explanation boxes use the purple `.learn-box` CSS class and include real-life analogies to make abstract concepts accessible.

---

## Key Math Concepts Cheat Sheet <a name="math-cheat-sheet"></a>

| Concept | Formula | What it means |
|---------|---------|---------------|
| **Standardization (z-score)** | z = (x - μ) / σ | How many standard deviations from the mean |
| **Covariance Matrix** | cov(X, Y) = Σ(xᵢ-x̄)(yᵢ-ȳ) / (n-1) | How pairs of features vary together |
| **Eigenvalue** | From Av = λv | How much variance a factor captures |
| **Loading** | Correlation between feature and factor | How strongly a feature "connects" to a factor |
| **Communality** | Σ(loadings²) per feature | % of a feature's variance explained by factors |
| **Inertia** | Σ distance²(point, centroid) | Total "spread" within all clusters (lower = tighter) |
| **Silhouette Score** | (b-a)/max(a,b) | How well-separated clusters are (-1 to +1, higher = better) |
| **Calinski-Harabasz** | between_var / within_var | Ratio of between-cluster to within-cluster variance |
| **Davies-Bouldin** | avg cluster similarity | Average similarity between clusters (lower = better) |
| **Ward Distance** | ΔESS when merging | Increase in error sum of squares from merging two clusters |
| **Correlation** | Σ((x-x̄)(y-ȳ))/√(Σ(x-x̄)²Σ(y-ȳ)²) | How strongly two variables move together (-1 to +1) |

---

## Full Glossary <a name="glossary"></a>

| Term | Simple Definition |
|------|------------------|
| **Unsupervised Learning** | Teaching a computer to find patterns WITHOUT labeled answers |
| **Supervised Learning** | Teaching a computer WITH labeled answers (Week 5) |
| **Feature** | An input variable (column) — e.g., alcohol, pH |
| **Dimensionality Reduction** | Reducing many features to fewer, simpler ones |
| **Factor Analysis** | Finding hidden variables that explain why features are correlated |
| **PCA** | Principal Component Analysis — finds directions of maximum variance |
| **Loading** | How strongly a feature connects to a factor (range: -1 to +1) |
| **Eigenvalue** | How much information (variance) a factor captures |
| **Communality** | Fraction of a feature's variance explained by all factors combined |
| **Scree Plot** | Chart of eigenvalues used to choose the number of factors (look for the "elbow") |
| **Kaiser Criterion** | Rule: keep factors with eigenvalue > 1 |
| **Clustering** | Grouping similar items together based on their features |
| **K-Means** | Algorithm: place k centers, assign points to nearest, adjust centers, repeat |
| **Hierarchical Clustering** | Algorithm: start with each point alone, merge closest pairs until k groups remain |
| **Ward's Method** | A way to decide which clusters to merge (minimizes within-cluster variance) |
| **Dendrogram** | Tree diagram showing the merging history of hierarchical clustering |
| **Centroid** | The "center point" of a cluster (average of all points in the cluster) |
| **Inertia** | Total distance from each point to its cluster center (lower = tighter clusters) |
| **Silhouette Score** | Measures how well each point fits its cluster (-1 to +1, higher = better) |
| **Calinski-Harabasz** | Ratio of between-cluster to within-cluster variance (higher = better) |
| **Davies-Bouldin** | Average similarity between clusters (lower = better) |
| **Standardization** | Scaling features to mean=0, std=1 so they're comparable |
| **Standard Deviation (σ)** | A measure of how spread out data is from the mean |
| **Covariance Matrix** | A table showing how each pair of features varies together |
| **Latent Variable** | A hidden variable you can't measure directly but can infer |
| **Elbow Method** | Plotting inertia vs. k and looking for the "bend" to choose k |
| **Decorator** | A `@something` wrapper that adds behavior to a function (like `@st.cache_data`) |

---

## How the File Connects to the Assignment Requirements

| Assignment Requirement | Where in week6.py | Lines |
|----------------------|-------------------|-------|
| Dataset ≥200 rows, 6-8+ numeric variables | Wine Quality (1,599 rows, 11 features) | 74–98 |
| Define analysis problems | Problem definition boxes for FA and Clustering | 213–223, 423–434 |
| Data preparation & standardization | `standardize()` function, no missing values check | 129–135, 225–232 |
| Factor Analysis (number of factors via Kaiser/Scree) | Eigenvalue computation, Scree plot | 242–279 |
| Factor Analysis results (loadings, communalities) | `fit_fa()`, loadings table, communalities | 284–319 |
| Factor interpretation (name the factors) | Dynamic factor naming | 330–350 |
| Factor visualizations (heatmap, scatter) | Loadings heatmap, factor scatter | 353–384 |
| Optional PCA comparison | PCA section with checkbox toggle | 387–416 |
| At least 2 clustering algorithms | K-Means AND Hierarchical (Ward) | 530–544 |
| Determine number of clusters (Elbow, Silhouette) | `compute_cluster_metrics()`, plots | 444–525 |
| Cluster evaluation metrics | Silhouette, Calinski-Harabasz, Davies-Bouldin | 546–574 |
| Cluster interpretation (profiles, descriptions) | Mean feature values per cluster, descriptions | 577–631 |
| Cluster visualizations (scatter, dendrogram) | Scatter in factor space, dendrogram | 633–687 |
| Joint interpretation (factors + clusters) | Factor means per cluster, dominant factor | 692–740 |
| 3D visualization | 3D scatter colored by cluster | 724–740 |
| Discussion: expected/surprising results | Discussion section | 745–785 |
| Limitations & future work | Limitations table | 777–785 |
| Documentation & plain-language explanations | Learning section with explanation boxes | 790–858 |
| Interactive sidebar controls | Slider, selectbox, checkbox | 104–124 |

---

> **🎓 Tip:** The best way to learn from this code is to **change things and see what happens.** Try:
> - Set factors to 2 vs 6 — watch how interpretability changes
> - Switch between K-Means and Hierarchical — compare cluster assignments
> - Set k=2 vs k=8 — see how silhouette score changes
> - Enable PCA comparison — notice how PCA explains more total variance but factors are harder to name
> - Look at which factor separates clusters most — that's the "hidden variable" driving wine diversity
