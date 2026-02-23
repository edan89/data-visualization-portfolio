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

st.set_page_config(page_title="Week 5: Machine Learning", page_icon="🤖", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .insight-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        padding: 18px; border-radius: 10px;
        border-left: 5px solid #43a047; margin: 12px 0;
        color: #1b5e20 !important;
    }
    .insight-box b { color: #1b5e20 !important; }
    .task-box {
        background: linear-gradient(135deg, #e3f2fd, #e8eaf6);
        padding: 18px; border-radius: 10px;
        border-left: 5px solid #1976d2; margin: 12px 0;
        color: #0d47a1 !important;
    }
    .task-box b { color: #0d47a1 !important; }
    .warn-box {
        background: linear-gradient(135deg, #fff3e0, #fbe9e7);
        padding: 18px; border-radius: 10px;
        border-left: 5px solid #ef6c00; margin: 12px 0;
        color: #bf360c !important;
    }
    .warn-box b { color: #bf360c !important; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    .stMetric label { color: #333 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1f1f1f !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("🤖 Week 5: Supervised Machine Learning")
st.markdown("""
### What is Supervised Learning?
Think of it like teaching a student with an answer key. We show the computer **many examples** 
(data) along with the **correct answers** (labels), and it learns patterns so it can make 
predictions on new data it has never seen.

We tackle **two types** of predictions:
- **Regression** — predicting a **number** (e.g., "How much will the tip be?")
- **Classification** — predicting a **category** (e.g., "Will the tip be Low, Medium, or High?")
""")

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Interactive Controls
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Model Settings")
st.sidebar.markdown("Use these controls to experiment:")

available_features = ['total_bill', 'size', 'sex', 'smoker', 'day', 'time']
selected_features = st.sidebar.multiselect(
    "📋 Features to use", available_features, default=available_features,
    help="Pick which information the model can use to make predictions"
)
if not selected_features:
    selected_features = available_features
    st.sidebar.warning("Need at least one feature — using all.")

n_estimators = st.sidebar.slider(
    "🌲 Number of trees (Random Forest)", 10, 200, 100, step=10,
    help="More trees = usually better, but slower"
)
max_depth = st.sidebar.slider(
    "📏 Max tree depth", 2, 20, 5,
    help="Deeper trees learn more detail but risk 'memorizing' instead of learning"
)
test_size = st.sidebar.slider(
    "📊 Test set size (%)", 20, 40, 30, step=5,
    help="Percentage of data reserved to test the model"
) / 100

# ════════════════════════════════════════════════════════════════════════════
# PREPARE DATA (shared by both tasks)
# ════════════════════════════════════════════════════════════════════════════
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
    
    # Split — same random seed for reproducibility
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=test_size, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=test_size, random_state=42
    )
    
    return (X_train_r, X_test_r, y_train_r, y_test_r,
            X_train_c, X_test_c, y_train_c, y_test_c,
            feature_names, class_names)

(X_train_r, X_test_r, y_train_r, y_test_r,
 X_train_c, X_test_c, y_train_c, y_test_c,
 feature_names, class_names) = prepare_features(df, selected_features, test_size)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA & EDA
# ════════════════════════════════════════════════════════════════════════════
st.header("1. 📋 Data & Exploratory Analysis")

st.markdown("""
🍽️ **Our Dataset: Restaurant Tips**

This data records 244 restaurant visits: the bill amount, the tip left, party size, 
and details like the customer's gender, whether they smoke, day of the week, and meal time.
""")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Rows", df.shape[0])
with col2: st.metric("Columns", df.shape[1])
with col3: st.metric("Numeric Vars", len(df.select_dtypes(include=[np.number]).columns))
with col4: st.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("📈 Summary Statistics")
st.dataframe(df.describe().round(2), use_container_width=True)

# Distribution + Correlation
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Tips & Bills")
    fig_dist = make_subplots(rows=1, cols=2,
                              subplot_titles=["Total Bill ($)", "Tip Amount ($)"])
    fig_dist.add_trace(
        go.Histogram(x=df['total_bill'], marker_color='#3366cc', opacity=0.75, name="Bill"),
        row=1, col=1
    )
    fig_dist.add_trace(
        go.Histogram(x=df['tip'], marker_color='#dc3912', opacity=0.75, name="Tip"),
        row=1, col=2
    )
    fig_dist.update_xaxes(title_text="Total Bill ($)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Tip ($)", row=1, col=2)
    fig_dist.update_yaxes(title_text="Count", row=1, col=1)
    fig_dist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("Tip Quality Distribution")
    tip_counts = df['tip_quality'].value_counts().reindex(['Low', 'Medium', 'High'])
    fig_tq = px.bar(
        x=tip_counts.index, y=tip_counts.values,
        color=tip_counts.index,
        color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60'},
        labels={'x': 'Tip Quality', 'y': 'Count'},
        title="Classification Target: Tip Quality"
    )
    fig_tq.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_tq, use_container_width=True)

# Correlation heatmap
st.subheader("🔗 Correlation Heatmap")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig_corr = px.imshow(
    corr.round(2), text_auto=True, color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1, title="How strongly are the numeric variables related?"
)
fig_corr.update_layout(height=400)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption("📊 Values close to +1 or −1 mean strong relationships. Total bill and tip are strongly correlated (~0.68).")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REGRESSION
# ════════════════════════════════════════════════════════════════════════════
st.header("2. 📈 Regression: Predicting Tip Amount")

st.markdown("""
<div class="task-box">
<b>🎯 Prediction Task:</b> Can we predict how much a customer will tip based on their bill, 
party size, and other details?<br><br>
<b>Target variable:</b> <code>tip</code> (a number in dollars)<br>
<b>Why it matters:</b> Understanding tipping patterns helps restaurants plan staffing and estimate 
server earnings. Even a rough prediction is better than just guessing the average!
</div>
""", unsafe_allow_html=True)

# --- Data Preparation Info ---
with st.expander("📋 How we prepared the data"):
    st.markdown(f"""
    - **Missing values:** None in this dataset ✅
    - **Categorical encoding:** Converted text categories (e.g., Male/Female) to numbers 
      using Label Encoding so the ML models can understand them
    - **Feature scaling:** Not applied — tree-based models don't need it, and linear 
      regression handles it naturally for this dataset
    - **Train–Test split:** {int((1-test_size)*100)}% training / {int(test_size*100)}% testing 
      (random split, test data never used during training)
    """)

# Train models
@st.cache_data
def train_regression_models(X_train, X_test, y_train, y_test, n_est, m_depth):
    results = {}
    
    # Baseline: predict the mean
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_pred_base_train = baseline.predict(X_train)
    results['Baseline (Mean)'] = {
        'model': baseline, 'predictions': y_pred_base,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_base)),
        'mae': mean_absolute_error(y_test, y_pred_base),
        'r2': r2_score(y_test, y_pred_base),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_base_train)),
        'r2_train': r2_score(y_train, y_pred_base_train),
    }
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_lr_train = lr.predict(X_train)
    results['Linear Regression'] = {
        'model': lr, 'predictions': y_pred_lr,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'r2': r2_score(y_test, y_pred_lr),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_lr_train)),
        'r2_train': r2_score(y_train, y_pred_lr_train),
    }
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=m_depth, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_rf_train = rf.predict(X_train)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    
    results['Random Forest'] = {
        'model': rf, 'predictions': y_pred_rf,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_rf_train)),
        'r2_train': r2_score(y_train, y_pred_rf_train),
        'cv_rmse_mean': -cv_scores.mean(),
        'cv_rmse_std': cv_scores.std(),
    }
    
    return results

reg_results = train_regression_models(X_train_r, X_test_r, y_train_r, y_test_r,
                                       n_estimators, max_depth)

# --- 2.1 Baseline ---
st.subheader("2.1 Baseline Model")
st.markdown("""
Our simplest "model" always guesses the **average tip** no matter what. 
Any good model must do better than this!
""")
col1, col2, col3 = st.columns(3)
base = reg_results['Baseline (Mean)']
with col1: st.metric("RMSE", f"${base['rmse']:.3f}")
with col2: st.metric("MAE", f"${base['mae']:.3f}")
with col3: st.metric("R²", f"{base['r2']:.3f}")

st.caption("💡 RMSE/MAE = how far off predictions are (lower = better). R² = how much variation the model explains (higher = better, max 1.0).")

# --- 2.2 Model Comparison ---
st.subheader("2.2 Model Comparison")

comp_data = []
for name, r in reg_results.items():
    row = {
        'Model': name,
        'RMSE (Test)': f"${r['rmse']:.3f}",
        'MAE (Test)': f"${r['mae']:.3f}",
        'R² (Test)': f"{r['r2']:.3f}",
        'RMSE (Train)': f"${r['rmse_train']:.3f}",
        'R² (Train)': f"{r['r2_train']:.3f}",
    }
    if 'cv_rmse_mean' in r:
        row['CV RMSE (5-fold)'] = f"${r['cv_rmse_mean']:.3f} ± {r['cv_rmse_std']:.3f}"
    else:
        row['CV RMSE (5-fold)'] = '—'
    comp_data.append(row)

st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

# Best model identification
best_reg = min(reg_results.items(), key=lambda x: x[1]['rmse'])
best_reg_name = best_reg[0]
baseline_rmse = reg_results['Baseline (Mean)']['rmse']
improvement = ((baseline_rmse - best_reg[1]['rmse']) / baseline_rmse) * 100

st.markdown(f"""
<div class="insight-box">
<b>🏆 Best Regression Model: {best_reg_name}</b><br><br>
• <b>RMSE = ${best_reg[1]['rmse']:.3f}</b> — predictions are off by about ${best_reg[1]['rmse']:.2f} on average<br>
• <b>R² = {best_reg[1]['r2']:.3f}</b> — the model explains {best_reg[1]['r2']*100:.1f}% of the variation in tips<br>
• <b>{improvement:.1f}% improvement</b> over the baseline (just guessing the average)<br><br>
This means our model actually learned useful patterns from the data!
</div>
""", unsafe_allow_html=True)

# Overfitting check
st.subheader("2.3 Overfitting Check (Train vs Test)")
st.markdown("""
If a model scores **much better on training data** than on test data, it may be 
"memorizing" rather than learning — like a student who memorizes answers without understanding.
""")

overfit_data = []
for name, r in reg_results.items():
    if name == 'Baseline (Mean)':
        continue
    gap = r['r2_train'] - r['r2']
    status = "✅ Good" if abs(gap) < 0.15 else "⚠️ Slight overfitting" if abs(gap) < 0.3 else "❌ Overfitting"
    overfit_data.append({
        'Model': name,
        'R² (Train)': f"{r['r2_train']:.3f}",
        'R² (Test)': f"{r['r2']:.3f}",
        'Gap': f"{gap:.3f}",
        'Status': status
    })
st.dataframe(pd.DataFrame(overfit_data), use_container_width=True, hide_index=True)

# ── Regression Visualizations ──
st.subheader("2.4 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Predicted vs Actual
    best_preds = reg_results[best_reg_name]['predictions']
    fig_pva = go.Figure()
    fig_pva.add_trace(go.Scatter(
        x=y_test_r, y=best_preds, mode='markers',
        marker=dict(color='#3366cc', size=7, opacity=0.6),
        name='Predictions'
    ))
    # 45-degree reference line
    min_val = min(y_test_r.min(), best_preds.min())
    max_val = max(y_test_r.max(), best_preds.max())
    fig_pva.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='red', dash='dash', width=2),
        name='Perfect prediction'
    ))
    fig_pva.update_layout(
        title=f"Predicted vs Actual Tips ({best_reg_name})",
        xaxis_title="Actual Tip ($)", yaxis_title="Predicted Tip ($)", height=420,
    )
    fig_pva.add_annotation(
        x=0.05, y=0.95, xref='paper', yref='paper',
        text=f"RMSE = ${reg_results[best_reg_name]['rmse']:.3f}<br>R² = {reg_results[best_reg_name]['r2']:.3f}",
        showarrow=False, font=dict(size=12), bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#ccc', borderwidth=1
    )
    st.plotly_chart(fig_pva, use_container_width=True)
    st.caption("📊 Points near the red dashed line = accurate predictions. Scattered points = larger errors.")

with col2:
    # Residual plot
    residuals = y_test_r - best_preds
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=best_preds, y=residuals, mode='markers',
        marker=dict(color='#e74c3c', size=7, opacity=0.6),
        name='Residuals'
    ))
    fig_res.add_hline(y=0, line_dash='dash', line_color='gray', line_width=2)
    fig_res.update_layout(
        title=f"Residuals ({best_reg_name})",
        xaxis_title="Predicted Tip ($)", yaxis_title="Error (Actual − Predicted)",
        height=420,
    )
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption("📊 Residuals should be scattered randomly around zero. Patterns may indicate the model is missing something.")

# Feature Importance (Random Forest)
if 'Random Forest' in reg_results:
    st.subheader("2.5 Feature Importance (Random Forest)")
    rf_model = reg_results['Random Forest']['model']
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names, 'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig_fi = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Blues',
        title="Which features matter most for predicting tips?"
    )
    fig_fi.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    top_feature = fi_df.iloc[-1]['Feature']
    st.caption(f"📊 **{top_feature}** is the most important predictor. This makes sense intuitively — larger bills tend to have larger tips!")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
st.header("3. 🏷️ Classification: Predicting Tip Quality")

st.markdown("""
<div class="task-box">
<b>🎯 Prediction Task:</b> Can we classify whether a customer's tip will be 
<b>Low</b> (≤ $2), <b>Medium</b> ($2–$3.50), or <b>High</b> (> $3.50)?<br><br>
<b>Target variable:</b> <code>tip_quality</code> (a category we created by binning the tip amount)<br>
<b>Why it matters:</b> A restaurant could use this to identify potentially high-tipping 
customers for premium seating or to understand what drives generous tipping.
</div>
""", unsafe_allow_html=True)

# Class distribution
st.subheader("3.0 Class Distribution")
class_counts = pd.Series(y_train_c).value_counts().sort_index()
class_labels = class_names
class_dist = pd.DataFrame({
    'Category': class_labels,
    'Training Samples': [class_counts.get(i, 0) for i in range(len(class_labels))],
})
class_dist['Percentage'] = (class_dist['Training Samples'] / class_dist['Training Samples'].sum() * 100).round(1)
st.dataframe(class_dist, use_container_width=True, hide_index=True)

# Check imbalance
max_pct = class_dist['Percentage'].max()
if max_pct > 60:
    st.markdown("""
    <div class="warn-box">
    <b>⚠️ Class Imbalance Detected!</b><br>
    One category has many more samples than the others. This means accuracy alone can be 
    misleading — a model that always guesses the biggest category could look "accurate" 
    without actually being useful. We'll look at <b>F1-score</b> and <b>Recall</b> for a 
    fairer picture.
    </div>
    """, unsafe_allow_html=True)

# Train classification models
@st.cache_data
def train_classification_models(X_train, X_test, y_train, y_test, n_est, m_depth, n_classes):
    results = {}
    
    # Baseline: predict most frequent class
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_pred_base_train = baseline.predict(X_train)
    results['Baseline (Majority)'] = {
        'model': baseline, 'predictions': y_pred_base,
        'accuracy': accuracy_score(y_test, y_pred_base),
        'precision': precision_score(y_test, y_pred_base, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_base, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_base, average='weighted', zero_division=0),
        'accuracy_train': accuracy_score(y_train, y_pred_base_train),
        'cm': confusion_matrix(y_test, y_pred_base),
    }
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    y_pred_log_train = log_reg.predict(X_train)
    results['Logistic Regression'] = {
        'model': log_reg, 'predictions': y_pred_log,
        'accuracy': accuracy_score(y_test, y_pred_log),
        'precision': precision_score(y_test, y_pred_log, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_log, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_log, average='weighted', zero_division=0),
        'accuracy_train': accuracy_score(y_train, y_pred_log_train),
        'cm': confusion_matrix(y_test, y_pred_log),
    }
    
    # Random Forest Classifier
    rf_cls = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, random_state=42)
    rf_cls.fit(X_train, y_train)
    y_pred_rf = rf_cls.predict(X_test)
    y_pred_rf_train = rf_cls.predict(X_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_cls, X_train, y_train, cv=5, scoring='f1_weighted')
    
    results['Random Forest'] = {
        'model': rf_cls, 'predictions': y_pred_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'accuracy_train': accuracy_score(y_train, y_pred_rf_train),
        'cm': confusion_matrix(y_test, y_pred_rf),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
    }
    
    return results

cls_results = train_classification_models(X_train_c, X_test_c, y_train_c, y_test_c,
                                           n_estimators, max_depth, len(class_names))

# --- 3.1 Baseline ---
st.subheader("3.1 Baseline Model")
st.markdown("Our baseline always guesses the **most common category** — like always betting on the favorite team.")
base_cls = cls_results['Baseline (Majority)']
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Accuracy", f"{base_cls['accuracy']:.1%}")
with col2: st.metric("Precision", f"{base_cls['precision']:.1%}")
with col3: st.metric("Recall", f"{base_cls['recall']:.1%}")
with col4: st.metric("F1-score", f"{base_cls['f1']:.1%}")

# --- 3.2 Model Comparison ---
st.subheader("3.2 Model Comparison")

cls_comp = []
for name, r in cls_results.items():
    row = {
        'Model': name,
        'Accuracy': f"{r['accuracy']:.1%}",
        'Precision': f"{r['precision']:.1%}",
        'Recall': f"{r['recall']:.1%}",
        'F1-score': f"{r['f1']:.1%}",
        'Accuracy (Train)': f"{r['accuracy_train']:.1%}",
    }
    if 'cv_f1_mean' in r:
        row['CV F1 (5-fold)'] = f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}"
    else:
        row['CV F1 (5-fold)'] = '—'
    cls_comp.append(row)

st.dataframe(pd.DataFrame(cls_comp), use_container_width=True, hide_index=True)

# Best model
best_cls = max(cls_results.items(), key=lambda x: x[1]['f1'])
best_cls_name = best_cls[0]
baseline_f1 = cls_results['Baseline (Majority)']['f1']
cls_improvement = ((best_cls[1]['f1'] - baseline_f1) / max(baseline_f1, 0.001)) * 100

st.markdown(f"""
<div class="insight-box">
<b>🏆 Best Classification Model: {best_cls_name}</b><br><br>
• <b>F1-score = {best_cls[1]['f1']:.1%}</b> — balances precision and recall<br>
• <b>Accuracy = {best_cls[1]['accuracy']:.1%}</b><br>
• <b>{cls_improvement:.0f}% improvement</b> in F1 over the baseline<br><br>
We prioritize F1-score over accuracy because our classes are somewhat imbalanced. 
F1 ensures the model is good at finding ALL categories, not just the most common one.
</div>
""", unsafe_allow_html=True)

# Overfitting check
st.subheader("3.3 Overfitting Check")
overfit_cls = []
for name, r in cls_results.items():
    if name == 'Baseline (Majority)':
        continue
    gap = r['accuracy_train'] - r['accuracy']
    status = "✅ Good" if abs(gap) < 0.1 else "⚠️ Slight overfitting" if abs(gap) < 0.2 else "❌ Overfitting"
    overfit_cls.append({
        'Model': name,
        'Accuracy (Train)': f"{r['accuracy_train']:.1%}",
        'Accuracy (Test)': f"{r['accuracy']:.1%}",
        'Gap': f"{gap:.1%}",
        'Status': status
    })
st.dataframe(pd.DataFrame(overfit_cls), use_container_width=True, hide_index=True)

# ── Classification Visualizations ──
st.subheader("3.4 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Confusion Matrix
    best_cm = cls_results[best_cls_name]['cm']
    fig_cm = px.imshow(
        best_cm, text_auto=True, color_continuous_scale='Blues',
        x=class_names, y=class_names,
        title=f"Confusion Matrix ({best_cls_name})"
    )
    fig_cm.update_xaxes(title="Predicted Category")
    fig_cm.update_yaxes(title="Actual Category")
    fig_cm.update_layout(height=420)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("📊 Diagonal = correct predictions. Off-diagonal = mistakes. Darker blue = more samples.")

with col2:
    # ROC Curves (One-vs-Rest)
    fig_roc = go.Figure()
    colors_roc = ['#e74c3c', '#f39c12', '#27ae60']
    
    y_test_bin = label_binarize(y_test_c, classes=list(range(len(class_names))))
    
    for i, (cls_name_i, color) in enumerate(zip(class_names, colors_roc)):
        try:
            if hasattr(cls_results[best_cls_name]['model'], 'predict_proba'):
                y_prob = cls_results[best_cls_name]['model'].predict_proba(X_test_c)
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f'{cls_name_i} (AUC = {roc_auc:.2f})',
                    line=dict(color=color, width=2)
                ))
        except Exception:
            pass
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='gray', dash='dash'), name='Random guess'
    ))
    fig_roc.update_layout(
        title=f"ROC Curves ({best_cls_name})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("📊 Curves closer to the top-left corner = better performance. AUC near 1.0 = excellent; 0.5 = random guess.")

# Feature importance (Classification RF)
if 'Random Forest' in cls_results:
    st.subheader("3.5 Feature Importance (Classification)")
    rf_cls_model = cls_results['Random Forest']['model']
    imp_cls = rf_cls_model.feature_importances_
    fi_cls_df = pd.DataFrame({
        'Feature': feature_names, 'Importance': imp_cls
    }).sort_values('Importance', ascending=True)
    
    fig_fi_cls = px.bar(
        fi_cls_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Greens',
        title="Which features matter most for classifying tip quality?"
    )
    fig_fi_cls.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_fi_cls, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INTERPRETATION & DISCUSSION
# ════════════════════════════════════════════════════════════════════════════
st.header("4. 💡 Interpretation & Discussion")

st.subheader("Summary of All Models")

summary_all = []
for name, r in reg_results.items():
    summary_all.append({
        'Task': 'Regression', 'Model': name,
        'Key Metric': f"RMSE = ${r['rmse']:.3f}",
        'Secondary Metric': f"R² = {r['r2']:.3f}",
        'vs Baseline': f"{((baseline_rmse - r['rmse'])/baseline_rmse*100):.1f}% better" if name != 'Baseline (Mean)' else '—'
    })
for name, r in cls_results.items():
    summary_all.append({
        'Task': 'Classification', 'Model': name,
        'Key Metric': f"F1 = {r['f1']:.1%}",
        'Secondary Metric': f"Accuracy = {r['accuracy']:.1%}",
        'vs Baseline': f"{((r['f1'] - baseline_f1)/max(baseline_f1, 0.001)*100):.0f}% better" if name != 'Baseline (Majority)' else '—'
    })
st.dataframe(pd.DataFrame(summary_all), use_container_width=True, hide_index=True)

st.subheader("Key Insights")

# Determine insights dynamically
lr_rmse = reg_results['Linear Regression']['rmse']
rf_rmse = reg_results['Random Forest']['rmse']
rmse_diff_pct = abs(lr_rmse - rf_rmse) / lr_rmse * 100

st.markdown(f"""
### 1. 📈 Regression Results — Predicting Tip Amounts

The **{best_reg_name}** achieved the lowest error (RMSE = ${best_reg[1]['rmse']:.3f}, 
MAE = ${best_reg[1]['mae']:.3f}), clearly outperforming the baseline that just guesses 
the average tip.

{"**Interesting finding:** The difference between Linear Regression and Random Forest is relatively small (≈" + f"{rmse_diff_pct:.0f}%" + "), suggesting that the relationship between features and tips is mostly **linear** — bigger bills simply lead to bigger tips." if rmse_diff_pct < 15 else "**Interesting finding:** Random Forest outperforms Linear Regression by a meaningful margin (" + f"{rmse_diff_pct:.0f}%" + "), suggesting there are **non-linear patterns** in tipping behavior that a simple line can't capture."}

This matches our expectations: total bill is the dominant predictor, and the relationship 
is fairly straightforward.

### 2. 🏷️ Classification Results — Tip Quality Categories

The **{best_cls_name}** performed best with F1 = {best_cls[1]['f1']:.1%}, a clear improvement 
over always guessing the most common category.

{"**Surprising result:** Despite using the same underlying data, classification is notably harder than regression. This is because putting tips into categories (Low/Medium/High) loses information — a $3.49 tip and a $3.51 tip are almost identical but end up in different categories!" if best_cls[1]['f1'] < 0.7 else "The classifier does a good job distinguishing between tip quality levels, especially after considering the class imbalance in the data."}

### 3. 🤔 What Makes People Tip More?

Both regression and classification models agree: **total bill** is the #1 predictor. This makes 
intuitive sense — tips are usually a percentage of the bill. **Party size** is the second most 
important factor, likely because larger groups have larger bills.
""")

st.subheader("Limitations & What We'd Try Next")

st.markdown("""
| Limitation | Impact | Potential Fix |
|-----------|--------|---------------|
| **Small dataset** (244 rows) | Models have limited data to learn from | Collect more restaurant data |
| **Single restaurant** | Results may not apply to other restaurants | Test on data from multiple venues |
| **Few features** | Can't capture all factors (food quality, service speed, etc.) | Add more variables |
| **Category boundaries** | Low/Medium/High cutoffs are arbitrary | Try different binning strategies |
| **Historical data** (1990s) | Tipping norms have likely changed | Use more recent data |

**Next steps we would try:**
- 🔧 **Hyperparameter tuning** — systematic search for the best model settings
- 📊 **More models** — Gradient Boosting (XGBoost) might capture more complex patterns
- 🔍 **Feature engineering** — create new features like "tip percentage" or "bill per person"
- ✅ **More data** — more observations would improve all models
""")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
**📚 Methods Used:**
- **Regression:** Baseline (Mean), Linear Regression, Random Forest Regressor
- **Classification:** Baseline (Majority Class), Logistic Regression, Random Forest Classifier
- **Evaluation:** Train/Test split, 5-fold Cross-Validation, multiple metrics per task
- **Data:** seaborn "tips" dataset (n = 244)
""")
st.caption("Week 5 Assignment: Supervised Machine Learning | Data Visualization Course")
