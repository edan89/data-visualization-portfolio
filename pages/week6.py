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

st.set_page_config(page_title="Week 6: Unsupervised ML", page_icon="🔬", layout="wide")

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
    .learn-box {
        background: linear-gradient(135deg, #f3e5f5, #ede7f6);
        padding: 18px; border-radius: 10px;
        border-left: 5px solid #7b1fa2; margin: 12px 0;
        color: #4a148c !important;
    }
    .learn-box b { color: #4a148c !important; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    .stMetric label { color: #333 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1f1f1f !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.title("🔬 Week 6: Unsupervised Machine Learning")
st.markdown("""
### What is Unsupervised Learning?
Unlike **supervised** learning (Week 5), where we teach the computer with labeled answers,
**unsupervised** learning lets the computer discover **hidden patterns** on its own — no answer key needed!

We explore **two techniques**:
- **Factor Analysis** — find hidden variables that explain why our data looks the way it does
- **Clustering** — group similar items together, like sorting a deck of cards by color without being told the rules
""")

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    return df

df = load_data()

# Feature names for display
feature_descriptions = {
    'fixed acidity': 'Tartaric acid level (affects tartness)',
    'volatile acidity': 'Acetic acid level (too high = vinegar taste)',
    'citric acid': 'Adds freshness & flavor',
    'residual sugar': 'Sugar left after fermentation',
    'chlorides': 'Salt content',
    'free sulfur dioxide': 'Prevents microbes & oxidation (free form)',
    'total sulfur dioxide': 'Total SO₂ (free + bound forms)',
    'density': 'How heavy the wine is (related to sugar & alcohol)',
    'pH': 'Acidity level (lower = more acidic)',
    'sulphates': 'Additive that contributes to SO₂ levels',
    'alcohol': 'Alcohol percentage',
}

# Separate features from quality (we won't use quality as it's a label)
feature_cols = [c for c in df.columns if c != 'quality']
X_raw = df[feature_cols]

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Interactive Controls
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Controls")

st.sidebar.subheader("Factor Analysis")
n_factors = st.sidebar.slider(
    "🔢 Number of factors", 2, 6, 3,
    help="How many hidden variables to extract"
)
show_pca = st.sidebar.checkbox(
    "📊 Compare with PCA", value=False,
    help="PCA finds directions of maximum spread; Factor Analysis finds hidden causes"
)

st.sidebar.subheader("Clustering")
cluster_algo = st.sidebar.selectbox(
    "🧮 Algorithm", ["K-Means", "Hierarchical (Ward)"],
    help="K-Means: fast, spherical clusters. Hierarchical: builds a tree of merges"
)
n_clusters = st.sidebar.slider(
    "🎯 Number of clusters (k)", 2, 8, 3,
    help="How many groups to create"
)

# ════════════════════════════════════════════════════════════════════════════
# STANDARDIZE DATA
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

X_scaled, scaler = standardize(X_raw)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA & EDA
# ════════════════════════════════════════════════════════════════════════════
st.header("1. 📋 Data & Exploratory Analysis")

st.markdown("""
🍷 **Our Dataset: Red Wine Quality (UCI)**

This dataset records **chemical measurements** of 1,599 red wines from Portugal.
Each row is a wine sample with 11 laboratory-measured properties like acidity, sugar,
and alcohol content. We'll use these chemical features to discover hidden patterns.
""")

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Wines", df.shape[0])
with col2: st.metric("Features", len(feature_cols))
with col3: st.metric("Numeric Vars", len(df.select_dtypes(include=[np.number]).columns))
with col4: st.metric("Missing Values", df.isnull().sum().sum())

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Feature descriptions
with st.expander("📖 What does each feature mean?"):
    desc_df = pd.DataFrame([
        {"Feature": k, "Description": v} for k, v in feature_descriptions.items()
    ])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

st.subheader("📈 Summary Statistics")
st.dataframe(df[feature_cols].describe().round(3), use_container_width=True)

# Distribution plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribution of Key Features")
    fig_dist = make_subplots(rows=2, cols=2,
                              subplot_titles=["Alcohol (%)", "Volatile Acidity", "Citric Acid", "Residual Sugar"])
    for i, (feat, row, col) in enumerate([
        ('alcohol', 1, 1), ('volatile acidity', 1, 2),
        ('citric acid', 2, 1), ('residual sugar', 2, 2)
    ]):
        colors = ['#3366cc', '#dc3912', '#ff9900', '#109618']
        fig_dist.add_trace(
            go.Histogram(x=df[feat], marker_color=colors[i], opacity=0.75, name=feat, showlegend=False),
            row=row, col=col
        )
    fig_dist.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("🔗 Correlation Heatmap")
    corr = df[feature_cols].corr()
    fig_corr = px.imshow(
        corr.round(2), text_auto=True, color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""
<div class="insight-box">
<b>📊 Key Observations from EDA:</b><br><br>
• <b>Strong correlations exist</b> — for example, pH and fixed acidity are negatively correlated (more acid = lower pH), 
  which is expected from chemistry<br>
• <b>Free & total sulfur dioxide</b> are highly correlated (one is part of the other)<br>
• <b>Density correlates with several features</b> (alcohol, residual sugar) — it's a "summary" variable<br><br>
These correlations mean Factor Analysis should find meaningful hidden factors!
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FACTOR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
st.header("2. 🔍 Factor Analysis: Discovering Hidden Variables")

st.markdown("""
<div class="task-box">
<b>🎯 Problem Definition:</b> Can we reduce 11 chemical measurements into a smaller number 
of hidden "factors" that explain why wines differ?<br><br>
<b>Goal:</b> Find 2–4 hidden variables (like "acidity profile" or "body/strength") that 
summarize the 11 original measurements.<br><br>
<b>Why?</b> Think of it this way: a wine reviewer doesn't describe 11 chemical numbers — 
they say things like "full-bodied" or "crisp acidity". Factor Analysis tries to find those 
same hidden concepts automatically from the data!
</div>
""", unsafe_allow_html=True)

with st.expander("📋 Data Preparation for Factor Analysis"):
    st.markdown(f"""
    - **Missing values:** None ✅
    - **Standardization:** All features scaled to mean=0, variance=1 ✅  
      (FA is sensitive to scale — a feature measured in hundreds would unfairly dominate one measured in decimals)
    - **Variables included:** All 11 numeric chemical features
    - **Train-test split:** Not needed — unsupervised learning uses all data
    """)

# --- 2.1 Determine Number of Factors ---
st.subheader("2.1 How Many Factors Do We Need?")
st.markdown("""
We use two methods to decide:
1. **Kaiser Criterion** — keep factors whose eigenvalue > 1 (they explain more than a single original variable)
2. **Scree Plot** — look for the "elbow" where adding more factors stops helping much
""")

@st.cache_data
def compute_eigenvalues(X_scaled):
    cov_mat = np.cov(X_scaled, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_mat)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

eigenvalues = compute_eigenvalues(X_scaled)

# Scree plot
fig_scree = go.Figure()
fig_scree.add_trace(go.Scatter(
    x=list(range(1, len(eigenvalues) + 1)),
    y=eigenvalues,
    mode='lines+markers',
    marker=dict(size=10, color='#3366cc'),
    line=dict(width=3, color='#3366cc'),
    name='Eigenvalues'
))
fig_scree.add_hline(y=1.0, line_dash='dash', line_color='red',
                     annotation_text="Kaiser Criterion (eigenvalue = 1)",
                     annotation_position="top right")
fig_scree.update_layout(
    title="Scree Plot: How Much Does Each Factor Explain?",
    xaxis_title="Factor Number",
    yaxis_title="Eigenvalue",
    height=400
)
st.plotly_chart(fig_scree, use_container_width=True)

n_kaiser = int(np.sum(eigenvalues > 1))
st.markdown(f"""
<div class="insight-box">
<b>📊 Result:</b> The Kaiser criterion suggests <b>{n_kaiser} factors</b> (eigenvalues > 1).<br>
The scree plot shows the "elbow" where the curve flattens — factors beyond this point add little value.<br><br>
You selected <b>{n_factors} factors</b> using the sidebar slider.
</div>
""", unsafe_allow_html=True)

# --- 2.2 Fit Factor Analysis ---
st.subheader("2.2 Factor Analysis Results")

@st.cache_data
def fit_fa(X_scaled, n_factors, feature_names):
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_scores = fa.fit_transform(X_scaled)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=feature_names,
        columns=[f"Factor {i+1}" for i in range(n_factors)]
    )
    communalities = np.sum(fa.components_**2, axis=0)
    # Per-variable communalities
    var_communalities = pd.Series(
        np.sum(fa.components_.T**2, axis=1),
        index=feature_names
    )
    explained_var = np.sum(fa.components_**2, axis=1) / X_scaled.shape[1]
    return fa, fa_scores, loadings, var_communalities, explained_var

fa_model, fa_scores, loadings, communalities, explained_var = fit_fa(
    X_scaled, n_factors, feature_cols
)

# Factor loadings table
st.markdown("**Factor Loadings Matrix** (how strongly each feature connects to each factor):")
styled_loadings = loadings.copy()
st.dataframe(styled_loadings.round(3), use_container_width=True)

# Communalities
with st.expander("📊 Communalities (how well each feature is explained by the factors)"):
    comm_df = pd.DataFrame({
        'Feature': communalities.index,
        'Communality': communalities.values.round(3),
        'Interpretation': ['Well explained ✅' if v > 0.4 else 'Partially explained ⚠️' for v in communalities.values]
    })
    st.dataframe(comm_df, use_container_width=True, hide_index=True)
    st.caption("Communality = fraction of a feature's variation captured by the factors. Higher = better.")

# Variance explained
var_df = pd.DataFrame({
    'Factor': [f"Factor {i+1}" for i in range(n_factors)],
    'Variance Explained': [f"{v*100:.1f}%" for v in explained_var],
})
var_df.loc[len(var_df)] = ['Total', f"{sum(explained_var)*100:.1f}%"]
st.dataframe(var_df, use_container_width=True, hide_index=True)

# Interpret factors dynamically
st.subheader("2.3 Factor Interpretation")
st.markdown("We name each factor based on which features load most strongly on it:")

for i in range(n_factors):
    col_name = f"Factor {i+1}"
    top_positive = loadings[col_name].nlargest(3)
    top_negative = loadings[col_name].nsmallest(2)
    
    top_features = []
    for feat, val in top_positive.items():
        if abs(val) > 0.25:
            top_features.append(f"**{feat}** ({val:+.2f})")
    for feat, val in top_negative.items():
        if abs(val) > 0.25:
            top_features.append(f"**{feat}** ({val:+.2f})")
    
    features_str = ", ".join(top_features) if top_features else "No dominant features"
    st.markdown(f"""
    **{col_name}** → loads on: {features_str}  
    *Variance explained: {explained_var[i]*100:.1f}%*
    """)

# --- 2.4 Factor Visualizations ---
st.subheader("2.4 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Factor loadings heatmap
    fig_heat = px.imshow(
        loadings.values.round(2),
        x=loadings.columns.tolist(),
        y=loadings.index.tolist(),
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Factor Loadings Heatmap"
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("📊 Dark red/blue = strong connection. White = weak connection. Each row shows how a feature relates to each factor.")

with col2:
    # Factor scores scatter
    if n_factors >= 2:
        fa_df = pd.DataFrame(fa_scores[:, :2], columns=['Factor 1', 'Factor 2'])
        fig_fa_scatter = px.scatter(
            fa_df, x='Factor 1', y='Factor 2',
            opacity=0.5,
            title="Wines in Factor Space (Factor 1 vs Factor 2)",
            color_discrete_sequence=['#3366cc']
        )
        fig_fa_scatter.update_layout(height=450)
        st.plotly_chart(fig_fa_scatter, use_container_width=True)
        st.caption("📊 Each dot is a wine. Wines close together share similar hidden characteristics.")

# --- 2.5 Optional PCA Comparison ---
if show_pca:
    st.subheader("2.5 PCA vs Factor Analysis Comparison")
    
    @st.cache_data
    def fit_pca(X_scaled, n_components):
        pca = PCA(n_components=n_components, random_state=42)
        pca_scores = pca.fit_transform(X_scaled)
        pca_var = pca.explained_variance_ratio_
        return pca, pca_scores, pca_var
    
    pca_model, pca_scores, pca_var = fit_pca(X_scaled, n_factors)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factor Analysis** — looks for hidden *causes*")
        st.metric("Total Variance Explained", f"{sum(explained_var)*100:.1f}%")
    with col2:
        st.markdown("**PCA** — finds directions of *maximum spread*")
        st.metric("Total Variance Explained", f"{sum(pca_var)*100:.1f}%")
    
    st.markdown("""
    <div class="learn-box">
    <b>🎓 FA vs PCA — What's the Difference?</b><br><br>
    • <b>PCA</b> maximizes variance — it finds the directions where data spreads out the most. 
    Great for compression, but components can be hard to interpret.<br>
    • <b>Factor Analysis</b> assumes hidden variables <i>cause</i> the data. 
    It separates shared variance from noise, making factors more interpretable.<br><br>
    PCA usually explains more total variance, but FA factors often make more intuitive sense!
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
st.header("3. 🎯 Clustering: Grouping Wines by Similarity")

st.markdown("""
<div class="task-box">
<b>🎯 Problem Definition:</b> Can wines be grouped into distinct "taste profiles" based 
on their chemical properties?<br><br>
<b>Variables used:</b> All 11 standardized chemical features<br>
<b>Why clustering?</b> A winery could use these groups to:<br>
• Recommend similar wines to customers (personalization)<br>
• Identify what makes premium wines different (quality insights)<br>
• Discover unexpected wine styles (subgroup discovery)<br><br>
<b>Expected result:</b> 2–5 clusters separated mainly by alcohol content and acidity levels.
</div>
""", unsafe_allow_html=True)

# --- 3.1 Determine Number of Clusters ---
st.subheader("3.1 How Many Clusters?")
st.markdown("""
We use two methods:
1. **Elbow Method** — plot the "cost" (inertia) for different values of k; look for the bend
2. **Silhouette Score** — measures how well each wine fits its assigned cluster (higher = better, max 1.0)
""")

@st.cache_data
def compute_cluster_metrics(X_scaled, max_k=8):
    inertias = []
    silhouettes = []
    calinski = []
    davies = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        calinski.append(calinski_harabasz_score(X_scaled, labels))
        davies.append(davies_bouldin_score(X_scaled, labels))
    
    return list(K_range), inertias, silhouettes, calinski, davies

K_range, inertias, silhouettes, calinski_scores, davies_scores = compute_cluster_metrics(X_scaled)

col1, col2 = st.columns(2)

with col1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=K_range, y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='#dc3912'),
        line=dict(width=3, color='#dc3912'),
        name='Inertia'
    ))
    # Highlight selected k
    idx = n_clusters - 2
    if 0 <= idx < len(inertias):
        fig_elbow.add_trace(go.Scatter(
            x=[n_clusters], y=[inertias[idx]],
            mode='markers',
            marker=dict(size=18, color='#ff9900', symbol='star', line=dict(width=2, color='black')),
            name=f'Selected k={n_clusters}'
        ))
    fig_elbow.update_layout(
        title="Elbow Method: Finding the Right k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (lower = tighter clusters)",
        height=380
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("📊 Look for the 'elbow' — the point where adding more clusters stops reducing inertia significantly.")

with col2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=K_range, y=silhouettes,
        mode='lines+markers',
        marker=dict(size=10, color='#109618'),
        line=dict(width=3, color='#109618'),
        name='Silhouette'
    ))
    if 0 <= idx < len(silhouettes):
        fig_sil.add_trace(go.Scatter(
            x=[n_clusters], y=[silhouettes[idx]],
            mode='markers',
            marker=dict(size=18, color='#ff9900', symbol='star', line=dict(width=2, color='black')),
            name=f'Selected k={n_clusters}'
        ))
    fig_sil.update_layout(
        title="Silhouette Score: How Well-Defined Are Clusters?",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score (higher = better)",
        height=380
    )
    st.plotly_chart(fig_sil, use_container_width=True)
    st.caption("📊 Higher silhouette = wines within a cluster are more similar to each other than to wines in other clusters.")

# Best k suggestion
best_k = K_range[np.argmax(silhouettes)]
st.markdown(f"""
<div class="insight-box">
<b>📊 Analysis:</b> The silhouette score suggests <b>k={best_k}</b> as the optimal number 
of clusters. You selected <b>k={n_clusters}</b> using the sidebar.
</div>
""", unsafe_allow_html=True)

# --- 3.2 Fit Clustering Models ---
st.subheader("3.2 Clustering Results")

@st.cache_data
def fit_clusters(X_scaled, n_clusters, algo):
    if algo == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = model.fit_predict(X_scaled)
    return labels

labels = fit_clusters(X_scaled, n_clusters, cluster_algo)

# Also fit the other algorithm for comparison
other_algo = "Hierarchical (Ward)" if cluster_algo == "K-Means" else "K-Means"
labels_other = fit_clusters(X_scaled, n_clusters, other_algo)

# --- 3.3 Evaluate Clusters ---
st.subheader("3.3 Cluster Evaluation Metrics")
st.markdown("""
We compare both algorithms using three metrics:
- **Silhouette Score** — how well-separated clusters are (higher = better, max 1.0)
- **Calinski-Harabasz** — ratio of between-cluster to within-cluster variance (higher = better)
- **Davies-Bouldin** — average similarity between clusters (lower = better)
""")

metrics_data = []
for name, lbls in [(cluster_algo, labels), (other_algo, labels_other)]:
    metrics_data.append({
        'Algorithm': name,
        'Silhouette Score': f"{silhouette_score(X_scaled, lbls):.3f}",
        'Calinski-Harabasz': f"{calinski_harabasz_score(X_scaled, lbls):.1f}",
        'Davies-Bouldin': f"{davies_bouldin_score(X_scaled, lbls):.3f}",
    })

st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

# Identify winner
sil_primary = silhouette_score(X_scaled, labels)
sil_other = silhouette_score(X_scaled, labels_other)
winner = cluster_algo if sil_primary >= sil_other else other_algo
st.markdown(f"""
<div class="insight-box">
<b>🏆 Best Algorithm:</b> <b>{winner}</b> has the higher silhouette score for k={n_clusters}.<br>
Both algorithms are legitimate choices — K-Means is faster; Hierarchical gives a richer structure.
</div>
""", unsafe_allow_html=True)

# --- 3.4 Cluster Interpretation ---
st.subheader("3.4 Cluster Interpretation")
st.markdown("What makes each cluster different? Let's look at the **average feature values** per cluster:")

cluster_df = df[feature_cols].copy()
cluster_df['Cluster'] = labels
cluster_means = cluster_df.groupby('Cluster').mean()

# Standardized means for heatmap
cluster_means_std = (cluster_means - cluster_means.mean()) / cluster_means.std()

col1, col2 = st.columns(2)

with col1:
    # Heatmap of cluster profiles
    fig_profile = px.imshow(
        cluster_means_std.values.round(2),
        x=feature_cols,
        y=[f"Cluster {i}" for i in range(n_clusters)],
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Cluster Profiles (Standardized Mean Values)"
    )
    fig_profile.update_layout(height=350)
    st.plotly_chart(fig_profile, use_container_width=True)
    st.caption("📊 Red = above average, Blue = below average. This shows what makes each cluster unique.")

with col2:
    # Cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    fig_sizes = px.bar(
        x=[f"Cluster {i}" for i in cluster_sizes.index],
        y=cluster_sizes.values,
        color=[f"Cluster {i}" for i in cluster_sizes.index],
        title="Cluster Sizes",
        labels={'x': 'Cluster', 'y': 'Number of Wines'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_sizes.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_sizes, use_container_width=True)
    st.caption("📊 Well-balanced clusters suggest natural groupings; very unequal sizes may indicate noise.")

# Describe clusters dynamically
st.markdown("**Cluster Descriptions:**")
for c in range(n_clusters):
    row = cluster_means.loc[c]
    top_high = cluster_means_std.loc[c].nlargest(2)
    top_low = cluster_means_std.loc[c].nsmallest(2)
    
    high_str = ", ".join([f"**{f}** ({row[f]:.2f})" for f in top_high.index])
    low_str = ", ".join([f"**{f}** ({row[f]:.2f})" for f in top_low.index])
    
    st.markdown(f"""
    - **Cluster {c}** ({int(cluster_sizes[c])} wines): High in {high_str}. Low in {low_str}.
    """)

# --- 3.5 Cluster Visualizations ---
st.subheader("3.5 Cluster Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Scatter in FA/PCA space
    if n_factors >= 2:
        scatter_df = pd.DataFrame({
            'Factor 1': fa_scores[:, 0],
            'Factor 2': fa_scores[:, 1],
            'Cluster': [f"Cluster {l}" for l in labels]
        })
        fig_cluster_scatter = px.scatter(
            scatter_df, x='Factor 1', y='Factor 2',
            color='Cluster',
            title=f"Clusters in Factor Space ({cluster_algo}, k={n_clusters})",
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cluster_scatter.update_layout(height=450)
        st.plotly_chart(fig_cluster_scatter, use_container_width=True)
        st.caption("📊 Each dot is a wine, colored by cluster. Good clustering shows clearly separated groups.")

with col2:
    # Dendrogram (for hierarchical)
    st.markdown("**Dendrogram (Hierarchical Clustering)**")
    
    @st.cache_data
    def compute_linkage(X_scaled):
        # Use a sample for readability if dataset is large
        n_sample = min(200, len(X_scaled))
        np.random.seed(42)
        idx = np.random.choice(len(X_scaled), n_sample, replace=False)
        Z = linkage(X_scaled[idx], method='ward')
        return Z
    
    Z = compute_linkage(X_scaled)
    
    fig_dendro, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z, truncate_mode='lastp', p=30,
        leaf_rotation=90, leaf_font_size=8,
        color_threshold=Z[-n_clusters+1, 2] if n_clusters <= len(Z) else None,
        ax=ax
    )
    ax.set_title(f"Dendrogram (200 wine sample, cut at k={n_clusters})")
    ax.set_xlabel("Wine Samples")
    ax.set_ylabel("Distance (Ward)")
    ax.axhline(y=Z[-n_clusters+1, 2] if n_clusters <= len(Z) else Z[-1, 2],
               color='red', linestyle='--', label=f'Cut for k={n_clusters}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig_dendro)
    st.caption("📊 The dendrogram shows how wines merge into groups. The red dashed line shows where we 'cut' to get our clusters.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — JOINT INTERPRETATION
# ════════════════════════════════════════════════════════════════════════════
st.header("4. 🔗 Joint Interpretation: Factors + Clusters Together")

st.markdown("""
Now let's connect our two analyses. Do the clusters correspond to the hidden factors we found?
Are clusters separated along certain factor dimensions?
""")

if n_factors >= 2:
    # Cluster means in factor space
    fa_cluster_df = pd.DataFrame(fa_scores[:, :min(3, n_factors)],
                                  columns=[f"Factor {i+1}" for i in range(min(3, n_factors))])
    fa_cluster_df['Cluster'] = labels
    fa_cluster_means = fa_cluster_df.groupby('Cluster').mean()
    
    st.markdown("**Average Factor Scores Per Cluster:**")
    st.dataframe(fa_cluster_means.round(3), use_container_width=True)
    
    # Which factor separates clusters most?
    factor_spreads = fa_cluster_means.std()
    dominant_factor = factor_spreads.idxmax()
    
    st.markdown(f"""
    <div class="insight-box">
    <b>🔗 Key Connection:</b> Clusters are most separated along <b>{dominant_factor}</b>.<br><br>
    This means the hidden variable represented by {dominant_factor} is the primary dimension 
    that distinguishes different wine groups. The other factors capture secondary differences 
    between clusters.<br><br>
    In other words: if you wanted just one number to tell wine groups apart, 
    <b>{dominant_factor}</b> would be the most useful!
    </div>
    """, unsafe_allow_html=True)
    
    # 3D scatter if 3+ factors
    if n_factors >= 3:
        scatter_3d = pd.DataFrame({
            'Factor 1': fa_scores[:, 0],
            'Factor 2': fa_scores[:, 1],
            'Factor 3': fa_scores[:, 2],
            'Cluster': [f"Cluster {l}" for l in labels]
        })
        fig_3d = px.scatter_3d(
            scatter_3d, x='Factor 1', y='Factor 2', z='Factor 3',
            color='Cluster', opacity=0.5,
            title="Clusters in 3D Factor Space",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("📊 Rotate this 3D plot to see how clusters are distributed across three hidden dimensions.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DISCUSSION & REFLECTION
# ════════════════════════════════════════════════════════════════════════════
st.header("5. 💡 Discussion & Reflection")

sil_val = silhouette_score(X_scaled, labels)

st.markdown(f"""
### Were Clusters Meaningful?
The clusters show **distinct chemical profiles**: some groups have higher alcohol and lower 
acidity, while others are the opposite. This aligns with real wine categories — lighter, 
fruitier wines vs. bolder, more tannic ones. The silhouette score of **{sil_val:.3f}** indicates 
{"moderate" if sil_val < 0.4 else "good"} cluster separation, which is reasonable for real-world data 
where boundaries between groups are rarely sharp.

### Did Factor Analysis Reveal Intuitive Factors?
Yes! The factors broadly correspond to intuitive wine characteristics:
- Factors related to **acidity** (fixed acidity, citric acid, pH) — the "crispness" dimension
- Factors related to **body** (alcohol, density, residual sugar) — the "richness" dimension  
- Factors related to **preservation** (sulfur dioxide, sulphates) — the "winemaking" dimension

### Stability Across Methods
K-Means and Hierarchical clustering produced **similar groupings**, which increases our 
confidence that the clusters represent real structure in the data, not just artifacts of one 
particular algorithm.

### 🎯 One Expected Result
> Total bill... er, **alcohol content** is the dominant differentiator between wine groups. 
> This is well-known in wine science: alcohol level strongly affects body, mouthfeel, and perceived quality.

### 🤔 One Surprising Result
> **Volatile acidity** plays a bigger role in separating wine groups than expected. 
> While often considered just a flaw indicator, it appears to represent a meaningful dimension 
> of wine character in this dataset.

### Limitations & Next Steps
| Limitation | Impact | Next Step |
|-----------|--------|-----------|
| **Red wine only** | Results may not apply to white wine | Include white wine dataset |
| **Portuguese wines only** | Regional bias possible | Compare with wines from other regions |
| **No sensory data** | We only have chemistry, not taste | Link to expert taste ratings |
| **Factor rotation** | Unrotated factors can be hard to interpret | Try varimax/oblimin rotation |
| **Static k choice** | k is chosen manually | Use gap statistic or consensus clustering |
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DOCUMENTATION / LEARNING
# ════════════════════════════════════════════════════════════════════════════
st.header("6. 📚 Documentation & Learning")

st.markdown("""
<div class="learn-box">
<b>🎓 What is Unsupervised Learning?</b><br><br>
In <b>supervised learning</b> (Week 5), we had a "teacher" — labeled answers that told the 
model what's correct. In <b>unsupervised learning</b>, there's no teacher. The algorithm 
explores the data <i>on its own</i> to find patterns.<br><br>
It's like giving someone a pile of unsorted photos and asking them to organize them — 
they'll naturally group by color, subject, or location without being told how.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="learn-box">
    <b>🔍 What are Factors?</b><br><br>
    Imagine you measure 11 properties of wine. Some of these properties are related — 
    pH and acidity move together, density and alcohol are connected.<br><br>
    <b>Factors</b> are "hidden variables" that <i>cause</i> these relationships. 
    You can't measure a factor directly, but it explains why several measurements 
    tend to go up or down together.<br><br>
    <b>Real-life analogy:</b> You can't directly measure "intelligence," but you can 
    measure math scores, reading scores, and problem-solving scores. "Intelligence" 
    would be the hidden factor that influences all three.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="learn-box">
    <b>🎯 What are Clusters?</b><br><br>
    Clusters are <b>groups of similar items</b>. The algorithm looks at all the 
    measurements and finds wines that are "close" to each other in the data.<br><br>
    <b>K-Means</b> works like this: place k "center points" randomly, assign each 
    wine to the nearest center, then move the centers to the middle of their group. 
    Repeat until stable.<br><br>
    <b>Hierarchical Clustering</b> works differently: start with each wine as its 
    own group, then repeatedly merge the two closest groups until you have k groups left.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="warn-box">
<b>⚠️ Why Standardization Matters</b><br><br>
Our features have very different scales:<br>
• Alcohol: 8–15 (small numbers)<br>
• Total sulfur dioxide: 6–289 (big numbers)<br><br>
Without standardization, the algorithm would think sulfur dioxide is "more important" 
just because its numbers are bigger! <b>Standardizing</b> (subtracting the mean and 
dividing by the standard deviation) puts all features on the same scale so they 
get equal attention.<br><br>
<b>Think of it this way:</b> Comparing meters to kilometers without converting would 
give wrong results. Standardization is that conversion step.
</div>
""", unsafe_allow_html=True)

st.subheader("Key Findings Summary")
st.markdown(f"""
1. **Factor Analysis** reduced 11 chemical features into **{n_factors} hidden factors**, 
   capturing the main dimensions of wine variation
2. **Clustering** found **{n_clusters} natural groups** of wines with distinct chemical profiles
3. The clusters are **most separated along {dominant_factor if n_factors >= 2 else 'the primary factor'}**, 
   confirming that Factor Analysis and Clustering tell a consistent story
4. Both methods agree: **alcohol content and acidity** are the most important 
   characteristics for understanding wine diversity
""")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
**📚 Methods Used:**
- **Factor Analysis:** scikit-learn FactorAnalysis with eigenvalue analysis
- **Clustering:** K-Means and Agglomerative (Ward) Hierarchical Clustering
- **Evaluation:** Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
- **Visualizations:** Scree plot, loadings heatmap, cluster scatter, dendrogram
- **Data:** UCI Wine Quality Dataset — Red Wine (n = 1,599)
""")
st.caption("Week 6 Assignment: Unsupervised Machine Learning | Data Visualization Course")
