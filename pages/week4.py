import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Week 4: Statistical Analysis", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: #333333 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
    }
    .stat-result {
        background-color: #e8f4ea;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
        color: #1f1f1f !important;
    }
    .stat-result b {
        color: #1f1f1f !important;
    }
    .hypothesis-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
        color: #333333 !important;
    }
    .hypothesis-box b {
        color: #1f1f1f !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Week 4: Statistical Analysis and Hypothesis Testing")
st.markdown("""
### What is this about?
Imagine you're a restaurant manager wondering: "Do smokers tip differently?" or "Do men spend more than women?"

Instead of guessing, we use **statistical tests** to find answers in data! This app analyzes **real restaurant data** 
to answer questions like these - and shows you exactly how we reach our conclusions.
""")

# --- Load Dataset ---
@st.cache_data
def load_data():
    """Load the tips dataset from seaborn."""
    df = sns.load_dataset("tips")
    return df

df = load_data()

# ============================================================================
# SECTION 1: DATA & EDA
# ============================================================================
st.header("1. 📋 Data & Exploratory Data Analysis")

st.subheader("Dataset Overview: Restaurant Tips")
st.markdown("""
🍽️ **What data are we using?**

This is real data from a restaurant! Each row represents one customer visit, recording:
- How much they spent (total bill)
- How much they tipped
- Their gender, whether they smoke, and when they visited
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Observations", len(df))
with col2:
    st.metric("Numerical Variables", len(df.select_dtypes(include=[np.number]).columns))
with col3:
    st.metric("Categorical Variables", len(df.select_dtypes(include=['category', 'object']).columns))
with col4:
    st.metric("Missing Values", df.isnull().sum().sum())

# Display first rows
st.subheader("📄 First Rows of Dataset")
st.dataframe(df.head(10), use_container_width=True)

# Summary statistics
st.subheader("📈 Summary Statistics")
tab1, tab2 = st.tabs(["Numerical Variables", "Categorical Variables"])

with tab1:
    st.dataframe(df.describe().round(2), use_container_width=True)

with tab2:
    cat_summary = pd.DataFrame({
        'Variable': ['sex', 'smoker', 'day', 'time'],
        'Unique Values': [df['sex'].nunique(), df['smoker'].nunique(), df['day'].nunique(), df['time'].nunique()],
        'Categories': [str(df['sex'].unique().tolist()), str(df['smoker'].unique().tolist()), 
                       str(df['day'].unique().tolist()), str(df['time'].unique().tolist())]
    })
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

# Distribution plots
st.subheader("📊 Distribution of Key Variables")
fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Total Bill Distribution", "Tip Amount Distribution"])

fig_dist.add_trace(
    go.Histogram(x=df['total_bill'], name="Total Bill", marker_color='#3366cc', opacity=0.7),
    row=1, col=1
)
fig_dist.add_trace(
    go.Histogram(x=df['tip'], name="Tip", marker_color='#dc3912', opacity=0.7),
    row=1, col=2
)
fig_dist.update_xaxes(title_text="Total Bill ($)", row=1, col=1)
fig_dist.update_xaxes(title_text="Tip ($)", row=1, col=2)
fig_dist.update_yaxes(title_text="Frequency", row=1, col=1)
fig_dist.update_yaxes(title_text="Frequency", row=1, col=2)
fig_dist.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================================
# SECTION 2: RESEARCH QUESTIONS & HYPOTHESES
# ============================================================================
st.header("2. 🔬 Our Questions")

st.markdown("""
### What do we want to find out?
We have **3 simple questions** about restaurant customers. For each question, we'll use data to find a real answer:
""")

# RQ1
st.subheader("❓ Question 1: Do smokers tip differently?")
st.markdown("""
<div class="hypothesis-box">
<b>🤔 The Question:</b> Do smokers leave different tips than non-smokers?<br><br>
<b>What we're comparing:</b> Average tip amounts between two groups (smokers vs non-smokers)<br><br>
<b>Our starting assumption (H₀):</b> There's NO real difference - any variation is just random chance<br>
<b>What we're testing (H₁):</b> There IS a real difference between the groups<br><br>
<b>How we'll test it:</b> Compare the average tips from both groups using a "t-test" (a method that tells us if differences are real or just luck)
</div>
""", unsafe_allow_html=True)

# RQ2
st.subheader("❓ Question 2: Do smokers prefer lunch or dinner?")
st.markdown("""
<div class="hypothesis-box">
<b>🤔 The Question:</b> Is there a pattern between smoking and when people eat?<br><br>
<b>What we're checking:</b> Whether smokers tend to come at certain times (lunch vs dinner)<br><br>
<b>Our starting assumption (H₀):</b> Smoking and meal time are unrelated - smokers and non-smokers visit at similar times<br>
<b>What we're testing (H₁):</b> There IS a connection between smoking and meal time<br><br>
<b>How we'll test it:</b> Use a "Chi-square test" (counts how many people fall into each category and checks if the pattern is meaningful)
</div>
""", unsafe_allow_html=True)

# RQ3
st.subheader("❓ Question 3: Do men spend more than women?")
st.markdown("""
<div class="hypothesis-box">
<b>🤔 The Question:</b> Do men and women have different average bills?<br><br>
<b>What we're comparing:</b> Average total bill between male and female customers<br><br>
<b>Our starting assumption (H₀):</b> There's NO real spending difference between genders<br>
<b>What we're testing (H₁):</b> There IS a real difference<br><br>
<b>How we'll test it:</b> Compare averages with a "t-test" and show how confident we are with a "confidence interval" (a range showing where the true difference likely falls)
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 3: ASSUMPTION CHECKING
# ============================================================================
st.header("3. ✅ Checking Our Data Quality")

st.markdown("""
### Before we test, let's make sure our data is suitable!

Statistical tests make certain assumptions about data. Think of it like checking the ingredients before cooking:

1. **Is the data "normal"?** - Does it follow a typical bell-curve pattern? (We use the Shapiro-Wilk test)
2. **Are the groups similar in spread?** - Do both groups have similar variability? (We use Levene's test)

Don't worry too much about the technical details - the key is that we're being thorough! ✅
""")

# Prepare groups
smokers_tip = df[df['smoker'] == 'Yes']['tip']
non_smokers_tip = df[df['smoker'] == 'No']['tip']
male_bill = df[df['sex'] == 'Male']['total_bill']
female_bill = df[df['sex'] == 'Female']['total_bill']

st.subheader("3.1 Normality Tests (Shapiro-Wilk)")

normality_results = []

# Test normality for RQ1 groups
stat1, p1 = stats.shapiro(smokers_tip)
stat2, p2 = stats.shapiro(non_smokers_tip)
normality_results.append({"Variable": "Tip (Smokers)", "W-statistic": round(stat1, 4), "P-value": round(p1, 4), "Normal?": "Yes" if p1 > 0.05 else "No"})
normality_results.append({"Variable": "Tip (Non-smokers)", "W-statistic": round(stat2, 4), "P-value": round(p2, 4), "Normal?": "Yes" if p2 > 0.05 else "No"})

# Test normality for RQ3 groups
stat3, p3 = stats.shapiro(male_bill)
stat4, p4 = stats.shapiro(female_bill)
normality_results.append({"Variable": "Total Bill (Male)", "W-statistic": round(stat3, 4), "P-value": round(p3, 4), "Normal?": "Yes" if p3 > 0.05 else "No"})
normality_results.append({"Variable": "Total Bill (Female)", "W-statistic": round(stat4, 4), "P-value": round(p4, 4), "Normal?": "Yes" if p4 > 0.05 else "No"})

normality_df = pd.DataFrame(normality_results)
st.dataframe(normality_df, use_container_width=True, hide_index=True)

# QQ Plots
st.subheader("3.2 QQ Plots for Visual Normality Check")
fig_qq = make_subplots(rows=1, cols=2, subplot_titles=["Tip (All Data)", "Total Bill (All Data)"])

# QQ plot data
tip_sorted = np.sort(df['tip'])
tip_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(tip_sorted)))

bill_sorted = np.sort(df['total_bill'])
bill_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(bill_sorted)))

fig_qq.add_trace(go.Scatter(x=tip_theoretical, y=tip_sorted, mode='markers', name='Tip', marker=dict(color='#3366cc', size=5)), row=1, col=1)
fig_qq.add_trace(go.Scatter(x=[-3, 3], y=[tip_sorted.min(), tip_sorted.max()], mode='lines', name='Reference', line=dict(color='red', dash='dash')), row=1, col=1)

fig_qq.add_trace(go.Scatter(x=bill_theoretical, y=bill_sorted, mode='markers', name='Total Bill', marker=dict(color='#dc3912', size=5)), row=1, col=2)
fig_qq.add_trace(go.Scatter(x=[-3, 3], y=[bill_sorted.min(), bill_sorted.max()], mode='lines', name='Reference', line=dict(color='red', dash='dash')), row=1, col=2)

fig_qq.update_xaxes(title_text="Theoretical Quantiles", row=1, col=1)
fig_qq.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
fig_qq.update_yaxes(title_text="Sample Quantiles", row=1, col=1)
fig_qq.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
fig_qq.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_qq, use_container_width=True)

st.subheader("3.3 Homogeneity of Variances (Levene's Test)")

# Levene's test for RQ1
lev_stat1, lev_p1 = stats.levene(smokers_tip, non_smokers_tip)
# Levene's test for RQ3
lev_stat2, lev_p2 = stats.levene(male_bill, female_bill)

levene_results = pd.DataFrame([
    {"Comparison": "Smokers vs Non-smokers (Tip)", "Levene Statistic": round(lev_stat1, 4), "P-value": round(lev_p1, 4), "Equal Variances?": "Yes" if lev_p1 > 0.05 else "No"},
    {"Comparison": "Male vs Female (Total Bill)", "Levene Statistic": round(lev_stat2, 4), "P-value": round(lev_p2, 4), "Equal Variances?": "Yes" if lev_p2 > 0.05 else "No"}
])
st.dataframe(levene_results, use_container_width=True, hide_index=True)

# Decision summary
st.info("""
📌 **What this means (in plain English):**
- Our data isn't perfectly "normal," but that's okay! With 244 customers, we have enough data to get reliable results anyway.
- Just to be safe, we'll run TWO types of tests and compare them.
- If the tests agree, we can be confident in our conclusions! ✅
""")

# ============================================================================
# SECTION 4: STATISTICAL TESTS
# ============================================================================
st.header("4. 📈 Statistical Tests")

# --- RQ1: T-test for Tips by Smoking Status ---
st.subheader("4.1 RQ1: Tips by Smoking Status")

col1, col2 = st.columns([2, 3])

with col1:
    # Descriptive stats
    st.markdown("**Group Statistics:**")
    group_stats_rq1 = df.groupby('smoker')['tip'].agg(['count', 'mean', 'std']).round(3)
    group_stats_rq1.columns = ['N', 'Mean', 'Std Dev']
    st.dataframe(group_stats_rq1, use_container_width=True)

with col2:
    # Perform tests
    # Independent t-test (Welch's)
    t_stat1, t_p1 = stats.ttest_ind(smokers_tip, non_smokers_tip, equal_var=False)
    
    # Mann-Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(smokers_tip, non_smokers_tip, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt(((len(smokers_tip)-1)*smokers_tip.std()**2 + (len(non_smokers_tip)-1)*non_smokers_tip.std()**2) / (len(smokers_tip)+len(non_smokers_tip)-2))
    cohens_d = (smokers_tip.mean() - non_smokers_tip.mean()) / pooled_std
    
    # 95% CI for mean difference
    mean_diff = smokers_tip.mean() - non_smokers_tip.mean()
    se_diff = np.sqrt(smokers_tip.var()/len(smokers_tip) + non_smokers_tip.var()/len(non_smokers_tip))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    st.markdown("**Test Results:**")
    results_rq1 = pd.DataFrame([
        {"Test": "Welch's t-test", "Statistic": f"t = {t_stat1:.3f}", "P-value": f"{t_p1:.4f}", "Significant?": "Yes" if t_p1 < 0.05 else "No"},
        {"Test": "Mann-Whitney U", "Statistic": f"U = {u_stat:.1f}", "P-value": f"{u_p:.4f}", "Significant?": "Yes" if u_p < 0.05 else "No"}
    ])
    st.dataframe(results_rq1, use_container_width=True, hide_index=True)

# Effect size and CI
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cohen's d", f"{cohens_d:.3f}", help="Effect size: |d|<0.2=negligible, 0.2-0.5=small, 0.5-0.8=medium, >0.8=large")
with col2:
    effect_interp = "Negligible" if abs(cohens_d) < 0.2 else "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
    st.metric("Effect Size", effect_interp)
with col3:
    st.metric("95% CI for Difference", f"[{ci_lower:.2f}, {ci_upper:.2f}]")

# Interpretation
st.markdown(f"""
<div class="stat-result">
<b>📝 Interpretation (RQ1):</b><br>
The independent samples t-test revealed <b>{"a statistically significant" if t_p1 < 0.05 else "no statistically significant"}</b> 
difference in tip amounts between smokers (M = ${smokers_tip.mean():.2f}, SD = ${smokers_tip.std():.2f}) and 
non-smokers (M = ${non_smokers_tip.mean():.2f}, SD = ${non_smokers_tip.std():.2f}), 
t = {t_stat1:.2f}, p = {t_p1:.4f}, Cohen's d = {cohens_d:.3f}.<br><br>
The effect size is <b>{effect_interp.lower()}</b>, and the 95% CI for the mean difference [{ci_lower:.2f}, {ci_upper:.2f}] 
{"includes" if ci_lower <= 0 <= ci_upper else "does not include"} zero, 
{"supporting the conclusion of no practical difference" if ci_lower <= 0 <= ci_upper else "suggesting a meaningful difference"}.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- RQ2: Chi-square test for Smoking Status vs Time ---
st.subheader("4.2 RQ2: Smoking Status vs Meal Time (χ² Test)")

col1, col2 = st.columns([2, 3])

with col1:
    # Contingency table
    st.markdown("**Contingency Table:**")
    contingency = pd.crosstab(df['smoker'], df['time'], margins=True, margins_name='Total')
    st.dataframe(contingency, use_container_width=True)

with col2:
    # Chi-square test
    contingency_no_margins = pd.crosstab(df['smoker'], df['time'])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_no_margins)
    
    # Cramer's V
    n = contingency_no_margins.sum().sum()
    min_dim = min(contingency_no_margins.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    st.markdown("**Chi-Square Test Results:**")
    results_rq2 = pd.DataFrame([
        {"Metric": "Chi-square (χ²)", "Value": f"{chi2:.3f}"},
        {"Metric": "Degrees of Freedom", "Value": f"{dof}"},
        {"Metric": "P-value", "Value": f"{chi_p:.4f}"},
        {"Metric": "Cramer's V", "Value": f"{cramers_v:.3f}"},
        {"Metric": "Significant?", "Value": "Yes" if chi_p < 0.05 else "No"}
    ])
    st.dataframe(results_rq2, use_container_width=True, hide_index=True)

# Expected frequencies
with st.expander("📊 View Expected Frequencies"):
    expected_df = pd.DataFrame(expected, index=contingency_no_margins.index, columns=contingency_no_margins.columns).round(2)
    st.dataframe(expected_df, use_container_width=True)
    st.caption("All expected frequencies > 5, so χ² test is appropriate.")

# Effect size interpretation
cramers_interp = "Negligible" if cramers_v < 0.1 else "Small" if cramers_v < 0.3 else "Medium" if cramers_v < 0.5 else "Large"

st.markdown(f"""
<div class="stat-result">
<b>📝 Interpretation (RQ2):</b><br>
The chi-square test of independence showed <b>{"a statistically significant" if chi_p < 0.05 else "no statistically significant"}</b> 
association between smoking status and meal time, χ²({dof}) = {chi2:.2f}, p = {chi_p:.4f}.<br><br>
Cramer's V = {cramers_v:.3f} indicates a <b>{cramers_interp.lower()}</b> effect size.
{"This suggests that smoking habits may differ between lunch and dinner customers." if chi_p < 0.05 else "Smokers and non-smokers appear to visit at similar times."}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- RQ3: T-test for Total Bill by Gender ---
st.subheader("4.3 RQ3: Total Bill by Gender")

col1, col2 = st.columns([2, 3])

with col1:
    # Descriptive stats
    st.markdown("**Group Statistics:**")
    group_stats_rq3 = df.groupby('sex')['total_bill'].agg(['count', 'mean', 'std']).round(3)
    group_stats_rq3.columns = ['N', 'Mean', 'Std Dev']
    st.dataframe(group_stats_rq3, use_container_width=True)

with col2:
    # Perform tests
    t_stat3, t_p3 = stats.ttest_ind(male_bill, female_bill, equal_var=False)
    
    # Mann-Whitney U
    u_stat3, u_p3 = stats.mannwhitneyu(male_bill, female_bill, alternative='two-sided')
    
    # Cohen's d
    pooled_std3 = np.sqrt(((len(male_bill)-1)*male_bill.std()**2 + (len(female_bill)-1)*female_bill.std()**2) / (len(male_bill)+len(female_bill)-2))
    cohens_d3 = (male_bill.mean() - female_bill.mean()) / pooled_std3
    
    # 95% CI
    mean_diff3 = male_bill.mean() - female_bill.mean()
    se_diff3 = np.sqrt(male_bill.var()/len(male_bill) + female_bill.var()/len(female_bill))
    ci_lower3 = mean_diff3 - 1.96 * se_diff3
    ci_upper3 = mean_diff3 + 1.96 * se_diff3
    
    st.markdown("**Test Results:**")
    results_rq3 = pd.DataFrame([
        {"Test": "Welch's t-test", "Statistic": f"t = {t_stat3:.3f}", "P-value": f"{t_p3:.4f}", "Significant?": "Yes" if t_p3 < 0.05 else "No"},
        {"Test": "Mann-Whitney U", "Statistic": f"U = {u_stat3:.1f}", "P-value": f"{u_p3:.4f}", "Significant?": "Yes" if u_p3 < 0.05 else "No"}
    ])
    st.dataframe(results_rq3, use_container_width=True, hide_index=True)

# Effect size and CI
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Cohen's d", f"{cohens_d3:.3f}")
with col2:
    effect_interp3 = "Negligible" if abs(cohens_d3) < 0.2 else "Small" if abs(cohens_d3) < 0.5 else "Medium" if abs(cohens_d3) < 0.8 else "Large"
    st.metric("Effect Size", effect_interp3)
with col3:
    st.metric("95% CI for Difference", f"[{ci_lower3:.2f}, {ci_upper3:.2f}]")

st.markdown(f"""
<div class="stat-result">
<b>📝 Interpretation (RQ3):</b><br>
The independent samples t-test revealed <b>{"a statistically significant" if t_p3 < 0.05 else "no statistically significant"}</b> 
difference in total bill amounts between male (M = ${male_bill.mean():.2f}, SD = ${male_bill.std():.2f}) and 
female (M = ${female_bill.mean():.2f}, SD = ${female_bill.std():.2f}) customers, 
t = {t_stat3:.2f}, p = {t_p3:.4f}, Cohen's d = {cohens_d3:.3f}.<br><br>
The 95% CI for the mean difference is [${ci_lower3:.2f}, ${ci_upper3:.2f}]. 
{"Although statistically significant," if t_p3 < 0.05 else "With"} Cohen's d = {cohens_d3:.3f} ({effect_interp3.lower()} effect), 
{"the practical difference may be limited." if abs(cohens_d3) < 0.5 else "this represents a meaningful practical difference."}
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================
st.header("5. 📊 Visualizations of Test Results")

st.subheader("5.1 Tips by Smoking Status (RQ1)")

col1, col2 = st.columns(2)

with col1:
    # Boxplot with individual points
    fig_box1 = px.box(df, x='smoker', y='tip', color='smoker',
                       title="Tip Amount by Smoking Status",
                       labels={'tip': 'Tip ($)', 'smoker': 'Smoker'},
                       color_discrete_map={'Yes': '#e74c3c', 'No': '#27ae60'},
                       points='all')
    fig_box1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_box1, use_container_width=True)

with col2:
    # Bar chart with error bars (CI)
    tip_by_smoker = df.groupby('smoker')['tip'].agg(['mean', 'std', 'count']).reset_index()
    tip_by_smoker['se'] = tip_by_smoker['std'] / np.sqrt(tip_by_smoker['count'])
    tip_by_smoker['ci'] = 1.96 * tip_by_smoker['se']
    
    fig_bar1 = go.Figure()
    fig_bar1.add_trace(go.Bar(
        x=tip_by_smoker['smoker'],
        y=tip_by_smoker['mean'],
        error_y=dict(type='data', array=tip_by_smoker['ci'], visible=True),
        marker_color=['#27ae60', '#e74c3c'],
        text=[f"${m:.2f}" for m in tip_by_smoker['mean']],
        textposition='outside'
    ))
    fig_bar1.update_layout(
        title="Mean Tip with 95% CI",
        xaxis_title="Smoker",
        yaxis_title="Mean Tip ($)",
        height=400
    )
    fig_bar1.add_annotation(
        x=0.5, y=1.1, xref='paper', yref='paper',
        text=f"p = {t_p1:.4f}, Cohen's d = {cohens_d:.3f}",
        showarrow=False, font=dict(size=12)
    )
    st.plotly_chart(fig_bar1, use_container_width=True)

st.caption("📊 The boxplot shows the distribution of tips within each group. The bar chart displays mean tips with 95% confidence intervals. Overlapping CIs suggest no significant difference.")

st.subheader("5.2 Smoking Status vs Meal Time (RQ2)")

col1, col2 = st.columns(2)

with col1:
    # Grouped bar chart
    fig_grouped = px.histogram(df, x='time', color='smoker', barmode='group',
                                title="Distribution of Smokers by Meal Time",
                                labels={'time': 'Meal Time', 'count': 'Count'},
                                color_discrete_map={'Yes': '#e74c3c', 'No': '#27ae60'},
                                category_orders={'time': ['Lunch', 'Dinner']})
    fig_grouped.update_layout(height=400, legend_title="Smoker")
    st.plotly_chart(fig_grouped, use_container_width=True)

with col2:
    # Proportions
    proportions = df.groupby(['time', 'smoker']).size().unstack(fill_value=0)
    proportions_pct = proportions.div(proportions.sum(axis=1), axis=0) * 100
    
    fig_prop = go.Figure()
    for smoker_status in ['No', 'Yes']:
        fig_prop.add_trace(go.Bar(
            x=proportions_pct.index,
            y=proportions_pct[smoker_status],
            name=f"Smoker: {smoker_status}",
            marker_color='#27ae60' if smoker_status == 'No' else '#e74c3c',
            text=[f"{v:.1f}%" for v in proportions_pct[smoker_status]],
            textposition='inside'
        ))
    fig_prop.update_layout(
        title="Proportion of Smokers by Meal Time",
        xaxis_title="Meal Time",
        yaxis_title="Percentage (%)",
        barmode='stack',
        height=400
    )
    fig_prop.add_annotation(
        x=0.5, y=1.1, xref='paper', yref='paper',
        text=f"χ² = {chi2:.2f}, p = {chi_p:.4f}, Cramer's V = {cramers_v:.3f}",
        showarrow=False, font=dict(size=12)
    )
    st.plotly_chart(fig_prop, use_container_width=True)

st.caption("📊 The grouped bar chart shows raw counts, while the stacked bar chart shows proportions. Notice the higher proportion of smokers at dinner vs lunch.")

st.subheader("5.3 Total Bill by Gender (RQ3)")

col1, col2 = st.columns(2)

with col1:
    # Violin plot with box
    fig_violin = px.violin(df, x='sex', y='total_bill', color='sex', box=True,
                            title="Total Bill Distribution by Gender",
                            labels={'total_bill': 'Total Bill ($)', 'sex': 'Gender'},
                            color_discrete_map={'Male': '#3498db', 'Female': '#e91e63'})
    fig_violin.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_violin, use_container_width=True)

with col2:
    # Bar chart with error bars
    bill_by_sex = df.groupby('sex')['total_bill'].agg(['mean', 'std', 'count']).reset_index()
    bill_by_sex['se'] = bill_by_sex['std'] / np.sqrt(bill_by_sex['count'])
    bill_by_sex['ci'] = 1.96 * bill_by_sex['se']
    
    fig_bar3 = go.Figure()
    fig_bar3.add_trace(go.Bar(
        x=bill_by_sex['sex'],
        y=bill_by_sex['mean'],
        error_y=dict(type='data', array=bill_by_sex['ci'], visible=True),
        marker_color=['#e91e63', '#3498db'],
        text=[f"${m:.2f}" for m in bill_by_sex['mean']],
        textposition='outside'
    ))
    fig_bar3.update_layout(
        title="Mean Total Bill with 95% CI",
        xaxis_title="Gender",
        yaxis_title="Mean Total Bill ($)",
        height=400
    )
    fig_bar3.add_annotation(
        x=0.5, y=1.1, xref='paper', yref='paper',
        text=f"p = {t_p3:.4f}, Cohen's d = {cohens_d3:.3f}",
        showarrow=False, font=dict(size=12)
    )
    st.plotly_chart(fig_bar3, use_container_width=True)

st.caption("📊 The violin plot shows full distribution shapes with embedded boxplots. The bar chart displays mean values with 95% confidence intervals for quick comparison.")

# ============================================================================
# SECTION 6: INTERPRETATION & DISCUSSION
# ============================================================================
st.header("6. 💡 Interpretation and Discussion")

st.subheader("Summary of Findings")

# Create summary table
summary_data = [
    {
        "Research Question": "RQ1: Tips by Smoking Status",
        "Test Used": "Welch's t-test",
        "Result": f"t = {t_stat1:.2f}, p = {t_p1:.4f}",
        "Effect Size": f"Cohen's d = {cohens_d:.3f} ({effect_interp})",
        "H₀ Decision": "Reject" if t_p1 < 0.05 else "Fail to Reject"
    },
    {
        "Research Question": "RQ2: Smoking vs Meal Time",
        "Test Used": "χ² test",
        "Result": f"χ² = {chi2:.2f}, p = {chi_p:.4f}",
        "Effect Size": f"Cramer's V = {cramers_v:.3f} ({cramers_interp})",
        "H₀ Decision": "Reject" if chi_p < 0.05 else "Fail to Reject"
    },
    {
        "Research Question": "RQ3: Total Bill by Gender",
        "Test Used": "Welch's t-test",
        "Result": f"t = {t_stat3:.2f}, p = {t_p3:.4f}",
        "Effect Size": f"Cohen's d = {cohens_d3:.3f} ({effect_interp3})",
        "H₀ Decision": "Reject" if t_p3 < 0.05 else "Fail to Reject"
    }
]
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("Key Insights")

st.markdown(f"""
### 1. Smoking and Tipping Behavior
{"The analysis found **no significant difference** in tipping behavior between smokers and non-smokers." if t_p1 >= 0.05 else "The analysis found a **significant difference** in tipping between smokers and non-smokers."}
With Cohen's d = {cohens_d:.3f}, the effect size is {effect_interp.lower()}, suggesting that 
**smoking status has minimal practical impact on tip amounts**.

### 2. Smoking Patterns by Meal Time  
{"The chi-square test revealed a **significant association**" if chi_p < 0.05 else "The chi-square test found **no significant association**"} 
between smoking status and meal time.
{"A higher proportion of smokers dine during dinner compared to lunch, which could reflect social or cultural factors." if chi_p < 0.05 else "Smokers and non-smokers appear equally likely to dine at lunch or dinner."}
The effect size (Cramer's V = {cramers_v:.3f}) is {cramers_interp.lower()}.

### 3. Gender and Spending Patterns
{"Male customers have **significantly higher** total bills on average compared to female customers." if t_p3 < 0.05 else "There is **no significant difference** in total bills between male and female customers."}
{"However," if t_p3 < 0.05 else "And"} the effect size (Cohen's d = {cohens_d3:.3f}) indicates this difference is 
{effect_interp3.lower()}, {"raising questions about practical relevance." if t_p3 < 0.05 and abs(cohens_d3) < 0.5 else "which aligns with the statistical finding."}
""")

st.subheader("Statistical vs Practical Significance")

st.warning("""
⚠️ **Important Note on Interpretation:**

A statistically significant result (p < 0.05) does NOT necessarily mean the effect is practically meaningful:
- **p-values** tell us about evidence against H₀, not about effect magnitude
- **Effect sizes** (Cohen's d, Cramer's V) measure practical importance
- **Confidence intervals** help understand the range of plausible effects

For example, even if p < 0.05, a Cohen's d < 0.2 suggests negligible practical difference.
""")

st.subheader("Potential Confounding Variables")

st.markdown("""
Several factors could influence these results:

| Variable | Potential Confound |
|----------|-------------------|
| **Party Size** | Larger groups may have higher bills regardless of gender |
| **Day of Week** | Weekend vs weekday dining patterns differ |
| **Restaurant Type** | Dataset from one restaurant limits generalizability |
| **Time Period** | Data collected in early 1990s; tipping norms may have changed |
| **Geographic Location** | Regional tipping customs vary |
""")

st.subheader("Limitations")

st.markdown("""
1. **Sample Size**: While adequate (n=244), larger samples would provide more precise estimates
2. **Single Restaurant**: Results may not generalize to other establishments
3. **Historical Data**: Collected in 1990s; modern patterns may differ
4. **No Random Sampling**: Convenience sample may introduce selection bias
5. **Limited Variables**: Cannot control for all potential confounders
""")

# Footer
st.markdown("---")
st.markdown("""
**📚 Methodology Summary:**
- Tests used: Independent t-test (Welch's), Mann-Whitney U, Chi-square (χ²)
- Effect sizes: Cohen's d, Cramer's V
- All tests conducted at α = 0.05 significance level
- Data source: seaborn "tips" dataset (n = 244)
""")
st.caption("Week 4 Assignment: Statistical Analysis and Tests | Data Visualization Course")
