import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Week 3: Correlation Analysis", page_icon="🔗", layout="wide")

st.title("🔗 Week 3: Correlation Analysis")
st.markdown("""
This app explores relationships between stock prices using correlation analysis. 
We'll calculate correlation coefficients, visualize correlations, and interpret the results.
""")

# --- 1. Dataset Selection ---
st.header("1. Dataset Selection")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.date.today())

# Stock selection
available_stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL", 
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "NVIDIA (NVDA)": "NVDA"
}

selected_stocks = st.multiselect(
    "Select Stocks for Correlation Analysis (minimum 2)",
    list(available_stocks.keys()),
    default=["Apple (AAPL)", "Google (GOOGL)", "Microsoft (MSFT)", "Amazon (AMZN)"]
)

if len(selected_stocks) < 2:
    st.warning("Please select at least 2 stocks for correlation analysis.")
    st.stop()

tickers = [available_stocks[s] for s in selected_stocks]

@st.cache_data
def load_multi_stock_data(tickers, start, end):
    """Load stock data for multiple tickers and compute daily returns."""
    data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker')
    
    if data.empty:
        return None, None
    
    # Extract close prices for each ticker
    close_prices = pd.DataFrame()
    for ticker in tickers:
        if len(tickers) == 1:
            close_prices[ticker] = data['Close']
        else:
            if ticker in data.columns.get_level_values(0):
                close_prices[ticker] = data[ticker]['Close']
    
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices = close_prices.dropna()
    
    # Compute daily returns (percentage change)
    daily_returns = close_prices.pct_change().dropna() * 100  # Convert to percentage
    
    return close_prices, daily_returns

try:
    close_prices, daily_returns = load_multi_stock_data(tickers, start_date, end_date)
    if close_prices is None or len(close_prices) < 30:
        st.warning("Not enough data points. Please select a longer date range (need 30+ observations).")
        st.stop()
    st.success(f"✅ Loaded {len(close_prices)} trading days of data for {len(tickers)} stocks")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. Exploratory Data Analysis (EDA) ---
st.header("2. Exploratory Data Analysis (EDA)")

st.subheader("📋 First Rows of Dataset")
tab1, tab2 = st.tabs(["Close Prices (USD)", "Daily Returns (%)"])
with tab1:
    st.dataframe(close_prices.head(10), use_container_width=True)
with tab2:
    st.dataframe(daily_returns.head(10).round(2), use_container_width=True)

# Missing values check
st.subheader("🔍 Data Quality Check")
missing_close = close_prices.isnull().sum().sum()
missing_returns = daily_returns.isnull().sum().sum()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Observations", len(close_prices))
with col2:
    st.metric("Missing Values (Prices)", missing_close)
with col3:
    st.metric("Missing Values (Returns)", missing_returns)

# Descriptive statistics
st.subheader("📊 Descriptive Statistics")
tab1, tab2 = st.tabs(["Close Prices (USD)", "Daily Returns (%)"])
with tab1:
    st.dataframe(close_prices.describe().round(2), use_container_width=True)
with tab2:
    st.dataframe(daily_returns.describe().round(2), use_container_width=True)

# Histograms
st.subheader("📈 Distribution of Daily Returns")
fig_hist = make_subplots(
    rows=1, cols=len(tickers),
    subplot_titles=[f"{t} Returns (%)" for t in tickers]
)

colors = px.colors.qualitative.Set2
for i, ticker in enumerate(tickers):
    fig_hist.add_trace(
        go.Histogram(x=daily_returns[ticker], name=ticker, marker_color=colors[i % len(colors)], opacity=0.7),
        row=1, col=i+1
    )
    fig_hist.update_xaxes(title_text="Daily Return (%)", row=1, col=i+1)
    fig_hist.update_yaxes(title_text="Frequency", row=1, col=i+1)

fig_hist.update_layout(height=350, showlegend=False, title_text="Distribution of Daily Returns by Stock")
st.plotly_chart(fig_hist, use_container_width=True)

# Scatter plots for initial exploration
st.subheader("🔘 Scatter Plots: Initial Relationship Exploration")
if len(tickers) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        x_stock = st.selectbox("X-axis Stock", tickers, index=0, key="scatter_x")
    with col2:
        y_stock = st.selectbox("Y-axis Stock", tickers, index=1, key="scatter_y")
    
    fig_scatter_eda = px.scatter(
        daily_returns, x=x_stock, y=y_stock,
        trendline="ols",
        labels={x_stock: f"{x_stock} Daily Return (%)", y_stock: f"{y_stock} Daily Return (%)"},
        title=f"Scatter Plot: {x_stock} vs {y_stock} Daily Returns"
    )
    fig_scatter_eda.update_traces(marker=dict(size=8, opacity=0.6))
    st.plotly_chart(fig_scatter_eda, use_container_width=True)

# --- 3. Normality Testing ---
st.header("3. Normality Testing")
st.markdown("""
To determine the appropriate correlation test, we first check if the data follows a normal distribution 
using the **Shapiro-Wilk test**. If p-value > 0.05, data is likely normal → use **Pearson**. 
Otherwise, use **Spearman** (non-parametric).
""")

normality_results = []
for ticker in tickers:
    # Shapiro-Wilk test (use first 5000 samples if dataset is large)
    sample = daily_returns[ticker].dropna()[:5000]
    stat, p_value = stats.shapiro(sample)
    is_normal = "Yes" if p_value > 0.05 else "No"
    normality_results.append({
        "Stock": ticker,
        "Shapiro-Wilk Statistic": round(stat, 4),
        "P-Value": round(p_value, 4),
        "Normally Distributed?": is_normal
    })

normality_df = pd.DataFrame(normality_results)
st.dataframe(normality_df, use_container_width=True, hide_index=True)

# Recommendation
any_non_normal = any(r["Normally Distributed?"] == "No" for r in normality_results)
if any_non_normal:
    st.info("📌 **Recommendation:** Some variables are not normally distributed. **Spearman correlation** is recommended, though Pearson can also provide insights.")
else:
    st.info("📌 **Recommendation:** All variables appear normally distributed. **Pearson correlation** is appropriate.")

# --- 4. Correlation Computation ---
st.header("4. Correlation Computation")

col1, col2 = st.columns([1, 2])
with col1:
    corr_method = st.radio(
        "Select Correlation Method",
        ["Pearson", "Spearman"],
        index=1 if any_non_normal else 0,
        help="Pearson measures linear relationships; Spearman measures monotonic relationships"
    )

method = corr_method.lower()

# Compute correlation matrix
corr_matrix = daily_returns.corr(method=method)

st.subheader(f"📊 {corr_method} Correlation Matrix")
st.dataframe(corr_matrix.round(3).style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)

# Individual correlations with p-values
st.subheader("🔢 Individual Correlation Coefficients with P-Values")

correlation_pairs = []
for i, t1 in enumerate(tickers):
    for j, t2 in enumerate(tickers):
        if i < j:  # Only upper triangle
            if method == 'pearson':
                corr, p_val = stats.pearsonr(daily_returns[t1], daily_returns[t2])
            else:
                corr, p_val = stats.spearmanr(daily_returns[t1], daily_returns[t2])
            
            strength = "Strong" if abs(corr) >= 0.7 else "Moderate" if abs(corr) >= 0.4 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            
            correlation_pairs.append({
                "Pair": f"{t1} ↔ {t2}",
                "Correlation": round(corr, 4),
                "P-Value": f"{p_val:.2e}",
                "Strength": strength,
                "Direction": direction,
                "Significant": "Yes" if p_val < 0.05 else "No"
            })

pairs_df = pd.DataFrame(correlation_pairs)
pairs_df = pairs_df.sort_values("Correlation", ascending=False)
st.dataframe(pairs_df, use_container_width=True, hide_index=True)

# Highlight interesting correlations
st.subheader("⭐ Notable Correlations")

# Find strongest positive and negative
strongest_pos = pairs_df[pairs_df["Direction"] == "Positive"].iloc[0] if len(pairs_df[pairs_df["Direction"] == "Positive"]) > 0 else None
strongest_neg = pairs_df[pairs_df["Direction"] == "Negative"].iloc[0] if len(pairs_df[pairs_df["Direction"] == "Negative"]) > 0 else None

col1, col2 = st.columns(2)
with col1:
    if strongest_pos is not None:
        st.success(f"**Strongest Positive:** {strongest_pos['Pair']}\n\n"
                   f"Correlation: **{strongest_pos['Correlation']:.3f}** ({strongest_pos['Strength']})")
    else:
        st.info("No positive correlations found.")
        
with col2:
    if strongest_neg is not None and strongest_neg["Correlation"] < 0:
        st.error(f"**Strongest Negative:** {strongest_neg['Pair']}\n\n"
                 f"Correlation: **{strongest_neg['Correlation']:.3f}** ({strongest_neg['Strength']})")
    else:
        st.info("No notable negative correlations found (all correlations are positive).")

# --- 5. Correlation Visualization ---
st.header("5. Correlation Visualization")

# Heatmap
st.subheader("🗺️ Correlation Heatmap")

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdYlGn',
    zmin=-1, zmax=1,
    text=corr_matrix.round(2).values,
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
))

fig_heatmap.update_layout(
    title=f"{corr_method} Correlation Heatmap - Daily Stock Returns",
    xaxis_title="Stock",
    yaxis_title="Stock",
    height=500
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Scatter plots with trend lines for important pairs
st.subheader("📈 Scatter Plots with Trend Lines")

if len(pairs_df) > 0:
    # Let user select a pair to visualize
    pair_options = pairs_df["Pair"].tolist()
    selected_pair = st.selectbox("Select Stock Pair to Visualize", pair_options)
    
    # Parse selected pair
    stocks = selected_pair.split(" ↔ ")
    stock1, stock2 = stocks[0], stocks[1]
    
    # Get correlation info
    pair_info = pairs_df[pairs_df["Pair"] == selected_pair].iloc[0]
    
    fig_scatter = px.scatter(
        daily_returns, x=stock1, y=stock2,
        trendline="ols",
        labels={stock1: f"{stock1} Daily Return (%)", stock2: f"{stock2} Daily Return (%)"},
        title=f"Scatter Plot: {stock1} vs {stock2}"
    )
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.6, color='#3366cc'))
    fig_scatter.update_layout(height=450)
    
    # Add annotation with correlation value
    fig_scatter.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"r = {pair_info['Correlation']:.3f} ({pair_info['Strength']} {pair_info['Direction']})",
        showarrow=False,
        font=dict(size=14, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 6. Interpretation and Discussion ---
st.header("6. Interpretation and Discussion")

# Dynamic interpretation based on results
avg_corr = pairs_df["Correlation"].mean()
all_positive = all(pairs_df["Direction"] == "Positive")

st.markdown(f"""
### 📝 Key Findings

#### 1. Overall Correlation Pattern
- **Average correlation** across all pairs: **{avg_corr:.3f}**
- {"All correlations are **positive**" if all_positive else "Correlations show **mixed directions**"}, indicating that {"these stocks tend to move together" if all_positive else "some stocks move in opposite directions"}.

#### 2. Strongest Relationships
""")

if strongest_pos is not None:
    st.markdown(f"""
- **{strongest_pos['Pair']}** shows the strongest positive correlation (**r = {strongest_pos['Correlation']:.3f}**).
  - This {strongest_pos['Strength'].lower()} positive correlation suggests these stocks tend to rise and fall together.
  - Visible in the scatter plot as an **upward trend**.
""")

if strongest_neg is not None and strongest_neg["Correlation"] < 0:
    st.markdown(f"""
- **{strongest_neg['Pair']}** shows the strongest negative correlation (**r = {strongest_neg['Correlation']:.3f}**).
  - A negative correlation means when one rises, the other tends to fall.
  - This could indicate diversification potential.
""")

st.markdown(f"""
#### 3. Why Tech Stocks Correlate

These correlations **make sense** given the context:
- **Market factors**: All selected stocks are influenced by similar macroeconomic factors (interest rates, inflation, investor sentiment).
- **Sector correlation**: Technology stocks often move together as the sector responds to tech-specific news and trends.
- **Index membership**: Many are in the same indices (S&P 500, NASDAQ-100), creating shared investor flows.

#### 4. Potential Confounding Variables

Several factors could influence these correlations:
- **Market-wide movements**: Bull/bear market conditions affect all stocks
- **Economic events**: Federal Reserve announcements, earnings seasons
- **Time period**: Correlations may vary in different market conditions
- **Industry overlap**: Cloud computing, advertising, hardware competition

#### 5. Practical Implications

- **Portfolio diversification**: High correlation between stocks means less diversification benefit
- **Risk management**: Correlated assets can amplify portfolio volatility
- **Trading strategies**: Pairs trading may exploit temporary deviations from typical correlation

---

**Correlation Method Used:** {corr_method}  
**Data Source:** Yahoo Finance (yfinance API)  
**Analysis Period:** {start_date} to {end_date} ({len(daily_returns)} trading days)
""")

# Footer
st.markdown("---")
st.caption("Week 3 Assignment: Correlation Analysis | Data Visualization Course")
