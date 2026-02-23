import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

st.set_page_config(page_title="Week 2: Time Series Analysis", page_icon="📈")

st.title("📈 Week 2: Stock Price Time Series Analysis")
st.markdown("This app analyzes stock price trends, seasonality, and performs linear regression forecasting.")

# --- 1. Dataset Selection ---
st.header("1. Dataset Selection")

# Dropdown menu for ticker selection
ticker_options = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Microsoft": "MSFT"
}
selected_ticker_name = st.selectbox("Select Stock Ticker", list(ticker_options.keys()))
ticker = ticker_options[selected_ticker_name]

start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

@st.cache_data
def load_data(ticker, start, end):
    # We use group_by='column' to ensure a more predictable structure in newer yfinance versions
    data = yf.download(ticker, start=start, end=end, progress=False, group_by='column')
    
    if data.empty:
        return data

    # Handle MultiIndex columns (common in newer yfinance versions: ('Close', 'AAPL'))
    if isinstance(data.columns, pd.MultiIndex):
        # We try to get 'Close' for our specific ticker
        if 'Close' in data.columns.get_level_values(0):
            data = data['Close']
            # If it's still a DataFrame (could happen if multi-ticker, though we pass one)
            if isinstance(data, pd.DataFrame) and ticker in data.columns:
                data = data[[ticker]]
    else:
        # Standard SingleIndex case
        if 'Close' in data.columns:
            data = data[['Close']]
    
    # Ensure we have a DataFrame with a single column named 'Close'
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    data.columns = ['Close']
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('B') # Business days frequency 
    return data

try:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        if end_date >= datetime.date.today():
             st.warning(f"No data found for {ticker} up to {end_date}. This might be because the market hasn't opened yet or data for the current day isn't available yet. **Try choosing yesterday or an earlier date as the End Date.**")
        else:
            st.warning(f"No data found for {ticker} in the selected date range. Please try a wider range.")
        st.stop()
    st.success(f"Loaded {len(df)} days of data for {ticker}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. EDA ---
st.header("2. Exploratory Data Analysis (EDA)")
st.subheader("Raw Time Series")
# st.line_chart(df['Close'])
fig_eda = go.Figure()
fig_eda.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_eda.update_layout(
    yaxis_title="Price (USD)",
    xaxis_title="Date",
    title=f"{ticker} Daily Close Price"
)
st.plotly_chart(fig_eda)

st.subheader("Statistics")
st.write(df.describe())
st.markdown("_*Note: Count, Mean, Std, Min, 25%, 50%, 75%, Max values for 'Close' are in USD._")

missing_count = df.isnull().sum().sum()
st.write(f"**Missing Values:** {missing_count}")
if missing_count > 0:
    st.warning("Handling missing values using Forward Fill...")
    df = df.ffill()
    st.success("Missing values filled.")

# --- 3. Trend and Seasonality Analysis ---
st.header("3. Trend and Seasonality Analysis")

# Decompose needs no missing values and enough observations
period = 30
min_obs = 2 * period
df_clean = df['Close'].dropna()

if len(df_clean) < min_obs:
    st.warning(f"Not enough data points for seasonal decomposition. Need at least {min_obs} observations, but only have {len(df_clean)}. Try selecting a longer timeframe.")
else:
    decomposition = seasonal_decompose(df_clean, model='additive', period=period) 
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trend")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
        fig_trend.update_layout(yaxis_title="Price (USD)", xaxis_title="Date")
        st.plotly_chart(fig_trend)

    with col2:
        st.subheader("Seasonality")
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonality'))
        fig_seasonal.update_layout(yaxis_title="Price Deviation (USD)", xaxis_title="Date")
        st.plotly_chart(fig_seasonal)

    st.subheader("Residuals")
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residuals'))
    fig_resid.update_layout(yaxis_title="Price Deviation (USD)", xaxis_title="Date")
    st.plotly_chart(fig_resid)

# --- 4. Regression Analysis ---
st.header("4. Regression Analysis (Trend Modeling)")

# Prepare data for regression
# X needs to be numeric (e.g., ordinal date)
df_reg = df.reset_index().dropna()
df_reg['Date_Ordinal'] = df_reg['Date'].apply(lambda x: x.toordinal())

X = df_reg[['Date_Ordinal']]
y = df_reg['Close']

model = LinearRegression()
model.fit(X, y)
df_reg['Predicted_Trend'] = model.predict(X)

# Metrics
rmse = np.sqrt(mean_squared_error(y, df_reg['Predicted_Trend']))
mae = mean_absolute_error(y, df_reg['Predicted_Trend'])

st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
st.caption("Represents the typical magnitude of the prediction error (standard deviation of residuals). Lower is better.")

st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
st.caption("Represents the average absolute difference between the actual and predicted prices in USD. Lower is better.")

# --- 5. Visualization & Forecasting ---
st.header("5. Visualization & Forecast")

# Future Prediction
days_to_predict = st.slider("Days to Forecast", 10, 365, 30)
last_date = df_reg['Date'].max()
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days_to_predict + 1)]
future_ordinals = [[d.toordinal()] for d in future_dates]
future_preds = model.predict(future_ordinals)

# Create Plotly Graph
fig = go.Figure()

# Original Data
fig.add_trace(go.Scatter(x=df_reg['Date'], y=df_reg['Close'], mode='lines', name='Actual Price'))
# Fitted Regression Line
fig.add_trace(go.Scatter(x=df_reg['Date'], y=df_reg['Predicted_Trend'], mode='lines', name='Linear Trend', line=dict(dash='dash')))
# Forecast
fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Forecast', line=dict(color='red')))

fig.update_layout(title=f"{ticker} Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig)

# --- 6. Key Findings ---
st.header("📝 Key Findings")

# Calculate slope for the summary
slope = model.coef_[0]
trend_direction = "upward" if slope > 0 else "downward"

st.markdown(f"""
### Summary of Analysis: **{ticker}**

1. **Long-term Trend:**
    * The stock displays a clear **{trend_direction}** trend over the selected period.
    * **Rate of change:** approximately **{slope:.4f}** USD per business day.
    * This indicates the overall momentum of {ticker} in the analyzed timeframe.

2. **Seasonality:**
    * A 30-day decomposition was used to identify monthly patterns.
    * The seasonal chart reveals recurring fluctuations that may correspond to monthly market cycles or earnings periods.
    * These patterns appear relatively consistent, though they are smaller in magnitude compared to the primary trend.

3. **Decomposition Analysis:**
    * **Trend:** Shows the smoothed underlying movement of the stock price, removing daily noise.
    * **Seasonality:** Highlights the periodic ups and downs within a 30-day window.
    * **Residuals:** Represent the 'random' volatility or unexpected market events that aren't explained by trend or seasonality.

4. **Forecast ({days_to_predict} Days):**
    * Based on the Linear Regression model, the price is projected to reach approximately **${future_preds[-1]:.2f}** by the end of the forecast period.
    * This projection assumes the historical linear trend continues without major market shifts.

5. **Model Performance & Limitations:**
    * **RMSE:** {rmse:.2f} | **MAE:** {mae:.2f}
    * Linear regression provides a simplified "baseline" view but does not account for complex market volatilty or sudden news events.
    * For financial data, more advanced models like LSTM or Prophet might capture non-linear movements more accurately.

**Data Source:** Yahoo Finance (YFinance API)
**Analysis Performed:** Linear Regression and Additive Time Series Decomposition.
""")
