import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Top 20 Stock Dashboard", layout="wide")

# Title & Instructions
st.title("S&P 500 Top 20 Stocks – Return & Risk Dashboard")
st.markdown("""
Analyze historical performance, risk metrics, and correlations for the top 20 S&P 500 companies by market cap.

### How to Use:
- Use the **default dataset** (auto-loaded with the latest daily returns from Yahoo Finance).
- Or, **upload your own CSV** with daily price data (columns = tickers, rows = dates).
""")

# Top 20 tickers (as of 2025, update as needed)
top_20_tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "UNH", "JPM", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK"
]

# Download adjusted close prices
@st.cache_data
def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle single and multi-index cases
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Adj Close"]
    else:
        adj_close = data.to_frame(name=tickers[0])

    return adj_close

# Date range: past 1 year
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# Load default data
try:
    default_data = load_data(top_20_tickers, start_date, end_date)
    returns = default_data.pct_change().dropna()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Option to upload your own CSV
uploaded_file = st.file_uploader("Upload your own CSV (daily prices, tickers as columns)", type=["csv"])
if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        returns = user_data.pct_change().dropna()
        st.success("Successfully loaded custom dataset.")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        st.stop()

# Summary Stats
st.subheader("Summary Statistics")
st.dataframe(returns.describe().T.style.format("{:.4f}"))

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Return Distributions
st.subheader("Daily Return Distributions")
selected = st.multiselect("Select stocks to plot", options=returns.columns.tolist(), default=returns.columns[:5])
fig, ax = plt.subplots(figsize=(12, 6))
for ticker in selected:
    sns.kdeplot(returns[ticker], label=ticker, ax=ax)
ax.set_title("Distribution of Daily Returns")
ax.legend()
st.pyplot(fig)

# Risk Metrics: Volatility & Max Drawdown
st.subheader("Risk Metrics")
volatility = returns.std() * np.sqrt(252)
cumulative = (1 + returns).cumprod()
max_drawdown = (cumulative / cumulative.cummax() - 1).min()
risk_df = pd.DataFrame({
    "Annualized Volatility": volatility,
    "Max Drawdown": max_drawdown
}).sort_values("Annualized Volatility", ascending=False)
st.dataframe(risk_df.style.format("{:.2%}"))
