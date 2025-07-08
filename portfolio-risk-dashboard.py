import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Top 20 Stock Returns", layout="wide")

# App title and description
st.title("S&P 500 Top 20 Stocks – Return Analysis Dashboard")
st.markdown("""
This application fetches historical return data for the top 20 S&P 500 stocks by market cap and displays return distributions, correlation heatmaps, and risk metrics.

### How to Use
- Use the **default dataset** (preloaded from Yahoo Finance).
- Or, **upload your own CSV file** with daily returns (columns = tickers, rows = daily prices).
""")

# Top 20 S&P 500 stocks by market cap (approximate list as of 2025; can be updated)
top_20_tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "TSLA",
    "UNH", "JPM", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK"
]

# Fetch historical stock prices
@st.cache_data
def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Extract adjusted close prices (use "Close" with auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data["Close"]
    else:
        adj_close = data.to_frame(name=tickers[0])

    return adj_close

# Date range: past 1 year
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

default_data = load_data(top_20_tickers, start_date, end_date)
returns = default_data.pct_change().dropna()

# File uploader
uploaded_file = st.file_uploader("Or upload your own CSV file (tickers as columns, daily prices)", type=["csv"])

if uploaded_file:
    user_data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    returns = user_data.pct_change().dropna()

# Display metrics
st.subheader("Summary Statistics")
st.write(returns.describe())

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Return distribution plots
st.subheader("Daily Return Distributions")
selected_tickers = st.multiselect("Select stocks to view distributions", returns.columns.tolist(), default=returns.columns[:5])

fig, ax = plt.subplots(figsize=(12, 6))
for ticker in selected_tickers:
    sns.kdeplot(returns[ticker], label=ticker, ax=ax)
ax.set_title("Distribution of Daily Returns")
ax.legend()
st.pyplot(fig)

# Risk metrics (volatility and max drawdown)
st.subheader("Risk Metrics")

volatility = returns.std() * np.sqrt(252)
cumulative = (1 + returns).cumprod()
max_drawdown = (cumulative / cumulative.cummax() - 1).min()

risk_df = pd.DataFrame({
    "Annualized Volatility": volatility,
    "Max Drawdown": max_drawdown
}).sort_values("Annualized Volatility", ascending=False)

st.dataframe(risk_df.style.format("{:.2%}"))
