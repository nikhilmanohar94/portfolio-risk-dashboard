import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Dashboard")

st.markdown("""
This interactive app calculates key portfolio risk metrics based on real or uploaded asset return data.  
It includes correlation analysis, Value at Risk (VaR), Sharpe ratio, volatility, and return visualizations.

### How to use:
1. View the default **Top 20 Stocks** dataset (1 year of daily returns).
2. Adjust the **asset weights** in the sidebar.
3. Explore metrics, return distribution, and cumulative performance.
4. (Optional) Upload your own dataset using the sidebar.

💡 **Expected format**: CSV with numeric daily returns, one asset per column.
""")

# Static list of tickers (as requested)
tickers = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "WMT",
    "V", "LLY", "ORCL", "NFLX", "MA", "XOM", "COST", "PG", "JNJ", "HD"
]
benchmark_ticker = "^GSPC"

# Define date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data
def load_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    return data['Close'].dropna()

# Load price data
all_prices = load_prices(tickers + [benchmark_ticker], start_date, end_date)
returns = all_prices.pct_change().dropna()
benchmark_returns = returns[benchmark_ticker]
returns = returns.drop(columns=[benchmark_ticker])

# Sidebar for file upload
st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    df = df.dropna()
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = returns.copy()
    st.sidebar.info("Using real return data for top 20 US stocks")

# Preview return data
st.subheader("1. Preview of Return Data")
st.dataframe(df.head())

# CSV download
csv_buffer = io.StringIO()
df.to_csv(csv_buffer)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📥 Download Raw Return Data as CSV",
    data=csv_data,
    file_name="portfolio_return_data.csv",
    mime="text/csv",
)

st.markdown("""
This table shows the first few rows of your dataset.  
Each column corresponds to an asset's daily return, typically calculated as:

$$ r_{t} = \\frac{P_t - P_{t-1}}{P_{t-1}} $$

where $P_t$ is the asset price at day $t$.  
Proper formatting and clean data are essential for accurate portfolio analysis.
""")

# Correlation Matrix
st.subheader("2. Correlation Matrix")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
The correlation matrix quantifies how pairs of asset returns move together.  
Correlation values range from -1 to +1:

- $+1$ indicates perfect positive correlation (assets move exactly together),
- $-1$ indicates perfect negative correlation (assets move exactly opposite),
- $0$ means no linear relationship.

Mathematically:

$$ \\rho_{i,j} = \\frac{\\text{Cov}(r_i, r_j)}{\\sigma_i \\sigma_j} $$

Assets with low or negative correlation help reduce overall portfolio risk through diversification.
""")

# Portfolio Metrics
st.subheader("3. Portfolio Metrics")

default_weights = ", ".join(["0.05"] * numeric_df.shape[1])
weights_input = st.sidebar.text_input("Asset Weights (comma-separated)", value=default_weights)

try:
    weights = np.array([float(w.strip()) for w in weights_input.split(",")])
except ValueError:
    st.error("Invalid weights input. Please enter numeric comma-separated values.")
    st.stop()

if len(weights) != numeric_df.shape[1]:
    st.error(f"Number of weights ({len(weights)}) doesn't match number of columns ({numeric_df.shape[1]}).")
    st.stop()

weights = weights / np.sum(weights)

cov_matrix = numeric_df.cov() * 252
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

port_returns = numeric_df.dot(weights)
var_95 = np.percentile(port_returns, 5)
rf_daily = 0.02 / 252
excess_returns = port_returns - rf_daily
sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

benchmark_vol = benchmark_returns.std() * np.sqrt(252)
benchmark_sharpe = ((benchmark_returns.mean() - rf_daily) / benchmark_returns.std()) * np.sqrt(252)

col1, col2, col3 = st.columns(3)
col1.metric("📉 Annualized Volatility", f"{port_vol:.2%}")
col2.metric("⚠️ 1-Day VaR (95%)", f"{abs(var_95):.2%}")
col3.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")

st.markdown(f"""
**Interpretation**  
- Your portfolio's annualized volatility is **{port_vol:.2%}**, compared to the S&P 500's **{benchmark_vol:.2%}**.  
  This indicates your portfolio has {"more" if port_vol > benchmark_vol else "less"} variability in returns.

- The 1-day 95% VaR of **{abs(var_95):.2%}** means that on 95% of days, you are unlikely to lose more than this amount.

- The Sharpe Ratio of **{sharpe_ratio:.2f}** {"exceeds" if sharpe_ratio > benchmark_sharpe else "is lower than"} the S&P 500’s ratio of **{benchmark_sharpe:.2f}**, indicating your portfolio has {"better" if sharpe_ratio > benchmark_sharpe else "worse"} risk-adjusted returns.
""")

# Return Distribution
st.subheader("4. Portfolio Return Distribution")
fig2 = px.histogram(port_returns, nbins=50, title="Portfolio Return Distribution")
st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"""
This histogram shows the frequency of daily portfolio returns.  

- **Shape**: Look for skewness or fat tails indicating abnormal risk.
- **Center**: Your average daily return is **{port_returns.mean():.4%}** vs the benchmark's **{benchmark_returns.mean():.4%}**.
- **Spread**: The portfolio's return distribution is {"more" if port_returns.std() > benchmark_returns.std() else "less"} volatile than the S&P 500.

Understanding return distribution helps gauge expected gains/losses and extreme outcomes.
""")

# Cumulative Returns
st.subheader("5. Cumulative Returns")
cum_port_returns = (1 + port_returns).cumprod()
cum_benchmark_returns = (1 + benchmark_returns).cumprod()

cum_df = pd.DataFrame({
    "Portfolio": cum_port_returns,
    "S&P 500": cum_benchmark_returns
})

fig3 = px.line(cum_df, title="Cumulative Return Comparison")
st.plotly_chart(fig3, use_container_width=True)

total_return = cum_port_returns.iloc[-1] - 1
benchmark_total = cum_benchmark_returns.iloc[-1] - 1

st.markdown(f"""
The cumulative return shows how your portfolio has grown over time compared to the benchmark.

- Your portfolio’s total return over the past year is **{total_return:.2%}**, while the S&P 500 returned **{benchmark_total:.2%}**.

This helps evaluate whether your investment strategy is outperforming or lagging the market.
""")
