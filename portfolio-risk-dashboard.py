import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("📊 Portfolio Risk Dashboard")

st.markdown("""
This interactive dashboard calculates key portfolio risk metrics and compares your portfolio to the S&P 500 benchmark.  

### How to Use:
1. By default, the app analyzes a portfolio of 20 major US stocks using 1 year of daily price data.
2. You can adjust portfolio weights in the sidebar.
3. Optionally upload your own return dataset.
4. Visualizations include correlation, VaR, Sharpe ratio, volatility, distribution, and cumulative returns.

**Required Format for Upload:** CSV with dates as index and columns as asset returns.
""")

# --- SETTINGS ---
tickers = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "WMT",
    "V", "LLY", "ORCL", "NFLX", "MA", "XOM", "COST", "PG", "JNJ", "HD"
]
benchmark_ticker = "^GSPC"
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# --- DATA LOADING ---
@st.cache_data
def load_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    return data['Close'].dropna()

all_prices = load_prices(tickers + [benchmark_ticker], start_date, end_date)
returns = all_prices.pct_change().dropna()
benchmark_returns = returns[benchmark_ticker]
returns = returns.drop(columns=[benchmark_ticker])

# --- DATA SELECTION ---
st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    df = df.dropna()
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = returns.copy()
    st.sidebar.info("Using default data for top 20 stocks")

# --- PREVIEW ---
st.subheader("1. Preview of Return Data")
st.dataframe(df.head())

csv_buffer = io.StringIO()
df.to_csv(csv_buffer)
csv_data = csv_buffer.getvalue()
st.download_button("📥 Download Return Data", data=csv_data, file_name="portfolio_returns.csv", mime="text/csv")

st.markdown("""
Each value in the dataset represents the daily return for a particular asset.  
Returns are calculated as:

$$
r_t = \\frac{P_t - P_{t-1}}{P_{t-1}}
$$

where:
- $P_t$: Price on day $t$  
- $r_t$: Daily return  

This data forms the basis for all portfolio risk and performance calculations.
""")

# --- CORRELATION ---
st.subheader("2. Correlation Matrix")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("The correlation matrix indicates how asset returns move together.")
st.markdown("It is calculated using the Pearson correlation coefficient:")

st.latex(r"\rho_{i,j} = \frac{\mathrm{Cov}(r_i, r_j)}{\sigma_i \sigma_j}")

st.markdown("""
Where:  
- $\mathrm{Cov}(r_i, r_j)$ is the covariance between returns $r_i$ and $r_j$  
- $\sigma_i$, $\sigma_j$ are the standard deviations of the respective returns  

**Interpretation:**  
- $\\rho = 1$: Perfect positive correlation  
- $\\rho = -1$: Perfect negative correlation  
- $\\rho = 0$: No linear relationship  

Lower or negative correlations between assets improve diversification and reduce portfolio risk.
""")

# --- PORTFOLIO METRICS ---
st.subheader("3. Portfolio Metrics")

default_weights = ", ".join(["0.05"] * numeric_df.shape[1])
weights_input = st.sidebar.text_input("Asset Weights (comma-separated)", value=default_weights)

try:
    weights = np.array([float(w.strip()) for w in weights_input.split(",")])
except ValueError:
    st.error("Please enter valid numeric weights.")
    st.stop()

if len(weights) != numeric_df.shape[1]:
    st.error("Number of weights must match number of assets.")
    st.stop()

weights = weights / np.sum(weights)

cov_matrix = numeric_df.cov() * 252  # Annualize
port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

port_returns = numeric_df.dot(weights)
rf_daily = 0.02 / 252
excess_returns = port_returns - rf_daily
sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
var_95 = np.percentile(port_returns, 5)

benchmark_vol = benchmark_returns.std() * np.sqrt(252)
benchmark_sharpe = ((benchmark_returns.mean() - rf_daily) / benchmark_returns.std()) * np.sqrt(252)

col1, col2, col3 = st.columns(3)
col1.metric("📉 Annualized Volatility", f"{port_vol:.2%}")
col2.metric("⚠️ 1-Day VaR (95%)", f"{abs(var_95):.2%}")
col3.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")

st.markdown("### Annualized Volatility")
st.markdown(f"""
Measures the total risk of the portfolio, calculated as:

$$
\\sigma_p = \\sqrt{{ \\mathbf{{w}}^T \\mathbf{{\\Sigma}} \\mathbf{{w}} }}
$$

Where:  
- $\\mathbf{{w}}$: Vector of asset weights  
- $\\mathbf{{\\Sigma}}$: Covariance matrix of returns  
- $\\sigma_p$: Annualized portfolio volatility

Your portfolio's annualized volatility is **{port_vol:.2%}**, compared to the S&P 500's **{benchmark_vol:.2%}**.

---
""")

st.markdown("### Value at Risk (VaR) at 95% Confidence")
st.markdown(f"""
Estimates the maximum expected loss over one day with 95% confidence.

$$
\\text{{VaR}}_{{95\\%}} = -\\text{{Percentile}}_5(r_p)
$$

Where $r_p$ are daily portfolio returns.

Your 1-day VaR is **{abs(var_95):.2%}**, meaning that in 95% of cases, losses should not exceed this value.

---
""")

st.markdown("### Sharpe Ratio")
st.markdown(f"""
Measures the portfolio's risk-adjusted return.

$$
S = \\frac{{E[R_p - R_f]}}{{\\sigma_p}} \\times \\sqrt{{252}}
$$

Where:  
- $R_p$: Portfolio return  
- $R_f$: Risk-free return  
- $\\sigma_p$: Volatility of portfolio returns

Your Sharpe ratio is **{sharpe_ratio:.2f}**, while the S&P 500’s Sharpe ratio is **{benchmark_sharpe:.2f}**.  
A higher Sharpe ratio indicates better risk-adjusted performance.
""")

# --- RETURN DISTRIBUTION ---
st.subheader("4. Portfolio Return Distribution")

fig2 = px.histogram(port_returns, nbins=50, title="Daily Return Distribution")
st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"""
This histogram shows how often different daily returns occurred in your portfolio.

- The **center** reflects your average daily return: **{port_returns.mean():.4%}**  
- The **spread** (standard deviation) is **{port_returns.std():.4%}**, compared to **{benchmark_returns.std():.4%}** for the S&P 500

---

### Interpretation:
- **Symmetry** indicates normal return behavior  
- **Skew** shows whether large gains or losses dominate  
- **Fat tails** suggest potential for extreme outcomes

Understanding return distributions helps assess downside risk and tail events.
""")

# --- CUMULATIVE RETURNS ---
st.subheader("5. Cumulative Returns")

cum_port = (1 + port_returns).cumprod()
cum_bench = (1 + benchmark_returns).cumprod()
cum_df = pd.DataFrame({
    "Portfolio": cum_port,
    "S&P 500": cum_bench
})

fig3 = px.line(cum_df, title="Cumulative Return Comparison")
st.plotly_chart(fig3, use_container_width=True)

total_return = cum_port.iloc[-1] - 1
benchmark_return = cum_bench.iloc[-1] - 1

st.markdown(f"""
This chart shows how an investment of $1 would have grown over the past year.

Cumulative return is calculated as:

$$
V_t = V_0 \\times \\prod_{{i=1}}^t (1 + r_i)
$$

Where:  
- $V_0$: Initial value (normalized to 1)  
- $r_i$: Return on day $i$

---

### Interpretation:
- Your portfolio grew by **{total_return:.2%}** over the past year  
- The S&P 500 grew by **{benchmark_return:.2%}** over the same period

This provides a direct benchmark comparison of total performance.
""")
