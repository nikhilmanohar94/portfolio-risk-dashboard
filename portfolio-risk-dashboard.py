import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Dashboard")

@st.cache_data(show_spinner=True)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    tickers = table['Symbol'].tolist()
    return tickers, table

@st.cache_data(show_spinner=True)
def get_market_caps(tickers):
    # Fetch info for all tickers in batches to avoid rate limits
    info_list = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        tickers_str = " ".join(batch)
        data = yf.Tickers(tickers_str)
        for ticker in batch:
            info = data.tickers[ticker].info
            info_list.append({
                "Ticker": ticker,
                "Market Cap": info.get("marketCap", np.nan),
                "Name": info.get("shortName", ""),
                "Price": info.get("regularMarketPrice", np.nan),
            })
    df_info = pd.DataFrame(info_list)
    df_info = df_info.dropna(subset=["Market Cap"])
    return df_info

# Get full S&P 500 list and tickers
all_tickers, sp500_table = get_sp500_tickers()

# Get market caps and other info
market_info_df = get_market_caps(all_tickers)

# Select top 20 by market cap
top_20_info = market_info_df.sort_values("Market Cap", ascending=False).head(20)
top_20_tickers = top_20_info["Ticker"].tolist()

st.subheader("Current Market Data for Top 20 S&P 500 Stocks")
st.dataframe(
    top_20_info.reset_index(drop=True).style.format({
        "Price": "${:,.2f}",
        "Market Cap": "${:,.0f}"
    })
)

# Define date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

@st.cache_data
def load_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data.to_frame()
        prices.columns = [tickers[0]]
    return prices.dropna()

prices = load_prices(top_20_tickers, start_date, end_date)
returns = prices.pct_change().dropna()

st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    df = df.dropna()
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = returns.copy()
    st.sidebar.info("Using real return data for top 20 S&P 500 stocks")

# Then continue with the rest of your app (show data preview, correlation matrix, metrics, etc.)
# ...



# Fetch live market data summary for tickers
@st.cache_data
def load_market_info(tickers):
    info_list = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            info_list.append({
                "Ticker": ticker,
                "Name": info.get("shortName", ""),
                "Price": info.get("regularMarketPrice", np.nan),
                "Market Cap": info.get("marketCap", np.nan),
            })
        except Exception as e:
            info_list.append({
                "Ticker": ticker,
                "Name": "",
                "Price": np.nan,
                "Market Cap": np.nan,
            })
    df_info = pd.DataFrame(info_list)
    df_info = df_info.dropna(subset=["Market Cap"]).sort_values(by="Market Cap", ascending=False).reset_index(drop=True)
    return df_info

market_info_df = load_market_info(tickers)

st.subheader("Current Market Data for Top 20 S&P 500 Stocks")
st.dataframe(market_info_df.style.format({
    "Price": "${:,.2f}",
    "Market Cap": "${:,.0f}"
}))
# Then show the raw returns preview
st.subheader("1. Preview of Return Data")
st.dataframe(df.head())

# Prepare CSV bytes for download
csv_buffer = io.StringIO()
df.to_csv(csv_buffer)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📥 Download Raw Return Data as CSV",
    data=csv_data,
    file_name="portfolio_return_data.csv",
    mime="text/csv",
)

st.markdown(
    """
    This table shows the first few rows of your dataset.  
    Each column corresponds to an asset's daily return, typically calculated as:
    
    $$ r_{t} = \\frac{P_t - P_{t-1}}{P_{t-1}} $$
    
    where $P_t$ is the asset price at day $t$.  
    Proper formatting and clean data are essential for accurate portfolio analysis.
    """
)

st.subheader("2. Correlation Matrix")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
st.plotly_chart(fig1, use_container_width=True)
st.markdown(
    """
    The correlation matrix quantifies how pairs of asset returns move together.  
    Correlation values range from -1 to +1:
    
    - $+1$ indicates perfect positive correlation (assets move exactly together),
    - $-1$ indicates perfect negative correlation (assets move exactly opposite),
    - $0$ means no linear relationship.
    
    Mathematically, the Pearson correlation coefficient between assets $i$ and $j$ is:
    
    $$ \\rho_{i,j} = \\frac{\\text{Cov}(r_i, r_j)}{\\sigma_i \\sigma_j} $$
    
    where Cov is covariance, and $\sigma_i$ and $\sigma_j$ are standard deviations of returns.
    
    Assets with low or negative correlation help reduce overall portfolio risk through diversification.
    """
)

st.subheader("3. Portfolio Metrics")

default_weights = ", ".join(["0.05"] * numeric_df.shape[1])
weights_input = st.sidebar.text_input("Asset Weights (comma-separated)", value=default_weights)
try:
    weights = np.array([float(w.strip()) for w in weights_input.split(",")])
except ValueError:
    st.error("Invalid weights input. Please enter numeric comma-separated values.")
    st.stop()

if len(weights) != numeric_df.shape[1]:
    st.error(f"Number of weights ({len(weights)}) doesn't match number of numeric asset columns ({numeric_df.shape[1]}).")
else:
    if np.sum(weights) == 0:
        st.error("Sum of weights cannot be zero.")
        st.stop()

    weights = weights / np.sum(weights)

    cov_matrix = numeric_df.cov() * 252  # Annualized covariance
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    port_returns = numeric_df.dot(weights)
    var_95 = np.percentile(port_returns, 5)

    rf_daily = 0.02 / 252
    excess_returns = port_returns - rf_daily
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Annualized Volatility", f"{port_vol:.2%}")
    col2.metric("⚠️ 1-Day VaR (95%)", f"{abs(var_95):.2%}")
    col3.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")

    st.markdown(
        """
        ### Annualized Volatility
        Measures the portfolio's total risk by quantifying the standard deviation of returns on an annual basis.  
        It is calculated as:

        $$ \\sigma_p = \\sqrt{\\mathbf{w}^T \\mathbf{\\Sigma} \\mathbf{w}} $$

        where:
        - $\\mathbf{w}$ is the vector of asset weights,
        - $\\mathbf{\\Sigma}$ is the annualized covariance matrix of asset returns.

        A higher volatility indicates larger fluctuations and greater risk.

        ---

        ### 1-Day Value at Risk (VaR) at 95% Confidence Level
        VaR estimates the maximum expected loss over one trading day with 95% confidence, i.e., there's a 5% chance losses exceed this value.

        Computed as the 5th percentile of the portfolio return distribution:

        $$ \\text{VaR}_{95\\%} = -\\text{Percentile}_{5}(r_p) $$

        where $r_p$ are portfolio returns.

        For example, a VaR of 5% means that on 95% of days, losses will not exceed 5%.

        ---

        ### Sharpe Ratio
        The Sharpe ratio measures the portfolio's risk-adjusted return by comparing excess returns to volatility:

        $$ S = \\frac{E[R_p - R_f]}{\\sigma_p} \\times \\sqrt{252} $$

        where:
        - $R_p$ = portfolio return,
        - $R_f$ = risk-free return,
        - $\\sigma_p$ = standard deviation of portfolio returns,
        - 252 is annualization factor (trading days).

        A higher Sharpe ratio indicates better returns for the risk taken.
        """
    )

    st.subheader("4. Portfolio Return Distribution")
    fig2 = px.histogram(port_returns, nbins=50, title="Portfolio Return Distribution")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        """
        This histogram shows the frequency distribution of daily portfolio returns.  
        Key points to note:
        
        - The shape indicates how returns are distributed (normal, skewed, etc.).
        - The center shows average returns.
        - The tails highlight the probability of extreme losses or gains.
        
        Understanding this distribution is essential for assessing downside risk and portfolio behavior.
        """
    )

    st.subheader("5. Cumulative Returns")
    cum_returns = (1 + port_returns).cumprod()
    fig3 = px.line(cum_returns, title="Cumulative Return (Backtest)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        """
        This chart shows how an initial investment would have grown over time by compounding daily returns:

        $$ V_t = V_0 \\times \\prod_{i=1}^t (1 + r_i) $$

        where:
        - $V_t$ is portfolio value at time $t$,
        - $V_0$ is the initial investment (normalized to 1),
        - $r_i$ are daily portfolio returns.

        It provides a clear picture of growth trends and performance consistency.
        """
    )
