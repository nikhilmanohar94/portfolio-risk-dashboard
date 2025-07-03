import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("📊 Portfolio Risk Dashboard")

# Description and instructions
st.markdown("""
This interactive app calculates key portfolio risk metrics based on uploaded or sample asset return data.  
It includes correlation analysis, Value at Risk (VaR), Sharpe ratio, volatility, and return visualizations.

### 🧭 How to use:
1. View the default **sample dataset** (daily returns of 4 assets).
2. Adjust the **asset weights** in the sidebar (comma-separated).
3. Explore metrics, return distribution, and cumulative performance.
4. (Optional) Upload your own dataset using the sidebar.

💡 **Expected format**: CSV with numeric daily returns, one asset per column.
""")

# Sidebar: File uploader
st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Load sample data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = pd.read_csv("sample_data.csv")
    st.sidebar.info("Using built-in sample dataset (AAPL, MSFT, SPY, TSLA)")

df.dropna(inplace=True)
numeric_df = df.select_dtypes(include=np.number)

st.subheader("1. Preview of Data")
st.dataframe(df.head())
st.markdown(
    "This table shows the first few rows of your dataset. Each column represents an asset's daily returns. "
    "Ensure your data is clean and in the correct format for accurate analysis."
)

st.subheader("2. Correlation Matrix")
corr = numeric_df.corr()
fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
st.plotly_chart(fig1, use_container_width=True)
st.markdown(
    "The correlation matrix shows how asset returns move in relation to each other. "
    "Values close to 1 mean assets move very similarly, while values near -1 indicate they move in opposite directions. "
    "Diversification benefits come from assets with low or negative correlations."
)

st.subheader("3. Portfolio Metrics")

weights_input = st.sidebar.text_input("Asset Weights (comma-separated)", value="0.25,0.25,0.25,0.25")
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

    cov_matrix = numeric_df.cov() * 252
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
        "**Annualized Volatility:** Measures the portfolio's total risk or how much its returns fluctuate over a year. "
        "Higher volatility means more uncertainty.\n\n"
        "**1-Day Value at Risk (VaR) at 95% confidence:** The estimated maximum loss you could expect on 1 day, "
        "with 95% confidence. For example, a 5% VaR means there is a 5% chance the portfolio will lose more than this on any day.\n\n"
        "**Sharpe Ratio:** Measures risk-adjusted return — how much excess return you get per unit of risk. "
        "Higher Sharpe ratios indicate better risk-adjusted performance."
    )

    st.subheader("4. Portfolio Return Distribution")
    fig2 = px.histogram(port_returns, nbins=50, title="Portfolio Return Distribution")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "This histogram shows the distribution of daily portfolio returns. "
        "It helps visualize the frequency of gains and losses and assess the risk of extreme outcomes."
    )

    st.subheader("5. Cumulative Returns")
    cum_returns = (1 + port_returns).cumprod()
    fig3 = px.line(cum_returns, title="Cumulative Return (Backtest)")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "This line chart shows how your portfolio's value would have grown over time assuming all returns were reinvested. "
        "It provides a sense of overall performance and growth trajectory."
    )
