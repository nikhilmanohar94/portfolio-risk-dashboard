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

st.subheader("2. Correlation Matrix")
corr = numeric_df.corr()
fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("3. Portfolio Metrics")

weights = st.sidebar.text_input("Asset Weights (comma-separated)", value="0.25,0.25,0.25,0.25")
weights = np.array([float(w.strip()) for w in weights.split(",")])

if len(weights) != numeric_df.shape[1]:
    st.error(f"Number of weights ({len(weights)}) doesn't match number of numeric asset columns ({numeric_df.shape[1]}).")
else:
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
    col2.metric("⚠️ 1-Day VaR (95%)", f"{var_95:.2%}")
    col3.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.2f}")

    st.subheader("4. Portfolio Return Distribution")
    fig2 = px.histogram(port_returns, nbins=50, title="Portfolio Return Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("5. Cumulative Returns")
    cum_returns = (1 + port_returns).cumprod()
    fig3 = px.line(cum_returns, title="Cumulative Return (Backtest)")
    st.plotly_chart(fig3, use_container_width=True)
