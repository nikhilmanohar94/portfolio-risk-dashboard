import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")

st.title("📊 Portfolio Risk Dashboard")

# Sidebar: File uploader
st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

st.sidebar.markdown("""
**File format tips:**
- Each column = asset (e.g., AAPL, SPY, etc.)
- Each row = daily returns (e.g., 0.002 = 0.2%)
""")

# Main
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)

    st.subheader("1. Preview of Uploaded Data")
    st.dataframe(df.head())

    st.subheader("2. Correlation Matrix")
    corr = df.select_dtypes(include=np.number).corr()
    fig1 = px.imshow(corr, text_auto=True, title="Asset Return Correlation")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("3. Portfolio Metrics")

    weights = st.sidebar.text_input("Asset Weights (comma-separated)", value="0.25,0.25,0.25,0.25")
    weights = np.array([float(w.strip()) for w in weights.split(",")])
    
    if len(weights) != df.shape[1]:
        st.error(f"Number of weights ({len(weights)}) doesn't match number of assets ({df.shape[1]}).")
    else:
        weights = weights / np.sum(weights)

        # Annualized volatility
        port_vol = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))
        
        # VaR (95% confidence)
        port_returns = df.dot(weights)
        var_95 = np.percentile(port_returns, 5)

        # Sharpe ratio (assuming 2% annual risk-free rate)
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

else:
    st.info("Please upload a CSV file to begin.")

