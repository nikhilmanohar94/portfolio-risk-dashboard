import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Dashboard")

# Create synthetic sample data if no upload
def generate_sample_data(num_assets=20, num_days=252*2):  # 2 years daily returns approx
    np.random.seed(42)
    # Simulate daily returns ~ Normal with small mean & std dev
    means = np.random.uniform(0.0001, 0.001, size=num_assets)
    stds = np.random.uniform(0.01, 0.03, size=num_assets)
    
    returns = np.array([
        np.random.normal(loc=means[i], scale=stds[i], size=num_days)
        for i in range(num_assets)
    ]).T
    
    columns = [f"Stock_{i+1}" for i in range(num_assets)]
    return pd.DataFrame(returns, columns=columns)

st.markdown("""
This interactive app calculates key portfolio risk metrics based on uploaded or sample asset return data.  
It includes correlation analysis, Value at Risk (VaR), Sharpe ratio, volatility, and return visualizations.

### How to use:
1. View the default **sample dataset** (daily returns of 20 assets).
2. Adjust the **asset weights** in the sidebar (comma-separated).
3. Explore metrics, return distribution, and cumulative performance.
4. (Optional) Upload your own dataset using the sidebar.

💡 **Expected format**: CSV with numeric daily returns, one asset per column.
""")

st.sidebar.header("Upload Returns Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = generate_sample_data()
    st.sidebar.info("Using synthetic sample dataset with 20 stocks")

df.dropna(inplace=True)
numeric_df = df.select_dtypes(include=np.number)

st.subheader("1. Preview of Data")
st.dataframe(df.head())
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
