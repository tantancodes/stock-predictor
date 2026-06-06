import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Global Processing Hyperparameters
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
BOLLINGER_WINDOW = 20
TRAIN_SPLIT_RATIO = 0.80

st.set_page_config(page_title="Quantitative Analytics Platform", layout="wide")


def sanitize_ticker_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flattens MultiIndex columns and normalizes schema for downstream processing."""
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.strip().str.capitalize()
    return df.dropna(subset=['Close']).reset_index(drop=True)


def engineer_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mathematical alpha features using vectorized operations."""
    # Moving Average Metrics
    df['Sma_5'] = df['Close'].rolling(window=5).mean()
    df['Sma_20'] = df['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_WINDOW).mean()
    avg_loss = loss.rolling(window=RSI_WINDOW).mean()
    df['Rsi_14'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # Bollinger Bands Volatility Channels
    df['Bb_mid'] = df['Close'].rolling(window=BOLLINGER_WINDOW).mean()
    df['Bb_std'] = df['Close'].rolling(window=BOLLINGER_WINDOW).std()
    df['Bb_upper'] = df['Bb_mid'] + 2 * df['Bb_std']
    df['Bb_lower'] = df['Bb_mid'] - 2 * df['Bb_std']

    # Moving Average Convergence Divergence (MACD)
    ema_12 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['Macd'] = ema_12 - ema_26

    # Volume & Temporal Vectors
    df['Volume_change'] = df['Volume'].pct_change()
    df['Day'] = np.arange(1, len(df) + 1)

    # Truncate lookback windows to eliminate analytical variance from backfilling
    return df.dropna().reset_index(drop=True)


# User Interface Input Architecture
st.title("Quantitative Analytics & ML Trend Engine")
st.sidebar.header("Data Ingestion Configurations")
ticker = st.sidebar.text_input("Asset Ticker Symbol", value="NVDA").upper()
days = st.sidebar.slider("Historical Observation Window (Days)", min_value=60, max_value=365, value=120)

if st.sidebar.button("Execute Predictive Models"):
    st.write(f"### Fetching historical metrics for target asset: {ticker}...")

    try:
        raw_telemetry = yf.download(ticker, period=f"{days}d")
        
        if raw_telemetry.empty:
            st.error("Error: Null dataset returned from remote data stream API.")
        else:
            # Operational Execution Pipeline
            df = sanitize_ticker_data(raw_telemetry)
            df = engineer_technical_indicators(df)

            features = ['Day', 'Sma_5', 'Sma_20', 'Rsi_14', 'Macd', 'Volume_change']
            X = df[features].values
            y = df['Close'].values

            # Sequential dataset partitioning to respect chronological time boundaries
            split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Normalize feature space parameters strictly mapping training sample bounds
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_all_scaled = scaler.transform(X)

            # Estimator Optimization Pipeline
            svr = SVR(kernel='rbf', C=1e3, gamma=0.05)
            svr.fit(X_train_scaled, y_train)
            svr_pred = svr.predict(X_test_scaled)
            svr_all = svr.predict(X_all_scaled)

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_all = rf.predict(X_all_scaled)

            # Performance Matrix Interface Layout
            st.write("### Statistical Evaluation & Benchmarks (Out-of-Sample Test Set)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("SVR R-Squared Variance", f"{r2_score(y_test, svr_pred):.4f}")
            col2.metric("SVR Mean Absolute Error", f"${mean_absolute_error(y_test, svr_pred):.2f}")
            col3.metric("Random Forest R-Squared", f"{r2_score(y_test, rf_pred):.4f}")
            col4.metric("Random Forest Mean Absolute Error", f"${mean_absolute_error(y_test, rf_pred):.2f}")

            # Plotly Dynamic Canvas Render Engine
            st.write("### Multi-Model Trend Analytics Dashboard")
            train_milestone = df['Day'].iloc[split_idx - 1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Close'], name='Asset Closing Price', line=dict(color='white', width=2)))
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Bb_upper'], name='Bollinger Upper Resistance Band', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Bb_lower'], name='Bollinger Lower Support Band', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            fig.add_trace(go.Scatter(x=df['Day'], y=svr_all, name='SVR Continuous Variance Curve', line=dict(color='cyan', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Day'], y=rf_all, name='Random Forest Ensemble Estimation Vector', line=dict(color='orange', width=1.5)))
            fig.add_vline(x=train_milestone, line_dash="dot", line_color="red", annotation_text="Out-of-Sample Threshold Boundary")
            fig.update_layout(template='plotly_dark', xaxis_title='Temporal Interval Vector', yaxis_title='Asset Price Valuation (USD)', legend=dict(orientation='h'))
            st.plotly_chart(fig, use_container_width=True)

            # Momentum Oscillator Framework
            st.write("### Momentum Analysis: Relative Strength Index")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df['Day'], y=df['Rsi_14'], name='RSI Line Vector', line=dict(color='purple')))
            rsi_fig.add_hline(y=70, line_color='red', line_dash='dash', annotation_text='Overbought Threshold Limit')
            rsi_fig.add_hline(y=30, line_color='green', line_dash='dash', annotation_text='Oversold Accumulation Boundary')
            rsi_fig.update_layout(template='plotly_dark', xaxis_title='Temporal Interval Vector', yaxis_title='RSI Value Range Matrix')
            st.plotly_chart(rsi_fig, use_container_width=True)

            # Forward Matrix Interpolation Stage
            last_matrix_row = df[features].iloc[-1].copy()
            last_matrix_row['Day'] = df['Day'].iloc[-1] + 1
            extrapolated_vector = scaler.transform([last_matrix_row.values])
            
            svr_forecast = float(svr.predict(extrapolated_vector)[0])
            rf_forecast = float(rf.predict(extrapolated_vector)[0])

            st.info(f"Support Vector Regressor Estimated Target Vector (Interval Model Day {int(last_matrix_row['Day'])}): **${svr_forecast:.2f}**")
            st.info(f"Random Forest Regression Ensemble Target Vector (Interval Model Day {int(last_matrix_row['Day'])}): **${rf_forecast:.2f}**")

    except Exception as e:
        st.error(f"Uncaught pipeline execution variance error: {e}")