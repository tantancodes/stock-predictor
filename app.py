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

# Page Configuration
st.set_page_config(page_title="Quant ML Dashboard", layout="wide")
st.title("📈 Quantitative Analytics & ML Trend Dashboard")
st.write("Fetches live stock data, engineers technical indicators, and compares SVR vs Random Forest models.")

# Sidebar UI Controls
st.sidebar.header("User Configurations")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="NVDA").upper()
days = st.sidebar.slider("Days of Historical Data", min_value=60, max_value=365, value=120)

if st.sidebar.button("Run Predictive Model"):
    st.write(f"### Fetching data for **{ticker}** over the last {days} days...")

    try:
        raw_data = yf.download(ticker, period=f"{days}d")

        if raw_data.empty:
            st.error("❌ No data found. Please check your ticker symbol.")
        else:
            # 1. Data Processing
            df = raw_data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.strip().str.capitalize()
            df = df.dropna(subset=['Close']).reset_index(drop=True)

            # 2. Feature Engineering
            # SMA
            df['Sma_5'] = df['Close'].rolling(window=5).mean()
            df['Sma_20'] = df['Close'].rolling(window=20).mean()

            # RSI (14-day)
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['Rsi_14'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['Bb_mid'] = df['Close'].rolling(window=20).mean()
            df['Bb_std'] = df['Close'].rolling(window=20).std()
            df['Bb_upper'] = df['Bb_mid'] + 2 * df['Bb_std']
            df['Bb_lower'] = df['Bb_mid'] - 2 * df['Bb_std']

            # MACD
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['Macd'] = ema_12 - ema_26

            # Volume change
            df['Volume_change'] = df['Volume'].pct_change()

            df['Day'] = np.arange(1, len(df) + 1)

            # Drop NaN rows from rolling windows (no bfill)
            df = df.dropna().reset_index(drop=True)

            # 3. ML Setup
            features = ['Day', 'Sma_5', 'Sma_20', 'Rsi_14', 'Macd', 'Volume_change']
            X = df[features].values
            y = df['Close'].values

            split = int(len(df) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_all_scaled = scaler.transform(X)

            # Train SVR
            svr = SVR(kernel='rbf', C=1e3, gamma=0.05)
            svr.fit(X_train_scaled, y_train)
            svr_pred = svr.predict(X_test_scaled)
            svr_all = svr.predict(X_all_scaled)

            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_all = rf.predict(X_all_scaled)

            # 4. Metrics
            st.write("### Model Comparison")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("SVR R²", f"{r2_score(y_test, svr_pred):.4f}")
            col2.metric("SVR MAE", f"${mean_absolute_error(y_test, svr_pred):.2f}")
            col3.metric("RF R²", f"{r2_score(y_test, rf_pred):.4f}")
            col4.metric("RF MAE", f"${mean_absolute_error(y_test, rf_pred):.2f}")

            # 5. Plotly Chart
            st.write("### Interactive Trend Analysis")
            train_end = df['Day'].iloc[split - 1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Close'], name='Actual Price', line=dict(color='white', width=2)))
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Bb_upper'], name='Bollinger Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=df['Day'], y=df['Bb_lower'], name='Bollinger Lower', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            fig.add_trace(go.Scatter(x=df['Day'], y=svr_all, name='SVR Prediction', line=dict(color='cyan', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Day'], y=rf_all, name='RF Prediction', line=dict(color='orange', width=1.5)))
            fig.add_vline(x=train_end, line_dash="dot", line_color="red", annotation_text="Train/Test Split")
            fig.update_layout(template='plotly_dark', xaxis_title='Day', yaxis_title='Price (USD)', legend=dict(orientation='h'))
            st.plotly_chart(fig, use_container_width=True)

            # 6. RSI Chart
            st.write("### RSI Indicator (Overbought/Oversold)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df['Day'], y=df['Rsi_14'], name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=70, line_color='red', line_dash='dash', annotation_text='Overbought')
            rsi_fig.add_hline(y=30, line_color='green', line_dash='dash', annotation_text='Oversold')
            rsi_fig.update_layout(template='plotly_dark', xaxis_title='Day', yaxis_title='RSI')
            st.plotly_chart(rsi_fig, use_container_width=True)

            # 7. Next Day Forecast
            last_row = df[features].iloc[-1].copy()
            last_row['Day'] = df['Day'].iloc[-1] + 1
            next_scaled = scaler.transform([last_row.values])
            svr_next = float(svr.predict(next_scaled)[0])
            rf_next = float(rf.predict(next_scaled)[0])

            st.success(f"🔮 SVR Forecast (Day {int(last_row['Day'])}): **${svr_next:.2f}**")
            st.success(f"🌲 RF Forecast (Day {int(last_row['Day'])}): **${rf_next:.2f}**")

    except Exception as e:
        st.error(f"Pipeline error: {e}")