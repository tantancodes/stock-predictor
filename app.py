import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#Page Configuration
st.set_page_config(page_title="Quant ML Dashboard", layout="centered")
st.title("📈 Stock Analytics & ML Prediction Dashboard")
st.write("An interactive app that fetches historical stock data, engineers indicators, and trains an SVR model.")

#Sidebar UI Controls
st.sidebar.header("User Configurations")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="NVDA").upper()
days = st.sidebar.slider("Days of Historical Data", min_value=15, max_value=120, value=60)

if st.sidebar.button("Run Predictive Model"):
    st.write(f"### Fetching data for **{ticker}** over the last {days} days...")
    
    # 1. Download live data based on UI parameters
    try:
        raw_data = yf.download(ticker, period=f"{days}d")
        
        if raw_data.empty:
            st.error("❌ Error: No data found. Please check your ticker symbol.")
        else:
            # 2. Pandas Processing Pipeline
            df = raw_data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.strip().str.capitalize()
            df = df.dropna(subset=['Close']).reset_index(drop=True)
            
            # Feature Engineering (5-Day SMA)
            df['Sma_5'] = df['Close'].rolling(window=5).mean().bfill()
            df['Day'] = np.arange(1, len(df) + 1)
            
            # 3. Machine Learning Setup (Train-Test Split)
            X = df['Day'].values.reshape(-1, 1)
            y = df['Close'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train the RBF Model
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.05)
            svr_rbf.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = svr_rbf.predict(X_test)
            df['ML_Prediction'] = svr_rbf.predict(X)
            
            # Calculate Evaluation Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # 4. Display Statistical Metrics UI Widgets
            col1, col2 = st.columns(2)
            col1.metric(label="Model Fit (R² Score)", value=f"{r2:.4f}")
            col2.metric(label="Mean Absolute Error (MAE)", value=f"${mae:.2f}")
            
            # 5. Render Interactive Web Charts Using Streamlit's Native Graphs
            # We will construct a clean visualization table for the browser
            chart_data = pd.DataFrame({
                'Actual Price': df['Close'],
                '5-Day SMA': df['Sma_5'],
                'ML Prediction Curve': df['ML_Prediction']
            }, index=df['Day'])
            
            st.write("### Interactive Trend Analysis")
            st.line_chart(chart_data)
            
            # Predict the absolute next market day
            next_day = np.array([[len(df) + 1]])
            pred_next = float(svr_rbf.predict(next_day)[0])
            st.success(f"🔮 **Engineered Forecast:** Predicted price for Day {len(df) + 1} is **${pred_next:.2f}**")
            
    except Exception as e:
        st.error(f"An unexpected pipeline error occurred: {e}")