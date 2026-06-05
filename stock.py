import pandas as pd
import numpy as np  
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def load_and_engineer_data(filename):
    df = pd.read_csv(filename, header=[0, 1])
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.columns = df.columns.str.strip().str.capitalize()
    
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    
    df['Sma_5'] = df['Close'].rolling(window=5).mean()
    
    df['Sma_5'] = df['Sma_5'].bfill()
    
    df['Day'] = np.arange(1, len(df) + 1)
    
    return df

def run_ml_dashboard(df):
    X_days = df['Day'].values.reshape(-1, 1)
    y_prices = df['Close'].values
    y_sma = df['Sma_5'].values

    print("🤖 Training advanced SVR machine learning curves...")
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.05)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    
    svr_rbf.fit(X_days, y_prices)
    svr_poly.fit(X_days, y_prices)

    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_days, y_prices, color='black', label='Actual Price (Close)', alpha=0.6)
    plt.plot(X_days, y_sma, color='orange', linestyle='--', label='5-Day Moving Avg (SMA)', linewidth=2)
    
    plt.plot(X_days, svr_rbf.predict(X_days), color='red', label='RBF ML Prediction', linewidth=2)
    plt.plot(X_days, svr_poly.predict(X_days), color='blue', label='Polynomial ML Prediction', linewidth=1)
    
    plt.xlabel('Timeline (Days)')
    plt.ylabel('Asset Value ($)')
    plt.title('Quant Portfolio Analytics & Trend Regression')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show() 

    next_day = np.array([[len(df) + 1]])
    pred_rbf = float(svr_rbf.predict(next_day)[0])
    
    return pred_rbf, len(df) + 1

stock_data = load_and_engineer_data('active_stock.csv')
next_day_prediction, target_day = run_ml_dashboard(stock_data)

print(f"\n--- 🔮 REVENUE & MARKET ANALYSIS DASHBOARD ---")
print(f"Target Timeline Frame: Day {target_day}")
print(f"Engineered RBF Forecast Model Price: ${next_day_prediction:.2f}")