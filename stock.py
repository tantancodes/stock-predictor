import pandas as pd
import numpy as np  
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def load_and_engineer_data(filename):
    df = pd.read_csv(filename, header=[0, 1])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    
    #PHASE 2: 5-Day Simple Moving Average Math!
    df['Sma_5'] = df['Close'].rolling(window=5).mean().bfill()
    
    df['Day'] = np.arange(1, len(df) + 1)
    return df

def run_ml_dashboard(df):
    X = df['Day'].values.reshape(-1, 1)
    y = df['Close'].values
    y_sma = df['Sma_5'].values 

    #PHASE 3: Train-Test Splitting (80% Train, 20% Hidden Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Training machine learning model on historical split...")
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.05)
    svr_rbf.fit(X_train, y_train)

    y_pred = svr_rbf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    
    plt.scatter(X_train, y_train, color='black', label='Training Data (AI Studied)', alpha=0.7)
    plt.scatter(X_test, y_test, color='blue', label='Testing Data (AI Evaluated On)', alpha=0.7)
    
    plt.plot(X, y_sma, color='orange', linestyle='--', label='5-Day Moving Avg (SMA)', linewidth=2)
    
    plt.plot(X, svr_rbf.predict(X), color='red', label='RBF ML Prediction Line', linewidth=2.5)
    
    plt.xlabel('Timeline (Days)')
    plt.ylabel('Asset Value ($)')
    plt.title('Ultimate Quantitative Portfolio Analytics & SVR Evaluation Engine')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show() 

    return r2, mae

stock_data = load_and_engineer_data('active_stock.csv')
r2, mae = run_ml_dashboard(stock_data)

print(f"\n---  COMPLETE MACHINE LEARNING METRICS ENGINE ---")
print(f"Model Fit Integrity (R² Score): {r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")