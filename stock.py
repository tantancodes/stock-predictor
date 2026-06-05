import csv 
import numpy as np  
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def get_data(filename):
    dates = []
    prices = []
    
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        day_counter = 1 
        
        for row in csvFileReader:
            if len(row) < 5:
                continue
            try:
                current_price = float(row[4])
                prices.append(current_price)
                dates.append(day_counter)
                day_counter += 1
            except ValueError:
                continue
                
    return dates, prices

def predict_prices(dates, prices, target_day):
    # Reshape dates dataset into an n x 1 matrix
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    print("Training machine learning models... please wait...")
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Actual Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model (Elastic)')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model (Trend)')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model (Curve)')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title('Advanced Stock Predictor Dashboard')
    plt.legend()
    plt.show() 

    res_rbf = float(svr_rbf.predict(np.array([[target_day]]))[0])
    res_lin = float(svr_lin.predict(np.array([[target_day]]))[0])
    res_poly = float(svr_poly.predict(np.array([[target_day]]))[0])

    return res_rbf, res_lin, res_poly

dates, prices = get_data('active_stock.csv')

total_days = len(dates)
target_day = total_days + 1

predicted_price = predict_prices(dates, prices, target_day)

print(f"\n--- 🔮 PREDICTIONS FOR DAY {target_day} ---")
print(f"RBF Prediction: ${predicted_price[0]:.2f}")
print(f"Linear Prediction: ${predicted_price[1]:.2f}")
print(f"Polynomial Prediction: ${predicted_price[2]:.2f}")