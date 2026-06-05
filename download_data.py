import yfinance as yf

data = yf.download("AAPL", period="30d")

data.to_csv("aapl.csv")

print("Successfully downloaded real AAPL data into aapl.csv!")