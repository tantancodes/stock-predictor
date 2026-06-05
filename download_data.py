import yfinance as yf

ticker = input("Enter a stock ticker symbol (e.g., AAPL, TSLA, NVDA): ").upper()
days = input("How many days of historical data do you want to pull? (e.g., 30, 60, 90): ")

print(f"\nFetching live data for {ticker} over the last {days} days...")

try:
    data = yf.download(ticker, period=f"{days}d")
    
    if data.empty:
        print("❌ Error: No data found. Make sure the ticker symbol is correct!")
    else:
        data.to_csv("active_stock.csv")
        print(f"✅ Success! Saved live {ticker} data directly into 'active_stock.csv'!")

except Exception as e:
    print(f"❌ An error occurred: {e}")