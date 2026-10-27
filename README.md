# Quantitative ML Stock Analytics Dashboard

A Streamlit dashboard that pulls daily price history for any ticker, computes technical indicators, and compares two regression models on an out-of-sample split.

**[Open the live app →](https://stock-predictor-tristan-project.streamlit.app/)**

## What it does

1. **Ingest.** Enter a ticker and a window of 60–365 days. Daily OHLCV data comes from Yahoo Finance via `yfinance`.
2. **Engineer features.** Vectorized pandas code computes:
   - 5- and 20-day simple moving averages
   - 14-day RSI
   - MACD (12/26 EMA)
   - 20-day Bollinger Bands (±2σ)
   - Day-over-day volume change
3. **Model.** Two regressors predict the closing price from those features:
   - Support Vector Regression (RBF kernel)
   - Random Forest (100 trees)
4. **Evaluate.** The data is split chronologically: the first 80% of days train, the last 20% test. There is no shuffling, so the test set is always in the future relative to training. The scaler is fit on the training rows only. R² and MAE are reported on the test set.
5. **Visualize.** Interactive Plotly charts show:
   - Price with Bollinger Bands, both models' fitted curves, and a marker where the test period begins
   - RSI with 70/30 overbought and oversold lines
   - A one-day-ahead estimate from each model

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`download_data.py` and `stock.py` are the earlier command-line version: download a ticker to CSV, then fit and plot an SVR on it with matplotlib.

## Limitations

This is a learning project, not a trading tool. Two things limit what the scores mean:

- **The features overlap the target.** The 5-day moving average includes the same day's close the model is predicting. That makes R² look better than a true forecast would.
- **Random Forests can't extrapolate.** A Random Forest can't predict a price outside the range it saw in training, so it flattens out when a stock trends to new highs or lows.

A sounder setup would predict next-day returns from features lagged by one day, and compare against a naive "tomorrow = today" baseline.

## Stack

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · yfinance
