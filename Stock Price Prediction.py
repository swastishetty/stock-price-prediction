#Phase 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = yf.download("AAPL", start= "2015-01-01", end="2024-12-31")
print(df.head())

df['Next Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X= df[['Close']]
y= df ['Next Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.pilot(y_test.values, label='Actual')
plt.pilot(predictions, label='Predicted')
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")


# Phase 1: Data collection & Feature Engineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = yf.download("AAPL", start= "2015-01-01", end="2024-12-31")

df= df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['Target'] = df['Close'].shift(-5)
df.dropna(inplace=True)

X= df[['Close']]
y= df ['Next Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.pilot(y_test.values, label='Actual')
plt.pilot(predictions, label='Predicted')
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")


# Phase 2: Machine Learning Models (Random Forecast, XGBoost)
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = yf.download("AAPL", start= "2015-01-01", end="2024-12-31")

df= df[['Open', 'High', 'Low', 'Close', 'Volume']]

df['Target'] = df['Close'].shift(-5)
df.dropna(inplace=True)

X= df[['Open', 'High', 'Low', 'Close', 'Volume']]
y= df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test=0.2, shuffle=False)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Phase 3: Time Series Forecasting (ARIMA & Prophet)
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA


df = yf.download("AAPL", start= "2015-01-01", end="2024-12-31")

series = df['Close']

model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forevsat(steps=5)
print(forecast)


# Phase 4 (Option2): Tinker GUI

import tkinter as tk
from tkinter import messagebox
import yfinance as yf

def fetch_price():
    ticker = entry.get()
    data = yf.download(ticker, period="1d")
    messagebox.showinfo("Result", f"Current Price: {data['Close'].iloc[-1]:.2f}")

app = tk.Tk()
app.title("Stock Price Checker")

tk.Label(app, text="Enter Stock Ticker").pack()
entry = tk.Entry(app)
entry.pack()

tk.Button(app, text= "Check Price", command=fetch_price).pack()
app.mainloop
