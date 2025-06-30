# Phase 1: Data collection & Feature Engineering / Data Handling & Preprocessing

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
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title("Actual vs Predicted Stock Prices")
plt.legend()
plt.show

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")