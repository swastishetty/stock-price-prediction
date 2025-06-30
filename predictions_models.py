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