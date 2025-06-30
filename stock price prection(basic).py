import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# tickers are short abbreviations used to identify publicly traded companies on a stock exchange
tickers = ['AAPL', 'MSFT', 'TSLA']

start_date = '2023-01-01'
end_date = '2024-01-01'
window_size = 5 
output_file = "multi_stock_prediction.csv"

#Data frame to hold all predictions
all_data = pd.DataFrame()

#Loop through each ticker
for ticker in tickers:
    print(f"Procession: {ticker}")

    #download data
    df= yf.download(ticker, start=start_date, end=end_date)
    print(df.head())
    df = df[['Close']]
    df.reset_index(inplace=True)

    #calculate prediction
    df['Prediction'] = df['Close'].rolling(window=window_size).mean().shift(1)

    #keep only last 30 predictions
    df = df[['Date', 'Close', 'Prediction']].tail(30)
    df['Ticker'] = ticker # add column to track which stock

    #add to master dataframe
    all_data = pd.concat([all_data, df], ignore_index=True)

    #plot individual stock
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Close'], label='Actual')
    plt.plot(df['Date'], df['Prediction'], label='Predicted', linestyle = '--')
    plt.title(f"{ticker} - Last 30 Days (Moving Average Prediction)")
    plt.xlabel("Date")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show

#Save all predictions to CSV
all_data.to_csv(output_file, index=False)
print(f"All predictions saved to {output_file}")