# Phase 5: Adf

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
