# Phase 4: GUI or Web App

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Stock Price Predictor")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

data = yf.download(ticker, start= "2020-01-01")
 
st.write(data.tail())


#plot
st.subheader("Closing Price Chart")
st.line_chart(data['Close'])

#python -m streamlit run app.py

# FOR COLOR
st.markdown("""
    <style>
    .stApp {
            background-color: #e6e6fa; /* Lavender*/
    }
    </style>
    """, unsafe_allow_html=True)