import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Mentor", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def get_stock_data(symbol, period="1y"):
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            return pd.DataFrame()
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_recommendation(df):
    if df.empty:
        return None
    try:
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["RSI"] = compute_rsi(df["Close"], 14)
        latest = df.iloc[-1]
        undervaluation = (latest["SMA_50"] - latest["Close"]) / latest["SMA_50"] * 100
        sentiment = "Bullish" if latest["SMA_20"] > latest["SMA_50"] else "Bearish"
        rec = {
            "signal": "BUY" if sentiment == "Bullish" and latest["RSI"] < 70 else "SELL" if latest["RSI"] > 70 else "HOLD",
            "undervaluation": round(undervaluation, 2),
            "RSI": round(latest["RSI"], 2),
            "sentiment": sentiment,
        }
        return rec
    except Exception as e:
        st.warning(f"Error in recommendation calculation: {e}")
        return None

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["🏠 Dashboard", "📈 Stocks", "💼 Portfolio", "🧠 Mentor AI", "⚙️ Settings"])

# -------------------------
# TAB: Dashboard
# -------------------------
with tabs[0]:
    st.title("🏠 Stock Mentor Dashboard")
    st.write("Welcome to **Stock Mentor** — your personal AI to guide you in the stock market.")
    st.metric("Market Sentiment", "Bullish", "+1.2%")
    st.metric("Top Performer", "RELIANCE.NS")
    st.metric("Most Active", "TCS.NS")

# -------------------------
# TAB: Stocks
# -------------------------
with tabs[1]:
    st.title("📈 Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS, RELIANCE.NS)", "RELIANCE.NS")
    period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    if st.button("Analyze"):
        hist = get_stock_data(symbol, period)
        if hist.empty:
            st.warning("No data found for the selected stock.")
        else:
            rec = calculate_recommendation(hist)
            st.subheader(f"📊 {symbol} Analysis Summary")
            if rec:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Signal", rec["signal"])
                col2.metric("RSI", rec["RSI"])
                col3.metric("Undervaluation %", rec["undervaluation"])
                col4.metric("Sentiment", rec["sentiment"])

            st.write("### Price Trend")
            # ✅ Fix: Ensure 'Date' exists before chart
            if "Date" in hist.columns:
                st.line_chart(hist.set_index("Date")["Close"], height=400)
            else:
                st.warning("No 'Date' column found in stock data.")

            st.write("### Raw Data")
            st.dataframe(hist.tail(10))

# -------------------------
# TAB: Portfolio
# -------------------------
with tabs[2]:
    st.title("💼 Portfolio Tracker")
    st.write("Track your holdings and analyze your profit or loss.")
    uploaded_file = st.file_uploader("Upload your portfolio (CSV format with Symbol, Qty, BuyPrice)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Portfolio:")
        st.dataframe(df)

        results = []
        for _, row in df.iterrows():
            price = yf.Ticker(row["Symbol"]).history(period="1d")["Close"].iloc[-1]
            profit = (price - row["BuyPrice"]) * row["Qty"]
            results.append({"Symbol": row["Symbol"], "Current Price": price, "Profit": profit})
        result_df = pd.DataFrame(results)
        st.write("### Portfolio Summary")
        st.dataframe(result_df)
        st.metric("Total Profit", f"₹{result_df['Profit'].sum():,.2f}")

# -------------------------
# TAB: Mentor AI
# -------------------------
with tabs[3]:
    st.title("🧠 Mentor AI")
    st.write("Ask your mentor anything about stocks, investment, or strategy.")
    query = st.text_area("Ask your question:")
    if st.button("Get Advice"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            # Mock AI response
            st.success("Based on recent market trends, consider diversifying your portfolio with strong mid-cap stocks.")

# -------------------------
# TAB: Settings
# -------------------------
with tabs[4]:
    st.title("⚙️ Settings")
    st.write("Customize your preferences below.")
    st.checkbox("Enable dark mode")
    st.selectbox("Default stock period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    st.text_input("Set default stock symbol", "RELIANCE.NS")
    st.button("Save Settings")

# End of file
