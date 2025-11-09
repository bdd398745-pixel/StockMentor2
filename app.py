import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# Title and Intro
# -------------------------
st.set_page_config(page_title="StockMentor - Long Term Stock Analyst (India)", layout="wide")
st.title("📊 StockMentor - Rule-based Long-Term Stock Analyst (India)")
st.caption("Smart stock screening, portfolio insights, and RJ-style scoring using yFinance data")

# -------------------------
# Load Watchlist
# -------------------------
def load_watchlist():
    try:
        df = pd.read_csv('watchlist.csv')
        return df['Symbol'].tolist()
    except Exception as e:
        st.warning(f"⚠️ Could not load watchlist.csv: {e}")
        return []

# -------------------------
# Fetch Data
# -------------------------
def fetch_info_and_history(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5y")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# Calculate Metrics
# -------------------------
def calculate_metrics(info):
    try:
        roe = info.get('returnOnEquity', np.nan) * 100 if info.get('returnOnEquity') else np.nan
        de = info.get('debtToEquity', np.nan)
        div_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        promoter = info.get('heldPercentInsiders', np.nan) * 100 if info.get('heldPercentInsiders') else np.nan
        return roe, de, div_yield, promoter
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

# -------------------------
# CAGR Calculation
# -------------------------
def calc_cagr(hist):
    try:
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        years = (hist.index[-1] - hist.index[0]).days / 365
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return np.nan

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Single Stock", "Portfolio", "Alerts", "Watchlist Editor", "RJ Score"])

# -------------------------
# Dashboard Tab
# -------------------------
with tab1:
    st.header("📈 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.warning("Add stocks to watchlist.csv to view data.")
    else:
        data = []
        for symbol in watchlist:
            info, hist = fetch_info_and_history(symbol)
            roe, de, div_yield, promoter = calculate_metrics(info)
            cagr = calc_cagr(hist)
            data.append({
                'Symbol': symbol,
                'ROE%': roe,
                'D/E': de,
                'Div Yield%': div_yield,
                'Promoter%': promoter,
                'CAGR%': cagr
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x='Symbol', y='CAGR%', color='CAGR%', title='CAGR Performance', text='CAGR%')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Single Stock Tab
# -------------------------
with tab2:
    st.header("🔍 Single Stock Deep Analysis")
    ticker = st.text_input("Enter Stock Symbol (e.g., TCS.NS, HDFCBANK.NS, INFY.NS):")
    if ticker:
        info, hist = fetch_info_and_history(ticker)
        if 'error' in info:
            st.error(info['error'])
        else:
            roe, de, div_yield, promoter = calculate_metrics(info)
            cagr = calc_cagr(hist)
            st.metric("ROE%", f"{roe:.2f}%" if not np.isnan(roe) else "N/A")
            st.metric("D/E", f"{de:.2f}" if not np.isnan(de) else "N/A")
            st.metric("Div Yield%", f"{div_yield:.2f}%")
            st.metric("Promoter%", f"{promoter:.2f}%" if not np.isnan(promoter) else "N/A")
            st.metric("5Y CAGR%", f"{cagr:.2f}%" if not np.isnan(cagr) else "N/A")
            if not hist.empty:
                st.line_chart(hist['Close'], use_container_width=True)

# -------------------------
# Portfolio Tab
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.info("Upload your portfolio (CSV with Symbol, Quantity, BuyPrice)")
    uploaded = st.file_uploader("Upload Portfolio CSV", type=["csv"])
    if uploaded:
        pf = pd.read_csv(uploaded)
        for idx, row in pf.iterrows():
            stock = yf.Ticker(row['Symbol'])
            price = stock.history(period='1d')['Close'].iloc[-1]
            pf.loc[idx, 'LTP'] = price
            pf.loc[idx, 'Value'] = row['Quantity'] * price
            pf.loc[idx, 'Gain/Loss%'] = ((price - row['BuyPrice']) / row['BuyPrice']) * 100
        st.dataframe(pf, use_container_width=True)
        total_val = pf['Value'].sum()
        gain = pf['Gain/Loss%'].mean()
        st.metric("Total Portfolio Value", f"₹{total_val:,.0f}")
        st.metric("Avg Gain/Loss%", f"{gain:.2f}%")

# -------------------------
# Alerts Tab
# -------------------------
with tab4:
    st.header("🔔 Alerts & Signals")
    st.info("System flags undervalued or high-performing stocks automatically.")
    if 'df' in locals():
        alert_df = df[(df['CAGR%'] > 20) & (df['ROE%'] > 15) & (df['D/E'] < 1)]
        if alert_df.empty:
            st.success("No high-alert stocks currently.")
        else:
            st.warning("Potential outperformers:")
            st.dataframe(alert_df)

# -------------------------
# Watchlist Editor Tab
# -------------------------
with tab5:
    st.header("📝 Watchlist Editor")
    st.info("Modify your watchlist.csv here directly.")
    watchlist_text = st.text_area("Edit symbols (one per line):", value='\n'.join(watchlist))
    if st.button("Save Watchlist"):
        with open('watchlist.csv', 'w') as f:
            f.write(watchlist_text)
        st.success("Watchlist updated successfully!")

# -------------------------
# RJ Score Tab
# -------------------------
with tab6:
    st.header("⭐ RJ Style Scoring")
    if 'df' in locals():
        df['RJ Score'] = (
            (df['ROE%'] / 20) * 0.25 +
            ((30 - df['D/E']) / 30) * 0.25 +
            (df['CAGR%'] / 20) * 0.25 +
            (df['Promoter%'] / 100) * 0.25
        ) * 100
        st.dataframe(df[['Symbol', 'RJ Score']], use_container_width=True)
        st.bar_chart(df.set_index('Symbol')['RJ Score'], use_container_width=True)
    else:
        st.info("Load your watchlist first in Dashboard tab.")

st.markdown("""---\n💡 **Pro Tip:** Keep your watchlist updated weekly for best accuracy. | Data Source: Yahoo Finance\n---""")
