# app.py
"""
📈 StockMentor — Rule-based Long-Term Stock Advisor (India + Global)
Enhanced with Financial Modeling Prep (FMP) API Support
Includes: Dashboard | Single Stock | Portfolio | Alerts | Watchlist | Screener
Author: Biswanath
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import math
from datetime import datetime

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (Enhanced)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India + Global)")
st.caption("Enhanced with Financial Modeling Prep (FMP) API & Nifty 500 Screener 🚀")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0

# Load API Key
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
if FMP_API_KEY:
    st.sidebar.success("✅ FMP key detected")
else:
    st.sidebar.warning("⚠️ No FMP key found — using yfinance fallback")

# -------------------------
# Load Nifty Stock List
# -------------------------
@st.cache_data
def load_nifty_stocks():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        return df[["Symbol", "Company Name", "Industry"]]
    except Exception as e:
        st.error(f"Could not load Nifty500 list: {e}")
        return pd.DataFrame(columns=["Symbol", "Company Name", "Industry"])

# -------------------------
# Watchlist Load/Save
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().tolist()
    except:
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        load_watchlist.clear()
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# FMP Data Fetch
# -------------------------
@st.cache_data(ttl=1200)
def fetch_fmp_data(symbol):
    try:
        base = "https://financialmodelingprep.com/api/v3"
        quote_url = f"{base}/quote/{symbol}?apikey={FMP_API_KEY}"
        profile_url = f"{base}/profile/{symbol}?apikey={FMP_API_KEY}"
        hist_url = f"{base}/historical-price-full/{symbol}?timeseries=365&apikey={FMP_API_KEY}"

        quote = requests.get(quote_url).json()
        profile = requests.get(profile_url).json()
        hist = requests.get(hist_url).json()

        if not quote or not profile:
            return {}, pd.DataFrame()

        q = quote[0] if isinstance(quote, list) else quote
        p = profile[0] if isinstance(profile, list) else profile

        info = {
            "symbol": symbol,
            "currentPrice": q.get("price"),
            "marketCap": q.get("marketCap"),
            "trailingPE": q.get("pe"),
            "priceToBook": q.get("pb"),
            "dividendYield": (q.get("lastDiv") or 0) / q.get("price") if q.get("price") else 0,
            "companyName": p.get("companyName"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "beta": p.get("beta"),
            "returnOnEquity": p.get("returnOnEquityTTM"),
            "debtToEquity": p.get("debtToEquityTTM"),
            "eps": p.get("epsTTM")
        }

        if "historical" in hist:
            df = pd.DataFrame(hist["historical"])
            df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        else:
            df = pd.DataFrame()

        return info, df
    except Exception as e:
        st.warning(f"FMP fetch error for {symbol}: {e}")
        return {}, pd.DataFrame()

# -------------------------
# yfinance Fallback
# -------------------------
@st.cache_data(ttl=900)
def fetch_yfinance_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y").reset_index()
        hist.rename(columns={"Date": "Date", "Close": "Close"}, inplace=True)
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# Helper Functions
# -------------------------
def safe_get(info, key, default=np.nan):
    v = info.get(key, default) if isinstance(info, dict) else default
    return v if v not in (None, "None", "") else default

def estimate_fair_value(info):
    eps = safe_get(info, "eps", safe_get(info, "trailingEps", np.nan))
    pe = safe_get(info, "trailingPE", DEFAULT_PE_TARGET)
    if isinstance(eps, (int, float)) and eps > 0:
        fv = eps * pe
        return round(fv, 2)
    return np.nan

def rule_based_recommendation(info, fair_value, price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    de = safe_get(info, "debtToEquity", np.nan)
    underval = None if not fair_value or not price else round(((fair_value - price) / fair_value) * 100, 2)
    score = 0
    if isinstance(roe, (int, float)):
        if roe >= 0.20: score += 3
        elif roe >= 0.12: score += 2
    if isinstance(de, (int, float)):
        if de <= 0.5: score += 2
        elif de <= 1.5: score += 1
    if isinstance(underval, (int, float)):
        if underval >= 25: score += 3
        elif underval >= 10: score += 2
    rec = "Hold"
    if score >= 7: rec = "Strong Buy"
    elif score >= 5: rec = "Buy"
    return {"score": score, "recommendation": rec, "undervaluation": underval}

def compute_buy_sell(fv):
    return round(fv * 0.9, 2), round(fv * 1.1, 2)

# -------------------------
# Tabs Setup
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard",
    "🔎 Single Stock",
    "💼 Portfolio",
    "📣 Alerts",
    "🧾 Watchlist Editor",
    "🧠 Stock Screener"
])

# -------------------------
# TAB 1: Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    elif st.button("🔍 Analyze Watchlist"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            info, _ = fetch_yfinance_data(f"{sym}.NS")
            if info.get("error"):
                continue
            price = safe_get(info, "currentPrice", safe_get(info, "regularMarketPrice", np.nan))
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)
            rows.append({
                "Symbol": sym,
                "LTP": price,
                "Fair Value": fv,
                "Underv%": rec["undervaluation"],
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec["recommendation"],
                "Score": rec["score"]
            })
            progress.progress(int(((i+1)/len(watchlist))*100))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

# -------------------------
# TAB 2: Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, AAPL):", "AAPL").upper()
    if symbol:
        if FMP_API_KEY:
            info, hist = fetch_fmp_data(symbol)
        else:
            info, hist = fetch_yfinance_data(f"{symbol}.NS")

        if info:
            st.subheader(f"{info.get('companyName', symbol)} ({info.get('symbol', symbol)})")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", info.get("currentPrice", "N/A"))
                st.metric("P/E", info.get("trailingPE", "N/A"))
            with col2:
                st.metric("ROE", info.get("returnOnEquity", "N/A"))
                st.metric("Debt/Equity", info.get("debtToEquity", "N/A"))
            with col3:
                st.metric("Industry", info.get("industry", "N/A"))
                st.metric("Beta", info.get("beta", "N/A"))
            if not hist.empty:
                st.line_chart(hist.set_index("Date")["Close"])
            else:
                st.warning("No price history available.")
        else:
            st.warning("No data found for this stock.")

# -------------------------
# TAB 3: Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    uploaded = st.file_uploader("Upload CSV (symbol, buy_price, quantity)", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf.columns = [c.lower() for c in pf.columns]
            rows = []
            for _, r in pf.iterrows():
                sym = str(r["symbol"]).strip().upper()
                buy = float(r["buy_price"])
                qty = float(r["quantity"])
                info, _ = fetch_yfinance_data(f"{sym}.NS")
                ltp = safe_get(info, "currentPrice", np.nan)
                invested = round(buy * qty, 2)
                current_value = round(ltp * qty, 2) if not np.isnan(ltp) else np.nan
                pl = current_value - invested if not np.isnan(current_value) else np.nan
                pl_pct = round((pl / invested) * 100, 2) if invested > 0 and not np.isnan(pl) else np.nan
                rows.append({"Symbol": sym, "Buy": buy, "LTP": ltp, "Qty": qty, "Invested": invested, "Current Value": current_value, "P/L": pl, "P/L%": pl_pct})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

# -------------------------
# TAB 4: Alerts
# -------------------------
with tab4:
    st.header("📣 Manual Alerts (Email setup placeholder)")
    st.info("Feature placeholder: you can extend this tab with SMTP or Push alerts later.")

# -------------------------
# TAB 5: Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    current = load_watchlist()
    new_txt = st.text_area("Edit your watchlist (one symbol per line):", "\n".join(current), height=300)
    if st.button("💾 Save Watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("✅ Watchlist saved successfully.")
        else:
            st.error("❌ Failed to save: " + msg)

# -------------------------
# TAB 6: Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty 500 Quick Scan")
    use_fmp = st.toggle("Use FMP API (recommended)", value=bool(FMP_API_KEY))
    nifty = load_nifty_stocks()
    if st.button("Run Screener"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(nifty["Symbol"].tolist()[:50]):  # limit to 50 for demo
            info, _ = fetch_fmp_data(f"{sym}.NS") if use_fmp else fetch_yfinance_data(f"{sym}.NS")
            price = safe_get(info, "currentPrice", np.nan)
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({"Symbol": sym, "LTP": price, "FairValue": fv, "Undervaluation%": rec["undervaluation"], "Score": rec["score"], "Recommendation": rec["recommendation"]})
            progress.progress(int(((i + 1) / 50) * 100))
        df = pd.DataFrame(rows)
        st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)

st.caption("Made by Biswanath 🔍 | Rule-based, API-optional, Fully Offline-capable")
