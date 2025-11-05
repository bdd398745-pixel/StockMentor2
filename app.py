# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Author: Biswanath Das

Features:
- Watchlist Dashboard
- Single Stock Detail
- Portfolio View
- Alerts
- Watchlist Editor
- RJ-Style Rule-based Scoring Logic
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
from datetime import datetime

st.set_page_config(page_title="StockMentor", layout="wide")

# -------------------------
# Load / Save Watchlist
# -------------------------
WATCHLIST_FILE = "watchlist.csv"

@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        symbols = [s.replace(".NS", "").strip().upper() for s in symbols if s and str(s).strip()]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading {WATCHLIST_FILE}: {e}")
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        try:
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Helper Functions
# -------------------------
def safe_get(info, key, default=np.nan):
    """Safely get data from yfinance info dict"""
    val = info.get(key)
    if val is None or val == "":
        return default
    return val

@st.cache_data(ttl=3600)
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        hist = ticker.history(period="1y")
        if hist.empty:
            hist = ticker.history(period="6mo")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# RJ Rule-based Scoring Logic
# -------------------------
def calculate_rj_score(info):
    """Score based on Rakesh Jhunjhunwala-style fundamentals"""
    try:
        roe = safe_get(info, "returnOnEquity", np.nan)
        if math.isnan(roe) or roe == 0:
            roe = safe_get(info, "returnOnAssets", 0) * 1.2

        pe = safe_get(info, "trailingPE", np.nan)
        pb = safe_get(info, "priceToBook", np.nan)
        de = safe_get(info, "debtToEquity", np.nan)
        div_yield = safe_get(info, "dividendYield", 0)
        profit_margin = safe_get(info, "profitMargins", np.nan)
        mcap = safe_get(info, "marketCap", np.nan)

        score = 0
        # Strong Return on Equity
        if roe > 0.15: score += 25
        elif roe > 0.10: score += 15
        elif roe > 0.05: score += 5

        # Low PE
        if 0 < pe < 20: score += 20
        elif 20 <= pe <= 30: score += 10

        # Good PB
        if 0 < pb < 3: score += 10

        # Low D/E
        if not math.isnan(de):
            if de < 0.5: score += 15
            elif de < 1: score += 10

        # Dividend Yield
        if div_yield and div_yield > 0.01: score += 5

        # Profit Margin
        if profit_margin and profit_margin > 0.15: score += 15
        elif profit_margin and profit_margin > 0.10: score += 10

        # Large Market Cap
        if mcap and mcap > 1e11: score += 10

        return min(score, 100)
    except Exception:
        return np.nan

# -------------------------
# Streamlit Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔎 Single Stock",
    "💼 Portfolio",
    "🚨 Alerts",
    "🧾 Watchlist Editor"
])

# ------------------------- #
# TAB 1: Dashboard
# ------------------------- #
with tab1:
    st.header("📊 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.warning("No symbols in watchlist. Please add from Watchlist Editor tab.")
    else:
        rows = []
        for sym in watchlist:
            info, hist = fetch_info_and_history(sym)
            if "error" in info:
                continue
            score = calculate_rj_score(info)
            price = safe_get(info, "currentPrice", np.nan)
            pe = safe_get(info, "trailingPE", np.nan)
            roe = safe_get(info, "returnOnEquity", np.nan)
            mcap = safe_get(info, "marketCap", np.nan)
            rows.append([sym, price, pe, roe, mcap, score])

        df = pd.DataFrame(rows, columns=["Symbol", "Price", "PE", "ROE", "MCap", "RJ Score"])
        df = df.sort_values("RJ Score", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.success("Top stocks indicate high RJ Score (strong fundamentals).")

# ------------------------- #
# TAB 2: Single Stock
# ------------------------- #
with tab2:
    st.header("🔎 Single Stock Detail")
    watchlist = load_watchlist()
    sel = st.selectbox("Select Stock", watchlist)
    if sel:
        info, hist = fetch_info_and_history(sel)
        if "error" in info:
            st.error("Error fetching data.")
        else:
            score = calculate_rj_score(info)
            st.subheader(f"{sel} — RJ Score: {score}/100")
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"₹{safe_get(info, 'currentPrice', np.nan):,.2f}")
            col2.metric("PE", f"{safe_get(info, 'trailingPE', np.nan):,.2f}")
            col3.metric("ROE", f"{safe_get(info, 'returnOnEquity', np.nan)*100:.1f}%")
            st.line_chart(hist["Close"])

# ------------------------- #
# TAB 3: Portfolio
# ------------------------- #
with tab3:
    st.header("💼 Portfolio View")
    watchlist = load_watchlist()
    if not watchlist:
        st.warning("No symbols to display.")
    else:
        portfolio_data = []
        for sym in watchlist:
            info, _ = fetch_info_and_history(sym)
            if "error" in info:
                continue
            score = calculate_rj_score(info)
            price = safe_get(info, "currentPrice", np.nan)
            pe = safe_get(info, "trailingPE", np.nan)
            portfolio_data.append([sym, price, pe, score])

        pdf = pd.DataFrame(portfolio_data, columns=["Symbol", "Price", "PE", "RJ Score"])
        st.dataframe(pdf, use_container_width=True)

# ------------------------- #
# TAB 4: Alerts
# ------------------------- #
with tab4:
    st.header("🚨 Alerts")
    st.info("Shows stocks crossing strong RJ score thresholds.")
    watchlist = load_watchlist()
    alert_data = []
    for sym in watchlist:
        info, _ = fetch_info_and_history(sym)
        if "error" in info:
            continue
        score = calculate_rj_score(info)
        if score >= 80:
            alert_data.append([sym, score, "🔥 Strong Buy"])
        elif score <= 40:
            alert_data.append([sym, score, "⚠️ Weak Fundamentals"])

    if alert_data:
        st.table(pd.DataFrame(alert_data, columns=["Symbol", "RJ Score", "Alert"]))
    else:
        st.success("No alert triggered.")

# ------------------------- #
# TAB 5: Watchlist Editor
# ------------------------- #
with tab5:
    st.header("🧾 Watchlist Editor")
    watchlist = load_watchlist()
    st.write("Current Watchlist:", watchlist)
    new_symbol = st.text_input("Add new stock symbol (e.g., RELIANCE):").upper().strip()
    if st.button("Add"):
        if new_symbol and new_symbol not in watchlist:
            watchlist.append(new_symbol)
            ok, msg = save_watchlist(watchlist)
            st.success("Added to watchlist.")
        else:
            st.warning("Already exists or empty input.")
    if st.button("Clear Watchlist"):
        save_watchlist([])
        st.warning("Cleared all symbols.")
