# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Author: Biswanath Das
Enhanced with RJ-style score system
Data: yfinance (free)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import math
from datetime import datetime

# -------------------------
# Helpers
# -------------------------
def safe_get(d, key, default=np.nan):
    return d[key] if key in d and d[key] is not None else default

@st.cache_data
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        hist = ticker.history(period="1y")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

def load_watchlist():
    if os.path.exists("watchlist.csv"):
        df = pd.read_csv("watchlist.csv")
        return df["symbol"].tolist()
    else:
        return []

def save_watchlist(symbols):
    pd.DataFrame({"symbol": symbols}).to_csv("watchlist.csv", index=False)

# -------------------------
# RJ Score Logic
# -------------------------
def rj_score(info):
    score = 0
    details = {}

    pe = safe_get(info, "trailingPE", np.nan)
    pb = safe_get(info, "priceToBook", np.nan)
    roe = safe_get(info, "returnOnEquity", np.nan)
    debt_eq = safe_get(info, "debtToEquity", np.nan)

    # --- ROE Fallbacks ---
    if not isinstance(roe, (int, float)) or math.isnan(roe):
        roe = safe_get(info, "returnOnEquityTTM", np.nan)
    if not isinstance(roe, (int, float)) or math.isnan(roe):
        roe = safe_get(info, "returnOnAssets", np.nan) * 2.5
    if isinstance(roe, (int, float)):
        roe_display = round(roe * 100, 2) if abs(roe) <= 3 else round(roe, 2)
    else:
        roe_display = 0

    # --- D/E Fallbacks ---
    if not isinstance(debt_eq, (int, float)) or math.isnan(debt_eq):
        total_debt = safe_get(info, "totalDebt", np.nan)
        total_assets = safe_get(info, "totalAssets", np.nan)
        if isinstance(total_debt, (int, float)) and isinstance(total_assets, (int, float)) and total_assets > 0:
            debt_eq = total_debt / total_assets
    if not isinstance(debt_eq, (int, float)) or math.isnan(debt_eq):
        debt_eq = 1.0

    # --- Rule-based scoring ---
    if isinstance(pe, (int, float)):
        if pe < 15: score += 2
        elif pe < 25: score += 1
    if isinstance(pb, (int, float)):
        if pb < 3: score += 1
        elif pb < 5: score += 0.5
    if roe_display > 15: score += 2
    elif roe_display > 10: score += 1
    if debt_eq < 1: score += 1
    if safe_get(info, "dividendYield", 0) > 0.01: score += 1

    details = {
        "PE": pe, "PB": pb, "ROE%": roe_display, "D/E": debt_eq,
        "DivYield%": round(safe_get(info, "dividendYield", 0) * 100, 2),
        "Score": score
    }
    return details

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="StockMentor", layout="wide")
st.title("📈 StockMentor — RJ Style Long-Term Stock Advisor")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🔎 Single Stock", "💼 Portfolio", "⚠️ Alerts", "📝 Watchlist Editor"
])

# ------------------------- #
# TAB 1: Dashboard
# ------------------------- #
with tab1:
    st.header("📊 RJ Scoring Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.warning("No watchlist found. Please add stocks in Watchlist Editor tab.")
    else:
        results = []
        for sym in watchlist:
            info, _ = fetch_info_and_history(sym)
            if "error" in info:
                continue
            data = rj_score(info)
            data["Symbol"] = sym
            results.append(data)

        if results:
            df = pd.DataFrame(results)
            df = df[["Symbol", "Score", "PE", "PB", "ROE%", "D/E", "DivYield%"]].sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No valid data fetched.")

# ------------------------- #
# TAB 2: Single Stock
# ------------------------- #
with tab2:
    st.header("🔎 Single Stock Detail")
    watchlist = load_watchlist()
    sel = st.selectbox("Select stock", watchlist) if watchlist else st.text_input("Enter symbol (e.g., RELIANCE)")
    if sel:
        info, hist = fetch_info_and_history(sel)
        if "error" in info:
            st.error("Data fetch error.")
        else:
            st.subheader(f"{sel} — {info.get('longName', '')}")
            st.write(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
            st.metric("Market Cap", f"{round(info.get('marketCap', 0)/1e9,2)} Cr")
            details = rj_score(info)
            st.write(details)
            st.line_chart(hist["Close"])

# ------------------------- #
# TAB 3: Portfolio
# ------------------------- #
with tab3:
    st.header("💼 Portfolio Summary")
    if os.path.exists("portfolio.csv"):
        df = pd.read_csv("portfolio.csv")
        st.dataframe(df)
    else:
        st.info("No portfolio data found. Create a portfolio.csv with symbol and quantity.")

# ------------------------- #
# TAB 4: Alerts
# ------------------------- #
with tab4:
    st.header("⚠️ Stock Alerts (Rule-based)")
    watchlist = load_watchlist()
    alerts = []
    for sym in watchlist:
        info, _ = fetch_info_and_history(sym)
        data = rj_score(info)
        if data["Score"] >= 6:
            alerts.append(f"🚀 {sym} looks strong (Score {data['Score']})")
        elif data["Score"] <= 2:
            alerts.append(f"⚠️ {sym} may be weak (Score {data['Score']})")
    if alerts:
        for a in alerts:
            st.write(a)
    else:
        st.info("No alerts triggered.")

# ------------------------- #
# TAB 5: Watchlist Editor
# ------------------------- #
with tab5:
    st.header("📝 Manage Your Watchlist")
    current = load_watchlist()
    st.write("Current watchlist:", current)
    new_symbols = st.text_area("Enter symbols separated by commas").upper()
    if st.button("Save Watchlist"):
        if new_symbols:
            symbols = [s.strip() for s in new_symbols.split(",") if s.strip()]
            save_watchlist(symbols)
            st.success("✅ Watchlist updated! Please refresh Dashboard tab.")
