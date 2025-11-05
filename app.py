# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- Uses yfinance for free data
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- Rule-based scoring, ranking & recommendation
Author: Adapted for Biswanath
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import math
from datetime import datetime

st.set_page_config(page_title="📈 StockMentor", layout="wide")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    return default if v in (None, "None", "", np.nan) else v


def load_watchlist():
    if os.path.exists("watchlist.csv"):
        df = pd.read_csv("watchlist.csv")
        return df["symbol"].dropna().tolist()
    return []


def save_watchlist(symbols):
    pd.DataFrame({"symbol": symbols}).to_csv("watchlist.csv", index=False)


@st.cache_data(ttl=3600)
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info
        hist = ticker.history(period="1y")

        # ---------- FIX: Auto-compute ROE and D/E if missing ----------
        if not info.get("returnOnEquity") and info.get("netIncomeToCommon") and info.get("totalStockholderEquity"):
            try:
                roe = (info["netIncomeToCommon"] / info["totalStockholderEquity"]) * 100
                info["returnOnEquity"] = round(roe, 2)
            except:
                info["returnOnEquity"] = np.nan

        if not info.get("debtToEquity") and info.get("totalDebt") and info.get("totalStockholderEquity"):
            try:
                de = (info["totalDebt"] / info["totalStockholderEquity"])
                info["debtToEquity"] = round(de, 2)
            except:
                info["debtToEquity"] = np.nan
        # --------------------------------------------------------------

        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()


def compute_buy_sell(fv):
    buy = round(fv * 0.75, 2)
    sell = round(fv * 1.25, 2)
    return buy, sell


def estimate_fair_value(info):
    pe = safe_get(info, "trailingPE", np.nan)
    eps = safe_get(info, "trailingEps", np.nan)
    growth = safe_get(info, "earningsQuarterlyGrowth", 0.1)
    if not pe or not eps:
        return np.nan, "insufficient data"
    fair_value = eps * pe * (1 + growth)
    return round(fair_value, 2), "PE-based"


def rule_based_recommendation(info, fair_value, ltp):
    roe = safe_get(info, "returnOnEquity", 0)
    de = safe_get(info, "debtToEquity", 0)
    peg = safe_get(info, "pegRatio", 1)
    div = safe_get(info, "dividendYield", 0)
    score = 0

    if roe > 15: score += 2
    if roe > 25: score += 3
    if de < 1: score += 2
    if de < 0.5: score += 3
    if peg < 1: score += 2
    if div > 0.02: score += 1
    if ltp < fair_value: score += 2
    if ltp < 0.8 * fair_value: score += 3

    if score >= 10:
        rec = "🚀 Strong Buy"
    elif score >= 7:
        rec = "✅ Buy"
    elif score >= 5:
        rec = "☯️ Hold"
    else:
        rec = "⚠️ Avoid"

    return rec


# --------------------------------------------------
# Tabs Setup
# --------------------------------------------------
st.title("📊 StockMentor — Rule-based Long-term Advisor (India)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏠 Dashboard", "🔎 Single Stock", "💼 Portfolio", "🚨 Alerts", "📝 Watchlist Editor", "💎 RJ Score"]
)

# -------------------------
# TAB 1: Dashboard
# -------------------------
with tab1:
    st.header("🏠 Dashboard Overview")

    watchlist = load_watchlist()
    if not watchlist:
        st.info("No stocks in watchlist. Add from the 'Watchlist Editor' tab.")
    else:
        data = []
        for sym in watchlist:
            info, hist = fetch_info_and_history(sym)
            if "error" in info:
                continue
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, method = estimate_fair_value(info)
            roe = safe_get(info, "returnOnEquity", np.nan)
            de = safe_get(info, "debtToEquity", np.nan)
            rec = rule_based_recommendation(info, fv, ltp)
            data.append([sym, ltp, fv, roe, de, rec])

        df = pd.DataFrame(data, columns=["Symbol", "LTP", "Fair Value", "ROE%", "D/E", "Recommendation"])
        st.dataframe(df, use_container_width=True)

# -------------------------
# TAB 2: Single Stock Detail
# -------------------------
with tab2:
    st.header("🔎 Single Stock Detail")

    watchlist = load_watchlist()
    sel = st.selectbox("Select stock", watchlist) if watchlist else st.text_input("Enter symbol (e.g., RELIANCE)")

    if sel:
        info, hist = fetch_info_and_history(sel)
        if info.get("error"):
            st.error("Data fetch error: " + info.get("error"))
        else:
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)

            st.subheader(f"📈 {sel} Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LTP", ltp)
            col2.metric("Fair Value", fv)
            col3.metric("ROE%", safe_get(info, "returnOnEquity"))
            col4.metric("D/E", safe_get(info, "debtToEquity"))
            st.write("**Method:**", method)
            st.write("**Recommendation:**", rec)
            st.write(f"💰 Buy below ₹{buy} | ⚡ Sell above ₹{sell}")

            if not hist.empty:
                st.line_chart(hist["Close"])

# -------------------------
# TAB 3: Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")

    uploaded = st.file_uploader("Upload your portfolio CSV (columns: Symbol, Qty, BuyPrice)", type=["csv"])
    if uploaded:
        pf = pd.read_csv(uploaded)
        rows = []
        for _, r in pf.iterrows():
            sym = r["Symbol"]
            qty = r["Qty"]
            bp = r["BuyPrice"]
            info, _ = fetch_info_and_history(sym)
            ltp = safe_get(info, "currentPrice", np.nan)
            gain = round((ltp - bp) * qty, 2)
            ret = round(((ltp / bp) - 1) * 100, 2)
            rows.append([sym, qty, bp, ltp, gain, ret])
        df = pd.DataFrame(rows, columns=["Symbol", "Qty", "BuyPrice", "LTP", "Gain ₹", "Return %"])
        st.dataframe(df, use_container_width=True)
        st.success(f"Total Portfolio Gain: ₹{df['Gain ₹'].sum():,.0f}")

# -------------------------
# TAB 4: Alerts
# -------------------------
with tab4:
    st.header("🚨 Alerts")

    watchlist = load_watchlist()
    alerts = []
    for sym in watchlist:
        info, _ = fetch_info_and_history(sym)
        if "error" in info:
            continue
        ltp = safe_get(info, "currentPrice", np.nan)
        fv, _ = estimate_fair_value(info)
        if ltp < 0.8 * fv:
            alerts.append((sym, ltp, fv, "Buy Alert"))
        elif ltp > 1.25 * fv:
            alerts.append((sym, ltp, fv, "Overvalued"))
    if alerts:
        st.dataframe(pd.DataFrame(alerts, columns=["Symbol", "LTP", "Fair Value", "Alert"]))
    else:
        st.info("No alerts triggered currently.")

# -------------------------
# TAB 5: Watchlist Editor
# -------------------------
with tab5:
    st.header("📝 Manage Watchlist")

    watchlist = load_watchlist()
    symbols = st.text_area("Watchlist symbols (one per line):", "\n".join(watchlist))
    if st.button("💾 Save Watchlist"):
        save_watchlist([s.strip().upper() for s in symbols.splitlines() if s.strip()])
        st.success("✅ Watchlist saved!")

# -------------------------
# TAB 6: RJ Score
# -------------------------
with tab6:
    st.header("💎 RJ Score — Buffett-style Quality Ranking")

    watchlist = load_watchlist()
    if not watchlist:
        st.info("Add stocks to watchlist first.")
    else:
        rows = []
        for sym in watchlist:
            info, _ = fetch_info_and_history(sym)
            if "error" in info:
                continue
            roe = safe_get(info, "returnOnEquity", 0)
            de = safe_get(info, "debtToEquity", 0)
            div = safe_get(info, "dividendYield", 0)
            score = 0
            if roe > 20: score += 3
            elif roe > 15: score += 2
            if de < 1: score += 2
            if de < 0.5: score += 3
            if div > 0.02: score += 1
            rows.append([sym, roe, de, div, score])
        df = pd.DataFrame(rows, columns=["Symbol", "ROE%", "D/E", "DivYield", "RJ Score"])
        st.dataframe(df.sort_values("RJ Score", ascending=False), use_container_width=True)
