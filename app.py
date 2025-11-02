# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary: Massive API (NSE/BSE)
Fallback: yfinance (.NS)
Includes: Watchlist Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, Screener
Author: Adapted for Biswanath Das
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import math
import json
from datetime import datetime

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (Massive)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: Massive API (NSE/BSE) — Fallback: yfinance (.NS)")

# -------------------------
# Constants & Keys
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 15.0
MASSIVE_API_KEY = st.secrets.get("MASSIVE_API_KEY", os.getenv("MASSIVE_API_KEY", ""))

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_nifty_stocks():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        return df[["Symbol", "Company Name", "Industry"]]
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Company Name", "Industry"])

@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().str.upper().tolist()
    except Exception:
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        load_watchlist.clear()
        return True, "Saved successfully"
    except Exception as e:
        return False, str(e)

def safe_get(info, key, default=np.nan):
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    return v if v not in (None, "", "None") else default

# -------------------------
# Core: Fair Value & Rules
# -------------------------
def estimate_fair_value(info):
    try:
        price = float(safe_get(info, "price", safe_get(info, "currentPrice", np.nan)))
        pe = safe_get(info, "pe", np.nan)
        eps = safe_get(info, "eps", np.nan)
        pb = safe_get(info, "pb", np.nan)

        if not pd.isna(eps) and eps > 0:
            fv = eps * DEFAULT_PE_TARGET
            method = "EPS-based"
        elif not pd.isna(pe) and pe > 0:
            fv = (price / pe) * DEFAULT_PE_TARGET
            method = "P/E reversion"
        elif not pd.isna(pb) and pb > 0:
            fv = pb * price
            method = "P/B heuristic"
        else:
            fv = price
            method = "fallback"

        return round(fv, 2), method
    except:
        return np.nan, "error"

def compute_buy_sell(fv, mos=0.25):
    try:
        fv = float(fv)
        buy = round(fv * (1 - mos), 2)
        sell = round(fv * (1 + mos / 1.5), 2)
        return buy, sell
    except:
        return None, None

def rule_based_recommendation(info, fair_value, price):
    roe = safe_get(info, "roe", np.nan)
    de = safe_get(info, "de", np.nan)
    underval = None
    if fair_value and price:
        underval = round(((fair_value - price) / fair_value) * 100, 2)

    score = 0
    if roe and roe >= 20:
        score += 3
    elif roe and roe >= 12:
        score += 2
    if de and de <= 0.5:
        score += 2
    elif de and de <= 1.5:
        score += 1
    if underval and underval >= 25:
        score += 3
    elif underval and underval >= 10:
        score += 2

    if score >= 7:
        rec = "Strong Buy"
    elif score >= 5:
        rec = "Buy"
    elif score <= 2:
        rec = "Avoid"
    else:
        rec = "Hold"

    return {"score": score, "recommendation": rec, "undervaluation": underval}

# -------------------------
# Massive API Fetch
# -------------------------
@st.cache_data(ttl=900)
def fetch_massive_data(symbol: str):
    """Try Massive API for Indian market (NSE/BSE)"""
    if not MASSIVE_API_KEY:
        return {}, pd.DataFrame()

    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    base = "https://api.massive.com/v1/stock"

    try:
        # 1️⃣ Quote / fundamentals
        url_info = f"{base}/quote/{symbol}?exchange=NSE"
        r = requests.get(url_info, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}, pd.DataFrame()
        data = r.json()
        if not isinstance(data, dict) or not data.get("symbol"):
            return {}, pd.DataFrame()

        info = {
            "symbol": data.get("symbol"),
            "companyName": data.get("name"),
            "exchange": data.get("exchange"),
            "price": data.get("price"),
            "pe": data.get("peRatio"),
            "pb": data.get("pbRatio"),
            "eps": data.get("eps"),
            "roe": data.get("returnOnEquity"),
            "de": data.get("debtToEquity"),
        }

        # 2️⃣ Historical data (past 1 year)
        url_hist = f"{base}/historical/{symbol}?interval=1d&range=1y"
        rh = requests.get(url_hist, headers=headers, timeout=15)
        hist = rh.json()
        if isinstance(hist, dict) and "data" in hist:
            hist_df = pd.DataFrame(hist["data"])
            hist_df["Date"] = pd.to_datetime(hist_df["date"])
            hist_df["Close"] = hist_df["close"]
            hist_df = hist_df.sort_values("Date")
        else:
            hist_df = pd.DataFrame()

        return info, hist_df
    except Exception as e:
        return {}, pd.DataFrame()

# -------------------------
# Fallback yfinance
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(symbol: str):
    try:
        t = yf.Ticker(f"{symbol}.NS")
        info = t.info or {}
        hist = t.history(period="1y", auto_adjust=False).reset_index()
        info["price"] = info.get("currentPrice")
        return info, hist
    except:
        return {}, pd.DataFrame()

# -------------------------
# Tabs Layout
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Screener"
])

# -------------------------
# Tab 1: Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    wl = load_watchlist()
    if not wl:
        st.info("Your watchlist is empty. Add stocks in the Watchlist Editor.")
    else:
        if st.button("🔍 Analyze Watchlist"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(wl):
                info, hist = fetch_massive_data(sym)
                if not info:
                    info, hist = fetch_yf(sym)
                ltp = safe_get(info, "price", np.nan)
                fv, _ = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)
                rows.append({
                    "Symbol": sym,
                    "LTP": ltp,
                    "Fair Value": fv,
                    "Underv%": rec["undervaluation"],
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"]
                })
                progress.progress(int(((i+1)/len(wl))*100))
            st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

# -------------------------
# Tab 2: Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    symbol = st.text_input("Enter NSE Symbol (e.g. RELIANCE, TCS)").upper().strip()
    if st.button("Analyze"):
        info, hist = fetch_massive_data(symbol)
        src = "Massive"
        if not info:
            info, hist = fetch_yf(symbol)
            src = "yfinance"
        if not info:
            st.error("No data found from Massive or yfinance.")
        else:
            price = safe_get(info, "price", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)
            st.metric("LTP", price)
            st.metric("Fair Value", fv, method)
            st.metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            st.metric("Undervaluation%", rec["undervaluation"])
            st.caption(f"Source: {src}")
            if not hist.empty:
                st.line_chart(hist.set_index("Date")["Close"], height=300)
            st.json(info)

# -------------------------
# Tab 3: Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV with: symbol,buy_price,quantity")
    up = st.file_uploader("Upload portfolio", type=["csv"])
    if up:
        pf = pd.read_csv(up)
        pf.columns = [c.lower() for c in pf.columns]
        rows = []
        progress = st.progress(0)
        for i, r in pf.iterrows():
            sym = r["symbol"].upper()
            info, hist = fetch_massive_data(sym)
            if not info:
                info, hist = fetch_yf(sym)
            ltp = safe_get(info, "price", np.nan)
            buy = r["buy_price"]
            qty = r["quantity"]
            invested = buy * qty
            current = (ltp or 0) * qty
            pl = current - invested
            rows.append({
                "Symbol": sym,
                "Buy": buy,
                "LTP": ltp,
                "Qty": qty,
                "Invested": invested,
                "Current": current,
                "P/L": round(pl, 2)
            })
            progress.progress(int(((i+1)/len(pf))*100))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------
# Tab 4: Alerts
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    st.write("Checks watchlist for undervaluation threshold.")
    th = st.number_input("Undervaluation ≥ %", value=10)
    wl = load_watchlist()
    if st.button("Check Alerts"):
        alerts = []
        for sym in wl:
            info, hist = fetch_massive_data(sym)
            if not info:
                info, hist = fetch_yf(sym)
            fv, _ = estimate_fair_value(info)
            ltp = safe_get(info, "price", np.nan)
            rec = rule_based_recommendation(info, fv, ltp)
            if rec["undervaluation"] and rec["undervaluation"] >= th:
                alerts.append({"Symbol": sym, "Underv%": rec["undervaluation"], "Rec": rec["recommendation"]})
        st.dataframe(pd.DataFrame(alerts) if alerts else pd.DataFrame({"Status": ["No alerts triggered."]}))

# -------------------------
# Tab 5: Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    cur = "\n".join(load_watchlist())
    txt = st.text_area("Edit watchlist", value=cur, height=300)
    if st.button("Save"):
        syms = [s.strip().upper() for s in txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(syms)
        if ok: st.success("Saved ✅")
        else: st.error(msg)

# -------------------------
# Tab 6: Screener
# -------------------------
with tab6:
    st.header("🧠 Nifty Screener")
    n = st.number_input("Top N symbols", 10, 500, 50)
    nifty = load_nifty_stocks()
    if st.button("Run Screener"):
        rows = []
        for sym in nifty["Symbol"].head(int(n)):
            info, hist = fetch_massive_data(sym)
            if not info:
                info, hist = fetch_yf(sym)
            ltp = safe_get(info, "price", np.nan)
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            rows.append({
                "Symbol": sym,
                "LTP": ltp,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

st.caption("Made by Biswanath 🔍 | Massive API primary (NSE/BSE), yfinance fallback (.NS)")
