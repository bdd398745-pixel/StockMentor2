# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary: Massive API (NSE/BSE)
Fallback: yfinance (.NS)
Includes: Watchlist Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, Screener
Author: Adapted for Biswanath
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import math
from datetime import datetime

# -----------------------------------------------------
# App Setup
# -----------------------------------------------------
st.set_page_config(page_title="StockMentor (Massive API)", page_icon="📊", layout="wide")
st.title("📊 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: Massive API (NSE/BSE) → Fallback: yfinance (.NS)")

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 15.0
MASSIVE_API_KEY = st.secrets.get("MASSIVE_API_KEY", os.getenv("MASSIVE_API_KEY", ""))

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
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
        return True, "Saved"
    except Exception as e:
        return False, str(e)

def safe_get(d, key, default=np.nan):
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    return val if val not in (None, "", "None") else default

# -----------------------------------------------------
# Fair Value Estimator
# -----------------------------------------------------
def estimate_fair_value(info):
    try:
        price = float(safe_get(info, "price", safe_get(info, "currentPrice", np.nan)))
        pe = float(safe_get(info, "peRatio", np.nan))
        eps = float(safe_get(info, "eps", np.nan))

        if not np.isnan(eps) and eps > 0:
            fv = eps * DEFAULT_PE_TARGET
            method = "EPS × PE heuristic"
        elif not np.isnan(pe) and pe > 0 and price:
            fv = (price / pe) * DEFAULT_PE_TARGET
            method = "PE reversion"
        else:
            fv = price
            method = "Fallback"

        return round(fv, 2), method
    except Exception:
        return np.nan, "error"

def compute_buy_sell(fv, mos=0.25):
    if not fv or fv <= 0:
        return None, None
    return round(fv * (1 - mos), 2), round(fv * (1 + mos / 1.5), 2)

# -----------------------------------------------------
# Rule-Based Recommendation
# -----------------------------------------------------
def rule_based_recommendation(info, fair_value, price):
    try:
        roe = float(safe_get(info, "roe", np.nan))
        de = float(safe_get(info, "debtToEquity", np.nan))
        underval = ((fair_value - price) / fair_value * 100) if fair_value and price else np.nan
        score = 0
        if roe >= 20: score += 3
        elif roe >= 12: score += 2
        if de <= 0.5: score += 2
        elif de <= 1.5: score += 1
        if underval >= 25: score += 3
        elif underval >= 10: score += 2
        if score >= 7: rec = "Strong Buy"
        elif score >= 5: rec = "Buy"
        elif score <= 2: rec = "Avoid / Monitor"
        else: rec = "Hold"
        return {"score": score, "recommendation": rec, "undervaluation": round(underval, 2)}
    except Exception:
        return {"score": 0, "recommendation": "Hold", "undervaluation": np.nan}

# -----------------------------------------------------
# Massive API Fetcher (Primary)
# -----------------------------------------------------
@st.cache_data(ttl=900)
def fetch_massive_data(symbol: str):
    """
    Fetch data from Massive API.
    Example: RELIANCE.NS → NSE | RELIANCE.BO → BSE
    """
    if not MASSIVE_API_KEY:
        return {}, pd.DataFrame()

    base_url = "https://api.massive.com/stock"
    for suffix in [".NS", ".BO"]:
        try:
            url = f"{base_url}/{symbol}{suffix}?apikey={MASSIVE_API_KEY}"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data or "price" not in data:
                continue

            info = {
                "symbol": symbol,
                "price": data.get("price"),
                "eps": data.get("eps"),
                "peRatio": data.get("peRatio"),
                "roe": data.get("roe"),
                "debtToEquity": data.get("debtToEquity"),
                "sector": data.get("sector"),
                "exchange": "NSE" if suffix == ".NS" else "BSE",
            }

            # Historical Data
            hist_url = f"{base_url}/{symbol}{suffix}/historical?period=1y&apikey={MASSIVE_API_KEY}"
            h_resp = requests.get(hist_url, timeout=10)
            hist = h_resp.json()
            hist_df = pd.DataFrame(hist)
            if not hist_df.empty:
                hist_df["Date"] = pd.to_datetime(hist_df["date"])
                hist_df["Close"] = hist_df["close"]
            return info, hist_df
        except Exception:
            continue

    return {}, pd.DataFrame()

# -----------------------------------------------------
# yfinance Fallback
# -----------------------------------------------------
@st.cache_data(ttl=300)
def fetch_yfinance_data(symbol: str):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        hist = hist.reset_index()
        hist.rename(columns={"Close": "Close"}, inplace=True)
        info["price"] = info.get("currentPrice")
        return info, hist
    except Exception as e:
        return {}, pd.DataFrame()

# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Screener"
])

# -----------------------------------------------------
# 1️⃣ Dashboard
# -----------------------------------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    wl = load_watchlist()
    if not wl:
        st.info("No symbols found in Watchlist.")
    else:
        if st.button("Analyze Watchlist"):
            rows = []
            for s in wl:
                info, hist = fetch_massive_data(s)
                if not info:
                    info, hist = fetch_yfinance_data(s)
                price = safe_get(info, "price", np.nan)
                fv, _ = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, price)
                buy, sell = compute_buy_sell(fv)
                rows.append({
                    "Symbol": s,
                    "LTP": price,
                    "FairValue": fv,
                    "Undervaluation%": rec["undervaluation"],
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Score": rec["score"],
                    "Recommendation": rec["recommendation"]
                })
            df = pd.DataFrame(rows)
            st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)

# -----------------------------------------------------
# 2️⃣ Single Stock
# -----------------------------------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    sym = st.text_input("Enter stock symbol (e.g. RELIANCE)").upper().strip()
    if st.button("Analyze"):
        info, hist = fetch_massive_data(sym)
        if not info:
            info, hist = fetch_yfinance_data(sym)
        if not info:
            st.error("No data found.")
        else:
            price = safe_get(info, "price", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            st.metric("LTP", f"{price}")
            st.metric("Fair Value", f"{fv} ({method})")
            st.metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            st.json(info)
            if not hist.empty:
                st.line_chart(hist.set_index("Date")["Close"], height=300)

# -----------------------------------------------------
# 3️⃣ Portfolio
# -----------------------------------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.write("Upload CSV with columns: symbol, buy_price, quantity.")
    f = st.file_uploader("Upload", type="csv")
    if f:
        df = pd.read_csv(f)
        rows = []
        for _, r in df.iterrows():
            s = r["symbol"].upper()
            buy, qty = r["buy_price"], r["quantity"]
            info, hist = fetch_massive_data(s)
            if not info:
                info, hist = fetch_yfinance_data(s)
            price = safe_get(info, "price", np.nan)
            invested = buy * qty
            current = price * qty if price else np.nan
            pl = current - invested if current else np.nan
            rows.append({
                "Symbol": s,
                "Buy Price": buy,
                "LTP": price,
                "Qty": qty,
                "Invested": invested,
                "Current Value": current,
                "P/L": pl,
                "P/L %": round(pl / invested * 100, 2) if invested else np.nan
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -----------------------------------------------------
# 4️⃣ Alerts
# -----------------------------------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    st.info("Highlight undervalued stocks from your Watchlist.")
    threshold = st.number_input("Undervaluation ≥ %", value=10.0)
    wl = load_watchlist()
    if st.button("Run Alerts"):
        flagged = []
        for s in wl:
            info, hist = fetch_massive_data(s)
            if not info:
                info, hist = fetch_yfinance_data(s)
            price = safe_get(info, "price", np.nan)
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            if rec["undervaluation"] and rec["undervaluation"] >= threshold:
                flagged.append({"Symbol": s, "Undervaluation%": rec["undervaluation"], "Rec": rec["recommendation"]})
        st.dataframe(pd.DataFrame(flagged), use_container_width=True)

# -----------------------------------------------------
# 5️⃣ Watchlist Editor
# -----------------------------------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    wl = load_watchlist()
    txt = st.text_area("Symbols", "\n".join(wl), height=300)
    if st.button("💾 Save"):
        new_list = [x.strip().upper() for x in txt.splitlines() if x.strip()]
        ok, msg = save_watchlist(new_list)
        st.success(msg if ok else msg)

# -----------------------------------------------------
# 6️⃣ Screener
# -----------------------------------------------------
with tab6:
    st.header("🧠 Screener — Top N (Manual List)")
    user_list = st.text_area("Enter comma-separated stock symbols").upper()
    if st.button("Run Screener"):
        symbols = [x.strip() for x in user_list.split(",") if x.strip()]
        rows = []
        for s in symbols:
            info, hist = fetch_massive_data(s)
            if not info:
                info, hist = fetch_yfinance_data(s)
            price = safe_get(info, "price", np.nan)
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": s,
                "LTP": price,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
        df = pd.DataFrame(rows)
        st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)

# -----------------------------------------------------
st.caption("Made by Biswanath 🔍 | Massive API primary (NSE/BSE), yfinance fallback (.NS)")
