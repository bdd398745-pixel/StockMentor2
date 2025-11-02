# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary API: Massive (NSE/BSE)
Fallback: Yahoo Finance (.NS)
Includes:
- Dashboard (Watchlist)
- Single Stock Analyzer
- Portfolio View
- Alerts
- Watchlist Editor
- Screener
With Sidebar API Monitor (Massive vs yfinance)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import datetime

# -----------------------------------------------------
# Streamlit Setup
# -----------------------------------------------------
st.set_page_config(page_title="StockMentor", page_icon="📈", layout="wide")
st.title("📊 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: Massive API → Fallback: Yahoo Finance (.NS)")

# -----------------------------------------------------
# Config & Constants
# -----------------------------------------------------
WATCHLIST_FILE = "watchlist.csv"
MASSIVE_API_KEY = st.secrets.get("MASSIVE_API_KEY", os.getenv("MASSIVE_API_KEY", ""))
DEFAULT_PE_TARGET = 15.0
api_log = []

# -----------------------------------------------------
# Sidebar Monitor
# -----------------------------------------------------
st.sidebar.header("🔍 API Status Monitor")

def log_api_status(symbol, source, status):
    """Append API status log"""
    api_log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Source": source,
        "Status": status
    })
    st.sidebar.dataframe(pd.DataFrame(api_log), use_container_width=True)

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
def safe_get(d, key, default=np.nan):
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    return val if val not in (None, "", "None") else default

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
        return True, "Saved successfully!"
    except Exception as e:
        return False, str(e)

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
# Recommendation Logic
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

        if score >= 7:
            rec = "Strong Buy"
        elif score >= 5:
            rec = "Buy"
        elif score <= 2:
            rec = "Avoid / Monitor"
        else:
            rec = "Hold"

        return {"score": score, "recommendation": rec, "undervaluation": round(underval, 2)}
    except Exception:
        return {"score": 0, "recommendation": "Hold", "undervaluation": np.nan}

# -----------------------------------------------------
# Massive API Fetcher
# -----------------------------------------------------
@st.cache_data(ttl=900)
def fetch_massive_data(symbol: str):
    if not MASSIVE_API_KEY:
        return {}, pd.DataFrame(), False

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

            hist_url = f"{base_url}/{symbol}{suffix}/historical?period=1y&apikey={MASSIVE_API_KEY}"
            h_resp = requests.get(hist_url, timeout=10)
            hist = h_resp.json()
            hist_df = pd.DataFrame(hist)
            if not hist_df.empty:
                hist_df["Date"] = pd.to_datetime(hist_df["date"])
                hist_df["Close"] = hist_df["close"]
            return info, hist_df, True
        except Exception:
            continue
    return {}, pd.DataFrame(), False

# -----------------------------------------------------
# Yahoo Finance Fallback
# -----------------------------------------------------
@st.cache_data(ttl=300)
def fetch_yfinance_data(symbol: str):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y").reset_index()
        info["price"] = info.get("currentPrice")
        return info, hist, True
    except Exception:
        return {}, pd.DataFrame(), False

# -----------------------------------------------------
# Tabs
# -----------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Screener"
])

# -----------------------------------------------------
# 📋 1. Dashboard
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
                info, hist, ok = fetch_massive_data(s)
                if ok:
                    log_api_status(s, "Massive API", "✅ Success")
                else:
                    info, hist, ok2 = fetch_yfinance_data(s)
                    log_api_status(s, "yfinance", "✅ Fallback" if ok2 else "❌ Failed")

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
# 🔎 2. Single Stock
# -----------------------------------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    sym = st.text_input("Enter stock symbol (e.g. RELIANCE)").upper().strip()
    if st.button("Analyze"):
        info, hist, ok = fetch_massive_data(sym)
        if ok:
            log_api_status(sym, "Massive API", "✅ Success")
        else:
            info, hist, ok2 = fetch_yfinance_data(sym)
            log_api_status(sym, "yfinance", "✅ Fallback" if ok2 else "❌ Failed")

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
# 💼 3. Portfolio
# -----------------------------------------------------
with tab3:
    st.header("💼 Portfolio View")
    st.info("Upload your holdings (CSV with columns: Symbol, Qty, AvgPrice)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df["Symbol"] = df["Symbol"].str.upper()
        results = []
        for _, row in df.iterrows():
            sym = row["Symbol"]
            qty = row["Qty"]
            avg = row["AvgPrice"]
            info, _, ok = fetch_massive_data(sym)
            if not ok:
                info, _, _ = fetch_yfinance_data(sym)
            ltp = safe_get(info, "price", np.nan)
            pnl = (ltp - avg) * qty
            results.append({
                "Symbol": sym, "Qty": qty, "AvgPrice": avg, "LTP": ltp,
                "PnL": round(pnl, 2)
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)

# -----------------------------------------------------
# 📣 4. Alerts
# -----------------------------------------------------
with tab4:
    st.header("📣 Alerts")
    st.info("Coming soon: email/SMS alerts for Buy/Sell triggers.")

# -----------------------------------------------------
# 🧾 5. Watchlist Editor
# -----------------------------------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    wl = load_watchlist()
    st.write("Current:", ", ".join(wl) if wl else "None")
    new_entry = st.text_input("Add symbol (e.g. TCS, INFY, RELIANCE):").upper().strip()
    if st.button("Add"):
        if new_entry and new_entry not in wl:
            wl.append(new_entry)
            save_watchlist(wl)
            st.success(f"Added {new_entry}")
    if st.button("Clear All"):
        save_watchlist([])
        st.warning("Watchlist cleared.")

# -----------------------------------------------------
# 🧠 6. Screener
# -----------------------------------------------------
with tab6:
    st.header("🧠 Basic Screener")
    st.info("Coming soon: Filter by ROE, PE, DE, and undervaluation.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.caption("Made by Biswanath 🔍 | API Monitor enabled in sidebar (Massive vs yfinance)")
