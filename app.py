# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary: NSE India API
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

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (NSE)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: NSE India API — Fallback: yfinance (.NS)")

# -------------------------
# Constants & Files
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 15.0

# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_nifty_stocks():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        cols = [c for c in ["Symbol", "Company Name", "Industry"] if c in df.columns]
        return df[cols]
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
        try:
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved successfully"
    except Exception as e:
        return False, str(e)

# -------------------------
# Helper Functions
# -------------------------
def safe_get(info, key, default=np.nan):
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if v in (None, "None", ""):
        return default
    return v

def estimate_fair_value(info):
    try:
        price = float(safe_get(info, "price", np.nan))
        eps = safe_get(info, "eps", np.nan)
        pe = safe_get(info, "pe", np.nan)
        pb = safe_get(info, "pb", np.nan)

        fv = np.nan
        method = "fallback"

        if eps and not pd.isna(eps) and eps > 0:
            fv = eps * DEFAULT_PE_TARGET
            method = "EPS-based"
        elif pe and not pd.isna(pe) and price:
            fv = (price / pe) * DEFAULT_PE_TARGET
            method = "P/E reversion"
        elif pb and not pd.isna(pb) and price:
            fv = pb * price
            method = "P/B heuristic"
        else:
            fv = price

        return round(float(fv), 2), method
    except Exception:
        return np.nan, "error"

def compute_buy_sell(fv, mos=0.25):
    try:
        fv = float(fv)
        if math.isnan(fv) or fv <= 0:
            return None, None
        buy = round(fv * (1 - mos), 2)
        sell = round(fv * (1 + mos / 1.5), 2)
        return buy, sell
    except Exception:
        return None, None

def rule_based_recommendation(info, fair_value, price):
    try:
        roe = safe_get(info, "roe", np.nan)
        de = safe_get(info, "de", np.nan)
        underval = None
        if fair_value and price and not pd.isna(fair_value) and not pd.isna(price):
            underval = round(((fair_value - price) / fair_value) * 100, 2)

        score = 0
        if isinstance(roe, (int, float)) and not pd.isna(roe):
            if roe >= 20:
                score += 3
            elif roe >= 12:
                score += 2
        if isinstance(de, (int, float)) and not pd.isna(de):
            if de <= 0.5:
                score += 2
            elif de <= 1.5:
                score += 1
        if isinstance(underval, (int, float)):
            if underval >= 25:
                score += 3
            elif underval >= 10:
                score += 2

        rec = "Hold"
        if score >= 7:
            rec = "Strong Buy"
        elif score >= 5:
            rec = "Buy"
        elif score <= 2:
            rec = "Avoid / Monitor"

        return {"score": score, "recommendation": rec, "undervaluation": underval}
    except Exception:
        return {"score": 0, "recommendation": "Hold", "undervaluation": None}

# -------------------------
# NSE API Fetcher (Primary)
# -------------------------
@st.cache_data(ttl=900)
def fetch_nse_data(symbol):
    """
    Fetch LTP and basic info from NSE India API.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return {}, pd.DataFrame()

        data = resp.json()
        price_info = data.get("priceInfo", {})
        meta = data.get("info", {})

        info = {
            "symbol": symbol.upper(),
            "price": price_info.get("lastPrice"),
            "pe": meta.get("pe"),
            "pb": meta.get("pb"),
            "eps": meta.get("eps"),
            "roe": meta.get("roe"),
            "de": meta.get("debtEquity"),
            "companyName": meta.get("companyName"),
            "industry": meta.get("industry")
        }

        hist = pd.DataFrame()
        return info, hist
    except Exception:
        return {}, pd.DataFrame()

# -------------------------
# yfinance Fetcher (Fallback)
# -------------------------
@st.cache_data(ttl=600)
def fetch_yf_data(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        hist = hist.reset_index() if not hist.empty else pd.DataFrame()
        price = info.get("currentPrice", np.nan)
        info["price"] = price
        return info, hist
    except Exception:
        return {}, pd.DataFrame()

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Stock Screener"
])

# -------------------------
# Dashboard
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    else:
        if st.button("🔍 Analyze Watchlist"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, hist = fetch_nse_data(sym)
                if not info:
                    info, hist = fetch_yf_data(sym)

                ltp = safe_get(info, "price")
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
                progress.progress(int(((i + 1) / len(watchlist)) * 100))
            st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = fetch_nse_data(symbol)
        source = "NSE India"
        if not info:
            info, hist = fetch_yf_data(symbol)
            source = "yfinance"

        if not info:
            st.error("No data found.")
        else:
            price = safe_get(info, "price")
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)

            cols = st.columns(4)
            cols[0].metric("Current Price", f"₹{price}")
            cols[1].metric("Fair Value", fv, method)
            cols[2].metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            cols[3].metric("Undervaluation %", rec["undervaluation"], source)

            st.json(info)
            if hist is not None and not hist.empty:
                st.line_chart(hist.set_index("Date")["Close"], height=300)

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    uploaded = st.file_uploader("Upload CSV (symbol, buy_price, quantity)", type=["csv"])
    if uploaded:
        pf = pd.read_csv(uploaded)
        pf.columns = [c.lower().strip() for c in pf.columns]
        rows = []
        for _, r in pf.iterrows():
            sym = str(r["symbol"]).strip().upper()
            buy = float(r["buy_price"])
            qty = float(r["quantity"])
            info, _ = fetch_nse_data(sym)
            if not info:
                info, _ = fetch_yf_data(sym)
            ltp = safe_get(info, "price")
            invested = buy * qty
            current_value = ltp * qty if not pd.isna(ltp) else np.nan
            pl = current_value - invested if not pd.isna(current_value) else np.nan
            pl_pct = (pl / invested) * 100 if invested else np.nan
            rows.append({
                "Symbol": sym,
                "Buy Price": buy,
                "Qty": qty,
                "LTP": ltp,
                "Invested": invested,
                "Current Value": current_value,
                "P/L": pl,
                "P/L %": pl_pct
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.metric("Total P/L (₹)", f"{df['P/L'].sum():,.2f}")

# -------------------------
# Alerts
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    threshold = st.number_input("Undervaluation % threshold", value=10.0)
    uploaded = st.file_uploader("Upload Watchlist (CSV one symbol per line)", type=["csv"])
    if uploaded:
        wl = pd.read_csv(uploaded, header=None)[0].astype(str).str.strip().str.upper().tolist()
        flagged = []
        for s in wl:
            info, _ = fetch_nse_data(s)
            if not info:
                info, _ = fetch_yf_data(s)
            fv, _ = estimate_fair_value(info)
            ltp = safe_get(info, "price")
            rec = rule_based_recommendation(info, fv, ltp)
            if rec["undervaluation"] and rec["undervaluation"] >= threshold:
                flagged.append({"symbol": s, "underv%": rec["undervaluation"], "rec": rec["recommendation"]})
        if flagged:
            st.success(f"Found {len(flagged)} flagged symbols")
            st.dataframe(pd.DataFrame(flagged))
        else:
            st.info("No symbols flagged.")

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    current = load_watchlist()
    new_txt = st.text_area("Edit Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save Watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("✅ Watchlist saved.")
        else:
            st.error(msg)

# -------------------------
# Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty 500")
    nifty = load_nifty_stocks()
    max_scan = st.number_input("Max symbols to scan", min_value=10, max_value=500, value=50, step=10)
    if st.button("Run Screener"):
        rows = []
        for sym in nifty["Symbol"].tolist()[:max_scan]:
            info, _ = fetch_nse_data(sym)
            if not info:
                info, _ = fetch_yf_data(sym)
            price = safe_get(info, "price")
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": sym,
                "LTP": price,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.caption("Made by Biswanath 🔍 | Primary: NSE API, Fallback: yfinance (.NS)")
