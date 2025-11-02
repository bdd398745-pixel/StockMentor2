# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary: Massive API (NSE)
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
st.set_page_config(page_title="StockMentor (Massive API)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: Massive API (NSE) — Fallback: yfinance (.NS)")

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

def safe_get(info, key, default=np.nan):
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if v in (None, "None", ""):
        return default
    return v

# -------------------------
# Massive API Fetcher
# -------------------------
@st.cache_data(ttl=900)
def fetch_massive_data(symbol: str):
    """
    Fetch stock data from Massive API (NSE)
    """
    if not MASSIVE_API_KEY:
        return {}, pd.DataFrame()

    base = "https://api.massive.app/api/v1"
    headers = {"x-api-key": MASSIVE_API_KEY}

    try:
        quote_url = f"{base}/stock/{symbol}.NS/quote"
        fund_url = f"{base}/stock/{symbol}.NS/fundamentals"
        hist_url = f"{base}/stock/{symbol}.NS/history?interval=1d&range=1y"

        q = requests.get(quote_url, headers=headers, timeout=10).json()
        f = requests.get(fund_url, headers=headers, timeout=10).json()
        h = requests.get(hist_url, headers=headers, timeout=15).json()

        price = safe_get(q, "price", np.nan)
        info = {
            "symbol": symbol,
            "exchange": "NSE",
            "price": price,
            "currentPrice": price,
            "marketCap": safe_get(f, "marketCap"),
            "trailingPE": safe_get(f, "peRatio"),
            "priceToBook": safe_get(f, "pbRatio"),
            "dividendYield": safe_get(f, "dividendYield"),
            "eps": safe_get(f, "eps"),
            "returnOnEquity": safe_get(f, "returnOnEquity"),
            "debtToEquity": safe_get(f, "debtToEquity"),
            "companyName": safe_get(f, "name"),
            "sector": safe_get(f, "sector"),
            "industry": safe_get(f, "industry"),
        }

        hist_df = pd.DataFrame()
        if "data" in h and isinstance(h["data"], list):
            hist_df = pd.DataFrame(h["data"])
            if {"date", "close"}.issubset(hist_df.columns):
                hist_df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
                hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                hist_df["Close"] = hist_df["Close"].astype(float)
                hist_df = hist_df.sort_values("Date")

        if info.get("price") is None:
            return {}, pd.DataFrame()

        return info, hist_df

    except Exception as e:
        print("Massive API fetch error:", e)
        return {}, pd.DataFrame()

# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=300)
def fetch_yf(symbol: str):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        if not hist.empty:
            hist = hist.reset_index()
        if "currentPrice" in info:
            info["price"] = info["currentPrice"]
        return info, hist
    except Exception as e:
        print("yfinance error:", e)
        return {}, pd.DataFrame()

# -------------------------
# Computation Helpers
# -------------------------
def estimate_fair_value(info):
    try:
        price = safe_get(info, "price", np.nan)
        eps = safe_get(info, "eps", np.nan)
        pe = safe_get(info, "trailingPE", np.nan)
        pb = safe_get(info, "priceToBook", np.nan)

        if eps and eps > 0:
            fv = eps * DEFAULT_PE_TARGET
            method = "EPS-based"
        elif pe and price:
            fv = (price / pe) * DEFAULT_PE_TARGET
            method = "P/E reversion"
        elif pb and price:
            fv = pb * price
            method = "P/B heuristic"
        else:
            fv = price
            method = "Fallback"

        return round(fv, 2), method
    except Exception:
        return np.nan, "Error"

def compute_buy_sell(fv, mos=0.25):
    if fv and fv > 0:
        buy = round(fv * (1 - mos), 2)
        sell = round(fv * (1 + mos / 1.5), 2)
        return buy, sell
    return None, None

def rule_based_recommendation(info, fair_value, price):
    try:
        roe = safe_get(info, "returnOnEquity", np.nan)
        de = safe_get(info, "debtToEquity", np.nan)
        underval = ((fair_value - price) / fair_value) * 100 if fair_value else np.nan

        score = 0
        if roe and roe >= 0.20:
            score += 3
        elif roe and roe >= 0.12:
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
            rec = "Avoid / Monitor"
        else:
            rec = "Hold"

        return {"score": score, "recommendation": rec, "undervaluation": round(underval, 2) if not pd.isna(underval) else None}
    except Exception:
        return {"score": 0, "recommendation": "Hold", "undervaluation": None}

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
                progress.progress(int(((i+1)/len(watchlist))*100))
            df = pd.DataFrame(rows)
            st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    symbol = st.text_input("Enter NSE symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = fetch_massive_data(symbol)
        if not info:
            info, hist = fetch_yf(symbol)
        if not info:
            st.error("No data found.")
        else:
            price = safe_get(info, "price", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)

            cols = st.columns(4)
            cols[0].metric("Current Price", round(price,2))
            cols[1].metric("Fair Value", fv, method)
            cols[2].metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            cols[3].metric("Undervaluation %", rec["undervaluation"])

            st.json(info)
            if not hist.empty:
                st.line_chart(hist.set_index("Date")["Close"], height=300)

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV with columns: symbol, buy_price, quantity.")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        pf = pd.read_csv(uploaded)
        pf.columns = [c.lower() for c in pf.columns]
        if not {"symbol", "buy_price", "quantity"}.issubset(pf.columns):
            st.error("CSV must contain symbol, buy_price, quantity.")
        else:
            rows = []
            progress = st.progress(0)
            for i, r in pf.iterrows():
                sym = str(r["symbol"]).strip().upper()
                buy = float(r["buy_price"])
                qty = float(r["quantity"])
                info, hist = fetch_massive_data(sym)
                if not info:
                    info, hist = fetch_yf(sym)
                ltp = safe_get(info, "price", np.nan)
                if pd.isna(ltp): continue
                invested = buy * qty
                current = ltp * qty
                pl = current - invested
                rows.append({
                    "Symbol": sym, "Qty": qty, "Buy Price": buy, "LTP": ltp,
                    "Invested": invested, "Current Value": current,
                    "P/L": pl, "P/L %": (pl/invested)*100
                })
                progress.progress(int(((i+1)/len(pf))*100))
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.metric("Total P/L", f"{out['P/L'].sum():,.2f}")

# -------------------------
# Alerts
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    threshold = st.number_input("Undervaluation % threshold", value=10.0, step=1.0)
    up = st.file_uploader("Upload watchlist CSV", type=["csv"])
    if up:
        wl = pd.read_csv(up, header=None)[0].astype(str).str.strip().str.upper().tolist()
        flagged = []
        progress = st.progress(0)
        for i, s in enumerate(wl):
            info, hist = fetch_massive_data(s)
            if not info:
                info, hist = fetch_yf(s)
            fv, _ = estimate_fair_value(info)
            ltp = safe_get(info, "price", np.nan)
            rec = rule_based_recommendation(info, fv, ltp)
            if rec["undervaluation"] and rec["undervaluation"] >= threshold:
                flagged.append({"Symbol": s, "Underv%": rec["undervaluation"], "Rec": rec["recommendation"]})
            progress.progress(int(((i+1)/len(wl))*100))
        if flagged:
            st.success(f"Found {len(flagged)} flagged symbols")
            st.dataframe(pd.DataFrame(flagged))
        else:
            st.info("No symbols matched criteria.")

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    current = load_watchlist()
    new_txt = st.text_area("Watchlist (one symbol per line)", "\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
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
    max_scan = st.number_input("Max symbols to scan", min_value=10, max_value=500, value=100, step=10)
    if st.button("Run Screener"):
        rows = []
        symbols = nifty["Symbol"].tolist()[:max_scan] if not nifty.empty else []
        progress = st.progress(0)
        for i, sym in enumerate(symbols):
            info, hist = fetch_massive_data(sym)
            if not info:
                info, hist = fetch_yf(sym)
            price = safe_get(info, "price", np.nan)
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
            progress.progress(int(((i+1)/max(1,len(symbols)))*100))
        df = pd.DataFrame(rows)
        if not df.empty:
            st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)
            st.success("✅ Screener completed.")
        else:
            st.info("No results.")
            
# -------------------------
# Footer
# -------------------------
st.caption("Made by Biswanath 🔍 | Data from Massive API (NSE) with yfinance fallback")
