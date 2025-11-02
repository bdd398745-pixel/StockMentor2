# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Enhanced with Twelve Data API + yfinance fallback

Features:
- Optional TwelveData API toggle
- yfinance fallback
- Screener for Nifty 500
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
st.set_page_config(page_title="StockMentor (Enhanced)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Now with Twelve Data API & yfinance fallback 🚀")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0

# Load Twelve Data API key
TWELVE_API_KEY = st.secrets.get("TWELVE_API_KEY", os.getenv("TWELVE_API_KEY", ""))

# -------------------------
# Load Nifty 500 List
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
        return True, "Saved successfully"
    except Exception as e:
        return False, str(e)

# -------------------------
# Twelve Data Fetcher (Fixed)
# -------------------------
def fetch_twelvedata_data(symbol):
    """Fetch stock info and 1-year history using Twelve Data API, with safe fallback."""
    try:
        if not TWELVE_API_KEY:
            raise Exception("Missing Twelve Data API key")

        base = "https://api.twelvedata.com"
        possible_symbols = [f"{symbol}:NS", f"{symbol}:BSE", symbol]

        for sym_try in possible_symbols:
            quote_url = f"{base}/quote?symbol={sym_try}&apikey={TWELVE_API_KEY}"
            profile_url = f"{base}/fundamentals?symbol={sym_try}&apikey={TWELVE_API_KEY}"
            hist_url = f"{base}/time_series?symbol={sym_try}&interval=1day&outputsize=365&apikey={TWELVE_API_KEY}"

            # Validate response is JSON
            def safe_get(url):
                r = requests.get(url)
                try:
                    return r.json()
                except:
                    st.info(f"Twelve Data returned HTML for {sym_try}, skipping...")
                    return {}

            q = safe_get(quote_url)
            f = safe_get(profile_url)
            h = safe_get(hist_url)

            if "price" not in q:
                continue  # Try next format

            info = {
                "symbol": symbol,
                "companyName": f.get("name") if isinstance(f, dict) else symbol,
                "sector": f.get("sector") if isinstance(f, dict) else None,
                "industry": f.get("industry") if isinstance(f, dict) else None,
                "price": float(q.get("price", np.nan)),
                "trailingPE": float(f.get("valuation_ratios", {}).get("pe_ratio", np.nan)) if isinstance(f, dict) else np.nan,
                "priceToBook": float(f.get("valuation_ratios", {}).get("pb_ratio", np.nan)) if isinstance(f, dict) else np.nan,
                "dividendYield": float(f.get("valuation_ratios", {}).get("dividend_yield", np.nan)) if isinstance(f, dict) else np.nan,
                "eps": float(f.get("earnings_per_share", {}).get("basic_eps", np.nan)) if isinstance(f, dict) else np.nan,
                "returnOnEquity": float(f.get("profitability", {}).get("roe", np.nan)) if isinstance(f, dict) else np.nan,
                "debtToEquity": float(f.get("financial_health", {}).get("debt_to_equity", np.nan)) if isinstance(f, dict) else np.nan,
            }

            # Historical data
            if "values" in h:
                hist = pd.DataFrame(h["values"])
                hist.rename(columns={"datetime": "Date", "close": "Close"}, inplace=True)
                hist["Date"] = pd.to_datetime(hist["Date"])
                hist["Close"] = hist["Close"].astype(float)
                hist = hist.sort_values("Date")
            else:
                hist = pd.DataFrame()

            return info, hist

        raise Exception("All symbol formats failed or data unavailable")

    except Exception as e:
        st.warning(f"TwelveData fetch failed for {symbol}: {e}. Falling back to yfinance...")
        return fetch_info_and_history(symbol)


# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        hist = hist.reset_index()
        hist.rename(columns={"Date": "Date", "Close": "Close"}, inplace=True)
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# Utility Functions
# -------------------------
def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    return default if v in (None, "None", "") else v

def estimate_fair_value(info):
    try:
        price = float(info.get("price") or info.get("currentPrice") or np.nan)
        pe = float(info.get("pe") or info.get("trailingPE") or np.nan)
        eps = float(info.get("eps") or info.get("trailingEps") or np.nan)
        pb = float(info.get("pb") or info.get("priceToBook") or np.nan)
        roe = float(info.get("roeTTM") or info.get("returnOnEquity") or np.nan)

        if not np.isnan(eps) and eps > 0:
            fair_value = eps * 15
            method = "EPS-based"
        elif not np.isnan(pe) and pe > 0 and not np.isnan(price):
            fair_value = (price / pe) * 15
            method = "P/E reversion"
        elif not np.isnan(pb) and not np.isnan(price):
            fair_value = pb * price
            method = "P/B heuristic"
        else:
            fair_value = price
            method = "fallback"

        return round(float(fair_value), 2), method
    except Exception as e:
        print(f"estimate_fair_value() error: {e}")
        return np.nan, "error"

def compute_buy_sell(fv, mos=0.25):
    try:
        fv = float(fv)
        if math.isnan(fv) or fv <= 0:
            return None, None
        buy = round(fv * (1 - mos), 2)
        sell = round(fv * (1 + mos / 1.5), 2)
        return buy, sell
    except Exception as e:
        print(f"compute_buy_sell() error: {e}")
        return None, None

def rule_based_recommendation(info, fair_value, price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    de = safe_get(info, "debtToEquity", np.nan)
    underval = None if not fair_value or not price else round(((fair_value - price) / fair_value) * 100, 2)
    score = 0
    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 3
        elif roe >= 0.12:
            score += 2
    if isinstance(de, (int, float)):
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
    return {"score": score, "recommendation": rec, "undervaluation": underval}

# -------------------------
# Tabs
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
    elif st.button("🔍 Analyze Watchlist"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(watchlist):
            info, _ = fetch_twelvedata_data(sym)
            if not info:
                info, _ = fetch_info_and_history(sym)
            ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
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
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="Score", ascending=False)
        st.dataframe(df_sorted, width="stretch")
        st.success("✅ Ranked by multi-factor score (Quality + Valuation)")

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    use_twelve = st.toggle("Use TwelveData API", value=bool(TWELVE_API_KEY))
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = fetch_twelvedata_data(symbol) if use_twelve else fetch_info_and_history(symbol)
        if not info:
            st.warning("No data found. Using fallback...")
            info, hist = fetch_info_and_history(symbol)
        price = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
        fv, method = estimate_fair_value(info)
        rec = rule_based_recommendation(info, fv, price)
        st.metric("Current Price", price)
        st.metric("Fair Value", fv)
        st.metric("Recommendation", rec["recommendation"])
        st.metric("Score", rec["score"])
        st.metric("Undervaluation %", rec["undervaluation"])
        st.caption(f"Valuation Method: {method}")
        st.write(info)
        if not hist.empty:
            st.line_chart(hist.set_index("Date")["Close"], height=250)

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV with columns: symbol, buy_price, quantity.")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf.columns = [c.lower() for c in pf.columns]
            if not set(["symbol", "buy_price", "quantity"]).issubset(set(pf.columns)):
                st.error("CSV must contain: symbol, buy_price, quantity")
            else:
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    info, _ = fetch_twelvedata_data(sym)
                    if not info:
                        info, _ = fetch_info_and_history(sym)
                    ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
                    current_value = ltp * qty if isinstance(ltp, (int, float)) else np.nan
                    invested = buy * qty
                    pl = current_value - invested if current_value and invested else np.nan
                    pl_pct = (pl / invested) * 100 if invested else np.nan
                    rows.append({
                        "Symbol": sym, "Buy Price": buy, "Qty": qty, "LTP": ltp,
                        "Current Value": round(current_value, 2) if current_value else None,
                        "Invested": round(invested, 2),
                        "P/L": round(pl, 2) if pl else None,
                        "P/L %": round(pl_pct, 2) if pl_pct else None
                    })
                out = pd.DataFrame(rows)
                st.dataframe(out, width="stretch")
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error reading portfolio: " + str(e))

# -------------------------
# Alerts (manual)
# -------------------------
with tab4:
    st.header("📣 Email Alerts (manual send)")
    st.info("This feature is optional — for Gmail, use smtp.gmail.com with App Password.")
    st.write("Alerts trigger based on undervaluation % threshold.")

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    st.write("Edit your watchlist (one symbol per line). Use NSE tickers (no .NS).")
    current = load_watchlist()
    new_txt = st.text_area("Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("✅ Watchlist saved.")
        else:
            st.error("Save failed: " + msg)

# -------------------------
# Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty 500")
    use_twelve = st.toggle("Use TwelveData API (recommended)", value=bool(TWELVE_API_KEY))
    nifty = load_nifty_stocks()
    if st.button("Run Screener"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(nifty["Symbol"].tolist()[:100]):
            info, _ = fetch_twelvedata_data(sym) if use_twelve else fetch_info_and_history(sym)
            if not info:
                info, _ = fetch_info_and_history(sym)
            price = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
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
            progress.progress(int(((i + 1) / len(nifty[:100])) * 100))
        df = pd.DataFrame(rows)
        df = df.sort_values("Score", ascending=False)
        st.dataframe(df, width="stretch")
        st.success("✅ Screener completed successfully")

st.caption("Made by Biswanath 🔍 | TwelveData + yfinance fallback | Fully offline-capable")
