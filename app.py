# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Enhanced with Financial Modeling Prep (FMP) API + yfinance fallback

Features:
- Optional FMP API toggle
- yfinance fallback for Indian tickers
- Screener for Nifty 500
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
st.set_page_config(page_title="StockMentor (FMP Edition)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Powered by Financial Modeling Prep (FMP) API & yfinance fallback 🚀")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))

# -------------------------
# Helper Functions
# -------------------------
def format_symbol_for_fmp(symbol):
    """Ensure FMP uses correct BSE format"""
    symbol = symbol.strip().upper()
    if not symbol.endswith(".BSE"):
        symbol += ".BSE"
    return symbol


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
# Fetching Data
# -------------------------
def fetch_fmp_data(symbol):
    """Fetch stock info and 1-year history from FMP"""
    try:
        if not FMP_API_KEY:
            raise Exception("Missing FMP API key")

        sym_fmp = format_symbol_for_fmp(symbol)

        base = "https://financialmodelingprep.com/api/v3"
        profile_url = f"{base}/profile/{sym_fmp}?apikey={FMP_API_KEY}"
        ratios_url = f"{base}/ratios-ttm/{sym_fmp}?apikey={FMP_API_KEY}"
        hist_url = f"{base}/historical-price-full/{sym_fmp}?timeseries=365&apikey={FMP_API_KEY}"

        profile = requests.get(profile_url).json()
        ratios = requests.get(ratios_url).json()
        hist = requests.get(hist_url).json()

        if not isinstance(profile, list) or len(profile) == 0:
            raise Exception("Profile not found")

        p = profile[0]
        r = ratios[0] if isinstance(ratios, list) and ratios else {}

        info = {
            "symbol": symbol,
            "companyName": p.get("companyName"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "price": float(p.get("price", np.nan)),
            "trailingPE": float(r.get("peRatioTTM", np.nan)),
            "priceToBook": float(r.get("pbRatioTTM", np.nan)),
            "dividendYield": float(r.get("dividendYielTTM", np.nan)),
            "eps": float(p.get("eps", np.nan)),
            "returnOnEquity": float(r.get("roeTTM", np.nan)),
            "debtToEquity": float(r.get("debtEquityRatioTTM", np.nan)),
        }

        if "historical" in hist:
            df = pd.DataFrame(hist["historical"])
            df.rename(columns={"date": "Date", "close": "Close"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Close"] = df["Close"].astype(float)
            df = df.sort_values("Date")
        else:
            df = pd.DataFrame()

        return info, df

    except Exception as e:
        st.warning(f"FMP fetch failed for {symbol}: {e}. Falling back to yfinance...")
        return fetch_info_and_history(symbol)


@st.cache_data(ttl=900)
def fetch_info_and_history(symbol):
    """yfinance fallback"""
    try:
        clean_symbol = symbol.replace(".NS", "").replace(".BSE", "").strip().upper()
        ticker = yf.Ticker(f"{clean_symbol}.NS")
        hist = ticker.history(period="1y")

        if hist.empty:
            ticker = yf.Ticker(f"{clean_symbol}.BO")
            hist = ticker.history(period="1y")

        info = ticker.info or {}
        hist = hist.reset_index()
        hist.rename(columns={"Date": "Date", "Close": "Close"}, inplace=True)

        return info, hist
    except Exception as e:
        st.warning(f"yfinance fetch failed for {symbol}: {e}")
        return {}, pd.DataFrame()


# -------------------------
# Analysis Functions
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
    except Exception:
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
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Screener"
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
            info, _ = fetch_fmp_data(sym)
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
        df = df.sort_values(by="Score", ascending=False)
        st.dataframe(df, width="stretch")
        st.success("✅ Ranked by Quality + Valuation")


# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    use_fmp = st.toggle("Use FMP API", value=bool(FMP_API_KEY))
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = fetch_fmp_data(symbol) if use_fmp else fetch_info_and_history(symbol)
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
                info, _ = fetch_fmp_data(sym)
                ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
                current_value = ltp * qty if isinstance(ltp, (int, float)) else np.nan
                invested = buy * qty
                pl = current_value - invested if current_value and invested else np.nan
                pl_pct = (pl / invested) * 100 if invested else np.nan
                rows.append({
                    "Symbol": sym, "Buy Price": buy, "Qty": qty, "LTP": ltp,
                    "Current Value": round(current_value, 2),
                    "Invested": round(invested, 2),
                    "P/L": round(pl, 2),
                    "P/L %": round(pl_pct, 2)
                })
            out = pd.DataFrame(rows)
            st.dataframe(out, width="stretch")
            st.metric("Total P/L (₹)", f"{out['P/L'].sum():,.2f}")


# -------------------------
# Alerts (manual)
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    st.info("This feature is optional — use for email or webhook alerts later.")


# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
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
    use_fmp = st.toggle("Use FMP API", value=bool(FMP_API_KEY))
    nifty = load_nifty_stocks()
    if st.button("Run Screener"):
        rows = []
        progress = st.progress(0)
        for i, sym in enumerate(nifty["Symbol"].tolist()[:100]):
            info, _ = fetch_fmp_data(sym) if use_fmp else fetch_info_and_history(sym)
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

st.caption("Made by Biswanath 🔍 | FMP + yfinance fallback | Fully offline-capable")
