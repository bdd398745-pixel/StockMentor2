# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
Enhanced: Tab6 = Stock Screener (FMP-powered with yfinance fallback)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import math
import time
from email.message import EmailMessage
from datetime import datetime

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="StockMentor (Rule-based)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("No OpenAI. FMP-powered screener (NIFTY500) + yfinance fallback.")

# -------------------------
# Constants & Settings
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
NIFTY500_FILE = "nifty500.csv"  # optional: upload full NIFTY500 list here (one symbol per line)
DEFAULT_PE_TARGET = 20.0
MOCK_SLEEP = 0.05   # polite pause to avoid hammering APIs
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_KEY = st.secrets.get("FMP_API_KEY")  # put your key in Streamlit secrets

# -------------------------
# Helper: watchlist load/save
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        return [s.replace(".NS", "").upper() for s in symbols if s and str(s).strip()]
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading {WATCHLIST_FILE}: {e}")
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        load_watchlist.clear()
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Helpers: safe-get & buy/sell
# -------------------------
def safe_get(info, key, default=np.nan):
    v = info.get(key, default) if info else default
    return default if v in (None, "None", "") else v

def compute_buy_sell(fv, mos=0.30):
    if fv is None or (isinstance(fv, float) and math.isnan(fv)):
        return None, None
    return round(fv * (1 - mos), 2), round(fv * (1 + mos/1.5), 2)

# -------------------------
# FMP fetch functions with fallback to yfinance
# -------------------------
def fmp_quote(symbol):
    """FMP quote endpoint for LTP and basic fields. symbol without .NS."""
    if not FMP_KEY:
        return None
    try:
        url = f"{FMP_BASE}/quote/{symbol}.NS?apikey={FMP_KEY}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0]
        return None
    except Exception:
        return None

def fmp_key_metrics(symbol):
    """FMP key metrics or ratios - try key-metrics endpoint."""
    if not FMP_KEY:
        return None
    try:
        url = f"{FMP_BASE}/key-metrics/{symbol}.NS?limit=1&apikey={FMP_KEY}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0]
        return None
    except Exception:
        return None

def fetch_info_and_history(symbol_no_suffix):
    """
    Try FMP first (quote + key metrics). If fails or missing, fallback to yfinance.
    Returns (info_dict, hist_df)
    info_dict will have keys used in app (currentPrice, trailingPE, trailingEps, returnOnEquity, debtToEquity, bookValue, targetMeanPrice, marketCap, earningsQuarterlyGrowth)
    """
    symbol_ns = f"{symbol_no_suffix}.NS"
    info = {}
    hist = pd.DataFrame()
    # FMP quote
    fmp_q = fmp_quote(symbol_no_suffix)
    fmp_km = fmp_key_metrics(symbol_no_suffix)
    if fmp_q:
        # normalize keys to match existing usage
        info["currentPrice"] = fmp_q.get("price") or fmp_q.get("previousClose")
        info["trailingPE"] = fmp_q.get("pe")
        info["marketCap"] = fmp_q.get("marketCap")
        info["symbol"] = fmp_q.get("symbol")
        # some FMP fields might not map; combine with key metrics
    if fmp_km:
        # example fields in key-metrics/ratios might be different names
        info["trailingEps"] = fmp_km.get("epsBasic") or fmp_km.get("eps")
        # ROE often under "returnOnEquity" or similar
        info["returnOnEquity"] = fmp_km.get("returnOnEquity")
        info["debtToEquity"] = fmp_km.get("debtToEquity")
        info["bookValue"] = fmp_km.get("bookValuePerShare")
        info["earningsQuarterlyGrowth"] = fmp_km.get("earningsQuarterlyGrowth")
        # FMP doesn't always provide targetMeanPrice; we'll skip
    # Fetch history via yfinance (fast)
    try:
        ticker = yf.Ticker(symbol_ns)
        hist = ticker.history(period="5y", interval="1d")
        # If missing some fields, attempt to fill from yfinance info
        yf_info = ticker.info or {}
        # fill missing
        for k in ["currentPrice","trailingPE","trailingEps","returnOnEquity","debtToEquity","bookValue","marketCap","targetMeanPrice","earningsQuarterlyGrowth","priceToBook","dividendYield","shortName","longName"]:
            if k not in info or info.get(k) in (None, np.nan):
                info[k] = safe_get(yf_info, k, np.nan)
    except Exception:
        # if yf fails, fallback: leave hist empty and use fmp data if present
        pass

    return info, hist

# -------------------------
# Valuation helpers (same logic)
# -------------------------
def estimate_fair_value(info):
    # Try analyst target first
    try:
        target = safe_get(info, "targetMeanPrice", np.nan)
        if isinstance(target, (int, float)) and target > 0:
            return round(float(target), 2), "AnalystTarget"
    except Exception:
        pass
    # EPS x PE fallback
    eps = safe_get(info, "trailingEps", np.nan)
    forward_pe = safe_get(info, "forwardPE", np.nan)
    trailing_pe = safe_get(info, "trailingPE", np.nan)
    if isinstance(forward_pe, (int,float)) and forward_pe > 0 and forward_pe < 200:
        pe_target = forward_pe
    elif isinstance(trailing_pe, (int,float)) and trailing_pe > 0 and trailing_pe < 200:
        pe_target = max(10.0, trailing_pe * 0.9)
    else:
        pe_target = DEFAULT_PE_TARGET
    if isinstance(eps, (int,float)) and eps > 0:
        fv = eps * pe_target
        return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"
    # book value fallback
    book = safe_get(info, "bookValue", np.nan)
    if isinstance(book,(int,float)) and book > 0 and isinstance(trailing_pe,(int,float)) and trailing_pe>0:
        fv = book * trailing_pe
        return round(float(fv),2), "BVxPE"
    return None, "InsufficientData"

# -------------------------
# Rule-based recommendation (same as your logic)
# -------------------------
def rule_based_recommendation(info, fair_value, current_price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and abs(roe) > 1: roe = roe/100.0
    de = safe_get(info, "debtToEquity", np.nan)
    earnings_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)
    underval = None
    if fair_value and current_price and fair_value>0:
        underval = round(((fair_value - current_price)/fair_value)*100,2)
    score = 0
    reasons = []
    # ROE
    if isinstance(roe,(int,float)) and not math.isnan(roe):
        if roe >= 0.20:
            score +=3; reasons.append("High ROE")
        elif roe >= 0.12:
            score +=2; reasons.append("Good ROE")
        elif roe>0:
            score +=1; reasons.append("Positive ROE")
    # D/E
    if isinstance(de,(int,float)) and not math.isnan(de):
        if de <= 0.5:
            score +=2; reasons.append("Low D/E")
        elif de <= 1.5:
            score +=1; reasons.append("Moderate D/E")
    # Growth
    if isinstance(earnings_growth,(int,float)) and not math.isnan(earnings_growth):
        if earnings_growth >= 0.20:
            score +=2; reasons.append("Strong growth")
        elif earnings_growth >= 0.05:
            score +=1; reasons.append("Moderate growth")
    # Valuation boost
    if isinstance(underval,(int,float)):
        if underval >=25:
            score +=3; reasons.append("Deep undervaluation")
        elif underval >=10:
            score +=2; reasons.append("Undervalued")
        elif underval >=3:
            score +=1; reasons.append("Slight undervaluation")
    # Recommendation
    rec = "Hold"
    if score >=7 and (isinstance(underval,(int,float)) and underval >=10):
        rec = "Strong Buy"
    elif score >=5 and (isinstance(underval,(int,float)) and underval >=5):
        rec = "Buy"
    elif (isinstance(pe,(int,float)) and pe>80) or (isinstance(roe,(int,float)) and roe<0):
        rec = "Avoid"
    return {"score": score, "reasons": reasons, "undervaluation_%": underval, "recommendation": rec, "market_cap": market_cap}

# -------------------------
# Load NIFTY lists (attempt CSV else default samples)
# -------------------------
@st.cache_data
def load_index_list(name):
    # If user provided a CSV with symbols use it (nifty500.csv). Otherwise small default sets
    try:
        if name == "NIFTY500":
            df = pd.read_csv(NIFTY500_FILE, header=None)
            symbols = df[0].astype(str).str.strip().tolist()
            return [s.replace(".NS","").upper() for s in symbols if s]
    except Exception:
        pass
    # fallback small lists (expand later or provide CSV)
    samples = {
        "NIFTY50": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","BHARTIARTL","KOTAKBANK",
                    "AXISBANK","HINDUNILVR","ASIANPAINT","BAJAJ-AUTO".replace("-",""), "MARUTI".upper()],
        "NIFTY100": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","SUNPHARMA","ASIANPAINT","AXISBANK","MARUTI","ULTRACEMCO","HINDUNILVR"],
        "NIFTY500": ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","ITC","LT","SBIN","SUNPHARMA","ASIANPAINT","AXISBANK","MARUTI","ULTRACEMCO","HINDUNILVR","PERSISTENT","CIPLA","NTPC","ONGC","TATASTEEL","JSWSTEEL"]
    }
    return samples.get(name, [])

# -------------------------
# UI Tabs (6 tabs including Screener)
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor", "🧠 Stock Screener"
])

# -------------------------
# (TAB1..TAB5 code omitted here for brevity - assume your existing code remains unchanged)
# You already have Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor earlier in the file.
# Make sure to keep that code above this point in your final app.py.
# -------------------------

# -------------------------
# TAB6: Stock Screener (manual run)
# -------------------------
with tab6:
    st.header("🧠 Stock Screener (FMP-powered, NIFTY500)")
    st.caption("Run manually. Uses your FMP API key (st.secrets['FMP_API_KEY']). Falls back to yfinance if needed.")

    # Universe selector
    universe = st.selectbox("Universe", ["NIFTY50", "NIFTY100", "NIFTY500", "Custom List"], index=2)
    if universe == "Custom List":
        custom = st.text_area("Enter comma-separated symbols (no .NS):", value="")
        symbols = [s.strip().upper() for s in custom.split(",") if s.strip()]
    else:
        symbols = load_index_list(universe)

    st.write(f"Symbols loaded: {len(symbols)}")

    # Filters
    col1, col2 = st.columns(2)
    min_score = col1.slider("Min Score (rule-based)", 0, 10, 3)
    min_underv = col2.number_input("Min Undervaluation %", value=5.0, step=1.0)

    run_button = st.button("🚀 Run Screener (manual)")

    if run_button:
        if not symbols:
            st.warning("No symbols available for the chosen universe.")
        else:
            rows = []
            prog = st.progress(0)
            for i, sym in enumerate(symbols):
                # Fetch info (FMP preferred, yfinance fallback)
                info, _ = fetch_info_and_history(sym)
                if not info or info.get("error"):
                    prog.progress(int(((i+1)/len(symbols))*100))
                    time.sleep(MOCK_SLEEP)
                    continue

                ltp = safe_get(info, "currentPrice", np.nan)
                fv, fv_method = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)

                # Apply filters
                if (rec["score"] < min_score) or (rec["undervaluation_%"] is None) or (rec["undervaluation_%"] < min_underv):
                    prog.progress(int(((i+1)/len(symbols))*100))
                    time.sleep(MOCK_SLEEP)
                    continue

                cap = rec.get("market_cap") or 0
                cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
                rank_score = (rec["score"] * 2) + ((rec["undervaluation_%"] or 0) / 10) + cap_weight

                rows.append({
                    "Symbol": sym,
                    "LTP": round(ltp,2) if isinstance(ltp,(int,float)) and not math.isnan(ltp) else np.nan,
                    "Fair Value": fv,
                    "Underv%": rec["undervaluation_%"],
                    "Valuation Method": fv_method,
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"],
                    "RankScore": round(rank_score,2),
                    "Reasons": "; ".join(rec.get("reasons", []))
                })
                prog.progress(int(((i+1)/len(symbols))*100))
                time.sleep(MOCK_SLEEP)

            result_df = pd.DataFrame(rows)
            if result_df.empty:
                st.info("No stocks passed the filters.")
            else:
                result_df = result_df.sort_values(by="RankScore", ascending=False).reset_index(drop=True)
                st.dataframe(result_df, use_container_width=True)
                st.subheader("🏆 Top 10 Picks")
                st.table(result_df.head(10)[["Symbol","Rec","Score","Underv%","Buy Below","Fair Value"]])
                st.download_button("📥 Download Screener Results", result_df.to_csv(index=False).encode("utf-8"), "screener_results.csv", "text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(f"StockMentor — rule-based long-term stock helper. {datetime.now().year}")
