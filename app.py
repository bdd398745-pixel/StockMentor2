# app.py
"""
StockMentor — Rule-based Long-Term Stock Advisor (India)
Primary: Financial Modeling Prep (FMP) API (uses .BO for BSE)
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
import json
from datetime import datetime

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (FMP)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Primary: FMP (.BO) — Fallback: yfinance (.NS)")

# -------------------------
# Constants & Keys
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 15.0   # used by fair value heuristic when EPS available

FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))

# -------------------------
# Helpers: Load lists & watchlist
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
# Utility functions
# -------------------------
def safe_get(info, key, default=np.nan):
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if v in (None, "None", ""):
        return default
    return v

def estimate_fair_value(info):
    """
    Estimate fair value with heuristics:
      - EPS * DEFAULT_PE_TARGET if EPS present
      - P/E reversion to DEFAULT_PE_TARGET if PE present
      - P/B * price if PB present
      - fallback to price
    """
    try:
        price = float(safe_get(info, "price", safe_get(info, "currentPrice", np.nan) or np.nan))
        # attempt multiple key names
        pe = np.nan
        for k in ("trailingPE", "pe", "peRatio", "pe_ratio", "pe_ratio_mrq"):
            v = safe_get(info, k, np.nan)
            try:
                pe = float(v) if not pd.isna(v) else pe
                if not pd.isna(pe):
                    break
            except Exception:
                continue

        eps = np.nan
        for k in ("eps", "epsTTM", "trailingEps", "basic_eps"):
            v = safe_get(info, k, np.nan)
            try:
                eps = float(v) if not pd.isna(v) else eps
                if not pd.isna(eps):
                    break
            except Exception:
                continue

        pb = np.nan
        for k in ("priceToBook", "pb", "pb_ratio"):
            v = safe_get(info, k, np.nan)
            try:
                pb = float(v) if not pd.isna(v) else pb
                if not pd.isna(pb):
                    break
            except Exception:
                continue

        if not pd.isna(eps) and eps > 0:
            fv = eps * DEFAULT_PE_TARGET
            method = "EPS-based"
        elif not pd.isna(pe) and pe > 0 and not pd.isna(price):
            fv = (price / pe) * DEFAULT_PE_TARGET
            method = "P/E reversion"
        elif not pd.isna(pb) and not pd.isna(price):
            fv = pb * price
            method = "P/B heuristic"
        else:
            fv = price
            method = "fallback"

        return round(float(fv), 2) if not pd.isna(fv) else np.nan, method
    except Exception as e:
        print("estimate_fair_value error:", e)
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
        print("compute_buy_sell error:", e)
        return None, None

def rule_based_recommendation(info, fair_value, price):
    try:
        roe = safe_get(info, "returnOnEquity", np.nan)
        de = safe_get(info, "debtToEquity", np.nan)
        # coerce numeric strings
        try:
            roe = float(roe) if roe not in (None, "", "None", np.nan) else np.nan
        except:
            roe = np.nan
        try:
            de = float(de) if de not in (None, "", "None", np.nan) else np.nan
        except:
            de = np.nan

        underval = None
        if fair_value and price and not pd.isna(fair_value) and not pd.isna(price) and fair_value != 0:
            underval = round(((fair_value - price) / fair_value) * 100, 2)

        score = 0
        if isinstance(roe, (int, float)) and not pd.isna(roe):
            if roe >= 0.20:
                score += 3
            elif roe >= 0.12:
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
    except Exception as e:
        print("rule_based_recommendation error:", e)
        return {"score": 0, "recommendation": "Hold", "undervaluation": None}

# -------------------------
# FMP Fetcher (primary)
# -------------------------
@st.cache_data(ttl=900)
def fetch_fmp_data(symbol: str):
    """
    Attempt to fetch quote, profile, and 1-year history from FMP.
    For Indian stocks, prefer .BO (BSE). Try formats in order: .BO, .NS, raw.
    Returns (info_dict, hist_df) or ({}, pd.DataFrame()) on failure.
    """
    if not FMP_API_KEY:
        return {}, pd.DataFrame()

    base = "https://financialmodelingprep.com/api/v3"
    possible_symbols = [f"{symbol}.BO", f"{symbol}.NS", symbol]

    for sym_try in possible_symbols:
        try:
            quote_url = f"{base}/quote/{sym_try}?apikey={FMP_API_KEY}"
            profile_url = f"{base}/profile/{sym_try}?apikey={FMP_API_KEY}"
            hist_url = f"{base}/historical-price-full/{sym_try}?timeseries=365&apikey={FMP_API_KEY}"

            q_resp = requests.get(quote_url, timeout=10)
            p_resp = requests.get(profile_url, timeout=10)
            h_resp = requests.get(hist_url, timeout=15)

            # parse safely (some endpoints can return non-json)
            try:
                quote = q_resp.json()
            except Exception:
                quote = []
            try:
                profile = p_resp.json()
            except Exception:
                profile = []
            try:
                hist = h_resp.json()
            except Exception:
                hist = {}

            # normalize quote: sometimes list, sometimes dict
            q = None
            if isinstance(quote, list) and len(quote) > 0:
                q = quote[0]
            elif isinstance(quote, dict) and quote:
                # FMP may return a dict with keys like 'symbol' if single
                q = quote if "symbol" in quote else None

            if not q:
                continue

            p = None
            if isinstance(profile, list) and len(profile) > 0:
                p = profile[0]
            elif isinstance(profile, dict) and profile:
                p = profile if "symbol" in profile else None

            info = {
                "symbol": symbol,
                "exchange": "BSE" if sym_try.endswith(".BO") else ("NSE" if sym_try.endswith(".NS") else None),
                "currentPrice": q.get("price"),
                "price": q.get("price"),
                "marketCap": q.get("marketCap"),
                "trailingPE": q.get("pe") or q.get("trailingPE"),
                "priceToBook": q.get("pb"),
                "dividendYield": (q.get("lastDiv") or 0) / q.get("price") if q.get("price") else np.nan,
                "companyName": p.get("companyName") if isinstance(p, dict) else None,
                "sector": p.get("sector") if isinstance(p, dict) else None,
                "industry": p.get("industry") if isinstance(p, dict) else None,
                "beta": p.get("beta") if isinstance(p, dict) else None,
                "returnOnEquity": p.get("returnOnEquityTTM") if isinstance(p, dict) else None,
                "debtToEquity": p.get("debtToEquityTTM") if isinstance(p, dict) else None,
                "eps": p.get("epsTTM") if isinstance(p, dict) else None,
            }

            hist_df = pd.DataFrame()
            if isinstance(hist, dict) and "historical" in hist:
                try:
                    hist_df = pd.DataFrame(hist["historical"])
                    hist_df = hist_df.rename(columns={"date": "Date", "close": "Close"})
                    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                    hist_df["Close"] = hist_df["Close"].astype(float)
                    hist_df = hist_df.sort_values("Date")
                except Exception:
                    hist_df = pd.DataFrame()

            # Ensure we actually have a price
            if info.get("price") is None and info.get("currentPrice") is None:
                continue

            return info, hist_df

        except Exception as e:
            # try next format
            continue

    # nothing found on FMP
    return {}, pd.DataFrame()

# -------------------------
# yfinance Fetcher (fallback)
# -------------------------
@st.cache_data(ttl=300)
def fetch_info_and_history(symbol: str):
    """
    Use yfinance as a robust fallback. Expects symbol without suffix; will query SYMBOL.NS.
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info or {}
        hist = ticker.history(period="1y", auto_adjust=False)
        if not hist.empty:
            hist = hist.reset_index()
            # unify Close column if needed
            if "Close" not in hist.columns and "close" in hist.columns:
                hist.rename(columns={"close": "Close"}, inplace=True)
        else:
            hist = pd.DataFrame()
        # normalize 'price' key for compatibility
        if "currentPrice" in info and "price" not in info:
            info["price"] = info.get("currentPrice")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dashboard", "🔎 Single Stock", "💼 Portfolio",
    "📣 Alerts", "🧾 Watchlist Editor", "🧠 Stock Screener"
])

# -------------------------
# Dashboard (Watchlist)
# -------------------------
with tab1:
    st.header("📋 Watchlist Dashboard")
    watchlist = load_watchlist()
    if not watchlist:
        st.info("Watchlist empty. Add symbols in Watchlist Editor.")
    else:
        use_fmp_dashboard = st.checkbox("Prefer FMP (BSE .BO) if available", value=True)
        if st.button("🔍 Analyze Watchlist"):
            rows = []
            progress = st.progress(0)
            for i, sym in enumerate(watchlist):
                info, hist = ({}, pd.DataFrame())
                if use_fmp_dashboard and FMP_API_KEY:
                    info, hist = fetch_fmp_data(sym)
                if not info:
                    info, hist = fetch_info_and_history(sym)
                ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
                fv, _ = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)
                rows.append({
                    "Symbol": sym,
                    "LTP": round(ltp,2) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else None,
                    "Fair Value": fv,
                    "Underv%": rec["undervaluation"],
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"]
                })
                progress.progress(int(((i + 1) / len(watchlist)) * 100))
            df_out = pd.DataFrame(rows)
            df_sorted = df_out.sort_values(by="Score", ascending=False)
            st.dataframe(df_sorted, use_container_width=True)
            st.success("✅ Ranked by multi-factor score (Quality + Valuation)")

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    use_fmp = st.checkbox("Use FMP API (recommended for fundamentals)", value=True)
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS)").upper().strip()
    if st.button("Analyze Stock") and symbol:
        info, hist = ({}, pd.DataFrame())
        source_used = "none"
        if use_fmp and FMP_API_KEY:
            info, hist = fetch_fmp_data(symbol)
            source_used = "FMP" if info else "FMP (no data)"
        if not info:
            info, hist = fetch_info_and_history(symbol)
            source_used = "yfinance" if info else source_used
        if not info:
            st.error("No data found from FMP or yfinance.")
        else:
            price = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)

            cols = st.columns(4)
            cols[0].metric("Current Price (LTP)", f"{round(price,2) if isinstance(price,(int,float)) and not pd.isna(price) else 'N/A'}")
            cols[1].metric("Fair Value", f"{fv if not pd.isna(fv) else 'N/A'}", method)
            cols[2].metric("Recommendation", rec["recommendation"], f"Score: {rec['score']}")
            cols[3].metric("Undervaluation %", f"{rec['undervaluation'] if rec['undervaluation'] is not None else 'N/A'}", f"Source: {source_used}")

            st.subheader("Info (partial)")
            try:
                info_display = {k: (round(v,4) if isinstance(v,(int,float)) and not pd.isna(v) else v) for k,v in info.items()}
            except Exception:
                info_display = info
            st.json(info_display)

            if hist is not None and not hist.empty:
                try:
                    st.line_chart(hist.set_index("Date")["Close"], height=300)
                except Exception:
                    st.info("Historical chart unavailable.")

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
            pf.columns = [c.lower().strip() for c in pf.columns]
            required = {"symbol", "buy_price", "quantity"}
            if not required.issubset(set(pf.columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity")
            else:
                rows = []
                progress = st.progress(0)
                for i, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    info, hist = ({}, pd.DataFrame())
                    if FMP_API_KEY:
                        info, hist = fetch_fmp_data(sym)
                    if not info:
                        info, hist = fetch_info_and_history(sym)
                    ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
                    current_value = (ltp * qty) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else np.nan
                    invested = buy * qty
                    pl = current_value - invested if not pd.isna(current_value) and not pd.isna(invested) else np.nan
                    pl_pct = (pl / invested) * 100 if invested else np.nan
                    rows.append({
                        "Symbol": sym,
                        "Buy Price": round(buy,2),
                        "Qty": qty,
                        "LTP": round(ltp,2) if isinstance(ltp,(int,float)) and not pd.isna(ltp) else None,
                        "Current Value": round(current_value,2) if not pd.isna(current_value) else None,
                        "Invested": round(invested,2),
                        "P/L": round(pl,2) if not pd.isna(pl) else None,
                        "P/L %": round(pl_pct,2) if not pd.isna(pl_pct) else None
                    })
                    progress.progress(int(((i+1)/len(pf))*100))
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error reading portfolio: " + str(e))

# -------------------------
# Alerts (manual)
# -------------------------
with tab4:
    st.header("📣 Alerts (Manual)")
    st.info("Manual alert scaffold — wire SMTP for automated sending.")
    threshold = st.number_input("Undervaluation % threshold (flag if underval >= this)", value=10.0, step=1.0)
    up = st.file_uploader("Watchlist CSV (one symbol per line)", type=["csv"])
    if up:
        try:
            wl = pd.read_csv(up, header=None)[0].astype(str).str.strip().str.upper().tolist()
            flagged = []
            progress = st.progress(0)
            for i, s in enumerate(wl):
                info, hist = ({}, pd.DataFrame())
                if FMP_API_KEY:
                    info, hist = fetch_fmp_data(s)
                if not info:
                    info, hist = fetch_info_and_history(s)
                fv, _ = estimate_fair_value(info)
                ltp = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
                rec = rule_based_recommendation(info, fv, ltp)
                if rec["undervaluation"] is not None and rec["undervaluation"] >= threshold:
                    flagged.append({"symbol": s, "underv%": rec["undervaluation"], "rec": rec["recommendation"]})
                progress.progress(int(((i+1)/len(wl))*100))
            if flagged:
                st.success(f"Found {len(flagged)} flagged symbols")
                st.dataframe(pd.DataFrame(flagged), use_container_width=True)
            else:
                st.info("No symbols flagged.")
        except Exception as e:
            st.error("Error processing file: " + str(e))

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
# Screener (Nifty)
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty 500 (first N sample)")
    use_fmp_screener = st.checkbox("Use FMP (recommended)", value=True)
    nifty = load_nifty_stocks()
    max_scan = st.number_input("Max symbols to scan (first N)", min_value=10, max_value=500, value=100, step=10)
    if st.button("Run Screener"):
        rows = []
        progress = st.progress(0)
        symbols_to_scan = nifty["Symbol"].tolist()[:max_scan] if not nifty.empty else []
        if not symbols_to_scan:
            st.info("Nifty 500 list not loaded; upload your own list to screen.")
        for i, sym in enumerate(symbols_to_scan):
            info, hist = ({}, pd.DataFrame())
            if use_fmp_screener and FMP_API_KEY:
                info, hist = fetch_fmp_data(sym)
            if not info:
                info, hist = fetch_info_and_history(sym)
            price = safe_get(info, "price", safe_get(info, "currentPrice", np.nan))
            fv, _ = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": sym,
                "LTP": round(price,2) if isinstance(price,(int,float)) and not pd.isna(price) else None,
                "FairValue": fv,
                "Undervaluation%": rec["undervaluation"],
                "Score": rec["score"],
                "Recommendation": rec["recommendation"]
            })
            progress.progress(int(((i + 1) / max(1, len(symbols_to_scan))) * 100))
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)
            st.success("✅ Screener completed successfully")
        else:
            st.info("Screener returned no results.")

# -------------------------
# Footer
# -------------------------
st.caption("Made by Biswanath 🔍 | FMP primary (.BO), yfinance fallback (.NS)")
