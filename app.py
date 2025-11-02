# app.py
"""
StockMentor - Rule-based Long-Term Stock Analyst (India)
- FMP API (optional via Streamlit secrets) with yfinance fallback
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, Screener
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import math
import smtplib
from email.message import EmailMessage
from datetime import datetime
import os
from typing import Tuple

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="StockMentor (Enhanced)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("Now with optional FMP API + yfinance fallback & Nifty screener")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))

# -------------------------
# Utilities / Helpers
# -------------------------
@st.cache_data
def load_nifty_stocks() -> pd.DataFrame:
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        # Some files use different column names; normalize
        if "Symbol" not in df.columns:
            df.columns = [c.strip() for c in df.columns]
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        # keep company name if present
        cols = [c for c in ["Symbol", "Company Name", "Company", "Industry"] if c in df.columns]
        if "Company Name" in df.columns:
            return df[["Symbol", "Company Name", "Industry"]].rename(columns={"Company Name": "CompanyName"})
        elif "Company" in df.columns:
            return df[["Symbol", "Company", "Industry"]].rename(columns={"Company": "CompanyName"})
        else:
            return df[["Symbol"]].assign(CompanyName="")
    except Exception as e:
        st.warning(f"Could not load Nifty list: {e}")
        return pd.DataFrame(columns=["Symbol", "CompanyName", "Industry"])


@st.cache_data
def load_watchlist() -> list:
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        return df[0].astype(str).str.strip().str.upper().tolist()
    except Exception:
        return []


def save_watchlist(symbols: list) -> Tuple[bool, str]:
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        return True, "Saved"
    except Exception as e:
        return False, str(e)


def add_ns_if_missing(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO") or s.endswith(".NSI"):
        return s
    return s + ".NS"


# -------------------------
# FMP Fetcher (Indian symbols supported with .NS)
# -------------------------
def fetch_fmp_data(symbol: str):
    """
    Returns: (info_dict, hist_df)
    info_dict: normalized keys similar to yfinance-like keys used in app
    hist_df: DataFrame with columns Date (datetime) and Close
    """
    if not FMP_API_KEY:
        return {}, pd.DataFrame()

    try:
        sym_ns = add_ns_if_missing(symbol)
        base = "https://financialmodelingprep.com/api/v3"

        # Quote (current price)
        quote_url = f"{base}/quote/{sym_ns.replace('.NS', '')}?apikey={FMP_API_KEY}"
        profile_url = f"{base}/profile/{sym_ns.replace('.NS', '')}?apikey={FMP_API_KEY}"
        hist_url = f"{base}/historical-price-full/{sym_ns.replace('.NS', '')}?timeseries=365&apikey={FMP_API_KEY}"

        q_res = requests.get(quote_url, timeout=8).json()
        p_res = requests.get(profile_url, timeout=8).json()
        h_res = requests.get(hist_url, timeout=8).json()

        # Normalize
        if isinstance(q_res, list):
            q = q_res[0] if q_res else {}
        elif isinstance(q_res, dict):
            # sometimes FMP returns dict on error
            q = q_res
        else:
            q = {}

        if isinstance(p_res, list):
            p = p_res[0] if p_res else {}
        elif isinstance(p_res, dict):
            p = p_res
        else:
            p = {}

        info = {
            "symbol": symbol.upper(),
            "currentPrice": q.get("price") or q.get("price"),
            "marketCap": q.get("marketCap") or p.get("mktCap") or q.get("marketCap"),
            "trailingPE": q.get("pe") or p.get("pe"),
            "priceToBook": q.get("pb") or p.get("pb"),
            "dividendYield": q.get("dividendYield") or p.get("lastDiv") and (q.get("price") and (q.get("lastDiv") / q.get("price"))),
            "companyName": p.get("companyName") or p.get("company"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "beta": p.get("beta"),
            "returnOnEquity": p.get("returnOnEquityTTM") or p.get("returnOnEquity"),
            "debtToEquity": p.get("debtToEquityTTM") or p.get("debtToEquity"),
            "eps": p.get("epsTTM") or p.get("eps")
        }

        hist_df = pd.DataFrame()
        if isinstance(h_res, dict) and "historical" in h_res:
            hist_df = pd.DataFrame(h_res["historical"])
            if not hist_df.empty:
                hist_df = hist_df.rename(columns={"date": "Date", "close": "Close"})
                hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                hist_df = hist_df.sort_values("Date").reset_index(drop=True)
        return info, hist_df
    except Exception as e:
        st.warning(f"FMP fetch error for {symbol}: {e}")
        return {}, pd.DataFrame()


# -------------------------
# yfinance fallback
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol: str):
    try:
        ns_symbol = add_ns_if_missing(symbol)
        ticker = yf.Ticker(ns_symbol)
        info = ticker.info or {}
        hist = ticker.history(period="1y")
        # normalize hist to Date and Close if needed
        if not hist.empty:
            hist = hist.reset_index().rename(columns={"Date": "Date", "Close": "Close"})
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()


# -------------------------
# Small helpers: buy/sell, safe_get, valuation, rules
# -------------------------
def safe_get(info, key, default=np.nan):
    try:
        v = info.get(key, default)
    except Exception:
        return default
    return default if v in (None, "None", "") else v


import numpy as np

def estimate_fair_value(info):
    """
    Estimate fair value of a stock based on simple valuation logic.
    Always returns a tuple: (fair_value, method)
    """
    try:
        price = float(info.get("price", np.nan))
        pe = float(info.get("pe", np.nan))
        eps = float(info.get("eps", np.nan))
        pb = float(info.get("pb", np.nan))
        roe = float(info.get("roeTTM", np.nan))

        # --- if enough data available, use a simple DCF-like estimate ---
        if not np.isnan(eps) and eps > 0:
            # assume moderate 10% annual growth and 10% discount
            fair_value = eps * (1 + 0.10) * 15
            method = "EPS-based DCF"
        elif not np.isnan(pb) and not np.isnan(roe):
            # fall back to P/B * ROE heuristic
            fair_value = price * (roe / 10 if roe > 0 else 1)
            method = "ROE-P/B heuristic"
        elif not np.isnan(pe) and pe > 0:
            # fallback using sector average PE = 15
            fair_value = (price / pe) * 15
            method = "P/E reversion"
        else:
            # last resort — return current price as fair value
            fair_value = price
            method = "No data fallback"

        return fair_value, method

    except Exception as e:
        # if anything goes wrong, still return a safe tuple
        print(f"Fair value estimation error: {e}")
        return np.nan, "error"



def compute_buy_sell(fv, mos=0.25):
    if fv is None or (isinstance(fv, float) and math.isnan(fv)):
        return None, None
    buy = round(fv * (1 - mos), 2)
    sell = round(fv * (1 + mos / 1.5), 2)
    return buy, sell


def rule_based_recommendation(info, fair_value, price):
    """
    Returns dict with:
    - score (int)
    - recommendation str
    - undervaluation (float %)
    - market_cap
    """
    roe = safe_get(info, "returnOnEquity", np.nan)
    # If FMP returns in percent form (e.g., 12.5) assume it's percent; if returned as 0.12 we keep it
    if isinstance(roe, (int, float)) and abs(roe) > 1:
        roe = roe / 100.0

    de = safe_get(info, "debtToEquity", np.nan)
    growth = safe_get(info, "eps", np.nan)  # a rough proxy
    try:
        market_cap = safe_get(info, "marketCap", np.nan)
    except Exception:
        market_cap = np.nan

    underval = None
    try:
        if fair_value and price and fair_value > 0:
            underval = round(((fair_value - price) / fair_value) * 100, 2)
    except Exception:
        underval = None

    score = 0
    # ROE scoring
    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 3
        elif roe >= 0.12:
            score += 2

    # DE scoring
    if isinstance(de, (int, float)):
        if de <= 0.5:
            score += 2
        elif de <= 1.5:
            score += 1

    # undervaluation scoring
    if isinstance(underval, (int, float)):
        if underval >= 25:
            score += 3
        elif underval >= 10:
            score += 2
        elif underval >= 3:
            score += 1

    # final rec
    rec = "Hold"
    if score >= 7:
        rec = "Strong Buy"
    elif score >= 5:
        rec = "Buy"
    elif isinstance(de, (int, float)) and de > 2:
        rec = "Avoid"

    return {
        "score": score,
        "recommendation": rec,
        "undervaluation": underval,
        "roe": roe,
        "de": de,
        "growth": growth,
        "market_cap": market_cap
    }


# -------------------------
# Simple SMTP sender for Alerts
# -------------------------
def send_email_smtp(host, port, username, password, sender, recipients_csv, subject, body):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender or username
        msg["To"] = [r.strip() for r in recipients_csv.split(",") if r.strip()]
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        return True, "OK"
    except Exception as e:
        return False, str(e)


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
                # prefer FMP if available
                info, hist = fetch_fmp_data(sym) if FMP_API_KEY else fetch_info_and_history(sym)
                if not info:
                    continue
                ltp = safe_get(info, "currentPrice", np.nan)
                fv = estimate_fair_value(info)
                rec = rule_based_recommendation(info, fv, ltp)
                buy, sell = compute_buy_sell(fv)
                mcap = rec.get("market_cap", np.nan)
                cap_weight = 2 if isinstance(mcap, (int, float)) and mcap and mcap > 5e11 else (1 if isinstance(mcap, (int, float)) and mcap and mcap > 1e11 else 0)
                underv = rec.get("undervaluation", np.nan)
                rank_score = (rec["score"] * 2) + (underv or 0)/10 + cap_weight

                rows.append({
                    "Symbol": sym,
                    "LTP": ltp,
                    "Fair Value": fv,
                    "Underv%": underv,
                    "Buy Below": buy,
                    "Sell Above": sell,
                    "Rec": rec["recommendation"],
                    "Score": rec["score"],
                    "RankScore": round(rank_score, 2)
                })
                progress.progress(int(((i + 1) / len(watchlist)) * 100))
            df = pd.DataFrame(rows).sort_values("RankScore", ascending=False)
            st.dataframe(df, use_container_width=True)
            st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Analysis")
    use_fmp = st.checkbox("Use FMP API (recommended if you set FMP_API_KEY)", value=bool(FMP_API_KEY))
    symbol = st.text_input("Enter stock symbol (e.g., RELIANCE, TCS, HDFCBANK):").upper().strip()

    if st.button("Analyze Stock") and symbol:
        if use_fmp and FMP_API_KEY:
            info, hist = fetch_fmp_data(symbol)
        else:
            info, hist = fetch_info_and_history(symbol)

        if not info:
            st.error("No data found for this symbol.")
        else:
            price = safe_get(info, "currentPrice", safe_get(info, "price", np.nan))
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            buy, sell = compute_buy_sell(fv)

            # Overview
            st.subheader(f"{info.get('companyName', symbol)} — {symbol}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("LTP", f"{price:,}" if isinstance(price, (int, float)) else price)
            c2.metric("Fair Value", f"{fv:,}" if isinstance(fv, (int, float)) and not math.isnan(fv) else "—")
            c3.metric("Recommendation", rec["recommendation"])
            c4.metric("Score", rec["score"])

            st.write("**Key fundamentals**")
            basics = {
                "ROE": rec.get("roe"),
                "Debt/Equity": rec.get("de"),
                "EPS (approx)": safe_get(info, "eps"),
                "PE": safe_get(info, "trailingPE") or safe_get(info, "pe"),
                "Market Cap": safe_get(info, "marketCap")
            }
            st.json(basics)

            if not hist.empty:
                # Ensure Date column name is standardized
                if "Date" in hist.columns:
                    xdf = hist.set_index("Date").sort_index()
                elif "date" in hist.columns:
                    xdf = hist.rename(columns={"date": "Date"}).set_index("Date").sort_index()
                elif "DateTime" in hist.columns:
                    xdf = hist.rename(columns={"DateTime": "Date"}).set_index("Date").sort_index()
                else:
                    xdf = hist.copy()

                if "Close" in xdf.columns:
                    st.write("### Price Trend (1 year)")
                    st.line_chart(xdf["Close"])
                else:
                    st.info("Price history available but 'Close' column missing.")
            else:
                st.info("No historical price data available.")

            st.write("**Buy / Sell guidance**")
            st.write(f"- Buy Below: {buy}")
            st.write(f"- Sell Above: {sell}")

# -------------------------
# Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    st.markdown("Upload CSV (columns: symbol, buy_price, quantity). Symbols should be without '.NS' (e.g., RELIANCE).")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf_columns = [c.lower() for c in pf.columns]
            if not set(["symbol", "buy_price", "quantity"]).issubset(set(pf_columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity (case-insensitive)")
            else:
                pf.columns = pf_columns
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    # use FMP if available else yfinance
                    info, _ = (fetch_fmp_data(sym) if FMP_API_KEY else fetch_info_and_history(sym))
                    ltp = safe_get(info, "currentPrice", safe_get(info, "price", np.nan))
                    current_value = round((ltp * qty), 2) if isinstance(ltp, (int, float)) and not math.isnan(ltp) else None
                    invested = round(buy * qty, 2)
                    pl = round((current_value - invested), 2) if current_value is not None else None
                    pl_pct = round((pl / invested * 100), 2) if (pl is not None and invested != 0) else None
                    rows.append({
                        "symbol": sym,
                        "buy_price": buy,
                        "quantity": qty,
                        "ltp": ltp,
                        "current_value": current_value,
                        "invested": invested,
                        "P/L": pl,
                        "P/L%": pl_pct
                    })
                out = pd.DataFrame(rows)
                st.dataframe(out, use_container_width=True)
                total_pl = out["P/L"].sum(skipna=True)
                st.metric("Total P/L (₹)", f"{total_pl:,.2f}")
        except Exception as e:
            st.error("Error reading portfolio: " + str(e))

# -------------------------
# Alerts (Email)
# -------------------------
with tab4:
    st.header("📣 Email Alerts (manual send)")
    st.write("This sends immediate email(s). For Gmail, use smtp.gmail.com port 587 and an App Password.")
    with st.form("alert_form"):
        smtp_host = st.text_input("SMTP host", value="smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", value=587)
        smtp_user = st.text_input("SMTP username (email)")
        smtp_pass = st.text_input("SMTP password (app password recommended)", type="password")
        sender = st.text_input("From (optional)", value=smtp_user)
        recipients = st.text_input("Recipients (comma separated)")
        underv_threshold = st.number_input("Send alerts when undervaluation% >= ", value=10)
        submit_alert = st.form_submit_button("Send Alerts Now")

    if submit_alert:
        if not smtp_user or not smtp_pass or not recipients:
            st.error("Provide SMTP username/password and recipient(s).")
        else:
            results = []
            wl = load_watchlist()
            for sym in wl:
                info, _ = (fetch_fmp_data(sym) if FMP_API_KEY else fetch_info_and_history(sym))
                if not info:
                    continue
                ltp = safe_get(info, "currentPrice", safe_get(info, "price", np.nan))
                fv = estimate_fair_value(info)
                underv = None
                if isinstance(fv, (int, float)) and isinstance(ltp, (int, float)) and fv > 0:
                    underv = round(((fv - ltp) / fv) * 100, 2)
                if isinstance(underv, (int, float)) and underv >= underv_threshold:
                    results.append(f"{sym}: LTP ₹{ltp} | Fair ₹{fv} | Underv {underv}%")

            if not results:
                st.info("No stocks passed the threshold.")
            else:
                body = "StockMentor alerts:\n\n" + "\n".join(results) + f"\n\nGenerated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ok, msg = send_email_smtp(smtp_host, int(smtp_port), smtp_user, smtp_pass, sender, recipients, "StockMentor Alerts", body)
                if ok:
                    st.success("Alerts sent successfully.")
                else:
                    st.error("Failed to send alerts: " + msg)

# -------------------------
# Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    st.write("Edit your watchlist (one symbol per line). Use NSE tickers (without .NS).")
    current = load_watchlist()
    new_txt = st.text_area("Watchlist", value="\n".join(current), height=300)
    if st.button("💾 Save watchlist"):
        new_list = [s.strip().upper() for s in new_txt.splitlines() if s.strip()]
        ok, msg = save_watchlist(new_list)
        if ok:
            st.success("Watchlist saved. Reload Dashboard to analyze.")
        else:
            st.error("Save failed: " + msg)

# -------------------------
# Screener
# -------------------------
with tab6:
    st.header("🧠 Stock Screener — Nifty Universe")
    use_fmp_for_screener = st.checkbox("Use FMP API (recommended)", value=bool(FMP_API_KEY))
    nifty = load_nifty_stocks()
    max_symbols = st.number_input("Max symbols to screen (for demo)", value=100, min_value=10, max_value=500)
    if st.button("Run Screener"):
        rows = []
        syms = nifty["Symbol"].tolist()[:int(max_symbols)]
        progress = st.progress(0)
        for i, sym in enumerate(syms):
            info, hist = (fetch_fmp_data(sym) if (use_fmp_for_screener and FMP_API_KEY) else fetch_info_and_history(sym))
            if not info:
                progress.progress(int(((i + 1) / len(syms)) * 100))
                continue
            price = safe_get(info, "currentPrice", safe_get(info, "price", np.nan))
            fv = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, price)
            rows.append({
                "Symbol": sym,
                "LTP": price,
                "FairValue": fv,
                "Underv%": rec.get("undervaluation"),
                "Score": rec.get("score"),
                "Recommendation": rec.get("recommendation")
            })
            progress.progress(int(((i + 1) / len(syms)) * 100))

        if not rows:
            st.warning("No stocks passed screening or data fetch failed.")
        else:
            df = pd.DataFrame(rows).sort_values("Score", ascending=False)
            st.dataframe(df, use_container_width=True)
            st.success("✅ Screener completed and ranked by score")

st.caption("Made by Biswanath 🔍 | Rule-based, API-optional, Fully Offline-capable")
