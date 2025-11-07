# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- Rule-based scoring, ranking & recommendation
Author: Biswanath Das
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
from email.message import EmailMessage
from datetime import datetime
import math
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="StockMentor (Rule-based)", page_icon="📈", layout="wide")
st.title("📈 StockMentor — Rule-based Long-Term Advisor (India)")
st.caption("No OpenAI. Pure rule-based valuation, ranking, and recommendations.")

# -------------------------
# Constants
# -------------------------
WATCHLIST_FILE = "watchlist.csv"
DEFAULT_PE_TARGET = 20.0
DISCOUNT_RATE = 0.10
MOCK_SLEEP = 0.02

# -------------------------
# Load/save watchlist
# -------------------------
@st.cache_data
def load_watchlist():
    try:
        df = pd.read_csv(WATCHLIST_FILE, header=None)
        symbols = df[0].astype(str).str.strip().tolist()
        symbols = [s.replace(".NS", "").strip().upper() for s in symbols if s and str(s).strip()]
        return symbols
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading {WATCHLIST_FILE}: {e}")
        return []

def save_watchlist(symbols):
    try:
        pd.DataFrame(symbols).to_csv(WATCHLIST_FILE, index=False, header=False)
        try:
            load_watchlist.clear()
        except Exception:
            pass
        return True, "Saved"
    except Exception as e:
        return False, str(e)

# -------------------------
# Data fetch
# -------------------------
@st.cache_data(ttl=900)
def fetch_info_and_history(symbol_no_suffix):
    symbol = f"{symbol_no_suffix}.NS"
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="5y", interval="1d")
        return info, hist
    except Exception as e:
        return {"error": str(e)}, pd.DataFrame()

def safe_get(info, key, default=np.nan):
    v = info.get(key, default)
    return default if v in (None, "None", "") else v

# -------------------------
# Compute ROE and D/E manually
# -------------------------
@st.cache_data(ttl=3600)
def compute_roe_de(symbol):
    """Compute ROE% and D/E using financial statements instead of yfinance.info"""
    try:
        t = yf.Ticker(f"{symbol}.NS")
        fin = t.financials
        bs = t.balance_sheet

        if fin is None or bs is None or fin.empty or bs.empty:
            return np.nan, np.nan

        net_income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else np.nan
        total_assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else np.nan
        total_liab = bs.loc["Total Liab"].iloc[0] if "Total Liab" in bs.index else np.nan

        equity = total_assets - total_liab if all(isinstance(x, (int, float)) for x in [total_assets, total_liab]) else np.nan

        roe = (net_income / equity) * 100 if isinstance(net_income, (int, float)) and isinstance(equity, (int, float)) and equity != 0 else np.nan
        debt_eq = (total_liab / equity) if isinstance(total_liab, (int, float)) and isinstance(equity, (int, float)) and equity != 0 else np.nan

        return round(roe, 2), round(debt_eq, 2)
    except Exception:
        return np.nan, np.nan

# -------------------------
# Fair Value Estimation
# -------------------------
def estimate_fair_value(info):
    try:
        target = safe_get(info, "targetMeanPrice", np.nan)
        if isinstance(target, (int, float)) and target > 0:
            return round(float(target), 2), "AnalystTarget"
    except Exception:
        pass

    eps = safe_get(info, "trailingEps", np.nan)
    trailing_pe = safe_get(info, "trailingPE", np.nan)

    pe_target = trailing_pe if isinstance(trailing_pe, (int, float)) and trailing_pe > 0 else DEFAULT_PE_TARGET
    if isinstance(eps, (int, float)) and eps > 0:
        fv = eps * pe_target
        return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"

    return None, "InsufficientData"

# -------------------------
# Buy/Sell price zones
# -------------------------
def compute_buy_sell(fair_value, mos=0.30):
    if fair_value is None or (isinstance(fair_value, float) and math.isnan(fair_value)):
        return None, None
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos / 1.5), 2)

# -------------------------
# Rule-based recommendation
# -------------------------
def rule_based_recommendation(info, fair_value, current_price):
    score = 0
    reasons = []

    roe = safe_get(info, "returnOnEquity", np.nan)
    de = safe_get(info, "debtToEquity", np.nan)
    cur_ratio = safe_get(info, "currentRatio", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)
    net_margin = safe_get(info, "profitMargins", np.nan)
    eps_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    sales_growth = safe_get(info, "revenueGrowth", np.nan)
    beta = safe_get(info, "beta", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)

    underv = None
    try:
        if fair_value and current_price and fair_value > 0:
            underv = round(((fair_value - current_price) / fair_value) * 100, 2)
    except Exception:
        underv = None

    if isinstance(de, (int, float)) and de < 1: score += 10; reasons.append("Healthy D/E (<1)")
    if isinstance(roe, (int, float)) and roe > 15: score += 10; reasons.append("ROE >15%")
    if isinstance(net_margin, (int, float)) and net_margin > 0.1: score += 10; reasons.append("Good Margin (>10%)")
    if isinstance(sales_growth, (int, float)) and sales_growth > 0.1: score += 10; reasons.append("Sales Growth >10%")
    if isinstance(eps_growth, (int, float)) and eps_growth > 0.1: score += 10; reasons.append("EPS Growth >10%")
    if isinstance(pe, (int, float)) and pe < 25: score += 10; reasons.append("Reasonable P/E (<25)")
    if isinstance(underv, (int, float)) and underv > 10: score += 10; reasons.append("Undervalued >10%")
    if isinstance(beta, (int, float)) and beta < 1: score += 10; reasons.append("Low Volatility (β<1)")

    final_score = min(score, 100)
    rec = "Hold"
    if final_score >= 85: rec = "Strong Buy"
    elif final_score >= 70: rec = "Buy"
    elif final_score < 55: rec = "Avoid"

    return {"score": final_score, "reasons": reasons, "undervaluation_%": underv, "recommendation": rec, "market_cap": market_cap}

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor", "🏆 RJ Score"])

# -------------------------
# TAB 1: Dashboard
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
            info, _ = fetch_info_and_history(sym)
            if info.get("error"):
                continue
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, _ = estimate_fair_value(info)
            roe, debt_eq = compute_roe_de(sym)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)
            rows.append({
                "Symbol": sym,
                "LTP": ltp,
                "Fair Value": fv,
                "ROE%": roe,
                "D/E": debt_eq,
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec["recommendation"],
                "Score": rec["score"]
            })
            progress.progress(int(((i + 1) / len(watchlist)) * 100))
            time.sleep(MOCK_SLEEP)
        st.dataframe(pd.DataFrame(rows).sort_values(by="Score", ascending=False), use_container_width=True)

# -------------------------
# TAB 2: Single Stock
# -------------------------
with tab2:
    st.header("🔎 Single Stock Detail")
    watchlist = load_watchlist()
    sel = st.selectbox("Select stock", watchlist) if watchlist else st.text_input("Enter symbol (e.g., RELIANCE)")
    if sel:
        info, hist = fetch_info_and_history(sel)
        if info.get("error"):
            st.error("Data fetch error: " + info.get("error"))
        else:
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, _ = estimate_fair_value(info)
            roe, debt_eq = compute_roe_de(sel)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)
            c1, c2, c3 = st.columns(3)
            c1.metric("LTP", f"₹{ltp:.2f}" if isinstance(ltp, (int, float)) else "-")
            c2.metric("Fair Value", f"₹{fv}" if fv else "-")
            c3.metric("Recommendation", rec["recommendation"])
            st.write("**ROE%**:", roe)
            st.write("**Debt/Equity**:", debt_eq)
            st.write("**Buy Below:**", buy)
            st.write("**Sell Above:**", sell)
            st.write("**Reasons:**", ", ".join(rec["reasons"]))
            if hist is not None and not hist.empty:
                st.line_chart(hist["Close"])

# -------------------------
# TAB 3: Portfolio
# -------------------------
with tab3:
    st.header("💼 Portfolio Tracker")
    uploaded = st.file_uploader("Upload CSV (columns: symbol, buy_price, quantity)")
    if uploaded:
        try:
            pf = pd.read_csv(uploaded)
            pf.columns = [c.lower() for c in pf.columns]
            if not set(["symbol", "buy_price", "quantity"]).issubset(set(pf.columns)):
                st.error("CSV must contain columns: symbol, buy_price, quantity")
            else:
                rows = []
                for _, r in pf.iterrows():
                    sym = str(r["symbol"]).strip().upper()
                    buy = float(r["buy_price"])
                    qty = float(r["quantity"])
                    info, _ = fetch_info_and_history(sym)
                    ltp = safe_get(info, "currentPrice", np.nan)
                    if isinstance(ltp, (int, float)):
                        invested = buy * qty
                        current_value = ltp * qty
                        pnl = current_value - invested
                        rows.append({"Symbol": sym, "Qty": qty, "Buy": buy, "LTP": ltp, "P/L": round(pnl, 2)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------
# TAB 4: Alerts
# -------------------------
with tab4:
    st.header("📣 Email Alerts")
    st.info("Set up basic SMTP alerts manually via script integration. (Feature placeholder)")

# -------------------------
# TAB 5: Watchlist Editor
# -------------------------
with tab5:
    st.header("🧾 Watchlist Editor")
    symbols = st.text_area("Enter stock symbols (one per line)", "\n".join(load_watchlist()))
    if st.button("💾 Save Watchlist"):
        syms = [s.strip().upper() for s in symbols.split("\n") if s.strip()]
        ok, msg = save_watchlist(syms)
        st.success("Saved successfully!" if ok else msg)

# -------------------------
# TAB 6: RJ Score
# -------------------------
with tab6:
    st.header("🏆 RJ Score Calculator")
    roe = st.number_input("ROE (%)", 0, 100, 15)
    de = st.number_input("Debt-Equity", 0.0, 10.0, 0.5)
    rev = st.number_input("Revenue CAGR (%)", 0, 100, 10)
    prof = st.number_input("Profit CAGR (%)", 0, 100, 10)
    pe = st.number_input("P/E Ratio", 0.0, 200.0, 20.0)
    pei = st.number_input("Industry P/E", 0.0, 200.0, 25.0)
    div = st.number_input("Dividend Yield (%)", 0.0, 10.0, 1.0)
    prom = st.number_input("Promoter Holding (%)", 0, 100, 60)
    if st.button("Calculate RJ Score"):
        score = (roe / 2) + (15 if de < 1 else 5) + (rev / 2) + (prof / 2) + (10 if pe < pei else 5) + (div) + (prom / 10)
        st.success(f"RJ Score: {round(score,1)} / 100")
