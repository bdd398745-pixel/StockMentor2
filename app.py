# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
- No OpenAI / no external LLMs
- Uses yfinance for data (free)
- Loads watchlist.csv (one symbol per line)
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor
- Rule-based scoring, ranking & recommendation
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
        load_watchlist.clear()
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
    forward_pe = safe_get(info, "forwardPE", np.nan)
    trailing_pe = safe_get(info, "trailingPE", np.nan)

    if isinstance(forward_pe, (int, float)) and forward_pe > 0 and forward_pe < 200:
        pe_target = forward_pe
    elif isinstance(trailing_pe, (int, float)) and trailing_pe > 0 and trailing_pe < 200:
        pe_target = max(10.0, trailing_pe * 0.9)
    else:
        pe_target = DEFAULT_PE_TARGET

    if isinstance(eps, (int, float)) and eps > 0:
        fv = eps * pe_target
        return round(float(fv), 2), f"EPSxPE({pe_target:.1f})"

    book = safe_get(info, "bookValue", np.nan)
    if isinstance(book, (int, float)) and book > 0 and isinstance(trailing_pe, (int, float)) and trailing_pe > 0:
        fv = book * trailing_pe
        return round(float(fv), 2), "BVxPE"

    return None, "InsufficientData"

# -------------------------
# Buy/Sell price zones
# -------------------------
def compute_buy_sell(fair_value, mos=0.30):
    if not fair_value or math.isnan(fair_value):
        return None, None
    return round(fair_value * (1 - mos), 2), round(fair_value * (1 + mos/1.5), 2)

# -------------------------
# Rule-based recommendation
# -------------------------
def rule_based_recommendation(info, fair_value, current_price):
    roe = safe_get(info, "returnOnEquity", np.nan)
    if roe and abs(roe) > 1:
        roe /= 100.0
    de = safe_get(info, "debtToEquity", np.nan)
    earnings_growth = safe_get(info, "earningsQuarterlyGrowth", np.nan)
    market_cap = safe_get(info, "marketCap", np.nan)
    pe = safe_get(info, "trailingPE", np.nan)

    underval = None
    if fair_value and current_price and fair_value > 0:
        underval = round(((fair_value - current_price)/fair_value) * 100, 2)

    score = 0
    reasons = []

    # Quality (ROE)
    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 3; reasons.append("High ROE")
        elif roe >= 0.12:
            score += 2; reasons.append("Good ROE")
        elif roe > 0:
            score += 1; reasons.append("Positive ROE")

    # Debt
    if isinstance(de, (int, float)):
        if de <= 0.5:
            score += 2; reasons.append("Low D/E")
        elif de <= 1.5:
            score += 1; reasons.append("Moderate D/E")

    # Growth
    if isinstance(earnings_growth, (int, float)):
        if earnings_growth >= 0.20:
            score += 2; reasons.append("Strong earnings growth")
        elif earnings_growth >= 0.05:
            score += 1; reasons.append("Moderate earnings growth")

    # Valuation
    if isinstance(underval, (int, float)):
        if underval >= 25:
            score += 3; reasons.append("Deep undervaluation")
        elif underval >= 10:
            score += 2; reasons.append("Undervalued")
        elif underval >= 3:
            score += 1; reasons.append("Slight undervaluation")

    rec = "Hold"
    if score >= 7 and underval >= 10:
        rec = "Strong Buy"
    elif score >= 5 and underval >= 5:
        rec = "Buy"
    elif (isinstance(pe, (int, float)) and pe > 80) or (isinstance(roe, (int, float)) and roe < 0):
        rec = "Avoid"

    return {
        "score": score,
        "reasons": reasons,
        "undervaluation_%": underval,
        "recommendation": rec,
        "market_cap": market_cap
    }

# -------------------------
# Email sender
# -------------------------
def send_email_smtp(smtp_host, smtp_port, username, password, sender, recipients, subject, body):
    try:
        if isinstance(recipients, str):
            recipients = [r.strip() for r in recipients.split(",") if r.strip()]
        msg = EmailMessage()
        msg["From"] = sender or username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        if smtp_port == 587:
            server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        return True, "Sent"
    except Exception as ex:
        return False, str(ex)

# -------------------------
# UI Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dashboard", "🔎 Single Stock", "💼 Portfolio", "📣 Alerts", "🧾 Watchlist Editor"])

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
            info, _ = fetch_info_and_history(sym)
            if info.get("error"):
                continue
            ltp = safe_get(info, "currentPrice", np.nan)
            fv, method = estimate_fair_value(info)
            rec = rule_based_recommendation(info, fv, ltp)
            buy, sell = compute_buy_sell(fv)
            cap = rec["market_cap"]
            cap_weight = 2 if cap and cap > 5e11 else (1 if cap and cap > 1e11 else 0)
            rank_score = (rec["score"] * 2) + (rec["undervaluation_%"] or 0)/10 + cap_weight
            rows.append({
                "Symbol": sym,
                "LTP": ltp,
                "Fair Value": fv,
                "Underv%": rec["undervaluation_%"],
                "Buy Below": buy,
                "Sell Above": sell,
                "Rec": rec["recommendation"],
                "Score": rec["score"],
                "RankScore": round(rank_score, 2),
                "Reasons": "; ".join(rec["reasons"])
            })
            progress.progress(int(((i+1)/len(watchlist))*100))
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="RankScore", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        st.success("✅ Ranked by multi-factor score (Quality + Valuation + Size)")

# -------------------------
# Single Stock, Portfolio, Alerts, Watchlist (same as before)
# -------------------------
# (You can retain the same code blocks from your previous version here unchanged)
