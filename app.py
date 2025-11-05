# app.py
"""
StockMentor - Rule-based long-term stock analyst (India)
Author: Biswanath Das
- No OpenAI or paid APIs
- Uses yfinance for stock data
- Tabs: Dashboard, Single Stock, Portfolio, Alerts, Watchlist Editor, RJ Score
- Computes ROE%, D/E manually if Yahoo data missing
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import smtplib, os
from datetime import datetime
from email.mime.text import MIMEText

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="StockMentor", layout="wide")
st.title("📈 StockMentor - Rule-based Long-Term Stock Analyst")

# -------------------- HELPERS --------------------
@st.cache_data(ttl=86400)
def fetch_info_and_history(symbol):
    """Fetch Yahoo Finance info + price history"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        return info, hist
    except Exception:
        return {}, pd.DataFrame()

def enrich_financial_ratios(symbol, info):
    """Compute missing ROE and D/E from financials"""
    try:
        ticker = yf.Ticker(symbol)
        bs = ticker.balance_sheet
        fin = ticker.financials

        if bs is not None and not bs.empty and fin is not None and not fin.empty:
            total_equity = bs.loc["Total Stockholder Equity"].iloc[0] if "Total Stockholder Equity" in bs.index else None
            total_liab = bs.loc["Total Liab"].iloc[0] if "Total Liab" in bs.index else None
            net_income = fin.loc["Net Income"].iloc[0] if "Net Income" in fin.index else None

            if total_equity and net_income:
                info["returnOnEquity"] = round((net_income / total_equity) * 100, 2)
            if total_equity and total_liab:
                info["debtToEquity"] = round(total_liab / total_equity, 2)
    except Exception:
        pass
    return info

def load_watchlist():
    """Load symbols from watchlist.csv"""
    if os.path.exists("watchlist.csv"):
        df = pd.read_csv("watchlist.csv")
        possible_cols = [c.lower() for c in df.columns]
        if "symbol" in possible_cols:
            return df[df.columns[possible_cols.index("symbol")]].dropna().tolist()
        elif df.shape[1] == 1:
            return df.iloc[:, 0].dropna().tolist()
    return []

def save_watchlist(symbols):
    """Save updated watchlist"""
    pd.DataFrame({"Symbol": symbols}).to_csv("watchlist.csv", index=False)

def calculate_rj_score(info):
    """Rule-based RJ Score (0–100)"""
    try:
        roe = info.get("returnOnEquity", 0)
        de = info.get("debtToEquity", 0)
        pe = info.get("trailingPE", 0)
        pb = info.get("priceToBook", 0)
        div_yield = info.get("dividendYield", 0)
        mcap = info.get("marketCap", 0)

        score = 0
        if roe and roe > 15: score += 20
        elif roe and roe > 10: score += 10

        if de and de < 0.5: score += 15
        elif de and de < 1: score += 10

        if pe and pe < 20: score += 15
        elif pe and pe < 30: score += 10

        if pb and pb < 3: score += 10

        if div_yield and div_yield > 0.01: score += 10

        if mcap and mcap > 1e11: score += 10  # large cap stability

        return min(score, 100)
    except Exception:
        return 0

def send_email_alert(subject, body, to_email):
    """Send email alert (optional feature)"""
    try:
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        if not smtp_user or not smtp_pass:
            return False

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception:
        return False

# -------------------- TAB SETUP --------------------
tabs = st.tabs(["📊 Dashboard", "🔍 Single Stock", "💼 Portfolio", "📧 Alerts", "📝 Watchlist Editor", "⭐ RJ Score"])

# -------------------- TAB 1: DASHBOARD --------------------
with tabs[0]:
    st.header("Market Dashboard")
    watchlist = load_watchlist()

    if not watchlist:
        st.info("No symbols in watchlist. Add them in 'Watchlist Editor' tab.")
    else:
        data = []
        for sym in watchlist:
            symbol = f"{sym}.NS"
            info, hist = fetch_info_and_history(symbol)
            info = enrich_financial_ratios(symbol, info)

            if info:
                price = info.get("currentPrice", np.nan)
                pe = info.get("trailingPE", np.nan)
                roe = info.get("returnOnEquity", np.nan)
                de = info.get("debtToEquity", np.nan)
                score = calculate_rj_score(info)
                data.append([sym, price, pe, roe, de, score])

        df = pd.DataFrame(data, columns=["Symbol", "Price", "P/E", "ROE%", "D/E", "RJ Score"])
        st.dataframe(df, use_container_width=True)

# -------------------- TAB 2: SINGLE STOCK --------------------
with tabs[1]:
    st.header("Single Stock Analyzer")
    sym = st.text_input("Enter NSE Symbol (e.g., RELIANCE):").strip().upper()

    if st.button("Analyze", type="primary"):
        if not sym:
            st.warning("Please enter a valid symbol (e.g., RELIANCE)")
        else:
            info, hist = fetch_info_and_history(f"{sym}.NS")
            info = enrich_financial_ratios(f"{sym}.NS", info)
            if info:
                score = calculate_rj_score(info)
                st.subheader(f"RJ Score: {score}/100")
                st.write("### Key Financials")
                st.write({
                    "Current Price": info.get("currentPrice"),
                    "P/E": info.get("trailingPE"),
                    "ROE%": info.get("returnOnEquity"),
                    "D/E": info.get("debtToEquity"),
                    "Div. Yield": info.get("dividendYield"),
                })
            else:
                st.error("Failed to fetch stock info.")

# -------------------- TAB 3: PORTFOLIO --------------------
with tabs[2]:
    st.header("Portfolio Tracker")
    if os.path.exists("portfolio.csv"):
        pf = pd.read_csv("portfolio.csv")
        st.dataframe(pf)
    else:
        st.info("No portfolio data found (portfolio.csv).")

# -------------------- TAB 4: ALERTS --------------------
with tabs[3]:
    st.header("Alerts Center")
    st.write("Send yourself an email when a stock meets a rule.")
    email = st.text_input("Alert Email ID")
    sym = st.text_input("Stock Symbol for Alert (e.g., INFY)").upper()
    target_price = st.number_input("Target Buy Price", min_value=0.0)

    if st.button("Check & Send Alert"):
        info, _ = fetch_info_and_history(f"{sym}.NS")
        current_price = info.get("currentPrice", 0)
        if current_price <= target_price:
            if send_email_alert(
                f"Buy Alert: {sym}",
                f"{sym} reached ₹{current_price}, below target ₹{target_price}",
                email,
            ):
                st.success("Email alert sent!")
            else:
                st.warning("Condition met but email not sent (check SMTP config).")
        else:
            st.info(f"{sym} is at ₹{current_price}, above your target ₹{target_price}.")

# -------------------- TAB 5: WATCHLIST EDITOR --------------------
with tabs[4]:
    st.header("Watchlist Editor")
    watchlist = load_watchlist()
    new_symbol = st.text_input("Add Symbol (e.g., TCS)").upper()

    if st.button("Add to Watchlist"):
        if new_symbol and new_symbol not in watchlist:
            watchlist.append(new_symbol)
            save_watchlist(watchlist)
            st.success(f"{new_symbol} added!")
        else:
            st.warning("Symbol already in list or invalid.")

    if watchlist:
        st.write("### Current Watchlist")
        st.dataframe(pd.DataFrame({"Symbol": watchlist}))

        remove = st.text_input("Remove Symbol").upper()
        if st.button("Remove"):
            if remove in watchlist:
                watchlist.remove(remove)
                save_watchlist(watchlist)
                st.success(f"{remove} removed!")
            else:
                st.warning("Symbol not found.")

# -------------------- TAB 6: RJ SCORE --------------------
with tabs[5]:
    st.header("RJ Score Analyzer")
    watchlist = load_watchlist()

    if not watchlist:
        st.info("No stocks in watchlist to analyze.")
    else:
        results = []
        for sym in watchlist:
            symbol = f"{sym}.NS"
            info, _ = fetch_info_and_history(symbol)
            info = enrich_financial_ratios(symbol, info)
            score = calculate_rj_score(info)
            roe = info.get("returnOnEquity", np.nan)
            de = info.get("debtToEquity", np.nan)
            pe = info.get("trailingPE", np.nan)
            pb = info.get("priceToBook", np.nan)
            dy = info.get("dividendYield", np.nan)
            results.append([sym, score, roe, de, pe, pb, dy])

        df = pd.DataFrame(results, columns=["Symbol", "RJ Score", "ROE%", "D/E", "P/E", "P/B", "Div. Yield"])
        st.dataframe(df.sort_values("RJ Score", ascending=False), use_container_width=True)
